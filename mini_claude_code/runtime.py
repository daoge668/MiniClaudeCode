"""Application composition root and the main agent loop."""

from __future__ import annotations

import threading
from typing import Any, Callable

from .agents import AgentService
from .client import (
    AnthropicGateway,
    ModelGateway,
    RetryController,
    block_type,
    extract_text,
    has_tool_use,
    is_prompt_too_long_error,
)
from .config import Settings
from .context import ContextManager
from .hooks import HookPipeline, register_default_hooks
from .mcp import MCPRegistry
from .messaging import MessageBus, ProtocolManager
from .models import RecoveryState
from .scheduling import BackgroundTaskManager, CronScheduler
from .skills import PromptAssembler, SkillRegistry
from .tools import ToolDispatcher, ToolRegistry
from .workspace import (
    FileTools,
    TaskRepository,
    TodoStore,
    WorktreeService,
)


def _block_value(block: Any, name: str, default: Any = None) -> Any:
    return (
        block.get(name, default)
        if isinstance(block, dict)
        else getattr(block, name, default)
    )


class Application:
    """Own all runtime services and their lifecycle."""

    CONTINUATION_PROMPT = (
        "Continue from the previous response. Do not repeat completed work."
    )

    def __init__(
        self,
        settings: Settings,
        *,
        gateway: ModelGateway | None = None,
        printer: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ):
        self.settings = settings
        self._provided_gateway = gateway
        self.printer = printer
        self.input_fn = input_fn
        self.gateway: ModelGateway | None = None

        self.history: list[dict] = []
        self.context: dict = {}
        self._agent_lock = threading.RLock()
        self._stop = threading.Event()
        self._started = False
        self._closed = False
        self._cron_consumer: threading.Thread | None = None
        self._rounds_since_todo = 0

        # Services are assigned during start(), keeping construction side-effect
        # free and making import/lifecycle tests straightforward.
        self.tasks: TaskRepository
        self.files: FileTools
        self.todos: TodoStore
        self.worktrees: WorktreeService
        self.skills: SkillRegistry
        self.bus: MessageBus
        self.protocols: ProtocolManager
        self.hooks: HookPipeline
        self.background: BackgroundTaskManager
        self.cron: CronScheduler
        self.mcp: MCPRegistry
        self.registry: ToolRegistry
        self.dispatcher: ToolDispatcher
        self.agents: AgentService
        self.context_manager: ContextManager
        self.prompt_assembler: PromptAssembler
        self.retry: RetryController

    @property
    def is_started(self) -> bool:
        return self._started and not self._closed

    def _gateway(self) -> ModelGateway:
        if self.gateway is None:
            raise RuntimeError("Application has not been started")
        return self.gateway

    def start(self) -> "Application":
        if self._closed:
            raise RuntimeError("Application is already closed")
        if self._started:
            return self

        self.settings.ensure_runtime_dirs()
        self.gateway = self._provided_gateway or AnthropicGateway.from_settings(
            self.settings
        )

        self.tasks = TaskRepository(
            self.settings.tasks_dir, printer=self.printer
        )
        self.files = FileTools(self.settings.workdir)
        self.todos = TodoStore(printer=self.printer)
        self.worktrees = WorktreeService(
            self.settings, self.tasks, printer=self.printer
        )
        self.skills = SkillRegistry(self.settings.skills_dir)
        self.bus = MessageBus(
            self.settings.mailbox_dir, printer=self.printer
        )
        self.protocols = ProtocolManager(self.bus)
        self.hooks = HookPipeline()
        self.background = BackgroundTaskManager(
            self.settings.tool_results_dir, printer=self.printer
        )
        self.cron = CronScheduler(
            self.settings.durable_cron_path, printer=self.printer
        )
        self.mcp = MCPRegistry(printer=self.printer)

        self.tasks.start()
        self.worktrees.start()
        self.bus.start()
        self.skills.scan()
        register_default_hooks(
            self.hooks,
            self.files,
            str(self.settings.workdir),
            input_fn=self.input_fn,
            printer=self.printer,
        )

        self.registry = ToolRegistry(
            self.settings,
            self.tasks,
            self.files,
            self.worktrees,
            self.todos,
            self.skills,
            self.bus,
            self.protocols,
            self.cron,
            self.mcp,
            printer=self.printer,
        )
        self.dispatcher = ToolDispatcher(
            self.registry, self.hooks, self.background
        )
        self.agents = AgentService(
            self.settings,
            self._gateway,
            self.tasks,
            self.worktrees,
            self.bus,
            self.protocols,
            self.registry,
            self.dispatcher,
            printer=self.printer,
        )
        self.registry.set_agent_service(self.agents)
        self.context_manager = ContextManager(
            self.settings, self._gateway, printer=self.printer
        )
        self.prompt_assembler = PromptAssembler(
            self.settings,
            self.skills,
            self.registry.tool_names,
            lambda: self.mcp.names,
        )
        self.retry = RetryController(
            self.settings, printer=self.printer
        )

        self.context = self._updated_context()
        self._stop.clear()
        self.cron.start()
        self._cron_consumer = threading.Thread(
            target=self._cron_autorun_loop,
            name="mini-claude-code-cron-consumer",
            daemon=True,
        )
        self._cron_consumer.start()
        self._started = True
        return self

    def _updated_context(self) -> dict:
        return self.context_manager.update_context(
            self.mcp.names, self.agents.active_names
        )

    def _inject_background_notifications(self) -> None:
        notes = self.background.collect_notifications()
        if notes:
            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": note} for note in notes
                    ],
                }
            )

    def _build_user_content(self, results: list[dict]) -> list[dict]:
        content = list(results)
        content.extend(
            {"type": "text", "text": note}
            for note in self.background.collect_notifications()
        )
        return content

    def _call_llm(
        self,
        state: RecoveryState,
        max_tokens: int,
    ) -> Any:
        tools, _ = self.registry.build()
        system = self.prompt_assembler.assemble(self.context)
        return self.retry.run(
            lambda: self._gateway().create_message(
                model=state.current_model,
                system=system,
                messages=self.history,
                tools=tools,
                max_tokens=max_tokens,
            ),
            state,
        )

    def _agent_loop(self) -> None:
        state = RecoveryState(current_model=self.settings.model_id)
        max_tokens = self.settings.default_max_tokens

        while not self._stop.is_set():
            self._inject_background_notifications()
            if self._rounds_since_todo >= 3:
                self.history.append(
                    {
                        "role": "user",
                        "content": "<reminder>Update your todos.</reminder>",
                    }
                )
                self._rounds_since_todo = 0

            self.context_manager.prepare(self.history)
            self.context = self._updated_context()

            try:
                response = self._call_llm(state, max_tokens)
            except Exception as exc:
                if (
                    is_prompt_too_long_error(exc)
                    and not state.has_attempted_reactive_compact
                ):
                    self.history[:] = (
                        self.context_manager.reactive_compact(self.history)
                    )
                    state.has_attempted_reactive_compact = True
                    continue
                self.history.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"[Error] {type(exc).__name__}: {exc}"
                                ),
                            }
                        ],
                    }
                )
                return

            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "max_tokens":
                if not state.has_escalated:
                    max_tokens = self.settings.escalated_max_tokens
                    state.has_escalated = True
                    self.printer(
                        f"  \033[33m[max_tokens] retry with "
                        f"{max_tokens}\033[0m"
                    )
                    continue
                self.history.append(
                    {"role": "assistant", "content": response.content}
                )
                if (
                    state.recovery_count
                    < self.settings.max_recovery_retries
                ):
                    self.history.append(
                        {
                            "role": "user",
                            "content": self.CONTINUATION_PROMPT,
                        }
                    )
                    state.recovery_count += 1
                    continue
                return

            max_tokens = self.settings.default_max_tokens
            state.has_escalated = False
            self.history.append(
                {"role": "assistant", "content": response.content}
            )
            if not has_tool_use(response.content):
                self.hooks.trigger("Stop", self.history)
                return

            results: list[dict] = []
            compacted = False
            for block in response.content:
                if block_type(block) != "tool_use":
                    continue
                name = str(_block_value(block, "name", ""))
                tool_use_id = str(_block_value(block, "id", ""))
                tool_input = _block_value(block, "input", {})
                self.printer(f"\033[36m> {name}\033[0m")

                if name == "compact":
                    self.history[:] = (
                        self.context_manager.compact_history(self.history)
                    )
                    self.history.append(
                        {
                            "role": "user",
                            "content": (
                                "[Compacted. Continue with summarized context.]"
                            ),
                        }
                    )
                    compacted = True
                    break

                output = self.dispatcher.execute(
                    name,
                    tool_input if isinstance(tool_input, dict) else {},
                    tool_use_id,
                )
                self.printer(output[:300])
                if name == "todo_write":
                    self._rounds_since_todo = 0
                else:
                    self._rounds_since_todo += 1
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": output,
                    }
                )

            if compacted:
                continue
            self.history.append(
                {
                    "role": "user",
                    "content": self._build_user_content(results),
                }
            )

    @staticmethod
    def _assistant_texts(messages: list[dict]) -> list[str]:
        texts: list[str] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            text = extract_text(message.get("content"))
            if text:
                texts.append(text)
        return texts

    def _append_lead_inbox(self) -> None:
        inbox = self.protocols.consume_lead_inbox(route_protocol=True)
        if not inbox:
            return

        def label(message: dict) -> str:
            request_id = message.get("metadata", {}).get(
                "request_id", ""
            )
            suffix = f" req:{request_id}" if request_id else ""
            return f"{message.get('type', 'message')}{suffix}"

        text = "\n".join(
            f"From {message.get('from', '?')} [{label(message)}]: "
            f"{str(message.get('content', ''))[:200]}"
            for message in inbox
        )
        self.history.append(
            {"role": "user", "content": f"[Inbox]\n{text}"}
        )

    def run(self, query: str) -> list[str]:
        """Run one user turn and return newly produced assistant text."""
        if not self.is_started:
            self.start()
        self.hooks.trigger("UserPromptSubmit", query)
        with self._agent_lock:
            turn_start = len(self.history)
            self.history.append({"role": "user", "content": query})
            self._agent_loop()
            self.context = self._updated_context()
            texts = self._assistant_texts(self.history[turn_start:])
            self._append_lead_inbox()
            return texts

    run_turn = run

    def _cron_autorun_loop(self) -> None:
        while not self._stop.is_set():
            jobs = self.cron.wait_for_jobs(timeout=0.5)
            if not jobs:
                continue
            with self._agent_lock:
                turn_start = len(self.history)
                for job in jobs:
                    self.history.append(
                        {
                            "role": "user",
                            "content": f"[Scheduled] {job.prompt}",
                        }
                    )
                    self.printer(
                        f"  \033[35m[cron auto] "
                        f"{job.prompt[:60]}\033[0m"
                    )
                self._agent_loop()
                self.context = self._updated_context()
                for text in self._assistant_texts(
                    self.history[turn_start:]
                ):
                    self.printer(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._started:
            self.cron.close()
            self.agents.close()
            self.background.close()
            consumer = self._cron_consumer
            if consumer and consumer is not threading.current_thread():
                consumer.join(timeout=2)
        self._started = False

    def __enter__(self) -> "Application":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
