"""Focused subagents and autonomous teammate threads."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .client import ModelGateway, block_type, extract_text, has_tool_use
from .config import Settings
from .messaging import MessageBus, ProtocolManager
from .tools import ToolDispatcher, ToolRegistry, _object_schema
from .workspace import FileTools, TaskRepository, WorktreeService


def _block_value(block: Any, name: str, default: Any = None) -> Any:
    return (
        block.get(name, default)
        if isinstance(block, dict)
        else getattr(block, name, default)
    )


@dataclass(slots=True)
class _TeammateHandle:
    thread: threading.Thread
    stop: threading.Event


class AgentService:
    NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
    IDLE_POLL_INTERVAL = 1.0
    IDLE_TIMEOUT = 60.0

    def __init__(
        self,
        settings: Settings,
        gateway_provider: Callable[[], ModelGateway],
        tasks: TaskRepository,
        worktrees: WorktreeService,
        bus: MessageBus,
        protocols: ProtocolManager,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        printer: Callable[[str], None] = print,
    ):
        self.settings = settings
        self.gateway_provider = gateway_provider
        self.tasks = tasks
        self.worktrees = worktrees
        self.bus = bus
        self.protocols = protocols
        self.registry = registry
        self.dispatcher = dispatcher
        self.printer = printer
        self._teammates: dict[str, _TeammateHandle] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()

    @property
    def active_names(self) -> list[str]:
        with self._lock:
            return list(self._teammates)

    def _definitions_for(self, names: set[str]) -> list[dict]:
        return [
            definition
            for definition in self.registry.definitions
            if definition["name"] in names
        ]

    def spawn_subagent(self, description: str) -> str:
        messages: list[dict] = [{"role": "user", "content": description}]
        names = {"bash", "read_file", "write_file", "edit_file", "glob"}
        tools = self._definitions_for(names)
        system = (
            f"You are a coding subagent at {self.settings.workdir}. "
            "Complete the task, then return a concise final summary. "
            "Do not spawn more agents."
        )
        for _ in range(30):
            try:
                response = self.gateway_provider().create_message(
                    model=self.settings.model_id,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=8_000,
                )
            except Exception as exc:
                return f"Subagent error: {type(exc).__name__}: {exc}"
            messages.append(
                {"role": "assistant", "content": response.content}
            )
            if not has_tool_use(response.content):
                break
            results: list[dict] = []
            for block in response.content:
                if block_type(block) != "tool_use":
                    continue
                name = str(_block_value(block, "name", ""))
                output = self.dispatcher.execute(
                    name,
                    _block_value(block, "input", {}),
                    str(_block_value(block, "id", "")),
                    allow_background=False,
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(_block_value(block, "id", "")),
                        "content": output,
                    }
                )
            messages.append({"role": "user", "content": results})
        for message in reversed(messages):
            if message.get("role") == "assistant":
                text = extract_text(message.get("content"))
                if text:
                    return text
        return "Subagent finished without a text summary."

    def spawn_teammate(self, name: str, role: str, prompt: str) -> str:
        if not self.NAME_PATTERN.fullmatch(name):
            return (
                f"Invalid teammate name '{name}': use 1-64 letters, "
                "digits, dots, underscores, or dashes"
            )
        with self._lock:
            if self._closed.is_set():
                return "Agent service is closed"
            if name in self._teammates:
                return f"Teammate '{name}' already exists"
            stop = threading.Event()
            thread = threading.Thread(
                target=self._run_teammate,
                args=(name, role, prompt, stop),
                name=f"mini-claude-code-teammate-{name}",
                daemon=True,
            )
            self._teammates[name] = _TeammateHandle(thread, stop)
            thread.start()
        self.printer(
            f"  \033[35m[teammate] spawned: {name} ({role})\033[0m"
        )
        return f"Spawned teammate '{name}' ({role})"

    def _list_tasks(self) -> str:
        tasks = self.tasks.list()
        if not tasks:
            return "No tasks."
        return "\n".join(
            f"  {task.id}: {task.subject} [{task.status}]"
            + (f" (wt:{task.worktree})" if task.worktree else "")
            for task in tasks
        )

    def _idle_poll(
        self,
        agent_name: str,
        messages: list,
        stop: threading.Event,
        worktree_context: dict,
    ) -> str:
        elapsed = 0.0
        while elapsed < self.IDLE_TIMEOUT:
            if self._closed.is_set() or stop.wait(self.IDLE_POLL_INTERVAL):
                return "shutdown"
            elapsed += self.IDLE_POLL_INTERVAL
            inbox = self.bus.read_inbox(agent_name)
            if inbox:
                for message in inbox:
                    if message.get("type") == "shutdown_request":
                        request_id = message.get("metadata", {}).get(
                            "request_id", ""
                        )
                        self.bus.send(
                            agent_name,
                            "lead",
                            "Shutting down.",
                            "shutdown_response",
                            {
                                "request_id": request_id,
                                "approve": True,
                            },
                        )
                        return "shutdown"
                messages.append(
                    {
                        "role": "user",
                        "content": "<inbox>"
                        + json.dumps(inbox, ensure_ascii=False)
                        + "</inbox>",
                    }
                )
                return "work"
            unclaimed = self.tasks.scan_unclaimed()
            if unclaimed:
                task = unclaimed[0]
                result = self.tasks.claim(task.id, owner=agent_name)
                if "Claimed" in result:
                    worktree_context["path"] = (
                        str(self.settings.worktrees_dir / task.worktree)
                        if task.worktree
                        else None
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[Autonomous task]\n{result}\n"
                                f"{task.description}"
                            ),
                        }
                    )
                    return "work"
        return "timeout"

    def _handle_protocol_message(
        self,
        name: str,
        message: dict,
        messages: list,
        protocol_context: dict,
    ) -> bool:
        message_type = message.get("type", "message")
        metadata = message.get("metadata", {})
        request_id = metadata.get("request_id", "")
        if message_type == "shutdown_request":
            self.bus.send(
                name,
                "lead",
                "Shutting down.",
                "shutdown_response",
                {"request_id": request_id, "approve": True},
            )
            return True
        if message_type == "plan_approval_response":
            approved = bool(metadata.get("approve", False))
            if request_id == protocol_context["waiting_plan"]:
                protocol_context["waiting_plan"] = None
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[Plan approved]"
                        if approved
                        else f"[Plan rejected] {message.get('content', '')}"
                    ),
                }
            )
        return False

    def _run_teammate(
        self,
        name: str,
        role: str,
        prompt: str,
        stop: threading.Event,
    ) -> None:
        protocol_context: dict[str, str | None] = {"waiting_plan": None}
        worktree_context: dict[str, str | None] = {"path": None}
        messages: list[dict] = [{"role": "user", "content": prompt}]
        system = (
            f"You are '{name}', a {role}. Use tools to complete tasks. "
            "If a task has a worktree, work in that directory."
        )
        teammate_names = {
            "bash",
            "read_file",
            "write_file",
            "send_message",
            "list_tasks",
            "claim_task",
            "complete_task",
        }
        tools = self._definitions_for(teammate_names)
        tools.append(
            {
                "name": "submit_plan",
                "description": "Submit a plan for Lead approval.",
                "input_schema": _object_schema(
                    {"plan": {"type": "string"}}, ["plan"]
                ),
            }
        )

        def current_files() -> FileTools:
            path = worktree_context.get("path")
            return FileTools(
                Path(path) if path else self.settings.workdir
            )

        def claim_task(task_id: str) -> str:
            try:
                result = self.tasks.claim(task_id, owner=name)
                if "Claimed" in result:
                    task = self.tasks.load(task_id)
                    worktree_context["path"] = (
                        str(self.settings.worktrees_dir / task.worktree)
                        if task.worktree
                        else None
                    )
                return result
            except FileNotFoundError:
                return f"Error: task {task_id} not found"

        def complete_task(task_id: str) -> str:
            try:
                result = self.tasks.complete(task_id)
                worktree_context["path"] = None
                return result
            except FileNotFoundError:
                return f"Error: task {task_id} not found"

        def overrides() -> dict[str, Callable[..., Any]]:
            files = current_files()

            def send_message(to: str, content: str) -> str:
                self.bus.send(name, to, content)
                return "Sent"

            return {
                "bash": files.run_bash,
                "read_file": files.read_file,
                "write_file": files.write_file,
                "send_message": send_message,
                "submit_plan": lambda plan: self.protocols.submit_plan(
                    name, plan
                ),
                "list_tasks": self._list_tasks,
                "claim_task": claim_task,
                "complete_task": complete_task,
            }

        try:
            while not (self._closed.is_set() or stop.is_set()):
                if len(messages) <= 3:
                    messages.insert(
                        0,
                        {
                            "role": "user",
                            "content": (
                                f"<identity>You are '{name}', role: {role}. "
                                "Continue your work.</identity>"
                            ),
                        },
                    )
                should_shutdown = False
                for _ in range(10):
                    if self._closed.is_set() or stop.is_set():
                        should_shutdown = True
                        break
                    inbox = self.bus.read_inbox(name)
                    for message in inbox:
                        if self._handle_protocol_message(
                            name,
                            message,
                            messages,
                            protocol_context,
                        ):
                            should_shutdown = True
                            break
                    if should_shutdown:
                        break
                    if protocol_context["waiting_plan"]:
                        stop.wait(self.IDLE_POLL_INTERVAL)
                        continue
                    ordinary = [
                        message
                        for message in inbox
                        if message.get("type") == "message"
                    ]
                    if ordinary:
                        messages.append(
                            {
                                "role": "user",
                                "content": "<inbox>"
                                + json.dumps(
                                    ordinary, ensure_ascii=False
                                )
                                + "</inbox>",
                            }
                        )
                    try:
                        response = self.gateway_provider().create_message(
                            model=self.settings.model_id,
                            system=system,
                            messages=messages[-20:],
                            tools=tools,
                            max_tokens=8_000,
                        )
                    except Exception as exc:
                        self.printer(
                            f"  \033[31m[teammate:{name}] "
                            f"{type(exc).__name__}: {exc}\033[0m"
                        )
                        break
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                        }
                    )
                    if not has_tool_use(response.content):
                        break
                    results: list[dict] = []
                    for block in response.content:
                        if block_type(block) != "tool_use":
                            continue
                        tool_name = str(
                            _block_value(block, "name", "")
                        )
                        output = self.dispatcher.execute(
                            tool_name,
                            _block_value(block, "input", {}),
                            str(_block_value(block, "id", "")),
                            allow_background=False,
                            handlers_override=overrides(),
                        )
                        if tool_name == "submit_plan":
                            match = re.search(r"\((req_\d+)\)", output)
                            protocol_context["waiting_plan"] = (
                                match.group(1) if match else output
                            )
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": str(
                                    _block_value(block, "id", "")
                                ),
                                "content": output,
                            }
                        )
                        if protocol_context["waiting_plan"]:
                            break
                    messages.append({"role": "user", "content": results})
                    if protocol_context["waiting_plan"]:
                        break
                if should_shutdown:
                    break
                if protocol_context["waiting_plan"]:
                    continue
                idle_result = self._idle_poll(
                    name, messages, stop, worktree_context
                )
                if idle_result in ("shutdown", "timeout"):
                    break
        finally:
            summary = "Done."
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    text = extract_text(message.get("content"))
                    if text:
                        summary = text
                        break
            self.bus.send(name, "lead", summary, "message")
            with self._lock:
                self._teammates.pop(name, None)
            self.printer(
                f"  \033[35m[teammate] stopped: {name}\033[0m"
            )

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            handles = list(self._teammates.values())
        for handle in handles:
            handle.stop.set()
        for handle in handles:
            if handle.thread is not threading.current_thread():
                handle.thread.join(timeout=2)
