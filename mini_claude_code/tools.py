"""Tool schemas, service adapters, and the unified dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .config import Settings
from .hooks import HookPipeline
from .mcp import MCPRegistry
from .messaging import MessageBus, ProtocolManager
from .models import ToolCall
from .rag.service import RagService
from .rag.tool import definitions_for as rag_definitions_for
from .rag.tool import handlers_for as rag_handlers_for
from .scheduling import BackgroundTaskManager, CronScheduler
from .skills import SkillRegistry
from .workspace import (
    FileTools,
    TaskRepository,
    TodoStore,
    WorktreeService,
)

if TYPE_CHECKING:
    from .agents import AgentService


def _object_schema(
    properties: dict | None = None,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
    }


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": _object_schema(
            {
                "command": {"type": "string"},
                "run_in_background": {"type": "boolean"},
            },
            ["command"],
        ),
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": _object_schema(
            {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            ["path"],
        ),
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": _object_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            ["path", "content"],
        ),
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": _object_schema(
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            ["path", "old_text", "new_text"],
        ),
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": _object_schema(
            {"pattern": {"type": "string"}}, ["pattern"]
        ),
    },
    {
        "name": "todo_write",
        "description": "Create and manage a task list for the current session.",
        "input_schema": _object_schema(
            {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            ["todos"],
        ),
    },
    {
        "name": "task",
        "description": "Launch a focused subagent. Returns only its final summary.",
        "input_schema": _object_schema(
            {"description": {"type": "string"}}, ["description"]
        ),
    },
    {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "input_schema": _object_schema(
            {"name": {"type": "string"}}, ["name"]
        ),
    },
    {
        "name": "compact",
        "description": (
            "Summarize earlier conversation and continue with compacted context."
        ),
        "input_schema": _object_schema(
            {"focus": {"type": "string"}}
        ),
    },
    {
        "name": "create_task",
        "description": "Create a task.",
        "input_schema": _object_schema(
            {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "blockedBy": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            ["subject"],
        ),
    },
    {
        "name": "list_tasks",
        "description": "List all tasks.",
        "input_schema": _object_schema(),
    },
    {
        "name": "get_task",
        "description": "Get full task details.",
        "input_schema": _object_schema(
            {"task_id": {"type": "string"}}, ["task_id"]
        ),
    },
    {
        "name": "claim_task",
        "description": "Claim a pending task.",
        "input_schema": _object_schema(
            {"task_id": {"type": "string"}}, ["task_id"]
        ),
    },
    {
        "name": "complete_task",
        "description": "Complete an in-progress task.",
        "input_schema": _object_schema(
            {"task_id": {"type": "string"}}, ["task_id"]
        ),
    },
    {
        "name": "schedule_cron",
        "description": (
            "Schedule a cron job. cron is 5-field: min hour dom month dow. "
            "For one-shot reminders, compute the target minute and set "
            "recurring=false."
        ),
        "input_schema": _object_schema(
            {
                "cron": {"type": "string"},
                "prompt": {"type": "string"},
                "recurring": {"type": "boolean"},
                "durable": {"type": "boolean"},
            },
            ["cron", "prompt"],
        ),
    },
    {
        "name": "list_crons",
        "description": "List registered cron jobs.",
        "input_schema": _object_schema(),
    },
    {
        "name": "cancel_cron",
        "description": "Cancel a cron job by ID.",
        "input_schema": _object_schema(
            {"job_id": {"type": "string"}}, ["job_id"]
        ),
    },
    {
        "name": "spawn_teammate",
        "description": "Spawn an autonomous teammate.",
        "input_schema": _object_schema(
            {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "prompt": {"type": "string"},
            },
            ["name", "role", "prompt"],
        ),
    },
    {
        "name": "send_message",
        "description": "Send message to a teammate.",
        "input_schema": _object_schema(
            {
                "to": {"type": "string"},
                "content": {"type": "string"},
            },
            ["to", "content"],
        ),
    },
    {
        "name": "check_inbox",
        "description": "Check inbox for messages and protocol responses.",
        "input_schema": _object_schema(),
    },
    {
        "name": "request_shutdown",
        "description": "Request a teammate to shut down.",
        "input_schema": _object_schema(
            {"teammate": {"type": "string"}}, ["teammate"]
        ),
    },
    {
        "name": "request_plan",
        "description": "Ask a teammate to submit a plan.",
        "input_schema": _object_schema(
            {
                "teammate": {"type": "string"},
                "task": {"type": "string"},
            },
            ["teammate", "task"],
        ),
    },
    {
        "name": "review_plan",
        "description": "Approve or reject a submitted plan.",
        "input_schema": _object_schema(
            {
                "request_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "feedback": {"type": "string"},
            },
            ["request_id", "approve"],
        ),
    },
    {
        "name": "create_worktree",
        "description": "Create an isolated git worktree.",
        "input_schema": _object_schema(
            {
                "name": {"type": "string"},
                "task_id": {"type": "string"},
            },
            ["name"],
        ),
    },
    {
        "name": "remove_worktree",
        "description": "Remove a worktree. Refuses if changes exist.",
        "input_schema": _object_schema(
            {
                "name": {"type": "string"},
                "discard_changes": {"type": "boolean"},
            },
            ["name"],
        ),
    },
    {
        "name": "keep_worktree",
        "description": "Keep a worktree for manual review.",
        "input_schema": _object_schema(
            {"name": {"type": "string"}}, ["name"]
        ),
    },
    {
        "name": "connect_mcp",
        "description": (
            "Connect to an MCP server (docs, deploy) and discover tools."
        ),
        "input_schema": _object_schema(
            {"name": {"type": "string"}}, ["name"]
        ),
    },
]


class ToolRegistry:
    """Build model-visible schemas and executable handlers from services."""

    def __init__(
        self,
        settings: Settings,
        tasks: TaskRepository,
        files: FileTools,
        worktrees: WorktreeService,
        todos: TodoStore,
        skills: SkillRegistry,
        bus: MessageBus,
        protocols: ProtocolManager,
        cron: CronScheduler,
        mcp: MCPRegistry,
        printer: Callable[[str], None] = print,
        rag: RagService | None = None,
    ):
        self.settings = settings
        self.tasks = tasks
        self.files = files
        self.worktrees = worktrees
        self.todos = todos
        self.skills = skills
        self.bus = bus
        self.protocols = protocols
        self.cron = cron
        self.mcp = mcp
        self.printer = printer
        self.rag = rag
        self._agents: AgentService | None = None

    def set_agent_service(self, agents: "AgentService") -> None:
        self._agents = agents

    @property
    def definitions(self) -> list[dict]:
        return [*TOOL_DEFINITIONS, *rag_definitions_for(self.rag)]

    def tool_names(self) -> list[str]:
        definitions, _ = self.build()
        return [str(item["name"]) for item in definitions]

    def _create_task(
        self,
        subject: str,
        description: str = "",
        blockedBy: list[str] | None = None,
    ) -> str:
        task = self.tasks.create(subject, description, blockedBy)
        dependencies = (
            f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
        )
        self.printer(
            f"  \033[34m[create] {task.subject}{dependencies}\033[0m"
        )
        return f"Created {task.id}: {task.subject}{dependencies}"

    def _list_tasks(self) -> str:
        tasks = self.tasks.list()
        if not tasks:
            return "No tasks."
        return "\n".join(
            f"  {task.id}: {task.subject} [{task.status}]"
            + (f" (wt:{task.worktree})" if task.worktree else "")
            for task in tasks
        )

    def _get_task(self, task_id: str) -> str:
        try:
            return self.tasks.get_json(task_id)
        except FileNotFoundError:
            return f"Error: task {task_id} not found"

    def _claim_task(self, task_id: str) -> str:
        try:
            return self.tasks.claim(task_id, owner="agent")
        except FileNotFoundError:
            return f"Error: task {task_id} not found"

    def _complete_task(self, task_id: str) -> str:
        try:
            return self.tasks.complete(task_id)
        except FileNotFoundError:
            return f"Error: task {task_id} not found"

    def _spawn_subagent(self, description: str) -> str:
        if not self._agents:
            return "Error: agent service is not ready"
        return self._agents.spawn_subagent(description)

    def _spawn_teammate(self, name: str, role: str, prompt: str) -> str:
        if not self._agents:
            return "Error: agent service is not ready"
        return self._agents.spawn_teammate(name, role, prompt)

    def _send_message(self, to: str, content: str) -> str:
        self.bus.send("lead", to, content)
        return f"Sent to {to}"

    def _check_inbox(self) -> str:
        messages = self.protocols.consume_lead_inbox(route_protocol=True)
        if not messages:
            return "(inbox empty)"
        lines: list[str] = []
        for message in messages:
            metadata = message.get("metadata", {})
            request_id = metadata.get("request_id", "")
            message_type = message.get("type", "message")
            tag = (
                f" [{message_type} req:{request_id}]"
                if request_id
                else f" [{message_type}]"
            )
            lines.append(
                f"  [{message.get('from', '?')}]{tag} "
                f"{str(message.get('content', ''))[:200]}"
            )
        return "\n".join(lines)

    def _handlers(self) -> dict[str, Callable[..., Any]]:
        handlers: dict[str, Callable[..., Any]] = {
            "bash": self.files.run_bash,
            "read_file": self.files.read_file,
            "write_file": self.files.write_file,
            "edit_file": self.files.edit_file,
            "glob": self.files.glob,
            "todo_write": self.todos.write,
            "task": self._spawn_subagent,
            "load_skill": self.skills.load,
            "create_task": self._create_task,
            "list_tasks": self._list_tasks,
            "get_task": self._get_task,
            "claim_task": self._claim_task,
            "complete_task": self._complete_task,
            "schedule_cron": self.cron.schedule_text,
            "list_crons": self.cron.list_text,
            "cancel_cron": self.cron.cancel,
            "spawn_teammate": self._spawn_teammate,
            "send_message": self._send_message,
            "check_inbox": self._check_inbox,
            "request_shutdown": self.protocols.request_shutdown,
            "request_plan": self.protocols.request_plan,
            "review_plan": self.protocols.review_plan,
            "create_worktree": self.worktrees.create,
            "remove_worktree": self.worktrees.remove,
            "keep_worktree": self.worktrees.keep,
            "connect_mcp": self.mcp.connect,
        }
        handlers.update(rag_handlers_for(self.rag))
        return handlers

    def build(self) -> tuple[list[dict], dict[str, Callable[..., Any]]]:
        definitions = self.definitions
        handlers = self._handlers()
        mcp_definitions, mcp_handlers = self.mcp.tool_pool()
        definitions.extend(mcp_definitions)
        handlers.update(mcp_handlers)
        return definitions, handlers


class ToolDispatcher:
    """The only path through which main, sub-, and teammate agents run tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        hooks: HookPipeline,
        background: BackgroundTaskManager,
    ):
        self.registry = registry
        self.hooks = hooks
        self.background = background

    def execute(
        self,
        name: str,
        tool_input: dict | None,
        tool_use_id: str = "",
        *,
        allow_background: bool = True,
        handlers_override: dict[str, Callable[..., Any]] | None = None,
    ) -> str:
        arguments = dict(tool_input or {})
        call = ToolCall(name=name, input=arguments, id=tool_use_id)
        blocked = self.hooks.trigger("PreToolUse", call)
        if blocked is not None:
            return str(blocked)

        _, handlers = self.registry.build()
        if handlers_override:
            handlers.update(handlers_override)
        handler = handlers.get(name)
        if not handler:
            return f"Unknown: {name}"

        def invoke() -> str:
            try:
                return str(handler(**arguments))
            except TypeError as exc:
                return f"Error: {exc}"
            except Exception as exc:
                return f"Error: {type(exc).__name__}: {exc}"

        if (
            allow_background
            and self.background.should_run(name, arguments)
        ):
            task_id = self.background.start(
                tool_use_id,
                name,
                arguments,
                invoke,
                after=lambda output: self.hooks.trigger(
                    "PostToolUse", call, output
                ),
            )
            return (
                f"[Background task {task_id} started] Result will arrive "
                "as a task_notification."
            )

        output = invoke()
        self.hooks.trigger("PostToolUse", call, output)
        return output
