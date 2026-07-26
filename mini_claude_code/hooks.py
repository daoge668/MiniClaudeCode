"""Tool permission and lifecycle hooks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .client import block_type
from .models import ToolCall
from .workspace import FileTools


class HookPipeline:
    def __init__(self):
        self._hooks: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def register(self, event: str, callback: Callable[..., Any]) -> None:
        self._hooks[event].append(callback)

    def trigger(self, event: str, *args: Any) -> Any:
        for callback in self._hooks.get(event, []):
            result = callback(*args)
            if result is not None:
                return result
        return None


def _call_name(block: Any) -> str:
    return (
        str(block.get("name", ""))
        if isinstance(block, dict)
        else str(getattr(block, "name", ""))
    )


def _call_input(block: Any) -> dict:
    value = (
        block.get("input", {})
        if isinstance(block, dict)
        else getattr(block, "input", {})
    )
    return value if isinstance(value, dict) else {}


class PermissionPolicy:
    DENY_LIST = ("rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=")
    DESTRUCTIVE = ("rm ", "> /etc/", "chmod 777")

    def __init__(
        self,
        files: FileTools,
        input_fn: Callable[[str], str] = input,
        printer: Callable[[str], None] = print,
    ):
        self.files = files
        self.input_fn = input_fn
        self.printer = printer

    def __call__(self, block: Any) -> str | None:
        name = _call_name(block)
        tool_input = _call_input(block)
        if name == "bash":
            command = str(tool_input.get("command", ""))
            for pattern in self.DENY_LIST:
                if pattern in command:
                    return (
                        f"Permission denied: '{pattern}' is on the deny list"
                    )
            if any(token in command for token in self.DESTRUCTIVE):
                self.printer("\n\033[33m[permission] destructive command\033[0m")
                self.printer(f"  {command}")
                choice = self.input_fn("  Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
        if name in ("write_file", "edit_file"):
            path = str(tool_input.get("path", ""))
            try:
                self.files.safe_path(path)
            except Exception:
                return f"Permission denied: path escapes workspace: {path}"
        if name.startswith("mcp__") and "deploy" in name:
            self.printer(
                "\n\033[33m[permission] MCP destructive-looking tool: "
                f"{name}\033[0m"
            )
            choice = self.input_fn("  Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
        return None


def register_default_hooks(
    pipeline: HookPipeline,
    files: FileTools,
    workdir: str,
    input_fn: Callable[[str], str] = input,
    printer: Callable[[str], None] = print,
) -> None:
    policy = PermissionPolicy(files, input_fn=input_fn, printer=printer)
    pipeline.register("PreToolUse", policy)

    def log_tool(block: Any) -> None:
        printer(f"\033[90m[HOOK] {_call_name(block)}\033[0m")
        return None

    def large_output(block: Any, output: Any) -> None:
        size = len(str(output))
        if size > 100_000:
            printer(
                f"\033[33m[HOOK] large output from {_call_name(block)}: "
                f"{size} chars\033[0m"
            )
        return None

    def user_prompt(_query: str) -> None:
        printer(f"\033[90m[HOOK] UserPromptSubmit: {workdir}\033[0m")
        return None

    def stop(messages: list) -> None:
        count = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                count += sum(
                    1
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "tool_result"
                )
        printer(f"\033[90m[HOOK] Stop: {count} tool result(s)\033[0m")
        return None

    pipeline.register("PreToolUse", log_tool)
    pipeline.register("PostToolUse", large_output)
    pipeline.register("UserPromptSubmit", user_prompt)
    pipeline.register("Stop", stop)
