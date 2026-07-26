"""Conversation memory, budgeting, transcripts, and compaction."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .client import ModelGateway, block_type, extract_text
from .config import Settings


def estimate_size(messages: list) -> int:
    return len(json.dumps(messages, default=str, ensure_ascii=False))


def message_has_tool_use(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "assistant"
        and isinstance(content, list)
        and any(block_type(block) == "tool_use" for block in content)
    )


def is_tool_result_message(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
    )


def collect_tool_results(messages: list) -> list[tuple[int, int, dict]]:
    found: list[tuple[int, int, dict]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                found.append((message_index, block_index, block))
    return found


class ContextManager:
    def __init__(
        self,
        settings: Settings,
        gateway_provider: Callable[[], ModelGateway],
        printer: Callable[[str], None] = print,
    ):
        self.settings = settings
        self.gateway_provider = gateway_provider
        self.printer = printer

    def update_context(
        self,
        mcp_names: list[str],
        teammate_names: list[str],
    ) -> dict:
        memories = ""
        if self.settings.memory_index.exists():
            memories = self.settings.memory_index.read_text(
                encoding="utf-8"
            )[:2_000]
        return {
            "memories": memories,
            "connected_mcp": list(mcp_names),
            "active_teammates": list(teammate_names),
        }

    def persist_large_output(self, tool_use_id: str, output: str) -> str:
        if len(output) <= self.settings.persist_threshold:
            return output
        self.settings.tool_results_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.tool_results_dir / f"{tool_use_id}.txt"
        if not path.exists():
            path.write_text(output, encoding="utf-8")
        return (
            f"<persisted-output>\nFull output: {path}\n"
            f"Preview:\n{output[:2000]}\n</persisted-output>"
        )

    def tool_result_budget(
        self, messages: list, max_bytes: int = 200_000
    ) -> list:
        if not messages:
            return messages
        last = messages[-1]
        content = last.get("content")
        if last.get("role") != "user" or not isinstance(content, list):
            return messages
        blocks = [
            (index, block)
            for index, block in enumerate(content)
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        total = sum(
            len(str(block.get("content", ""))) for _, block in blocks
        )
        for _, block in sorted(
            blocks,
            key=lambda pair: len(str(pair[1].get("content", ""))),
            reverse=True,
        ):
            if total <= max_bytes:
                break
            text = str(block.get("content", ""))
            block["content"] = self.persist_large_output(
                str(block.get("tool_use_id", "unknown")), text
            )
            total = sum(
                len(str(item.get("content", ""))) for _, item in blocks
            )
        return messages

    @staticmethod
    def snip_compact(messages: list, max_messages: int = 50) -> list:
        if len(messages) <= max_messages:
            return messages
        head_end = 3
        tail_start = len(messages) - (max_messages - 3)
        if head_end and message_has_tool_use(messages[head_end - 1]):
            while (
                head_end < len(messages)
                and is_tool_result_message(messages[head_end])
            ):
                head_end += 1
        if (
            0 < tail_start < len(messages)
            and is_tool_result_message(messages[tail_start])
            and message_has_tool_use(messages[tail_start - 1])
        ):
            tail_start -= 1
        if head_end >= tail_start:
            return messages
        snipped = tail_start - head_end
        return (
            messages[:head_end]
            + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
            + messages[tail_start:]
        )

    def micro_compact(self, messages: list) -> list:
        results = collect_tool_results(messages)
        if len(results) <= self.settings.keep_recent_tool_results:
            return messages
        cutoff = -self.settings.keep_recent_tool_results
        for _, _, block in results[:cutoff]:
            if len(str(block.get("content", ""))) > 120:
                block["content"] = (
                    "[Earlier tool result compacted. Re-run if needed.]"
                )
        return messages

    def prepare(self, messages: list) -> list:
        messages[:] = self.tool_result_budget(messages)
        messages[:] = self.snip_compact(messages)
        messages[:] = self.micro_compact(messages)
        if estimate_size(messages) > self.settings.context_limit:
            messages[:] = self.compact_history(messages)
        return messages

    def write_transcript(self, messages: list) -> Path:
        self.settings.transcript_dir.mkdir(parents=True, exist_ok=True)
        path = self.settings.transcript_dir / (
            f"transcript_{int(time.time() * 1000)}.jsonl"
        )
        with path.open("w", encoding="utf-8") as stream:
            for message in messages:
                stream.write(
                    json.dumps(
                        message, default=str, ensure_ascii=False
                    )
                    + "\n"
                )
        return path

    def summarize_history(self, messages: list) -> str:
        conversation = json.dumps(
            messages, default=str, ensure_ascii=False
        )[:80_000]
        prompt = (
            "Summarize this coding-agent conversation so work can continue. "
            "Preserve current goal, key findings, changed files, remaining "
            "work, and user constraints.\n\n"
            + conversation
        )
        response = self.gateway_provider().create_message(
            model=self.settings.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2_000,
        )
        return extract_text(response.content) or "(empty summary)"

    def compact_history(self, messages: list) -> list:
        transcript = self.write_transcript(messages)
        self.printer(
            f"  \033[36m[compact] transcript saved: {transcript}\033[0m"
        )
        summary = self.summarize_history(messages)
        return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]

    def reactive_compact(self, messages: list) -> list:
        transcript = self.write_transcript(messages)
        self.printer(
            "  \033[31m[reactive compact] transcript saved: "
            f"{transcript}\033[0m"
        )
        tail_start = max(0, len(messages) - 5)
        if (
            0 < tail_start < len(messages)
            and is_tool_result_message(messages[tail_start])
            and message_has_tool_use(messages[tail_start - 1])
        ):
            tail_start -= 1
        try:
            summary = self.summarize_history(messages[:tail_start])
        except Exception:
            summary = (
                "Earlier conversation was trimmed after a "
                "prompt-too-long error."
            )
        return [
            {"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
            *messages[tail_start:],
        ]
