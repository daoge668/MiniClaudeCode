"""Thread-safe JSONL mailboxes and teammate protocols."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from .models import ProtocolState


class MessageBus:
    """A small durable mailbox bus.

    Inbox consumption first atomically renames the current mailbox. Senders can
    immediately append to a fresh file, so messages arriving during a read are
    left for the next read rather than being deleted with the consumed batch.
    """

    AGENT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

    def __init__(
        self,
        mailbox_dir: Path,
        printer: Callable[[str], None] = print,
    ):
        self.mailbox_dir = mailbox_dir
        self.printer = printer
        self._lock = threading.RLock()

    def start(self) -> None:
        self.mailbox_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, agent: str) -> Path:
        if not self.AGENT_NAME.fullmatch(agent):
            raise ValueError(
                f"Invalid agent name '{agent}': expected 1-64 safe characters"
            )
        return self.mailbox_dir / f"{agent}.jsonl"

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict | None = None,
    ) -> None:
        message = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "ts": time.time(),
            "metadata": metadata or {},
        }
        inbox = self._inbox_path(to_agent)
        with self._lock:
            self.mailbox_dir.mkdir(parents=True, exist_ok=True)
            with inbox.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(message, ensure_ascii=False) + "\n")
                stream.flush()
        self.printer(
            f"  \033[33m[bus] {from_agent} → {to_agent}: "
            f"({msg_type}) {content[:50]}\033[0m"
        )

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = self._inbox_path(agent)
        consuming = self.mailbox_dir / (
            f".{agent}.{os.getpid()}.{uuid.uuid4().hex}.consuming"
        )
        with self._lock:
            if not inbox.exists():
                return []
            try:
                os.replace(inbox, consuming)
            except FileNotFoundError:
                return []
        try:
            messages: list[dict] = []
            for line in consuming.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    self.printer(
                        f"  \033[31m[bus] ignored malformed message for "
                        f"{agent}\033[0m"
                    )
            return messages
        finally:
            consuming.unlink(missing_ok=True)


class ProtocolManager:
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._pending: dict[str, ProtocolState] = {}
        self._lock = threading.RLock()

    @property
    def pending(self) -> dict[str, ProtocolState]:
        with self._lock:
            return dict(self._pending)

    def new_request(
        self,
        request_type: str,
        sender: str,
        target: str,
        payload: str,
    ) -> ProtocolState:
        with self._lock:
            while True:
                request_id = f"req_{random.randint(0, 999999):06d}"
                if request_id not in self._pending:
                    break
            state = ProtocolState(
                request_id=request_id,
                type=request_type,
                sender=sender,
                target=target,
                status="pending",
                payload=payload,
            )
            self._pending[request_id] = state
            return state

    def get(self, request_id: str) -> ProtocolState | None:
        with self._lock:
            return self._pending.get(request_id)

    def match_response(
        self,
        response_type: str,
        request_id: str,
        approve: bool,
    ) -> None:
        with self._lock:
            state = self._pending.get(request_id)
            if not state:
                return
            if (
                state.type == "shutdown"
                and response_type != "shutdown_response"
            ):
                return
            if (
                state.type == "plan_approval"
                and response_type != "plan_approval_response"
            ):
                return
            state.status = "approved" if approve else "rejected"

    def consume_lead_inbox(self, route_protocol: bool = True) -> list[dict]:
        messages = self.bus.read_inbox("lead")
        if route_protocol:
            for message in messages:
                metadata = message.get("metadata", {})
                request_id = metadata.get("request_id", "")
                message_type = message.get("type", "")
                if request_id and message_type.endswith("_response"):
                    self.match_response(
                        message_type,
                        request_id,
                        bool(metadata.get("approve", False)),
                    )
        return messages

    def submit_plan(self, from_name: str, plan: str) -> str:
        state = self.new_request(
            "plan_approval", from_name, "lead", plan
        )
        self.bus.send(
            from_name,
            "lead",
            plan,
            "plan_approval_request",
            {"request_id": state.request_id},
        )
        return f"Plan submitted ({state.request_id})"

    def request_shutdown(self, teammate: str) -> str:
        state = self.new_request("shutdown", "lead", teammate, "")
        self.bus.send(
            "lead",
            teammate,
            "Shut down.",
            "shutdown_request",
            {"request_id": state.request_id},
        )
        return f"Shutdown request sent to {teammate}"

    def request_plan(self, teammate: str, task: str) -> str:
        self.bus.send(
            "lead", teammate, f"Submit plan for: {task}", "message"
        )
        return f"Asked {teammate} to submit a plan"

    def review_plan(
        self,
        request_id: str,
        approve: bool,
        feedback: str = "",
    ) -> str:
        with self._lock:
            state = self._pending.get(request_id)
            if not state:
                return f"Request {request_id} not found"
            state.status = "approved" if approve else "rejected"
        self.bus.send(
            "lead",
            state.sender,
            feedback or ("Approved" if approve else "Rejected"),
            "plan_approval_response",
            {"request_id": request_id, "approve": approve},
        )
        return f"Plan {'approved' if approve else 'rejected'}"
