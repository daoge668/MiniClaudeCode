"""Shared data models and stable JSON-facing field names."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]
    worktree: str | None = None


@dataclass(slots=True)
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class CronJob:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


@dataclass(slots=True)
class RecoveryState:
    current_model: str
    has_escalated: bool = False
    recovery_count: int = 0
    consecutive_529: int = 0
    has_attempted_reactive_compact: bool = False


@dataclass(slots=True)
class ToolCall:
    name: str
    input: dict
    id: str = ""
