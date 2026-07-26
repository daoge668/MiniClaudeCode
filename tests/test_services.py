from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from mini_claude_code.mcp import MCPRegistry, normalize_mcp_name
from mini_claude_code.messaging import MessageBus, ProtocolManager
from mini_claude_code.scheduling import (
    BackgroundTaskManager,
    CronScheduler,
    cron_matches,
    validate_cron,
)


def test_concurrent_messages_are_not_lost(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path, printer=lambda _text: None)
    bus.start()

    def sender(number: int) -> None:
        for index in range(25):
            bus.send(f"sender-{number}", "lead", f"{number}:{index}")

    threads = [threading.Thread(target=sender, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(bus.read_inbox("lead")) == 100
    assert bus.read_inbox("lead") == []


def test_mailbox_name_cannot_escape_directory(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path, printer=lambda _text: None)
    bus.start()
    with pytest.raises(ValueError, match="Invalid agent name"):
        bus.send("lead", "../outside", "bad")
    assert not (tmp_path.parent / "outside.jsonl").exists()


def test_plan_protocol_round_trip(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path, printer=lambda _text: None)
    bus.start()
    protocols = ProtocolManager(bus)
    result = protocols.submit_plan("worker", "do the work")
    request_id = result.removeprefix("Plan submitted (").removesuffix(")")
    assert "approved" in protocols.review_plan(request_id, True).lower()
    assert bus.read_inbox("worker")[0]["metadata"]["approve"] is True


def test_cron_validation_and_matching() -> None:
    assert validate_cron("*/5 9-17 * * 1-5") is None
    assert "minute" in str(validate_cron("61 * * * *"))
    monday = datetime(2026, 7, 27, 10, 0)
    assert cron_matches("0 10 * * 1", monday)
    assert not cron_matches("0 10 * * 0", monday)


def test_one_shot_cron_has_single_consumer(tmp_path: Path) -> None:
    scheduler = CronScheduler(
        tmp_path / ".scheduled_tasks.json",
        printer=lambda _text: None,
    )
    result = scheduler.schedule(
        "0 10 27 7 *", "once", recurring=False, durable=False
    )
    assert not isinstance(result, str)
    scheduler._tick(datetime(2026, 7, 27, 10, 0))
    assert [job.prompt for job in scheduler.wait_for_jobs(timeout=0)] == [
        "once"
    ]
    assert scheduler.wait_for_jobs(timeout=0) == []


def test_background_failure_is_reported(tmp_path: Path) -> None:
    manager = BackgroundTaskManager(
        tmp_path, printer=lambda _text: None
    )

    def fail() -> str:
        raise RuntimeError("boom")

    manager.start("tool-1", "bash", {"command": "x"}, fail)
    deadline = time.time() + 2
    notifications: list[str] = []
    while time.time() < deadline and not notifications:
        notifications = manager.collect_notifications()
        time.sleep(0.01)
    assert "<status>failed</status>" in notifications[0]
    assert "boom" in notifications[0]


def test_mcp_discovery() -> None:
    registry = MCPRegistry(printer=lambda _text: None)
    assert normalize_mcp_name("docs server") == "docs_server"
    assert "Connected" in registry.connect("docs")
    definitions, handlers = registry.tool_pool()
    assert "mcp__docs__search" in {
        definition["name"] for definition in definitions
    }
    assert "Found 3 results" in handlers["mcp__docs__search"](query="cron")
