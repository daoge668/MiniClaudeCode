from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

from mini_claude_code.models import Task
from mini_claude_code.workspace import FileTools, TaskRepository, TodoStore


def test_file_tools_stay_inside_workspace(tmp_path: Path) -> None:
    files = FileTools(tmp_path)
    assert files.write_file("nested/example.txt", "hello").startswith("Wrote")
    assert files.read_file("nested/example.txt") == "hello"
    assert files.edit_file("nested/example.txt", "hello", "world") == (
        "Edited nested/example.txt"
    )
    assert files.read_file("nested/example.txt") == "world"
    assert files.write_file("../escape.txt", "bad").startswith("Error:")
    assert not (tmp_path.parent / "escape.txt").exists()


def test_todo_normalization() -> None:
    normalized, error = TodoStore.normalize(
        '[{"content": "x", "status": "pending"}]'
    )
    assert error is None
    assert normalized == [{"content": "x", "status": "pending"}]
    _, error = TodoStore.normalize(
        [{"content": "x", "status": "unknown"}]
    )
    assert "invalid status" in str(error)


def test_task_lifecycle_and_json_compatibility(tmp_path: Path) -> None:
    repository = TaskRepository(tmp_path, printer=lambda _text: None)
    repository.start()
    first = repository.create("first")
    second = repository.create("second", blockedBy=[first.id])
    assert repository.can_start(first.id)
    assert not repository.can_start(second.id)
    assert "Claimed" in repository.claim(first.id, "worker")
    assert "Completed" in repository.complete(first.id)
    assert repository.can_start(second.id)

    record = Task(
        id="task_existing",
        subject="existing",
        description="data",
        status="pending",
        owner=None,
        blockedBy=[],
        worktree=None,
    )
    (tmp_path / "task_existing.json").write_text(
        json.dumps(asdict(record)), encoding="utf-8"
    )
    assert repository.load("task_existing") == record


def test_task_claim_is_serialized(tmp_path: Path) -> None:
    repository = TaskRepository(tmp_path, printer=lambda _text: None)
    repository.start()
    task = repository.create("one owner")
    results: list[str] = []

    def claim(owner: str) -> None:
        results.append(repository.claim(task.id, owner))

    threads = [
        threading.Thread(target=claim, args=(f"worker-{index}",))
        for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum("Claimed" in result for result in results) == 1
