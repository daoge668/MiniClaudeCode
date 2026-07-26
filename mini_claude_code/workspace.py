"""Workspace storage, file tools, tasks, and Git worktrees."""

from __future__ import annotations

import ast
import glob as glob_module
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import Settings
from .models import Task


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file atomically from a temporary sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TaskRepository:
    def __init__(
        self,
        tasks_dir: Path,
        printer: Callable[[str], None] = print,
    ):
        self.tasks_dir = tasks_dir
        self.printer = printer
        self._lock = threading.RLock()

    def start(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def create(
        self,
        subject: str,
        description: str = "",
        blockedBy: list[str] | None = None,
    ) -> Task:
        with self._lock:
            while True:
                task_id = (
                    f"task_{int(time.time())}_"
                    f"{random.randint(0, 9999):04d}"
                )
                if not self._path(task_id).exists():
                    break
            task = Task(
                id=task_id,
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=list(blockedBy or []),
            )
            self.save(task)
            return task

    def save(self, task: Task) -> None:
        with self._lock:
            atomic_write_text(
                self._path(task.id),
                json.dumps(asdict(task), indent=2, ensure_ascii=False),
            )

    def load(self, task_id: str) -> Task:
        with self._lock:
            return Task(
                **json.loads(self._path(task_id).read_text(encoding="utf-8"))
            )

    def list(self) -> list[Task]:
        with self._lock:
            return [
                Task(**json.loads(path.read_text(encoding="utf-8")))
                for path in sorted(self.tasks_dir.glob("task_*.json"))
            ]

    def get_json(self, task_id: str) -> str:
        return json.dumps(asdict(self.load(task_id)), indent=2, ensure_ascii=False)

    def can_start(self, task_id: str) -> bool:
        with self._lock:
            task = self.load(task_id)
            for dependency_id in task.blockedBy:
                path = self._path(dependency_id)
                if not path.exists() or self.load(dependency_id).status != "completed":
                    return False
            return True

    def claim(self, task_id: str, owner: str = "agent") -> str:
        with self._lock:
            task = self.load(task_id)
            if task.status != "pending":
                return f"Task {task_id} is {task.status}, cannot claim"
            if task.owner:
                return f"Task {task_id} already owned by {task.owner}"
            if not self.can_start(task_id):
                incomplete = [
                    dependency
                    for dependency in task.blockedBy
                    if self._path(dependency).exists()
                    and self.load(dependency).status != "completed"
                ]
                missing = [
                    dependency
                    for dependency in task.blockedBy
                    if not self._path(dependency).exists()
                ]
                parts: list[str] = []
                if incomplete:
                    parts.append(f"blocked by: {incomplete}")
                if missing:
                    parts.append(f"missing deps: {missing}")
                return "Cannot start — " + ", ".join(parts)
            task.owner = owner
            task.status = "in_progress"
            self.save(task)
        self.printer(
            f"  \033[36m[claim] {task.subject} → in_progress\033[0m"
        )
        return f"Claimed {task.id} ({task.subject})"

    def complete(self, task_id: str) -> str:
        with self._lock:
            task = self.load(task_id)
            if task.status != "in_progress":
                return f"Task {task_id} is {task.status}, cannot complete"
            task.status = "completed"
            self.save(task)
            unblocked = [
                item.subject
                for item in self.list()
                if item.status == "pending"
                and item.blockedBy
                and self.can_start(item.id)
            ]
        self.printer(f"  \033[32m[complete] {task.subject} ✓\033[0m")
        result = f"Completed {task.id} ({task.subject})"
        if unblocked:
            result += f"\nUnblocked: {', '.join(unblocked)}"
        return result

    def scan_unclaimed(self) -> list[Task]:
        with self._lock:
            return [
                task
                for task in self.list()
                if task.status == "pending"
                and not task.owner
                and self.can_start(task.id)
            ]


class TodoStore:
    def __init__(self, printer: Callable[[str], None] = print):
        self.items: list[dict] = []
        self.printer = printer
        self._lock = threading.Lock()

    @staticmethod
    def normalize(todos: object) -> tuple[list[dict] | None, str | None]:
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                try:
                    todos = ast.literal_eval(todos)
                except (SyntaxError, ValueError):
                    return None, "Error: todos must be a list or JSON array string"
        if not isinstance(todos, list):
            return None, "Error: todos must be a list"
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                return None, f"Error: todos[{index}] must be an object"
            if "content" not in todo or "status" not in todo:
                return (
                    None,
                    f"Error: todos[{index}] missing 'content' or 'status'",
                )
            if todo["status"] not in ("pending", "in_progress", "completed"):
                return (
                    None,
                    f"Error: todos[{index}] has invalid status "
                    f"'{todo['status']}'",
                )
        return todos, None

    def write(self, todos: object) -> str:
        normalized, error = self.normalize(todos)
        if error:
            return error
        assert normalized is not None
        with self._lock:
            self.items = normalized
        self.printer(
            f"  \033[33m[todo] updated {len(normalized)} item(s)\033[0m"
        )
        return f"Updated {len(normalized)} todos"


class FileTools:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def safe_path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def run_bash(
        self,
        command: str,
        run_in_background: bool = False,
    ) -> str:
        del run_in_background
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout + result.stderr).strip()
            return output[:50_000] if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (120s)"
        except Exception as exc:
            return f"Error: {exc}"

    def read_file(
        self,
        path: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> str:
        try:
            lines = self.safe_path(path).read_text(encoding="utf-8").splitlines()
            normalized_offset = max(int(offset or 0), 0)
            normalized_limit = int(limit) if limit is not None else None
            lines = lines[normalized_offset:]
            if normalized_limit is not None and normalized_limit < len(lines):
                remaining = len(lines) - normalized_limit
                lines = lines[:normalized_limit] + [
                    f"... ({remaining} more lines)"
                ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {exc}"

    def write_file(self, path: str, content: str) -> str:
        try:
            target = self.safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as exc:
            return f"Error: {exc}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            target = self.safe_path(path)
            text = target.read_text(encoding="utf-8")
            if old_text not in text:
                return f"Error: text not found in {path}"
            target.write_text(
                text.replace(old_text, new_text, 1), encoding="utf-8"
            )
            return f"Edited {path}"
        except Exception as exc:
            return f"Error: {exc}"

    def glob(self, pattern: str) -> str:
        try:
            matches: list[str] = []
            for match in glob_module.glob(pattern, root_dir=self.root):
                if (self.root / match).resolve().is_relative_to(self.root):
                    matches.append(match)
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as exc:
            return f"Error: {exc}"


class WorktreeService:
    VALID_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

    def __init__(
        self,
        settings: Settings,
        tasks: TaskRepository,
        printer: Callable[[str], None] = print,
    ):
        self.settings = settings
        self.tasks = tasks
        self.printer = printer
        self.root = settings.worktrees_dir
        self._lock = threading.RLock()

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_name(cls, name: str) -> str | None:
        if not name:
            return "Worktree name cannot be empty"
        if name in (".", ".."):
            return f"'{name}' is not a valid worktree name"
        if not cls.VALID_NAME.fullmatch(name):
            return (
                f"Invalid worktree name '{name}': only letters, digits, "
                "dots, underscores, dashes (1-64 chars)"
            )
        return None

    def _run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd or self.settings.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            return result.returncode == 0, output[:5_000]
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"Error: {exc}"

    def _log_event(
        self,
        event_type: str,
        name: str,
        task_id: str = "",
        base_commit: str = "",
    ) -> None:
        event = {
            "type": event_type,
            "worktree": name,
            "task_id": task_id,
            "ts": time.time(),
        }
        if base_commit:
            event["base_commit"] = base_commit
        path = self.root / "events.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def create(self, name: str, task_id: str = "") -> str:
        error = self.validate_name(name)
        if error:
            return f"Error: {error}"
        if task_id:
            try:
                self.tasks.load(task_id)
            except FileNotFoundError:
                return f"Error: task {task_id} not found"
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root.resolve()):
            return "Error: worktree path escapes workspace"
        if path.exists():
            return f"Worktree '{name}' already exists at {path}"

        base_ok, base_commit = self._run_git(["rev-parse", "HEAD"])
        if not base_ok:
            return f"Git error: {base_commit}"
        ok, result = self._run_git(
            ["worktree", "add", str(path), "-b", f"wt/{name}", "HEAD"]
        )
        if not ok:
            return f"Git error: {result}"
        if task_id:
            task = self.tasks.load(task_id)
            task.worktree = name
            self.tasks.save(task)
        self._log_event("create", name, task_id, base_commit.splitlines()[0])
        self.printer(
            f"  \033[33m[worktree] created: {name} at {path}\033[0m"
        )
        return f"Worktree '{name}' created at {path}"

    def _base_commit(self, name: str) -> str | None:
        events = self.root / "events.jsonl"
        if not events.exists():
            return None
        found: str | None = None
        for raw in events.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                event.get("type") == "create"
                and event.get("worktree") == name
                and event.get("base_commit")
            ):
                found = str(event["base_commit"])
        return found

    def _count_changes(self, name: str, path: Path) -> tuple[int, int]:
        status_ok, status = self._run_git(
            ["status", "--porcelain"], cwd=path, timeout=10
        )
        if not status_ok:
            return -1, -1
        files = len([line for line in status.splitlines() if line.strip()])
        base = self._base_commit(name)
        if base:
            commits_ok, output = self._run_git(
                ["rev-list", "--count", f"{base}..HEAD"], cwd=path, timeout=10
            )
        else:
            commits_ok, output = self._run_git(
                ["rev-list", "--count", "@{upstream}..HEAD"],
                cwd=path,
                timeout=10,
            )
        if not commits_ok:
            return -1, -1
        try:
            return files, int(output.splitlines()[0])
        except (ValueError, IndexError):
            return -1, -1

    def remove(self, name: str, discard_changes: bool = False) -> str:
        error = self.validate_name(name)
        if error:
            return error
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root.resolve()):
            return "Error: worktree path escapes workspace"
        if not path.exists():
            return f"Worktree '{name}' not found"
        if not discard_changes:
            files, commits = self._count_changes(name, path)
            if files < 0:
                return "Cannot verify status. Use discard_changes=true to force."
            if files or commits:
                return (
                    f"Worktree '{name}' has {files} file(s), "
                    f"{commits} commit(s). Use discard_changes=true "
                    "or keep_worktree."
                )
        ok, output = self._run_git(
            ["worktree", "remove", str(path), "--force"]
        )
        if not ok:
            return f"Failed to remove worktree '{name}': {output}"
        self._run_git(["branch", "-D", f"wt/{name}"])
        self._log_event("remove", name)
        self.printer(f"  \033[33m[worktree] removed: {name}\033[0m")
        return f"Worktree '{name}' removed"

    def keep(self, name: str) -> str:
        error = self.validate_name(name)
        if error:
            return error
        self._log_event("keep", name)
        return f"Worktree '{name}' kept for review (branch: wt/{name})"
