"""Background tool execution and cron scheduling."""

from __future__ import annotations

import json
import queue
import random
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from .models import CronJob
from .workspace import atomic_write_text


class BackgroundTaskManager:
    def __init__(
        self,
        result_dir: Path,
        printer: Callable[[str], None] = print,
    ):
        self.result_dir = result_dir
        self.printer = printer
        self._counter = 0
        self._tasks: dict[str, dict] = {}
        self._results: dict[str, str] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()

    @staticmethod
    def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
        if tool_name != "bash":
            return False
        command = str(tool_input.get("command", "")).lower()
        keywords = (
            "install",
            "build",
            "test",
            "deploy",
            "compile",
            "docker build",
            "pip install",
            "npm install",
            "cargo build",
            "pytest",
            "make",
        )
        return any(keyword in command for keyword in keywords)

    @classmethod
    def should_run(cls, tool_name: str, tool_input: dict) -> bool:
        return tool_name == "bash" and (
            bool(tool_input.get("run_in_background"))
            or cls.is_slow_operation(tool_name, tool_input)
        )

    def start(
        self,
        tool_use_id: str,
        tool_name: str,
        tool_input: dict,
        operation: Callable[[], str],
        after: Callable[[str], None] | None = None,
    ) -> str:
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("background manager is closed")
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            command = tool_input.get("command", tool_name)
            self._tasks[task_id] = {
                "tool_use_id": tool_use_id,
                "command": command,
                "status": "running",
            }

        def worker() -> None:
            try:
                result = str(operation())
                if after:
                    after(result)
                status = "completed"
            except Exception as exc:  # a failed worker must still become visible
                result = f"Error: {type(exc).__name__}: {exc}"
                status = "failed"
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task["status"] = status
                    self._results[task_id] = result

        threading.Thread(
            target=worker,
            name=f"mini-claude-code-{task_id}",
            daemon=True,
        ).start()
        self.printer(
            f"  \033[33m[background] {task_id}: "
            f"{str(command)[:60]}\033[0m"
        )
        return task_id

    def collect_notifications(self) -> list[str]:
        with self._lock:
            ready = [
                task_id
                for task_id, task in self._tasks.items()
                if task["status"] in ("completed", "failed")
            ]
        notifications: list[str] = []
        for task_id in ready:
            with self._lock:
                task = self._tasks.pop(task_id, None)
                output = self._results.pop(task_id, "")
            if task is None:
                continue
            summary = output
            if len(output) > 200:
                self.result_dir.mkdir(parents=True, exist_ok=True)
                path = self.result_dir / f"{task_id}.txt"
                atomic_write_text(path, output)
                summary = f"{output[:200]}… [full output: {path}]"
            notifications.append(
                "<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{summary}</summary>\n"
                "</task_notification>"
            )
        return notifications

    def close(self) -> None:
        self._closed.set()


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(
            _cron_field_matches(part.strip(), value)
            for part in field.split(",")
        )
    if "-" in field:
        lower, upper = field.split("-", 1)
        return int(lower) <= value <= int(upper)
    return value == int(field)


def cron_matches(expression: str, value: datetime) -> bool:
    fields = expression.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, day_of_month, month, day_of_week = fields
    cron_weekday = (value.weekday() + 1) % 7
    minute_ok = _cron_field_matches(minute, value.minute)
    hour_ok = _cron_field_matches(hour, value.hour)
    day_ok = _cron_field_matches(day_of_month, value.day)
    month_ok = _cron_field_matches(month, value.month)
    weekday_ok = _cron_field_matches(day_of_week, cron_weekday)
    if not (minute_ok and hour_ok and month_ok):
        return False
    if day_of_month == "*" and day_of_week == "*":
        return True
    if day_of_month == "*":
        return weekday_ok
    if day_of_week == "*":
        return day_ok
    return day_ok or weekday_ok


def _validate_cron_field(field: str, lower: int, upper: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if (
            not step.isdigit()
            or int(step) <= 0
        ):
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            error = _validate_cron_field(part.strip(), lower, upper)
            if error:
                return error
        return None
    if "-" in field:
        left, right = field.split("-", 1)
        if not left.isdigit() or not right.isdigit():
            return f"Invalid range: {field}"
        start, end = int(left), int(right)
        if (
            start < lower
            or start > upper
            or end < lower
            or end > upper
        ):
            return f"Range {field} out of bounds [{lower}-{upper}]"
        if start > end:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    number = int(field)
    if number < lower or number > upper:
        return f"Value {number} out of bounds [{lower}-{upper}]"
    return None


def validate_cron(expression: str) -> str | None:
    fields = expression.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))
    names = ("minute", "hour", "day-of-month", "month", "day-of-week")
    for field, (lower, upper), name in zip(fields, bounds, names):
        error = _validate_cron_field(field, lower, upper)
        if error:
            return f"{name}: {error}"
    return None


class CronScheduler:
    """Durable cron registry with a single-consumer fired-job queue."""

    def __init__(
        self,
        durable_path: Path,
        printer: Callable[[str], None] = print,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.durable_path = durable_path
        self.printer = printer
        self.clock = clock
        self._jobs: dict[str, CronJob] = {}
        self._last_fired: dict[str, str] = {}
        self._fired: queue.Queue[CronJob] = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def jobs(self) -> list[CronJob]:
        with self._lock:
            return list(self._jobs.values())

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop.clear()
            self._load_durable_jobs()
            self._thread = threading.Thread(
                target=self._run,
                name="mini-claude-code-cron-scheduler",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _serialize_durable_locked(self) -> str:
        durable = [
            asdict(job) for job in self._jobs.values() if job.durable
        ]
        return json.dumps(durable, indent=2, ensure_ascii=False)

    def _save_durable_jobs(self) -> None:
        with self._lock:
            content = self._serialize_durable_locked()
        atomic_write_text(self.durable_path, content)

    def _load_durable_jobs(self) -> None:
        if not self.durable_path.exists():
            return
        try:
            raw = json.loads(
                self.durable_path.read_text(encoding="utf-8")
            )
            for item in raw:
                job = CronJob(**item)
                if not validate_cron(job.cron):
                    self._jobs[job.id] = job
        except Exception as exc:
            self.printer(
                "  \033[31m[cron] ignored invalid durable file: "
                f"{exc}\033[0m"
            )

    def schedule(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> CronJob | str:
        error = validate_cron(cron)
        if error:
            return error
        with self._lock:
            while True:
                job_id = f"cron_{random.randint(0, 999999):06d}"
                if job_id not in self._jobs:
                    break
            job = CronJob(
                id=job_id,
                cron=cron,
                prompt=prompt,
                recurring=recurring,
                durable=durable,
            )
            self._jobs[job.id] = job
        if durable:
            self._save_durable_jobs()
        return job

    def cancel(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            self._last_fired.pop(job_id, None)
        if not job:
            return f"Job {job_id} not found"
        if job.durable:
            self._save_durable_jobs()
        return f"Cancelled {job_id}"

    def list_text(self) -> str:
        jobs = self.jobs
        if not jobs:
            return "No cron jobs."
        return "\n".join(
            f"  {job.id}: '{job.cron}' -> {job.prompt[:40]} "
            f"[{'recurring' if job.recurring else 'one-shot'}, "
            f"{'durable' if job.durable else 'session'}]"
            for job in jobs
        )

    def schedule_text(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> str:
        result = self.schedule(cron, prompt, recurring, durable)
        if isinstance(result, str):
            return f"Error: {result}"
        return f"Scheduled {result.id}: '{cron}' -> {prompt}"

    def _tick(self, now: datetime) -> None:
        marker = now.strftime("%Y-%m-%d %H:%M")
        save_needed = False
        with self._lock:
            for job in list(self._jobs.values()):
                try:
                    if (
                        cron_matches(job.cron, now)
                        and self._last_fired.get(job.id) != marker
                    ):
                        self._fired.put(job)
                        self._last_fired[job.id] = marker
                        if not job.recurring:
                            self._jobs.pop(job.id, None)
                            save_needed = save_needed or job.durable
                except Exception as exc:
                    self.printer(
                        f"  \033[31m[cron error] {job.id}: {exc}\033[0m"
                    )
        if save_needed:
            self._save_durable_jobs()

    def _run(self) -> None:
        while not self._stop.wait(1):
            self._tick(self.clock())

    def wait_for_jobs(self, timeout: float = 0.5) -> list[CronJob]:
        try:
            first = self._fired.get(timeout=timeout)
        except queue.Empty:
            return []
        jobs = [first]
        while True:
            try:
                jobs.append(self._fired.get_nowait())
            except queue.Empty:
                return jobs
