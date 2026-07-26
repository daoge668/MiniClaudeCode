"""Configuration and runtime paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when required startup configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    project_dir: Path
    workdir: Path
    model_id: str
    fallback_model_id: str | None = None
    anthropic_base_url: str | None = None
    default_max_tokens: int = 8_000
    escalated_max_tokens: int = 16_000
    max_retries: int = 3
    max_consecutive_529: int = 2
    max_recovery_retries: int = 2
    base_delay_ms: int = 500
    context_limit: int = 50_000
    keep_recent_tool_results: int = 3
    persist_threshold: int = 30_000

    @classmethod
    def from_env(
        cls,
        project_dir: str | Path,
        workdir: str | Path | None = None,
    ) -> "Settings":
        """Load ``.env`` explicitly and validate the model configuration."""
        project = Path(project_dir).expanduser().resolve()
        runtime_root = (
            Path(workdir).expanduser().resolve() if workdir else project
        )

        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - installation guidance
            raise ConfigurationError(
                "Missing dependency 'python-dotenv'. Run: pip install -e ."
            ) from exc

        load_dotenv(project / ".env", override=True)
        model_id = os.getenv("MODEL_ID", "").strip()
        if not model_id:
            raise ConfigurationError(
                "MODEL_ID is required. Copy .env.example to .env and set it."
            )

        base_url = os.getenv("ANTHROPIC_BASE_URL") or None
        if base_url:
            # Preserve the original custom-endpoint authentication behavior.
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

        return cls(
            project_dir=project,
            workdir=runtime_root,
            model_id=model_id,
            fallback_model_id=os.getenv("FALLBACK_MODEL_ID") or None,
            anthropic_base_url=base_url,
        )

    @property
    def tasks_dir(self) -> Path:
        return self.workdir / ".tasks"

    @property
    def worktrees_dir(self) -> Path:
        return self.workdir / ".worktrees"

    @property
    def skills_dir(self) -> Path:
        return self.workdir / "skills"

    @property
    def mailbox_dir(self) -> Path:
        return self.workdir / ".mailboxes"

    @property
    def memory_dir(self) -> Path:
        return self.workdir / ".memory"

    @property
    def memory_index(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    @property
    def transcript_dir(self) -> Path:
        return self.workdir / ".transcripts"

    @property
    def tool_results_dir(self) -> Path:
        return self.workdir / ".task_outputs" / "tool-results"

    @property
    def durable_cron_path(self) -> Path:
        return self.workdir / ".scheduled_tasks.json"

    def ensure_runtime_dirs(self) -> None:
        """Create only directories needed immediately at application startup."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self.mailbox_dir.mkdir(parents=True, exist_ok=True)
