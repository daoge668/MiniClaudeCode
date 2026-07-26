"""RAG configuration loaded from the application's existing ``.env``."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


SCHEMA_VERSION = 1
COLLECTION_NAME = "project_resources_v1"
MAX_FILE_BYTES = 1024 * 1024
DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "rerank/text-rerank/text-rerank"
)

SUPPORTED_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".cs",
        ".css",
        ".go",
        ".graphql",
        ".h",
        ".hpp",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".kts",
        ".md",
        ".php",
        ".properties",
        ".proto",
        ".ps1",
        ".py",
        ".rb",
        ".rs",
        ".rst",
        ".scala",
        ".scss",
        ".sh",
        ".sql",
        ".svelte",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)
SUPPORTED_NAMES = frozenset(
    {
        "dockerfile",
        "makefile",
        "cmakelists.txt",
    }
)


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class RagConfig:
    project_dir: Path
    enabled: bool
    milvus_uri: str
    milvus_token: str = field(repr=False)
    embedding_base_url: str = ""
    embedding_api_key: str = field(default="", repr=False)
    embedding_model: str = ""
    rerank_enabled: bool = False
    rerank_url: str = DEFAULT_RERANK_URL
    rerank_api_key: str = field(default="", repr=False)
    rerank_model: str = "qwen3-vl-rerank"
    rerank_candidates: int = 20
    rerank_timeout_seconds: float = 30.0
    collection_name: str = COLLECTION_NAME
    schema_version: int = SCHEMA_VERSION
    max_file_bytes: int = MAX_FILE_BYTES
    embedding_batch_size: int = 32

    @classmethod
    def from_env(cls, project_dir: str | Path) -> "RagConfig":
        project = Path(project_dir).expanduser().resolve()
        embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
        try:
            rerank_candidates = int(
                os.getenv("RERANK_CANDIDATES", "20").strip()
            )
        except ValueError:
            rerank_candidates = 20
        try:
            rerank_timeout = float(
                os.getenv("RERANK_TIMEOUT_SECONDS", "30").strip()
            )
        except ValueError:
            rerank_timeout = 30.0
        return cls(
            project_dir=project,
            enabled=_enabled(os.getenv("RAG_ENABLED")),
            milvus_uri=(
                os.getenv("MILVUS_URI", "http://127.0.0.1:19530").strip()
            ),
            milvus_token=os.getenv("MILVUS_TOKEN", "root:Milvus").strip(),
            embedding_base_url=(
                os.getenv("EMBEDDING_BASE_URL", "").strip().rstrip("/")
            ),
            embedding_api_key=embedding_api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
            rerank_enabled=_enabled(os.getenv("RERANK_ENABLED")),
            rerank_url=os.getenv(
                "RERANK_URL", DEFAULT_RERANK_URL
            ).strip(),
            rerank_api_key=os.getenv(
                "RERANK_API_KEY", embedding_api_key
            ).strip(),
            rerank_model=os.getenv(
                "RERANK_MODEL", "qwen3-vl-rerank"
            ).strip(),
            rerank_candidates=max(1, min(rerank_candidates, 100)),
            rerank_timeout_seconds=max(1.0, min(rerank_timeout, 120.0)),
        )

    @property
    def resources_dir(self) -> Path:
        return self.project_dir / "resources"

    @property
    def state_dir(self) -> Path:
        return self.project_dir / ".rag"

    @property
    def manifest_path(self) -> Path:
        return self.state_dir / "manifest.json"

    @property
    def embedding_configured(self) -> bool:
        return bool(
            self.embedding_base_url
            and self.embedding_api_key
            and self.embedding_model
        )

    @property
    def rerank_configured(self) -> bool:
        return bool(
            self.rerank_enabled
            and self.rerank_url
            and self.rerank_api_key
            and self.rerank_model
        )

    @property
    def embedding_fingerprint(self) -> str:
        """Fingerprint semantics-affecting config without including the key."""
        payload = {
            "base_url": self.embedding_base_url,
            "model": self.embedding_model,
            "schema_version": self.schema_version,
        }
        raw = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
