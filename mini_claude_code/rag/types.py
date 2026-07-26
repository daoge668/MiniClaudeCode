"""Shared, dependency-free RAG data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A decoded source file that is eligible for indexing."""

    path: str
    text: str
    file_hash: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One structure-aware, source-addressable document chunk."""

    chunk_id: str
    text: str
    source_path: str
    start_line: int
    end_line: int
    symbol: str
    file_hash: str


@dataclass(slots=True)
class ScanResult:
    documents: list[SourceDocument] = field(default_factory=list)
    skipped: int = 0
    failed_paths: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SyncStats:
    scanned: int = 0
    skipped: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
