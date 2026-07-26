"""Startup synchronization and query service for the shared RAG snapshot."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Protocol

from .chunking import chunk_document, scan_resources
from .config import RagConfig
from .embeddings import OpenAIEmbeddingClient
from .milvus import MilvusStore
from .reranker import QwenVLReranker, RerankResult
from .types import DocumentChunk, ScanResult, SourceDocument, SyncStats


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class RerankProvider(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]: ...


StoreFactory = Callable[[RagConfig], MilvusStore]
EmbeddingFactory = Callable[[RagConfig], EmbeddingProvider]
RerankFactory = Callable[[RagConfig], RerankProvider]


def _default_manifest(config: RagConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "embedding_fingerprint": config.embedding_fingerprint,
        "embedding_dimension": None,
        "collection_name": config.collection_name,
        "files": {},
    }


class RagService:
    """Own one process-wide, read-only-at-query-time RAG snapshot."""

    def __init__(
        self,
        config: RagConfig,
        *,
        printer: Callable[[str], None] = print,
        store_factory: StoreFactory = MilvusStore.connect,
        embedding_factory: EmbeddingFactory = OpenAIEmbeddingClient,
        rerank_factory: RerankFactory = QwenVLReranker,
    ):
        self.config = config
        self.printer = printer
        self.store_factory = store_factory
        self.embedding_factory = embedding_factory
        self.rerank_factory = rerank_factory
        self.store: MilvusStore | None = None
        self.embedding: EmbeddingProvider | None = None
        self.reranker: RerankProvider | None = None
        self.available = False
        self.mode = "disabled"
        self.dimension: int | None = None
        self.stats = SyncStats()
        self._started = False

    def _warn(self, message: str) -> None:
        self.printer(f"  \033[33m[rag] warning: {message}\033[0m")

    def _info(self, message: str) -> None:
        self.printer(f"  \033[36m[rag] {message}\033[0m")

    def _load_manifest(self) -> dict[str, Any]:
        path = self.config.manifest_path
        if not path.exists():
            return _default_manifest(self.config)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(
                data.get("files"), dict
            ):
                raise ValueError("invalid manifest shape")
            return data
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._warn(f"manifest ignored ({exc})")
            return _default_manifest(self.config)

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.manifest_path
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)

    def _make_embedding(self) -> EmbeddingProvider | None:
        if not self.config.embedding_configured:
            self._warn(
                "embedding configuration is incomplete; dense indexing disabled"
            )
            return None
        try:
            return self.embedding_factory(self.config)
        except Exception as exc:
            self._warn(f"embedding client unavailable ({type(exc).__name__})")
            return None

    def _make_reranker(self) -> RerankProvider | None:
        if not self.config.rerank_enabled:
            return None
        if not self.config.rerank_configured:
            self._warn(
                "rerank configuration is incomplete; RRF results will be used"
            )
            return None
        try:
            return self.rerank_factory(self.config)
        except Exception as exc:
            self._warn(
                f"rerank client unavailable; RRF results will be used "
                f"({type(exc).__name__})"
            )
            return None

    @staticmethod
    def _manifest_matches(config: RagConfig, manifest: dict[str, Any]) -> bool:
        return (
            manifest.get("schema_version") == config.schema_version
            and manifest.get("embedding_fingerprint")
            == config.embedding_fingerprint
            and manifest.get("collection_name") == config.collection_name
            and isinstance(manifest.get("embedding_dimension"), int)
        )

    @staticmethod
    def _rows_for(
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        return chunk_document(document)

    @staticmethod
    def _validate_vectors(
        vectors: list[list[float]],
        expected_count: int,
    ) -> int:
        if len(vectors) != expected_count:
            raise RuntimeError("Embedding result count mismatch")
        if not vectors:
            return 0
        dimension = len(vectors[0])
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise RuntimeError("Embedding vectors have inconsistent dimensions")
        return dimension

    def _embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> tuple[list[list[float]], int]:
        if not chunks:
            return [], 0
        if self.embedding is None:
            raise RuntimeError("embedding service is unavailable")
        vectors = self.embedding.embed([chunk.text for chunk in chunks])
        return vectors, self._validate_vectors(vectors, len(chunks))

    def _rebuild(
        self,
        scan: ScanResult,
        *,
        old_exists: bool,
    ) -> dict[str, Any] | None:
        if self.store is None or self.embedding is None:
            return None
        temporary_name = (
            f"{self.config.collection_name}_build_{uuid.uuid4().hex[:8]}"
        )
        new_manifest = _default_manifest(self.config)
        files: dict[str, Any] = {}
        created = False
        old_dropped = False
        dimension: int | None = None
        failures = len(scan.failed_paths)

        try:
            for document in scan.documents:
                try:
                    chunks = self._rows_for(document)
                    vectors, candidate_dimension = self._embed_chunks(chunks)
                    if candidate_dimension:
                        if dimension is None:
                            dimension = candidate_dimension
                            self.store.create_collection(
                                temporary_name, dimension
                            )
                            created = True
                        elif candidate_dimension != dimension:
                            raise RuntimeError(
                                "Embedding dimension changed during rebuild"
                            )
                    if chunks:
                        if not created:
                            raise RuntimeError(
                                "No collection for non-empty chunks"
                            )
                        self.store.upsert(
                            temporary_name,
                            chunks,
                            vectors,
                        )
                    files[document.path] = {
                        "file_hash": document.file_hash,
                        "chunk_ids": [chunk.chunk_id for chunk in chunks],
                    }
                except Exception as exc:
                    failures += 1
                    self.stats.failed += 1
                    self._warn(
                        f"{document.path}: rebuild kept old data "
                        f"({type(exc).__name__})"
                    )

            if not created or dimension is None:
                self._warn("no indexable chunks were available for rebuild")
                return None
            if old_exists and failures:
                self._warn(
                    "rebuild was not activated because one or more files failed"
                )
                return None

            self.store.load(temporary_name)
            if old_exists:
                self.store.drop(self.config.collection_name)
                old_dropped = True
            self.store.rename(temporary_name, self.config.collection_name)
            created = False
            self.store.load(self.config.collection_name)

            new_manifest["embedding_dimension"] = dimension
            new_manifest["files"] = files
            self._save_manifest(new_manifest)
            self.stats.updated += len(files)
            self.dimension = dimension
            return new_manifest
        except Exception as exc:
            self._warn(f"collection rebuild failed ({type(exc).__name__}: {exc})")
            if old_dropped:
                try:
                    if self.store.has_collection(temporary_name):
                        self.store.rename(
                            temporary_name, self.config.collection_name
                        )
                        created = False
                except Exception:
                    pass
            return None
        finally:
            if created and not old_dropped:
                try:
                    self.store.drop(temporary_name)
                except Exception:
                    pass

    def _sync_incremental(
        self,
        scan: ScanResult,
        manifest: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Return the updated manifest and whether a full rebuild is needed."""
        if self.store is None:
            return manifest, False
        stored_files = dict(manifest.get("files", {}))
        current = {document.path: document for document in scan.documents}
        expected_dimension = int(manifest["embedding_dimension"])
        prepared: list[
            tuple[SourceDocument, list[DocumentChunk], list[list[float]]]
        ] = []

        for document in scan.documents:
            previous = stored_files.get(document.path, {})
            if previous.get("file_hash") == document.file_hash:
                self.stats.unchanged += 1
                continue
            try:
                chunks = self._rows_for(document)
                vectors, dimension = self._embed_chunks(chunks)
                if dimension and dimension != expected_dimension:
                    return manifest, True
                prepared.append((document, chunks, vectors))
            except Exception as exc:
                self.stats.failed += 1
                self._warn(
                    f"{document.path}: update skipped; old data retained "
                    f"({type(exc).__name__})"
                )

        changed = False
        for document, chunks, vectors in prepared:
            previous = stored_files.get(document.path, {})
            old_ids = list(previous.get("chunk_ids", []))
            new_ids = [chunk.chunk_id for chunk in chunks]
            try:
                self.store.upsert(
                    self.config.collection_name,
                    chunks,
                    vectors,
                )
                stale = sorted(set(old_ids) - set(new_ids))
                if stale:
                    try:
                        self.store.delete_chunks(
                            self.config.collection_name, stale
                        )
                    except Exception:
                        new_only = sorted(set(new_ids) - set(old_ids))
                        if new_only:
                            try:
                                self.store.delete_chunks(
                                    self.config.collection_name, new_only
                                )
                            except Exception:
                                pass
                        raise
                stored_files[document.path] = {
                    "file_hash": document.file_hash,
                    "chunk_ids": new_ids,
                }
                self.stats.updated += 1
                changed = True
            except Exception as exc:
                self.stats.failed += 1
                self._warn(
                    f"{document.path}: write failed; old data retained "
                    f"({type(exc).__name__})"
                )

        for path in sorted(set(stored_files) - set(current)):
            if path in scan.failed_paths:
                continue
            previous = stored_files[path]
            try:
                self.store.delete_chunks(
                    self.config.collection_name,
                    list(previous.get("chunk_ids", [])),
                )
                stored_files.pop(path, None)
                self.stats.deleted += 1
                changed = True
            except Exception as exc:
                self.stats.failed += 1
                self._warn(
                    f"{path}: deletion failed; manifest retained "
                    f"({type(exc).__name__})"
                )

        if changed:
            manifest["files"] = stored_files
            self._save_manifest(manifest)
        return manifest, False

    def start(self) -> "RagService":
        """Connect, synchronize exactly once, and decide tool availability."""
        if self._started:
            return self
        self._started = True
        started_at = time.perf_counter()
        if not self.config.enabled:
            return self

        try:
            self.store = self.store_factory(self.config)
        except Exception as exc:
            self._warn(
                f"Milvus unavailable; tool not registered "
                f"({type(exc).__name__})"
            )
            return self

        manifest = self._load_manifest()
        try:
            old_exists = self.store.has_collection(
                self.config.collection_name
            )
        except Exception as exc:
            self._warn(
                f"Milvus unavailable; tool not registered "
                f"({type(exc).__name__})"
            )
            return self
        self.embedding = self._make_embedding()
        scan = scan_resources(self.config)
        self.stats.scanned = len(scan.documents)
        self.stats.skipped = scan.skipped
        self.stats.failed = len(scan.failed_paths)
        for warning in scan.warnings:
            self._warn(warning)

        matches = self._manifest_matches(self.config, manifest)
        active_manifest: dict[str, Any] | None = manifest
        if not old_exists or not matches:
            active_manifest = self._rebuild(scan, old_exists=old_exists)
            if active_manifest is None:
                if not old_exists:
                    self.stats.elapsed_seconds = (
                        time.perf_counter() - started_at
                    )
                    self._warn("no usable collection; tool not registered")
                    return self
                self.dimension = (
                    int(manifest["embedding_dimension"])
                    if isinstance(manifest.get("embedding_dimension"), int)
                    else None
                )
                self.embedding = None
        else:
            self.dimension = int(manifest["embedding_dimension"])
            active_manifest, dimension_changed = self._sync_incremental(
                scan, manifest
            )
            if dimension_changed:
                rebuilt = self._rebuild(scan, old_exists=True)
                if rebuilt is not None:
                    active_manifest = rebuilt
                else:
                    self.embedding = None

        try:
            self.store.load(self.config.collection_name)
        except Exception as exc:
            self._warn(
                f"collection could not be loaded; tool not registered "
                f"({type(exc).__name__})"
            )
            return self

        self.available = True
        self.reranker = self._make_reranker()
        fingerprint_ok = bool(
            active_manifest
            and active_manifest.get("embedding_fingerprint")
            == self.config.embedding_fingerprint
        )
        self.mode = (
            "hybrid"
            if self.embedding is not None
            and self.dimension is not None
            and fingerprint_ok
            else "bm25_only"
        )
        self.stats.elapsed_seconds = time.perf_counter() - started_at
        self._info(
            "ready "
            f"mode={self.mode} scanned={self.stats.scanned} "
            f"rerank={'enabled' if self.reranker else 'disabled'} "
            f"skipped={self.stats.skipped} updated={self.stats.updated} "
            f"deleted={self.stats.deleted} failed={self.stats.failed} "
            f"elapsed={self.stats.elapsed_seconds:.2f}s"
        )
        return self

    @staticmethod
    def _hit_value(hit: Any, name: str, default: Any = None) -> Any:
        if isinstance(hit, dict):
            return hit.get(name, default)
        return getattr(hit, name, default)

    @classmethod
    def _format_hits(cls, raw: Any, mode: str) -> list[dict[str, Any]]:
        groups = list(raw or [])
        hits = list(groups[0]) if groups else []
        formatted: list[dict[str, Any]] = []
        for rank, hit in enumerate(hits, start=1):
            entity = cls._hit_value(hit, "entity", {}) or {}

            def field(name: str, default: Any = "") -> Any:
                if isinstance(entity, dict):
                    return entity.get(name, default)
                return getattr(entity, name, default)

            score = cls._hit_value(
                hit,
                "distance",
                cls._hit_value(hit, "score", 0.0),
            )
            item: dict[str, Any] = {
                "rank": rank,
                "path": str(field("source_path")),
                "start_line": int(field("start_line", 0)),
                "end_line": int(field("end_line", 0)),
                "symbol": str(field("symbol")),
                "content": str(field("text")),
            }
            score_name = (
                "fused_score" if mode == "hybrid" else "bm25_score"
            )
            item[score_name] = float(score)
            formatted.append(item)
        return formatted

    @staticmethod
    def _rerank_text(hit: dict[str, Any]) -> str:
        source = str(hit.get("path", ""))
        symbol = str(hit.get("symbol", "")).strip()
        lines = f"{hit.get('start_line', 0)}-{hit.get('end_line', 0)}"
        prefix = f"Source: {source}:{lines}"
        if symbol:
            prefix += f"\nSymbol: {symbol}"
        return f"{prefix}\nContent:\n{hit.get('content', '')}"

    def _apply_rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str]:
        if self.reranker is None or not hits:
            return hits[:top_k], "disabled"
        try:
            results = self.reranker.rerank(
                query,
                [self._rerank_text(hit) for hit in hits],
                min(top_k, len(hits)),
            )
            reranked: list[dict[str, Any]] = []
            for rank, result in enumerate(results, start=1):
                item = dict(hits[result.index])
                item["rank"] = rank
                item["rerank_score"] = result.score
                reranked.append(item)
            return reranked, self.config.rerank_model
        except Exception as exc:
            self._warn(
                f"rerank unavailable; using retrieval order "
                f"({type(exc).__name__})"
            )
            return hits[:top_k], "fallback"

    def search(self, query: str, top_k: int = 5) -> str:
        """Return a JSON tool result; never interpret retrieved text."""
        query = str(query).strip()
        if not query:
            return json.dumps(
                {"mode": self.mode, "query": query, "hits": []},
                ensure_ascii=False,
            )
        top_k = max(1, min(int(top_k), 8))
        if not self.available or self.store is None:
            return json.dumps(
                {
                    "mode": "unavailable",
                    "query": query,
                    "hits": [],
                    "error": "RAG service is unavailable",
                },
                ensure_ascii=False,
            )

        mode = self.mode
        candidate_limit = min(
            100,
            (
                max(top_k, self.config.rerank_candidates)
                if self.reranker is not None
                else top_k
            ),
        )
        raw: Any
        if (
            mode == "hybrid"
            and self.embedding is not None
            and self.dimension is not None
        ):
            try:
                vectors = self.embedding.embed([query])
                dimension = self._validate_vectors(vectors, 1)
                if dimension != self.dimension:
                    raise RuntimeError("query embedding dimension mismatch")
                raw = self.store.hybrid_search(
                    self.config.collection_name,
                    query,
                    vectors[0],
                    max(20, candidate_limit * 4),
                    candidate_limit,
                )
            except Exception as exc:
                mode = "bm25_only"
                self._warn(
                    f"dense query unavailable; using BM25 "
                    f"({type(exc).__name__})"
                )
                try:
                    raw = self.store.bm25_search(
                        self.config.collection_name, query, candidate_limit
                    )
                except Exception as fallback_exc:
                    return json.dumps(
                        {
                            "mode": "unavailable",
                            "query": query,
                            "hits": [],
                            "error": (
                                "RAG query failed: "
                                f"{type(fallback_exc).__name__}"
                            ),
                        },
                        ensure_ascii=False,
                    )
        else:
            try:
                raw = self.store.bm25_search(
                    self.config.collection_name, query, candidate_limit
                )
            except Exception as exc:
                return json.dumps(
                    {
                        "mode": "unavailable",
                        "query": query,
                        "hits": [],
                        "error": f"RAG query failed: {type(exc).__name__}",
                    },
                    ensure_ascii=False,
                )

        hits, rerank_mode = self._apply_rerank(
            query,
            self._format_hits(raw, mode),
            top_k,
        )
        return json.dumps(
            {
                "mode": mode,
                "rerank_mode": rerank_mode,
                "query": query,
                "score_note": (
                    "Retrieval and rerank scores are ranking signals. Rerank "
                    "scores are relative to this request, not confidence "
                    "probabilities."
                ),
                "hits": hits,
            },
            ensure_ascii=False,
        )
