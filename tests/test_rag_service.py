from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from mini_claude_code.rag.config import RagConfig
from mini_claude_code.rag.service import RagService
from mini_claude_code.rag.reranker import RerankResult


class FakeEmbedding:
    def __init__(self, *, fail: bool = False, dimension: int = 3):
        self.fail = fail
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("offline")
        return [
            [float((len(text) + index) % 7) for index in range(self.dimension)]
            for text in texts
        ]


class FakeStore:
    def __init__(self):
        self.collections: dict[str, dict[str, Any]] = {}
        self.deleted_batches: list[list[str]] = []
        self.hybrid_calls = 0
        self.bm25_calls = 0

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, dimension: int) -> None:
        self.collections[name] = {"dimension": dimension, "rows": {}}

    def load(self, name: str) -> None:
        if name not in self.collections:
            raise RuntimeError("missing")

    def upsert(self, name: str, chunks: list, vectors: list) -> None:
        rows = self.collections[name]["rows"]
        for chunk, vector in zip(chunks, vectors):
            rows[chunk.chunk_id] = {
                "chunk": chunk,
                "vector": vector,
            }

    def delete_chunks(self, name: str, chunk_ids: list[str]) -> None:
        self.deleted_batches.append(list(chunk_ids))
        rows = self.collections[name]["rows"]
        for chunk_id in chunk_ids:
            rows.pop(chunk_id, None)

    def drop(self, name: str) -> None:
        self.collections.pop(name, None)

    def rename(self, old_name: str, new_name: str) -> None:
        self.collections[new_name] = self.collections.pop(old_name)

    def hybrid_search(
        self,
        name: str,
        query: str,
        vector: list[float],
        candidates: int,
        limit: int,
    ) -> list[list[dict]]:
        self.hybrid_calls += 1
        rows = list(self.collections[name]["rows"].values())[:limit]
        return [
            [
                {
                    "distance": 0.03,
                    "entity": {
                        "text": row["chunk"].text,
                        "source_path": row["chunk"].source_path,
                        "start_line": row["chunk"].start_line,
                        "end_line": row["chunk"].end_line,
                        "symbol": row["chunk"].symbol,
                        "file_hash": row["chunk"].file_hash,
                    },
                }
                for row in rows
            ]
        ]

    def bm25_search(
        self, name: str, query: str, limit: int
    ) -> list[list[dict]]:
        self.bm25_calls += 1
        return self.hybrid_search(name, query, [], limit, limit)


class FakeReranker:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        self.calls.append((query, documents, top_n))
        if self.fail:
            raise RuntimeError("offline")
        indices = list(reversed(range(len(documents))))[:top_n]
        return [
            RerankResult(index=index, score=0.9 - rank * 0.1)
            for rank, index in enumerate(indices)
        ]


def make_config(project: Path) -> RagConfig:
    return RagConfig(
        project_dir=project,
        enabled=True,
        milvus_uri="http://milvus:19530",
        milvus_token="token",
        embedding_base_url="https://embedding.example/v1",
        embedding_api_key="secret",
        embedding_model="qwen3.7-text-embedding",
    )


def start_service(
    config: RagConfig,
    store: FakeStore,
    embedding: FakeEmbedding,
) -> RagService:
    return RagService(
        config,
        printer=lambda _: None,
        store_factory=lambda _: store,
        embedding_factory=lambda _: embedding,
    ).start()


def test_first_build_and_unchanged_restart_use_zero_embeddings(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "facts.md").write_text(
        "# Fact\nThe launch code is ORCHID-731.\n", encoding="utf-8"
    )
    config = make_config(tmp_path)
    store = FakeStore()
    first_embedding = FakeEmbedding()

    first = start_service(config, store, first_embedding)

    assert first.available
    assert first.mode == "hybrid"
    assert len(first_embedding.calls) == 1
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding_dimension"] == 3
    assert "resources/facts.md" in manifest["files"]

    second_embedding = FakeEmbedding()
    second = start_service(config, store, second_embedding)

    assert second.available
    assert second.stats.unchanged == 1
    assert second_embedding.calls == []


def test_incremental_modify_and_delete(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    document = resources / "facts.txt"
    document.write_text("old fact", encoding="utf-8")
    config = make_config(tmp_path)
    store = FakeStore()
    start_service(config, store, FakeEmbedding())

    document.write_text("new fact", encoding="utf-8")
    changed_embedding = FakeEmbedding()
    changed = start_service(config, store, changed_embedding)
    assert changed.stats.updated == 1
    assert len(changed_embedding.calls) == 1
    rows = store.collections[config.collection_name]["rows"]
    assert [row["chunk"].text for row in rows.values()] == ["new fact"]

    document.unlink()
    deleted_embedding = FakeEmbedding()
    deleted = start_service(config, store, deleted_embedding)
    assert deleted.stats.deleted == 1
    assert deleted_embedding.calls == []
    assert store.collections[config.collection_name]["rows"] == {}


def test_embedding_outage_retains_old_file_and_falls_back_to_bm25(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    document = resources / "facts.txt"
    document.write_text("stable fact", encoding="utf-8")
    config = make_config(tmp_path)
    store = FakeStore()
    start_service(config, store, FakeEmbedding())
    old_ids = set(store.collections[config.collection_name]["rows"])

    document.write_text("unindexed change", encoding="utf-8")
    failed = start_service(config, store, FakeEmbedding(fail=True))

    assert failed.available
    assert failed.stats.failed == 1
    assert set(store.collections[config.collection_name]["rows"]) == old_ids
    result = json.loads(failed.search("stable", top_k=5))
    assert result["mode"] == "bm25_only"
    assert "bm25_score" in result["hits"][0]
    assert store.bm25_calls == 1


def test_dimension_change_rebuilds_collection(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    document = resources / "facts.txt"
    document.write_text("version one", encoding="utf-8")
    config = make_config(tmp_path)
    store = FakeStore()
    start_service(config, store, FakeEmbedding(dimension=3))

    document.write_text("version two", encoding="utf-8")
    rebuilt = start_service(config, store, FakeEmbedding(dimension=4))

    assert rebuilt.available
    assert rebuilt.dimension == 4
    assert store.collections[config.collection_name]["dimension"] == 4


def test_query_reranks_rrf_candidates_and_preserves_both_scores(
    tmp_path: Path,
) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "a.txt").write_text("first candidate", encoding="utf-8")
    (resources / "b.txt").write_text("second candidate", encoding="utf-8")
    config = replace(
        make_config(tmp_path),
        rerank_enabled=True,
        rerank_api_key="secret",
        rerank_candidates=20,
    )
    store = FakeStore()
    reranker = FakeReranker()
    service = RagService(
        config,
        printer=lambda _: None,
        store_factory=lambda _: store,
        embedding_factory=lambda _: FakeEmbedding(),
        rerank_factory=lambda _: reranker,
    ).start()

    result = json.loads(service.search("candidate", top_k=2))

    assert result["mode"] == "hybrid"
    assert result["rerank_mode"] == "qwen3-vl-rerank"
    assert [hit["content"] for hit in result["hits"]] == [
        "second candidate",
        "first candidate",
    ]
    assert result["hits"][0]["fused_score"] == 0.03
    assert result["hits"][0]["rerank_score"] == 0.9
    assert len(reranker.calls) == 1
    assert reranker.calls[0][2] == 2


def test_rerank_outage_falls_back_to_rrf_order(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "a.txt").write_text("first candidate", encoding="utf-8")
    (resources / "b.txt").write_text("second candidate", encoding="utf-8")
    config = replace(
        make_config(tmp_path),
        rerank_enabled=True,
        rerank_api_key="secret",
    )
    store = FakeStore()
    service = RagService(
        config,
        printer=lambda _: None,
        store_factory=lambda _: store,
        embedding_factory=lambda _: FakeEmbedding(),
        rerank_factory=lambda _: FakeReranker(fail=True),
    ).start()

    result = json.loads(service.search("candidate", top_k=2))

    assert result["mode"] == "hybrid"
    assert result["rerank_mode"] == "fallback"
    assert [hit["content"] for hit in result["hits"]] == [
        "first candidate",
        "second candidate",
    ]
    assert all("rerank_score" not in hit for hit in result["hits"])
