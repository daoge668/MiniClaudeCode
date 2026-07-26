from __future__ import annotations

import os
import uuid

import pytest

from mini_claude_code.rag.config import RagConfig
from mini_claude_code.rag.milvus import MilvusStore
from mini_claude_code.rag.types import DocumentChunk


@pytest.mark.integration
def test_docker_milvus_chinese_code_and_hybrid_search(tmp_path) -> None:
    if os.getenv("RUN_MILVUS_TESTS") != "1":
        pytest.skip("set RUN_MILVUS_TESTS=1 with Docker Milvus running")

    config = RagConfig(
        project_dir=tmp_path,
        enabled=True,
        milvus_uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"),
        milvus_token=os.getenv("MILVUS_TOKEN", "root:Milvus"),
    )
    store = MilvusStore.connect(config)
    name = f"rag_test_{uuid.uuid4().hex[:12]}"
    chunks = [
        DocumentChunk(
            chunk_id="a" * 64,
            text="部署密钥保存在本地配置中。",
            source_path="guide.md",
            start_line=2,
            end_line=2,
            symbol="部署",
            file_hash="1" * 64,
        ),
        DocumentChunk(
            chunk_id="b" * 64,
            text="Use search_project_knowledge before private answers.",
            source_path="agent.py",
            start_line=10,
            end_line=10,
            symbol="tools",
            file_hash="2" * 64,
        ),
    ]
    try:
        store.create_collection(name, 3)
        store.upsert(name, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        store.load(name)

        chinese = store.bm25_search(name, "部署密钥", 2)
        code = store.bm25_search(name, "search_project_knowledge", 2)
        hybrid = store.hybrid_search(
            name, "部署密钥", [1.0, 0.0, 0.0], 20, 2
        )

        assert chinese and chinese[0]
        assert code and code[0]
        assert hybrid and hybrid[0]
    finally:
        store.drop(name)
