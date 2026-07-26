from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mini_claude_code.rag.config import RagConfig
from mini_claude_code.rag.reranker import QwenVLReranker


class FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_qwen_vl_reranker_uses_native_dashscope_shape(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def open_request(request: Any, *, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.91},
                        {"index": 0, "relevance_score": 0.42},
                    ]
                }
            }
        )

    config = RagConfig(
        project_dir=tmp_path,
        enabled=True,
        milvus_uri="http://milvus:19530",
        milvus_token="token",
        rerank_enabled=True,
        rerank_url="https://rerank.example/api",
        rerank_api_key="secret",
        rerank_model="qwen3-vl-rerank",
    )
    client = QwenVLReranker(config, opener=open_request)

    results = client.rerank("哪个文档？", ["第一段", "第二段"], 2)

    assert captured["url"] == "https://rerank.example/api"
    assert captured["authorization"] == "Bearer secret"
    assert captured["timeout"] == 30.0
    assert captured["payload"] == {
        "model": "qwen3-vl-rerank",
        "input": {
            "query": {"text": "哪个文档？"},
            "documents": [{"text": "第一段"}, {"text": "第二段"}],
        },
        "parameters": {
            "return_documents": False,
            "top_n": 2,
            "instruct": (
                "Given a project knowledge query, retrieve passages that "
                "directly answer it. Preserve exact code identifiers, "
                "configuration names, constraints, and version details."
            ),
        },
    }
    assert [(item.index, item.score) for item in results] == [
        (1, 0.91),
        (0, 0.42),
    ]


def test_rerank_api_key_defaults_to_embedding_key(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_API_KEY", "shared-secret")
    monkeypatch.delenv("RERANK_API_KEY", raising=False)

    config = RagConfig.from_env(tmp_path)

    assert config.rerank_configured
    assert config.rerank_api_key == "shared-secret"
