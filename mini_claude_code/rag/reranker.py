"""DashScope qwen3-vl-rerank HTTP adapter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib.request import Request, urlopen

from .config import RagConfig

UrlOpen = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class QwenVLReranker:
    """Rerank text candidates through the model's native DashScope API."""

    def __init__(
        self,
        config: RagConfig,
        *,
        opener: UrlOpen = urlopen,
    ):
        self.url = config.rerank_url
        self.api_key = config.rerank_api_key
        self.model = config.rerank_model
        self.timeout = config.rerank_timeout_seconds
        self._opener = opener

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankResult]:
        if not documents:
            return []
        requested = max(1, min(int(top_n), len(documents)))
        payload = {
            "model": self.model,
            "input": {
                "query": {"text": str(query)},
                "documents": [{"text": str(text)} for text in documents],
            },
            "parameters": {
                "return_documents": False,
                "top_n": requested,
                "instruct": (
                    "Given a project knowledge query, retrieve passages that "
                    "directly answer it. Preserve exact code identifiers, "
                    "configuration names, constraints, and version details."
                ),
            },
        }
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Rerank endpoint request failed ({type(exc).__name__})"
            ) from exc
        if status < 200 or status >= 300:
            raise RuntimeError(f"Rerank endpoint returned HTTP {status}")
        try:
            body = json.loads(raw.decode("utf-8"))
            rows = body["output"]["results"]
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                "Rerank endpoint returned an invalid response"
            ) from exc
        if not isinstance(rows, list) or len(rows) != requested:
            raise RuntimeError("Rerank endpoint returned an incomplete ranking")

        results: list[RerankResult] = []
        seen: set[int] = set()
        for row in rows:
            try:
                index = int(row["index"])
                score = float(row["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Rerank endpoint returned an invalid result"
                ) from exc
            if (
                index < 0
                or index >= len(documents)
                or index in seen
                or not math.isfinite(score)
                or score < 0.0
                or score > 1.0
            ):
                raise RuntimeError("Rerank endpoint returned an invalid ranking")
            seen.add(index)
            results.append(RerankResult(index=index, score=score))
        return results
