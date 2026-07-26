"""OpenAI-compatible embedding adapter."""

from __future__ import annotations

from typing import Any, Sequence

from .config import RagConfig


class OpenAIEmbeddingClient:
    """Send only ``model`` and batched ``input`` to the embeddings endpoint."""

    def __init__(self, config: RagConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'openai'; install requirements.txt"
            ) from exc
        self.model = config.embedding_model
        self.batch_size = config.embedding_batch_size
        self._client = OpenAI(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response: Any = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            ordered = sorted(response.data, key=lambda item: int(item.index))
            if len(ordered) != len(batch):
                raise RuntimeError(
                    "Embedding endpoint returned a different result count"
                )
            vectors.extend([list(item.embedding) for item in ordered])
        if not vectors or any(not vector for vector in vectors):
            raise RuntimeError("Embedding endpoint returned an empty vector")
        return vectors
