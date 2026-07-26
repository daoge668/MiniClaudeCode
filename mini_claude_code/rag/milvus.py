"""Small Milvus 2.6 storage adapter for dense + BM25 retrieval."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .config import RagConfig
from .types import DocumentChunk

OUTPUT_FIELDS = [
    "text",
    "source_path",
    "start_line",
    "end_line",
    "symbol",
    "file_hash",
]


class MilvusStore:
    def __init__(self, client: Any):
        self.client = client

    @classmethod
    def connect(cls, config: RagConfig) -> "MilvusStore":
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'pymilvus'; install requirements.txt"
            ) from exc
        client = MilvusClient(
            uri=config.milvus_uri,
            token=config.milvus_token,
        )
        client.list_collections()
        return cls(client)

    def has_collection(self, name: str) -> bool:
        return bool(self.client.has_collection(collection_name=name))

    def create_collection(self, name: str, dimension: int) -> None:
        from pymilvus import DataType, Function, FunctionType

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,
            analyzer_params={
                "tokenizer": "jieba",
                "filter": ["lowercase", "removepunct"],
            },
        )
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name="source_path",
            datatype=DataType.VARCHAR,
            max_length=1_024,
        )
        schema.add_field(
            field_name="start_line",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="end_line",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="symbol",
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name="file_hash",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_function(
            Function(
                name="text_bm25_embedding",
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                function_type=FunctionType.BM25,
            )
        )

        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        indexes.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )

    def load(self, name: str) -> None:
        self.client.load_collection(collection_name=name)

    def upsert(
        self,
        name: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk/vector count mismatch")
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "dense_vector": vector,
                "source_path": chunk.source_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol": chunk.symbol,
                "file_hash": chunk.file_hash,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        if rows:
            self.client.upsert(collection_name=name, data=rows)
            self.client.flush(collection_name=name)

    def delete_chunks(self, name: str, chunk_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(chunk_ids))
        for start in range(0, len(ids), 256):
            batch = ids[start : start + 256]
            if not batch:
                continue
            self.client.delete(
                collection_name=name,
                filter=f"chunk_id in {json.dumps(batch)}",
            )

    def drop(self, name: str) -> None:
        if self.has_collection(name):
            self.client.drop_collection(collection_name=name)

    def rename(self, old_name: str, new_name: str) -> None:
        self.client.rename_collection(old_name, new_name)

    def hybrid_search(
        self,
        name: str,
        query: str,
        vector: list[float],
        candidates: int,
        limit: int,
    ) -> Any:
        from pymilvus import AnnSearchRequest, Function, FunctionType

        dense = AnnSearchRequest(
            data=[vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=candidates,
        )
        sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=candidates,
        )
        ranker = Function(
            name="rrf",
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": 60},
        )
        return self.client.hybrid_search(
            collection_name=name,
            reqs=[dense, sparse],
            ranker=ranker,
            limit=limit,
            output_fields=OUTPUT_FIELDS,
            consistency_level="Strong",
        )

    def bm25_search(
        self,
        name: str,
        query: str,
        limit: int,
    ) -> Any:
        return self.client.search(
            collection_name=name,
            data=[query],
            anns_field="sparse_vector",
            search_params={"metric_type": "BM25", "params": {}},
            limit=limit,
            output_fields=OUTPUT_FIELDS,
            consistency_level="Strong",
        )
