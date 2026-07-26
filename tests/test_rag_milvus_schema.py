from __future__ import annotations

from pymilvus import MilvusClient

from mini_claude_code.rag.milvus import MilvusStore


class SchemaCapture:
    def create_schema(self, **kwargs):
        return MilvusClient.create_schema(**kwargs)

    def prepare_index_params(self):
        return MilvusClient.prepare_index_params()

    def create_collection(self, **kwargs):
        self.created = kwargs


def test_collection_schema_has_dense_bm25_and_reference_fields() -> None:
    capture = SchemaCapture()

    MilvusStore(capture).create_collection("schema_test", 1024)

    schema = capture.created["schema"]
    field_names = [field.name for field in schema.fields]
    assert field_names == [
        "chunk_id",
        "text",
        "dense_vector",
        "sparse_vector",
        "source_path",
        "start_line",
        "end_line",
        "symbol",
        "file_hash",
    ]
    assert [function.name for function in schema.functions] == [
        "text_bm25_embedding"
    ]
    assert len(capture.created["index_params"]) == 2
