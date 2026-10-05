"""本地 Chroma 索引：写入、重启后读取、删除。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chunking.models import Chunk
from app.core.exceptions import EmbeddingError, VectorStoreError
from app.ingestion.pipeline import IngestionPipeline
from app.vectorstore.chroma_store import ChromaVectorStore, probe_vector_store
from tests.fake_embedding import HashEmbedding
from tests.helpers import make_settings


def _chunk(chunk_id: str, document_id: str, text: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=index,
        metadata={
            "filename": "note.md",
            "file_type": "markdown",
            "chunk_index": index,
            "document_id": document_id,
            "content_hash": "hash",
        },
    )


def _store(path: Path, *, model: str = "fake-hash", dimension: int = 8) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=path,
        collection_name="kb_local",
        embedding_model=model,
        embedding_dimension=dimension,
    )


def test_persists_across_a_new_client(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    embedder = HashEmbedding()
    original = _chunk("chunk_a", "doc_a", "事务通过代理生效")
    first = _store(path)
    try:
        assert first.search(embedder.embed_query("没有资料"), top_k=3) == []
        first.add_chunks([original], embedder.embed_documents([original.text]))
    finally:
        first.close()

    second = _store(path)
    try:
        stored = second.get_by_document_id("doc_a")
        assert [chunk.chunk_id for chunk in stored] == ["chunk_a"]
        assert stored[0].text == original.text
        hits = second.search(embedder.embed_query(original.text), top_k=3)
        assert hits[0].chunk_id == "chunk_a"
        assert hits[0].score_kind == "distance"
        assert hits[0].score < 1e-5
    finally:
        second.close()


def test_delete_by_document_id_removes_every_chunk(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    embedder = HashEmbedding()
    chunks = [
        _chunk("chunk_0", "doc_a", "第一段事务说明", 0),
        _chunk("chunk_1", "doc_a", "第二段传播行为", 1),
    ]
    store = _store(path)
    try:
        store.add_chunks(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
        assert store.delete_by_document_id("doc_missing") == 0
        assert store.delete_by_document_id("doc_a") == 2
        assert store.get_by_document_id("doc_a") == []
        assert store.search(embedder.embed_query("事务"), top_k=3) == []
    finally:
        store.close()


def test_duplicate_ids_and_bad_dimensions_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path / "chroma")
    try:
        chunk = _chunk("chunk_a", "doc_a", "正文")
        with pytest.raises(VectorStoreError, match="重复"):
            store.add_chunks([chunk, chunk], [[0.0] * 8, [0.0] * 8])
        with pytest.raises(VectorStoreError, match="维度"):
            store.add_chunks([chunk], [[0.1, 0.2]])
        with pytest.raises(VectorStoreError, match="top_k"):
            store.search([0.0] * 8, top_k=0)
    finally:
        store.close()


def test_reopen_rejects_a_different_embedding_signature(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    store = _store(path)
    store.close()
    with pytest.raises(VectorStoreError, match="不兼容"):
        _store(path, model="another-model")
    assert (
        probe_vector_store(
            path,
            collection_name="kb_local",
            embedding_model="another-model",
            embedding_dimension=8,
        )
        == "error"
    )
    assert (
        probe_vector_store(
            path,
            collection_name="kb_local",
            embedding_model="fake-hash",
            embedding_dimension=8,
        )
        == "ok"
    )


def test_pipeline_skips_unchanged_documents_and_keeps_old_vectors_on_embed_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chroma"
    embedder = HashEmbedding()
    store = _store(path)
    pipeline = IngestionPipeline(
        make_settings(chunk_size=40, chunk_overlap=8),
        embedder=embedder,
        vector_store=store,
    )
    try:
        created = pipeline.ingest_bytes(
            "spring.md",
            "事务通过代理生效。".encode(),
            source_key="kb/spring",
        )
        assert created.status == "completed"
        assert embedder.document_calls == 1
        assert store.get_by_document_id(created.document.document_id)

        skipped = pipeline.ingest_bytes(
            "spring.md",
            "事务通过代理生效。".encode(),
            source_key="kb/spring",
        )
        assert skipped.status == "skipped"
        assert embedder.document_calls == 1

        embedder.fail = True
        with pytest.raises(EmbeddingError, match="模拟向量化失败"):
            pipeline.ingest_bytes(
                "spring.md",
                "这段内容已经改写。".encode(),
                source_key="kb/spring",
            )
        assert [
            chunk.chunk_id for chunk in store.get_by_document_id(created.document.document_id)
        ] == [chunk.chunk_id for chunk in created.chunks]

        embedder.fail = False
        replaced = pipeline.ingest_bytes(
            "spring.md",
            "这段内容已经改写。".encode(),
            source_key="kb/spring",
        )
        assert replaced.status == "completed"
        assert replaced.document.document_id == created.document.document_id
        stored_ids = [
            chunk.chunk_id for chunk in store.get_by_document_id(created.document.document_id)
        ]
        assert stored_ids == [chunk.chunk_id for chunk in replaced.chunks]
        assert created.chunks[0].chunk_id not in stored_ids
    finally:
        store.close()
