"""BM25 从已经写入的 Chroma 片段重建，不另存一份关键词库。"""

from __future__ import annotations

from pathlib import Path

from app.chunking.models import Chunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.tokenizer import JiebaTokenizer
from app.vectorstore.chroma_store import ChromaVectorStore
from tests.fake_embedding import HashEmbedding


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        text=text,
        chunk_index=0,
        metadata={
            "filename": f"{chunk_id}.md",
            "file_type": "markdown",
            "chunk_index": 0,
            "document_id": chunk_id,
        },
    )


def test_bm25_reads_chunks_persisted_in_chroma(tmp_path: Path) -> None:
    path = tmp_path / "chroma"
    embedder = HashEmbedding()
    chunks = [
        _chunk("overflow", "连接池参数 max_overflow 控制溢出连接数。"),
        _chunk("proxy", "事务通过代理生效。"),
    ]
    writer = ChromaVectorStore(
        persist_dir=path,
        collection_name="kb_local",
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )
    try:
        writer.add_chunks(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    finally:
        writer.close()

    reader = ChromaVectorStore(
        persist_dir=path,
        collection_name="kb_local",
        embedding_model=embedder.model_name,
        embedding_dimension=embedder.dimension,
    )
    try:
        stored = reader.list_chunks()
        assert {chunk.chunk_id for chunk in stored} == {"overflow", "proxy"}
        retriever = BM25Retriever(reader, JiebaTokenizer())
        hits = retriever.retrieve("max_overflow", top_k=3)
        assert [hit.chunk_id for hit in hits] == ["overflow"]
        assert all(hit.score_kind == "bm25" for hit in hits)
        assert reader.delete_by_document_id("overflow") == 1
        assert retriever.retrieve("max_overflow", top_k=3) == []
    finally:
        reader.close()
