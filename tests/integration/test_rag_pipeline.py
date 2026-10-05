"""导入、检索、生成和引用的本地闭环。Embedding 与 LLM 都是假的。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import EmbeddingError, LLMError
from app.ingestion.pipeline import IngestionPipeline
from app.rag.context_builder import ContextBuilder
from app.rag.pipeline import RAGPipeline
from app.rag.prompt import PromptBuilder
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from tests.fake_embedding import HashEmbedding
from tests.fake_llm import ScriptedLLM
from tests.helpers import make_settings

_SPRING = "事务通过代理生效。自调用不会开启事务。"
_POOL = "数据库连接池的大小需要单独配置。"


def _store(path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=path,
        collection_name="kb_local",
        embedding_model="fake-hash",
        embedding_dimension=8,
    )


def _ingest(embedder: HashEmbedding, store: ChromaVectorStore):
    ingestion = IngestionPipeline(
        make_settings(chunk_size=200, chunk_overlap=20),
        embedder=embedder,
        vector_store=store,
    )
    spring = ingestion.ingest_bytes("spring.md", _SPRING.encode(), source_key="kb/spring")
    pool = ingestion.ingest_bytes("pool.md", _POOL.encode(), source_key="kb/pool")
    return spring, pool


def _rag(
    embedder: HashEmbedding,
    store: ChromaVectorStore,
    llm: ScriptedLLM,
    *,
    max_distance: float | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        retriever=VectorRetriever(embedder, store, max_distance=max_distance),
        context_builder=ContextBuilder(2000),
        prompt_builder=PromptBuilder(),
        llm=llm,
        default_top_k=2,
        max_top_k=5,
        max_question_chars=200,
    )


def test_exact_question_retrieves_and_cites_the_same_chunk(tmp_path: Path) -> None:
    embedder = HashEmbedding()
    store = _store(tmp_path / "chroma")
    llm = ScriptedLLM("事务通过代理生效。[C1] [C99]")
    try:
        spring, pool = _ingest(embedder, store)
        response = _rag(embedder, store, llm).query(spring.chunks[0].text)
        assert response.retrieved_chunks[0].chunk_id == spring.chunks[0].chunk_id
        assert response.retrieved_chunks[0].score < 1e-5
        assert response.retrieved_chunks[0].document_id != pool.document.document_id
        assert [item.citation_id for item in response.citations] == ["C1"]
        citation = response.citations[0]
        assert citation.chunk_id == spring.chunks[0].chunk_id
        assert citation.document_id == spring.document.document_id
        assert citation.filename == "spring.md"
        assert citation.text == spring.chunks[0].text
        assert "事务通过代理" not in llm.calls[0][0]
        assert spring.chunks[0].text in llm.calls[0][1]
        assert response.metrics.prompt_tokens == 11
        assert response.metrics.completion_tokens == 7
    finally:
        store.close()


def test_distance_limit_and_failures_leave_the_index_intact(tmp_path: Path) -> None:
    embedder = HashEmbedding()
    store = _store(tmp_path / "chroma")
    llm = ScriptedLLM()
    try:
        spring, _pool = _ingest(embedder, store)
        limited = _rag(embedder, store, llm, max_distance=1e-4).query(spring.chunks[0].text)
        assert [item.chunk_id for item in limited.retrieved_chunks] == [spring.chunks[0].chunk_id]

        llm.fail = True
        with pytest.raises(LLMError, match="模拟生成失败"):
            _rag(embedder, store, llm).query(spring.chunks[0].text)
        assert store.get_by_document_id(spring.document.document_id)

        embedder.fail = True
        calls_before = len(llm.calls)
        with pytest.raises(EmbeddingError, match="模拟向量化失败"):
            _rag(embedder, store, llm).query("事务通过代理生效。")
        assert len(llm.calls) == calls_before
    finally:
        store.close()


def test_empty_index_asks_the_model_to_say_evidence_is_missing(tmp_path: Path) -> None:
    embedder = HashEmbedding()
    store = _store(tmp_path / "empty")
    llm = ScriptedLLM("不该使用这条。")
    try:
        response = _rag(embedder, store, llm).query("这里没有记载的问题")
        assert response.retrieved_chunks == []
        assert response.citations == []
        assert response.answer == "知识库中缺少相关信息。"
    finally:
        store.close()
