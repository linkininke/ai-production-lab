"""向量检索器。分数是距离，阈值把更远的命中丢掉。"""

from __future__ import annotations

import logging

import pytest

from app.core.exceptions import ConfigurationError, EmbeddingError, RetrievalError
from app.retrieval.factory import RetrieverFactory
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever
from tests.fake_embedding import HashEmbedding


class RankingStore:
    def __init__(self, hits: list[RetrievalResult], *, model: str = "fake-hash") -> None:
        self.hits = hits
        self._model = model
        self.last_top_k: int | None = None

    @property
    def embedding_model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return 8

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        self.last_top_k = top_k
        assert len(query_embedding) == 8
        return self.hits[:top_k]


def _hit(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc_a",
        text=chunk_id,
        score=score,
        metadata={
            "filename": "note.md",
            "file_type": "markdown",
            "chunk_index": 0,
            "document_id": "doc_a",
        },
    )


def test_distance_limit_drops_far_hits_and_keeps_order(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    store = RankingStore([_hit("near", 0.1), _hit("far", 0.9)])
    retriever = VectorRetriever(HashEmbedding(), store, max_distance=0.5)
    hits = retriever.retrieve("事务为什么失效", top_k=5)
    assert [hit.chunk_id for hit in hits] == ["near"]
    assert store.last_top_k == 5
    assert hits[0].score_kind == "distance"
    assert hits[0].retriever == "vector"
    assert hits[0].score == 0.1
    assert "retriever=vector" in caplog.text
    assert "top_k=5" in caplog.text
    assert "result_count=1" in caplog.text
    assert "latency_ms=" in caplog.text
    assert "事务为什么失效" not in caplog.text


def test_blank_query_does_not_embed() -> None:
    embedder = HashEmbedding()
    retriever = VectorRetriever(embedder, RankingStore([]))
    with pytest.raises(RetrievalError, match="不能为空"):
        retriever.retrieve("  ")
    assert embedder.document_calls == 0


def test_embedding_failure_is_not_rewritten() -> None:
    embedder = HashEmbedding()
    embedder.fail = True
    retriever = VectorRetriever(embedder, RankingStore([_hit("near", 0.1)]))
    with pytest.raises(EmbeddingError, match="模拟向量化失败"):
        retriever.retrieve("事务")


def test_mismatched_model_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="不一致"):
        VectorRetriever(HashEmbedding(), RankingStore([], model="other-model"))


def test_factory_returns_vector_retriever() -> None:
    store = RankingStore([_hit("near", 0.2)])
    retriever = RetrieverFactory(HashEmbedding(), store, max_distance=0.5).create()
    hits = retriever.retrieve("事务", top_k=3)
    assert hits[0].retriever == "vector"
    assert hits[0].score_kind == "distance"
    assert store.last_top_k == 3


def test_factory_rejects_unknown_mode() -> None:
    factory = RetrieverFactory(HashEmbedding(), RankingStore([]))
    with pytest.raises(ConfigurationError, match="未知检索模式"):
        factory.create("graph")
