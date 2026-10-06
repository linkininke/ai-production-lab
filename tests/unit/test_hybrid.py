"""混合检索先扩大候选，再按 RRF 截成最终 Top-K。"""

from __future__ import annotations

import logging

import pytest

from app.core.exceptions import ConfigurationError, RetrievalError
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult, RetrieverName, ScoreKind


class ScriptedRetriever:
    def __init__(self, hits: list[RetrievalResult]) -> None:
        self.hits = hits
        self.calls: list[int] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append(top_k)
        return self.hits[:top_k]


class BoomRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        raise RetrievalError("向量检索失败")


def _hit(
    chunk_id: str,
    retriever: RetrieverName,
    score: float,
    score_kind: ScoreKind,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=f"机密片段{chunk_id}",
        score=score,
        score_kind=score_kind,
        retriever=retriever,
        metadata={"filename": f"{chunk_id}.md"},
    )


def test_hybrid_fuses_candidates_and_logs_sources_without_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    vector = ScriptedRetriever(
        [
            _hit("A", "vector", 0.2, "distance"),
            _hit("B", "vector", 0.4, "distance"),
            _hit("C", "vector", 0.6, "distance"),
        ]
    )
    bm25 = ScriptedRetriever(
        [
            _hit("C", "bm25", 9.0, "bm25"),
            _hit("A", "bm25", 5.0, "bm25"),
            _hit("D", "bm25", 1.0, "bm25"),
        ]
    )
    retriever = HybridRetriever(vector, bm25, vector_top_k=20, bm25_top_k=20, final_top_k=10)
    hits = retriever.retrieve("机密问题唯一标记", top_k=10)
    assert [hit.chunk_id for hit in hits] == ["A", "C", "B", "D"]
    assert hits[0].score_kind == "rrf"
    assert vector.calls == [20]
    assert bm25.calls == [20]
    assert "vector_count=3" in caplog.text
    assert "bm25_count=3" in caplog.text
    assert "hybrid_hit chunk_id=A" in caplog.text
    assert "sources=vector:1,bm25:2" in caplog.text
    assert "机密问题唯一标记" not in caplog.text
    assert "机密片段A" not in caplog.text


def test_final_top_k_is_smaller_than_the_candidate_pool() -> None:
    hits = [_hit(f"c{index}", "vector", 0.1 * index, "distance") for index in range(15)]
    vector = ScriptedRetriever(hits)
    bm25 = ScriptedRetriever([])
    retriever = HybridRetriever(vector, bm25, vector_top_k=20, bm25_top_k=20, final_top_k=10)
    selected = retriever.retrieve("事务")
    assert len(selected) == 10
    assert vector.calls == [20]
    assert bm25.calls == [20]
    assert selected[0].sources[0].retriever == "vector"


def test_requested_top_k_can_expand_the_candidate_pool() -> None:
    vector = ScriptedRetriever([_hit("A", "vector", 0.1, "distance")])
    bm25 = ScriptedRetriever([])
    retriever = HybridRetriever(vector, bm25, vector_top_k=20, bm25_top_k=20, final_top_k=10)
    retriever.retrieve("事务", top_k=25)
    assert vector.calls == [25]
    assert bm25.calls == [25]


def test_blank_query_does_not_call_either_retriever() -> None:
    vector = ScriptedRetriever([])
    bm25 = ScriptedRetriever([])
    retriever = HybridRetriever(vector, bm25)
    with pytest.raises(RetrievalError, match="不能为空"):
        retriever.retrieve("  ")
    assert vector.calls == []
    assert bm25.calls == []


def test_vector_failure_is_not_replaced_by_bm25() -> None:
    bm25 = ScriptedRetriever([_hit("A", "bm25", 1.0, "bm25")])
    retriever = HybridRetriever(BoomRetriever(), bm25)
    with pytest.raises(RetrievalError, match="向量检索失败"):
        retriever.retrieve("事务")
    assert bm25.calls == []


def test_invalid_candidate_count_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="HYBRID_VECTOR_TOP_K"):
        HybridRetriever(ScriptedRetriever([]), ScriptedRetriever([]), vector_top_k=0)
