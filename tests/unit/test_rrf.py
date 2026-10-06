"""RRF 只按名次加分。同一 chunk 合并，某一路为空不影响另一路。"""

from __future__ import annotations

import pytest

from app.core.exceptions import RetrievalError
from app.retrieval.models import RetrievalResult, RetrieverName, ScoreKind
from app.retrieval.rrf import reciprocal_rank_fusion

_K = 60


def _hit(
    chunk_id: str,
    retriever: RetrieverName,
    score: float,
    score_kind: ScoreKind,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=f"正文标记{chunk_id}",
        score=score,
        score_kind=score_kind,
        retriever=retriever,
        metadata={"filename": f"{chunk_id}.md"},
    )


def _points(*ranks: int, k: int = _K) -> float:
    return sum(1 / (k + rank) for rank in ranks)


def test_overlapping_lists_rank_by_reciprocal_rank() -> None:
    vector = [
        _hit("A", "vector", 0.2, "distance"),
        _hit("B", "vector", 0.4, "distance"),
        _hit("C", "vector", 0.6, "distance"),
    ]
    bm25 = [
        _hit("C", "bm25", 9.0, "bm25"),
        _hit("A", "bm25", 5.0, "bm25"),
        _hit("D", "bm25", 1.0, "bm25"),
    ]
    hits = reciprocal_rank_fusion([vector, bm25], k=_K)
    assert [hit.chunk_id for hit in hits] == ["A", "C", "B", "D"]
    assert hits[0].score == pytest.approx(_points(1, 2))
    assert hits[1].score == pytest.approx(_points(3, 1))
    assert hits[0].score != pytest.approx(0.2 + 5.0)
    assert hits[0].score_kind == "rrf"
    assert hits[0].retriever == "hybrid"
    assert [(item.retriever, item.rank) for item in hits[0].sources] == [
        ("vector", 1),
        ("bm25", 2),
    ]
    assert [(item.retriever, item.rank, item.score) for item in hits[1].sources] == [
        ("vector", 3, 0.6),
        ("bm25", 1, 9.0),
    ]


def test_chunk_found_only_by_vector_keeps_that_rank() -> None:
    hits = reciprocal_rank_fusion(
        [
            [_hit("A", "vector", 0.1, "distance"), _hit("B", "vector", 0.2, "distance")],
            [_hit("A", "bm25", 3.0, "bm25")],
        ],
        k=_K,
    )
    only_vector = next(hit for hit in hits if hit.chunk_id == "B")
    assert only_vector.score == pytest.approx(_points(2))
    assert [(item.retriever, item.rank) for item in only_vector.sources] == [("vector", 2)]


def test_chunk_found_only_by_bm25_keeps_that_rank() -> None:
    hits = reciprocal_rank_fusion(
        [
            [_hit("A", "vector", 0.1, "distance")],
            [_hit("D", "bm25", 1.5, "bm25")],
        ],
        k=_K,
    )
    only_bm25 = next(hit for hit in hits if hit.chunk_id == "D")
    assert only_bm25.score == pytest.approx(_points(1))
    assert [(item.retriever, item.rank, item.score_kind) for item in only_bm25.sources] == [
        ("bm25", 1, "bm25"),
    ]


def test_both_retrievers_can_return_nothing() -> None:
    assert reciprocal_rank_fusion([[], []], k=_K) == []


def test_identical_rankings_keep_order_and_double_the_points() -> None:
    vector = [_hit("A", "vector", 0.1, "distance"), _hit("B", "vector", 0.2, "distance")]
    bm25 = [_hit("A", "bm25", 4.0, "bm25"), _hit("B", "bm25", 2.0, "bm25")]
    hits = reciprocal_rank_fusion([vector, bm25], k=_K)
    assert [hit.chunk_id for hit in hits] == ["A", "B"]
    assert hits[0].score == pytest.approx(_points(1, 1))
    assert hits[1].score == pytest.approx(_points(2, 2))
    assert len(hits[0].sources) == 2


def test_duplicate_chunk_in_one_list_uses_the_first_rank() -> None:
    vector = [
        _hit("A", "vector", 0.1, "distance"),
        _hit("A", "vector", 0.9, "distance"),
        _hit("B", "vector", 0.3, "distance"),
    ]
    bm25 = [_hit("A", "bm25", 2.0, "bm25")]
    hits = reciprocal_rank_fusion([vector, bm25], k=_K)
    merged = next(hit for hit in hits if hit.chunk_id == "A")
    assert [(item.retriever, item.rank, item.score) for item in merged.sources] == [
        ("vector", 1, 0.1),
        ("bm25", 1, 2.0),
    ]
    assert merged.score == pytest.approx(_points(1, 1))
    later = next(hit for hit in hits if hit.chunk_id == "B")
    assert [(item.retriever, item.rank) for item in later.sources] == [("vector", 3)]


def test_negative_k_is_rejected() -> None:
    with pytest.raises(RetrievalError, match="RRF_K"):
        reciprocal_rank_fusion([[]], k=-1)
