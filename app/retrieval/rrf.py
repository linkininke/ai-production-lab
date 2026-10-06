"""倒数排名融合。只看每一路的名次，不读取距离或 BM25 分数。

RRF(d) = Σ 1 / (k + rank(d))。rank 从 1 开始。同一路里重复的 chunk 只保留最先出现的名次。
某一路没有结果时，那一路不贡献分数，其他路照常合并。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import RetrievalError
from app.retrieval.models import RetrievalResult, RetrievalSource


@dataclass
class _Accumulator:
    hit: RetrievalResult
    sources: list[RetrievalSource]
    rrf: float
    sequence: int


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    if k < 0:
        raise RetrievalError("RRF_K 不能为负数")
    fused: dict[str, _Accumulator] = {}
    sequence = 0
    for results in result_lists:
        seen: set[str] = set()
        for rank, hit in enumerate(results, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            source = RetrievalSource(
                retriever=hit.retriever,
                rank=rank,
                score=hit.score,
                score_kind=hit.score_kind,
            )
            current = fused.get(hit.chunk_id)
            points = 1 / (k + rank)
            if current is None:
                sequence += 1
                fused[hit.chunk_id] = _Accumulator(hit, [source], points, sequence)
            else:
                current.sources.append(source)
                current.rrf += points
    ordered = sorted(fused.values(), key=lambda item: (-item.rrf, item.sequence))
    return [
        item.hit.model_copy(
            update={
                "score": item.rrf,
                "score_kind": "rrf",
                "retriever": "hybrid",
                "sources": item.sources,
            }
        )
        for item in ordered
    ]
