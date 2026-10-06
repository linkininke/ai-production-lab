"""向量检索和 BM25 各取一批候选，再用 RRF 合成最终结果。

候选数量可以大于最终 Top-K。融合分只来自名次。
日志记录两路候选数量、来源名次和 RRF 分数，不记录问题原文和片段正文。
"""

from __future__ import annotations

import logging
from time import perf_counter

from app.core.exceptions import ConfigurationError, RetrievalError
from app.retrieval.base import Retriever
from app.retrieval.models import RetrievalResult
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.trace import note_hybrid

logger = logging.getLogger(__name__)


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        vector: Retriever,
        bm25: Retriever,
        *,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._vector = vector
        self._bm25 = bm25
        self._vector_top_k = _positive("HYBRID_VECTOR_TOP_K", vector_top_k)
        self._bm25_top_k = _positive("HYBRID_BM25_TOP_K", bm25_top_k)
        self._final_top_k = _positive("HYBRID_FINAL_TOP_K", final_top_k)
        if rrf_k < 0:
            raise ConfigurationError("RRF_K 不能为负数")
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not query.strip():
            raise RetrievalError("问题不能为空")
        final_k = self._final_top_k if top_k is None else top_k
        if final_k < 1:
            raise RetrievalError("top_k 必须大于 0")
        started = perf_counter()
        vector_hits = self._vector.retrieve(query, top_k=max(self._vector_top_k, final_k))
        bm25_hits = self._bm25.retrieve(query, top_k=max(self._bm25_top_k, final_k))
        fusion_started = perf_counter()
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=self._rrf_k)
        fusion_ms = (perf_counter() - fusion_started) * 1000
        note_hybrid(
            fused,
            hybrid_latency_ms=(perf_counter() - started) * 1000,
            rrf_latency_ms=fusion_ms,
        )
        selected = fused[:final_k]
        logger.info(
            "retrieval_completed retriever=%s query_length=%s top_k=%s "
            "vector_count=%s bm25_count=%s fused_count=%s result_count=%s "
            "rrf_k=%s latency_ms=%.1f",
            self.name,
            len(query.strip()),
            final_k,
            len(vector_hits),
            len(bm25_hits),
            len(fused),
            len(selected),
            self._rrf_k,
            (perf_counter() - started) * 1000,
        )
        for hit in selected:
            origins = ",".join(f"{item.retriever}:{item.rank}" for item in hit.sources)
            logger.debug(
                "hybrid_hit chunk_id=%s rrf_score=%.6f sources=%s",
                hit.chunk_id,
                hit.score,
                origins,
            )
        return selected


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ConfigurationError(f"{name} 必须大于 0")
    return value
