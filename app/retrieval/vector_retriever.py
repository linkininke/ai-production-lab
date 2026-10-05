"""用向量距离检索片段。

先把问题变成向量，再交给向量库。分数保持为余弦距离，越小越近。
max_distance 是距离上限：超过它的命中会被丢掉。它不是相似度下限。
不填上限时，只按 Top-K 截断，不过滤远距离结果。
"""

from __future__ import annotations

import logging
from time import perf_counter

from app.core.exceptions import ConfigurationError, RetrievalError
from app.embeddings.base import EmbeddingProvider
from app.retrieval.models import RetrievalResult
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        max_distance: float | None = None,
    ) -> None:
        if max_distance is not None and max_distance < 0:
            raise ConfigurationError("RETRIEVAL_MAX_DISTANCE 不能为负数")
        if embedder.dimension != store.dimension:
            raise ConfigurationError(
                f"Embedding 维度与向量库不一致：{embedder.dimension} 对 {store.dimension}"
            )
        if embedder.model_name != store.embedding_model:
            raise ConfigurationError(
                f"Embedding 模型与向量库不一致：{embedder.model_name} 对 {store.embedding_model}"
            )
        self._embedder = embedder
        self._store = store
        self._max_distance = max_distance

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise RetrievalError("问题不能为空")
        if top_k < 1:
            raise RetrievalError("top_k 必须大于 0")
        started = perf_counter()
        vector = self._embedder.embed_query(query.strip())
        hits = self._store.search(vector, top_k)
        if self._max_distance is not None:
            hits = [hit for hit in hits if hit.score <= self._max_distance]
        logger.info(
            "retrieval_completed query_length=%s top_k=%s result_count=%s elapsed_ms=%.1f",
            len(query.strip()),
            top_k,
            len(hits),
            (perf_counter() - started) * 1000,
        )
        return hits
