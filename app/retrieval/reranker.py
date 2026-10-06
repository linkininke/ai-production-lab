"""重排接口。

重排只改变已有候选的顺序，并截短到 Top-K，不能把候选池变大。
MockReranker 只用于测试：它按调用方给的顺序重排，不访问模型，也不编造相关度。
线上要重排时使用配置好的 OpenAICompatibleReranker，不要把 Mock 接进问答。
"""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

from app.core.exceptions import ConfigurationError, RetrievalError
from app.retrieval.base import Retriever
from app.retrieval.models import RetrievalResult
from app.retrieval.trace import note_reranker

_MOCK_IS_NOT_PRODUCTION = "MockReranker 只用于测试，不能当作线上重排模型。"


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """重新排序并只保留 top_k 条。返回数量不会超过输入数量。"""


class MockReranker:
    """按给定 chunk_id 顺序重排。未点名的片段保持原来的相对顺序，排在后面。

    分数和 score_kind 保持检索阶段的原值。reranker 字段标成 mock，避免被看成模型分数。
    """

    name = "mock"

    def __init__(self, order: list[str] | None = None) -> None:
        self._order = list(order or [])

    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        _require_query_and_top_k(query, top_k)
        if not documents:
            return []
        rank = {chunk_id: index for index, chunk_id in enumerate(self._order)}
        indexed = list(enumerate(documents))
        indexed.sort(
            key=lambda item: (
                (0, rank[item[1].chunk_id]) if item[1].chunk_id in rank else (1, item[0])
            )
        )
        return [
            hit.model_copy(update={"reranker": "mock"})
            for _, hit in indexed[:top_k]
        ]


class RerankingRetriever:
    """先取出比最终条数更多的候选，再交给重排器截短。"""

    name = "hybrid_rerank"

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_top_k: int = 20,
        final_top_k: int = 5,
    ) -> None:
        if candidate_top_k < 1:
            raise ConfigurationError("RERANKER_CANDIDATE_TOP_K 必须大于 0")
        if final_top_k < 1:
            raise ConfigurationError("RERANKER_FINAL_TOP_K 必须大于 0")
        if candidate_top_k < final_top_k:
            raise ConfigurationError(
                "RERANKER_CANDIDATE_TOP_K 不能小于 RERANKER_FINAL_TOP_K"
            )
        self._retriever = retriever
        self._reranker = reranker
        self._candidate_top_k = candidate_top_k
        self._final_top_k = final_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        _require_query_and_top_k(query, top_k if top_k is not None else self._final_top_k)
        final_k = self._final_top_k if top_k is None else top_k
        candidate_k = max(self._candidate_top_k, final_k)
        hits = self._retriever.retrieve(query, top_k=candidate_k)
        if not hits:
            note_reranker(
                [],
                candidate_count=0,
                latency_ms=0.0,
                reranker_name=_reranker_name(self._reranker),
            )
            return []
        started = perf_counter()
        ranked = self._reranker.rerank(query, hits, final_k)
        note_reranker(
            ranked,
            candidate_count=len(hits),
            latency_ms=(perf_counter() - started) * 1000,
            reranker_name=_reranker_name(self._reranker),
        )
        return ranked


def require_production_reranker(reranker: Reranker | None) -> Reranker:
    if reranker is None or isinstance(reranker, MockReranker):
        raise ConfigurationError(
            "未启用重排。请设置 RERANKER_ENABLED=true，并填写重排服务的地址、密钥和模型名。"
            + _MOCK_IS_NOT_PRODUCTION
        )
    return reranker


def _reranker_name(reranker: Reranker) -> str | None:
    name = getattr(reranker, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _require_query_and_top_k(query: str, top_k: int) -> None:
    if not query.strip():
        raise RetrievalError("问题不能为空")
    if top_k < 1:
        raise RetrievalError("top_k 必须大于 0")
