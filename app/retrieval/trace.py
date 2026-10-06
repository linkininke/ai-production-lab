"""一次检索的分步记录。

只在当前请求里保存。日志使用数量和耗时，不使用这里的片段正文。
没有开始记录时，检索器照常返回结果。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

from pydantic import BaseModel

from app.retrieval.models import RetrievalResult

_current: ContextVar[RetrievalTrace | None] = ContextVar("retrieval_trace", default=None)


class RetrievalTrace(BaseModel):
    """各检索阶段的候选和耗时。未执行的阶段保持为空。"""

    retrieval_mode: str
    vector_hits: list[RetrievalResult] | None = None
    bm25_hits: list[RetrievalResult] | None = None
    rrf_hits: list[RetrievalResult] | None = None
    reranker_hits: list[RetrievalResult] | None = None
    vector_candidate_count: int | None = None
    bm25_candidate_count: int | None = None
    hybrid_candidate_count: int | None = None
    reranker_candidate_count: int | None = None
    reranker_enabled: bool = False
    final_result_count: int | None = None
    embedding_latency_ms: float | None = None
    vector_latency_ms: float | None = None
    bm25_latency_ms: float | None = None
    hybrid_latency_ms: float | None = None
    rrf_latency_ms: float | None = None
    reranker_latency_ms: float | None = None
    reranker_name: str | None = None


def begin_retrieval_trace(retrieval_mode: str) -> Token[RetrievalTrace | None]:
    return _current.set(RetrievalTrace(retrieval_mode=retrieval_mode))


def current_retrieval_trace() -> RetrievalTrace | None:
    return _current.get()


def end_retrieval_trace(token: Token[RetrievalTrace | None]) -> RetrievalTrace | None:
    trace = _current.get()
    _current.reset(token)
    return trace


def note_vector(
    hits: list[RetrievalResult],
    *,
    embedding_latency_ms: float,
    vector_latency_ms: float,
) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace.vector_hits = list(hits)
    trace.vector_candidate_count = len(hits)
    trace.embedding_latency_ms = embedding_latency_ms
    trace.vector_latency_ms = vector_latency_ms


def note_bm25(hits: list[RetrievalResult], *, latency_ms: float) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace.bm25_hits = list(hits)
    trace.bm25_candidate_count = len(hits)
    trace.bm25_latency_ms = latency_ms


def note_hybrid(
    fused: list[RetrievalResult],
    *,
    hybrid_latency_ms: float,
    rrf_latency_ms: float,
) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace.rrf_hits = list(fused)
    trace.hybrid_candidate_count = len(fused)
    trace.hybrid_latency_ms = hybrid_latency_ms
    trace.rrf_latency_ms = rrf_latency_ms


def note_reranker(
    hits: list[RetrievalResult],
    *,
    candidate_count: int,
    latency_ms: float,
    reranker_name: str | None,
) -> None:
    trace = _current.get()
    if trace is None:
        return
    trace.reranker_enabled = True
    trace.reranker_hits = list(hits)
    trace.reranker_candidate_count = candidate_count
    trace.reranker_latency_ms = latency_ms
    trace.reranker_name = reranker_name
