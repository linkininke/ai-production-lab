"""一次问答请求的 Trace。

Span 只记录阶段名、耗时、状态和数量摘要。
不写入问题原文、提示词、资料正文或密钥。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import request_id_var
from app.retrieval.trace import RetrievalTrace

SpanStatus = Literal["success", "error", "warning"]
_current: ContextVar[TraceContext | None] = ContextVar("request_trace", default=None)


class Span(BaseModel):
    """请求里的一个阶段。时间用 UTC，耗时用毫秒。"""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    trace_id: str
    name: str
    start_time: str
    end_time: str
    duration_ms: float
    status: SpanStatus
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    input_summary: str = ""
    output_summary: str = ""


class RequestTrace(BaseModel):
    """一次问答请求。question 不入库，只保留长度。"""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    request_id: str
    question_length: int
    retrieval_mode: str
    start_time: str
    end_time: str
    total_latency_ms: float
    status: SpanStatus
    spans: list[Span]
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SpanUpdate:
    """阶段执行过程中可以改状态和摘要。不能写入原文。"""

    def __init__(self) -> None:
        self.status: SpanStatus = "success"
        self.output_summary = ""
        self.metadata: dict[str, str | int | float | bool] = {}


class TraceContext:
    """当前请求的 Span 收集器。请求结束时收成 RequestTrace。"""

    def __init__(self, *, retrieval_mode: str, question_length: int) -> None:
        self.trace_id = uuid4().hex
        self.retrieval_mode = retrieval_mode
        self.question_length = question_length
        self.status: SpanStatus = "success"
        self._started = perf_counter()
        self._start_time = datetime.now(UTC)
        self._spans: list[Span] = []

    @contextmanager
    def span(self, name: str, *, input_summary: str = "") -> Iterator[SpanUpdate]:
        update = SpanUpdate()
        started = perf_counter()
        start_time = datetime.now(UTC)
        try:
            yield update
        except Exception:
            self._record(
                name,
                started,
                start_time,
                "error",
                input_summary,
                update.output_summary,
                update.metadata,
            )
            self.status = "error"
            raise
        self._record(
            name,
            started,
            start_time,
            update.status,
            input_summary,
            update.output_summary,
            update.metadata,
        )
        self._raise_status(update.status)

    def add_measured(
        self,
        name: str,
        duration_ms: float,
        *,
        status: SpanStatus = "success",
        input_summary: str = "",
        output_summary: str = "",
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        end = datetime.now(UTC)
        start = end - timedelta(milliseconds=max(duration_ms, 0))
        self._spans.append(
            Span(
                span_id=uuid4().hex,
                trace_id=self.trace_id,
                name=name,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
                duration_ms=duration_ms,
                status=status,
                metadata=metadata or {},
                input_summary=input_summary,
                output_summary=output_summary,
            )
        )
        self._raise_status(status)

    def finish(self) -> RequestTrace:
        end = datetime.now(UTC)
        return RequestTrace(
            trace_id=self.trace_id,
            request_id=request_id_var.get(),
            question_length=self.question_length,
            retrieval_mode=self.retrieval_mode,
            start_time=self._start_time.isoformat(),
            end_time=end.isoformat(),
            total_latency_ms=(perf_counter() - self._started) * 1000,
            status=self.status,
            spans=list(self._spans),
        )

    def _record(
        self,
        name: str,
        started: float,
        start_time: datetime,
        status: SpanStatus,
        input_summary: str,
        output_summary: str,
        metadata: dict[str, str | int | float | bool],
    ) -> None:
        end = datetime.now(UTC)
        self._spans.append(
            Span(
                span_id=uuid4().hex,
                trace_id=self.trace_id,
                name=name,
                start_time=start_time.isoformat(),
                end_time=end.isoformat(),
                duration_ms=(perf_counter() - started) * 1000,
                status=status,
                metadata=metadata,
                input_summary=input_summary,
                output_summary=output_summary,
            )
        )

    def _raise_status(self, status: SpanStatus) -> None:
        if status == "error" or (status == "warning" and self.status == "success"):
            self.status = status


_REQUEST_ATTR = "_lab_request_trace"
_RETRIEVAL_ATTR = "_lab_retrieval_trace"
_CITATION_ATTR = "_lab_context_citation_ids"


def bind_error_trace(
    exc: BaseException,
    request_trace: RequestTrace | None,
    retrieval_trace: RetrievalTrace | None,
    context_citation_ids: list[str],
) -> None:
    """失败时把已经结束的 Trace 挂在异常上，供评测报告取走。不写入日志。"""
    exc.__dict__[_REQUEST_ATTR] = request_trace
    exc.__dict__[_RETRIEVAL_ATTR] = retrieval_trace
    exc.__dict__[_CITATION_ATTR] = list(context_citation_ids)


def bound_error_trace(
    exc: BaseException,
) -> tuple[RequestTrace | None, RetrievalTrace | None, list[str]]:
    request = exc.__dict__.get(_REQUEST_ATTR)
    retrieval = exc.__dict__.get(_RETRIEVAL_ATTR)
    citations = exc.__dict__.get(_CITATION_ATTR)
    citation_ids = (
        [item for item in citations if isinstance(item, str)] if isinstance(citations, list) else []
    )
    return (
        request if isinstance(request, RequestTrace) else None,
        retrieval if isinstance(retrieval, RetrievalTrace) else None,
        citation_ids,
    )


def begin_request_trace(*, retrieval_mode: str, question_length: int) -> Token[TraceContext | None]:
    context = TraceContext(retrieval_mode=retrieval_mode, question_length=question_length)
    return _current.set(context)


def current_request_trace() -> TraceContext | None:
    return _current.get()


def end_request_trace(token: Token[TraceContext | None] | None) -> RequestTrace | None:
    if token is None:
        return None
    context = _current.get()
    _current.reset(token)
    if context is None:
        return None
    return context.finish()


def record_retrieval_spans(
    context: TraceContext,
    retrieval: RetrievalTrace | None,
    duration_ms: float,
    result_count: int,
) -> None:
    """把已有的检索耗时收成 Span。没有分步耗时时只记一次检索。"""
    recorded = False
    if retrieval is not None:
        recorded = _measured(
            context,
            "embedding",
            retrieval.embedding_latency_ms,
            output_summary="embedded",
        )
        recorded = (
            _measured(
                context,
                "vector_retrieval",
                retrieval.vector_latency_ms,
                output_summary=f"candidates={retrieval.vector_candidate_count or 0}",
            )
            or recorded
        )
        recorded = (
            _measured(
                context,
                "bm25_retrieval",
                retrieval.bm25_latency_ms,
                output_summary=f"candidates={retrieval.bm25_candidate_count or 0}",
            )
            or recorded
        )
        recorded = (
            _measured(
                context,
                "rrf",
                retrieval.rrf_latency_ms,
                output_summary=f"candidates={retrieval.hybrid_candidate_count or 0}",
            )
            or recorded
        )
        recorded = (
            _measured(
                context,
                "reranker",
                retrieval.reranker_latency_ms,
                output_summary=f"candidates={retrieval.reranker_candidate_count or 0}",
            )
            or recorded
        )
    if not recorded:
        context.add_measured("retrieval", duration_ms, output_summary=f"results={result_count}")


def _measured(
    context: TraceContext,
    name: str,
    duration_ms: float | None,
    *,
    output_summary: str,
) -> bool:
    if duration_ms is None:
        return False
    context.add_measured(name, duration_ms, output_summary=output_summary)
    return True
