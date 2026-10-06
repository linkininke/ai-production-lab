"""请求 Trace。警告不能盖过错误，摘要里不能出现问题原文。"""

from __future__ import annotations

import pytest

from app.observability.trace import (
    begin_request_trace,
    current_request_trace,
    end_request_trace,
    record_retrieval_spans,
)
from app.retrieval.trace import RetrievalTrace


def test_warning_stays_below_error() -> None:
    token = begin_request_trace(retrieval_mode="vector", question_length=4)
    context = current_request_trace()
    assert context is not None
    with context.span("citation_validation") as span:
        span.status = "warning"
        span.output_summary = "valid=1 invalid=1"
    with pytest.raises(RuntimeError, match="生成失败"):
        with context.span("llm_generation"):
            raise RuntimeError("生成失败")
    context.add_measured("later", 1.0, status="warning")
    trace = end_request_trace(token)
    assert trace is not None
    assert trace.status == "error"
    assert [item.status for item in trace.spans] == ["warning", "error", "warning"]
    assert "question" not in trace.model_dump()


def test_retrieval_latencies_become_separate_spans() -> None:
    token = begin_request_trace(retrieval_mode="hybrid", question_length=2)
    context = current_request_trace()
    assert context is not None
    record_retrieval_spans(
        context,
        RetrievalTrace(
            retrieval_mode="hybrid",
            embedding_latency_ms=1.0,
            vector_latency_ms=2.0,
            vector_candidate_count=3,
            bm25_latency_ms=0.0,
            bm25_candidate_count=1,
            rrf_latency_ms=0.5,
            hybrid_candidate_count=4,
        ),
        duration_ms=9.0,
        result_count=2,
    )
    trace = end_request_trace(token)
    assert trace is not None
    assert [item.name for item in trace.spans] == [
        "embedding",
        "vector_retrieval",
        "bm25_retrieval",
        "rrf",
    ]
    assert trace.spans[2].duration_ms == 0.0
    assert "retrieval" not in [item.name for item in trace.spans]
