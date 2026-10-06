"""把已有评测结果收成仪表盘数据。

失败分布只数失败记录。没有记录就是空，不用 0 去填每一种类型。
Trace 只留阶段、编号和分数。问题原文留在评测题上，提示词和片段正文不进入 Trace。
"""

from __future__ import annotations

from app.evaluation.models import (
    CaseTrace,
    EvaluationReport,
    NamedCount,
    QuestionResult,
    TraceHitView,
    TraceSpanView,
    TraceStageView,
)
from app.observability.trace import RequestTrace
from app.retrieval.models import RetrievalResult
from app.retrieval.trace import RetrievalTrace

_STAGE_FIELDS = (
    ("vector", "vector_hits"),
    ("bm25", "bm25_hits"),
    ("rrf", "rrf_hits"),
    ("reranker", "reranker_hits"),
)


def with_failure_counts(report: EvaluationReport) -> EvaluationReport:
    return report.model_copy(update=failure_count_update(report.questions))


def failure_count_update(questions: list[QuestionResult]) -> dict[str, list[NamedCount]]:
    categories = {item.id: item.category or "未分类" for item in questions}
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for item in questions:
        for record in item.failure_records:
            _bump(by_type, record.failure_type)
            _bump(by_severity, record.severity)
            _bump(by_stage, record.stage)
            label = categories.get(record.case_id or item.id, "未分类")
            _bump(by_category, label)
    return {
        "failure_by_type": _buckets(by_type),
        "failure_by_severity": _buckets(by_severity),
        "failure_by_stage": _buckets(by_stage),
        "failure_by_category": _buckets(by_category),
    }


def build_case_trace(
    *,
    request_trace: RequestTrace | None,
    retrieval_trace: RetrievalTrace | None,
    context_citation_ids: list[str],
) -> CaseTrace | None:
    """收成可保存的轨迹。检索候选去掉正文。没有请求 Trace 时不编造。"""
    if request_trace is None:
        return None
    return CaseTrace(
        trace_id=request_trace.trace_id,
        status=request_trace.status,
        retrieval_mode=request_trace.retrieval_mode,
        question_length=request_trace.question_length,
        spans=[
            TraceSpanView(
                name=item.name,
                duration_ms=item.duration_ms,
                status=item.status,
                output_summary=item.output_summary,
            )
            for item in request_trace.spans
        ],
        stages=_stages(retrieval_trace),
        context_citation_ids=list(context_citation_ids),
    )


def _stages(trace: RetrievalTrace | None) -> list[TraceStageView]:
    if trace is None:
        return []
    stages: list[TraceStageView] = []
    for name, field_name in _STAGE_FIELDS:
        hits = getattr(trace, field_name)
        if hits is None:
            continue
        stages.append(
            TraceStageView(
                stage=name,
                hits=[_hit(index, item) for index, item in enumerate(hits, start=1)],
            )
        )
    return stages


def _hit(rank: int, item: RetrievalResult) -> TraceHitView:
    return TraceHitView(
        rank=rank,
        chunk_id=item.chunk_id,
        document_id=item.document_id,
        score=item.score,
        score_kind=item.score_kind,
    )


def _bump(counts: dict[str, int], name: str) -> None:
    counts[name] = counts.get(name, 0) + 1


def _buckets(counts: dict[str, int]) -> list[NamedCount]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [NamedCount(name=name, count=count) for name, count in ordered]
