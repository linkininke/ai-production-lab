"""仪表盘只汇总已经产生的记录，不补造分数，也不把正文写进 Trace。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import EvaluationError
from app.evaluation.dashboard import build_case_trace, failure_count_update
from app.evaluation.experiment import load_saved_experiment
from app.evaluation.models import QuestionResult
from app.main import create_app
from app.observability.failure import FailureRecord
from app.observability.trace import begin_request_trace, current_request_trace, end_request_trace
from app.retrieval.models import RetrievalResult
from app.retrieval.trace import RetrievalTrace
from tests.helpers import make_settings

_BODY = "机密片段唯一标记"


def test_failure_counts_group_records_and_leave_missing_types_out() -> None:
    questions = [
        _question(
            "q1",
            "semantic",
            [
                _record("q1", "RETRIEVAL_FAILURE", "HIGH", "retrieval"),
                _record("q1", "CITATION_FAILURE", "MEDIUM", "citation_validation"),
            ],
        ),
        _question("q2", None, [_record("q2", "GENERATION_FAILURE", "HIGH", "llm_generation")]),
        _question("q3", "semantic", []),
    ]
    counts = failure_count_update(questions)
    assert {item.name: item.count for item in counts["failure_by_type"]} == {
        "RETRIEVAL_FAILURE": 1,
        "CITATION_FAILURE": 1,
        "GENERATION_FAILURE": 1,
    }
    assert "PROVIDER_ERROR" not in {item.name for item in counts["failure_by_type"]}
    assert {item.name: item.count for item in counts["failure_by_category"]} == {
        "semantic": 2,
        "未分类": 1,
    }
    assert "不会进入统计" not in str(counts)


def test_case_trace_keeps_scores_and_drops_chunk_text() -> None:
    token = begin_request_trace(retrieval_mode="hybrid", question_length=6)
    context = current_request_trace()
    assert context is not None
    with context.span("query_validation", input_summary="chars=6"):
        pass
    request = end_request_trace(token)
    retrieval = RetrievalTrace(
        retrieval_mode="hybrid",
        vector_hits=[_hit()],
        bm25_hits=[],
    )
    view = build_case_trace(
        request_trace=request,
        retrieval_trace=retrieval,
        context_citation_ids=["C1"],
    )
    assert view is not None
    payload = view.model_dump_json()
    assert _BODY not in payload
    assert "query_validation" in payload
    assert [item.stage for item in view.stages] == ["vector", "bm25"]
    assert view.stages[1].hits == []
    assert view.stages[0].hits[0].chunk_id == "chunk-1"
    assert view.context_citation_ids == ["C1"]
    assert build_case_trace(
        request_trace=None,
        retrieval_trace=retrieval,
        context_citation_ids=[],
    ) is None


def test_missing_retrieval_experiment_stays_missing(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="还没有检索实验报告"):
        load_saved_experiment(tmp_path / "latest.json")


def test_retrieval_experiment_api_reports_a_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.api.routes.evaluation.latest_experiment_path",
        lambda: tmp_path / "latest.json",
    )
    with TestClient(create_app(make_settings())) as client:
        response = client.get("/api/v1/evaluation/retrieval-experiment")
    assert response.status_code == 404
    assert "还没有检索实验报告" in response.json()["error"]["message"]


def _question(
    case_id: str,
    category: str | None,
    records: list[FailureRecord],
) -> QuestionResult:
    return QuestionResult(
        id=case_id,
        question="不会进入统计",
        category=category,
        expects_abstention=False,
        status="success",
        failure_records=records,
    )


def _record(case_id: str, failure_type: str, severity: str, stage: str) -> FailureRecord:
    return FailureRecord(
        case_id=case_id,
        failure_type=failure_type,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        stage=stage,
        description="固定说明",
        suggested_action="固定建议",
    )


def _hit() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc_0123456789abcdef",
        text=_BODY,
        score=0.2,
        score_kind="distance",
        metadata={"filename": "note.md"},
    )
