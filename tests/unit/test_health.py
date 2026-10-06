"""健康卡只聚合已有结果。缺指标保持为空，不合成总分。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT
from app.core.exceptions import EvaluationError
from app.evaluation.experiment import ExperimentReport, RetrieverMetrics
from app.evaluation.health import (
    HealthCard,
    HealthMetric,
    HealthWarningRules,
    build_health_card,
    gates_for_saved_report,
    load_health_context,
)
from app.evaluation.models import EvaluationHit, EvaluationReport, QuestionResult
from app.evaluation.regression import CaseDrop, RegressionReport
from app.main import create_app
from app.observability.failure import FailureRecord
from tests.helpers import make_settings

_SECRET = "机密片段唯一标记"


def test_health_card_uses_report_metrics_and_passes_when_checks_pass() -> None:
    report = _report(
        recall=0.94,
        correctness=0.82,
        questions=[
            _question("q1", "semantic", recall=1.0, judge=0),
            _question("q2", "semantic", recall=1.0, judge=4),
        ],
    )
    card = build_health_card(report, regression=_passed())
    assert card.status == "PASS"
    assert card.reasons == []
    assert card.quality_gate == "PASS"
    assert card.regression == "PASS"
    correctness = _metric(card, "judge_correctness")
    assert correctness.comparison.current == 0.82
    assert correctness.comparison.current != 0.5
    assert _metric(card, "recall_at_5").comparison.current == 0.94
    assert _metric(card, "judge_cost").comparison.current is None
    assert _metric(card, "judge_cost").status == "na"
    semantic = [
        item.value
        for item in card.categories
        if item.category == "semantic" and item.metric == "judge_correctness"
    ]
    assert semantic == [0.5]
    assert _SECRET not in card.model_dump_json()
    assert "q1" not in {item.id for item in card.failed_cases}


def test_missing_metric_stays_empty_and_baseline_delta_uses_points() -> None:
    current = _report(recall=0.82, correctness=0.82, groundedness=None, p95=2800, cost=0.012)
    baseline = _report(
        recall=0.78,
        correctness=0.78,
        groundedness=0.9,
        p95=2100,
        cost=0.009,
        filename="eval_20261006T000000000000Z_k5_base.json",
    )
    card = build_health_card(current, baseline=baseline, regression=_passed())
    grounded = _metric(card, "judge_groundedness")
    assert grounded.comparison.current is None
    assert grounded.status == "na"
    assert grounded.note == "没有这项实测。"
    correct = _metric(card, "judge_correctness")
    assert correct.comparison.delta == 0.04
    assert correct.comparison.delta_text == "+4.0 pp"
    latency = _metric(card, "p95_latency_ms")
    assert latency.comparison.delta_text == "+33%"
    assert card.status == "PASS"


def test_missing_baseline_does_not_fail() -> None:
    card = build_health_card(_report(recall=0.8, correctness=0.8), regression=_passed())
    correct = _metric(card, "judge_correctness")
    assert correct.comparison.baseline is None
    assert correct.comparison.delta is None
    assert correct.comparison.delta_text == ""
    assert card.status == "PASS"


def test_quality_gate_failure_sets_overall_fail() -> None:
    report = _report(recall=0.9, correctness=0.9, gate="QUALITY GATE FAILED")
    card = build_health_card(report, regression=_passed())
    assert card.status == "FAIL"
    assert card.quality_gate == "FAIL"
    assert any("质量门未通过" in item for item in card.reasons)


def test_regression_failure_sets_overall_fail_and_counts_cases() -> None:
    report = _report(recall=0.9, correctness=0.9, question_count=30)
    regression = RegressionReport(
        passed=False,
        dataset_match=True,
        checks=[],
        case_failures=[
            CaseDrop(
                case_id="q9",
                metric="recall_at_k",
                baseline=1.0,
                current=0.5,
                drop=0.5,
                max_drop=0.03,
                failed=True,
            )
        ],
        note="回归未通过。",
    )
    card = build_health_card(report, regression=regression)
    assert card.status == "FAIL"
    assert card.regression == "FAIL"
    assert card.regression_failed_cases == 1
    assert card.regression_case_total == 30
    assert any("1 / 30" in item for item in card.reasons)


def test_configured_p95_increase_warns_without_a_hard_coded_cutoff() -> None:
    current = _report(recall=0.9, correctness=0.9, p95=4000)
    baseline = _report(recall=0.9, correctness=0.9, p95=2000)
    quiet = build_health_card(current, baseline=baseline, regression=_passed())
    assert quiet.status == "PASS"
    warned = build_health_card(
        current,
        baseline=baseline,
        regression=_passed(),
        warning=HealthWarningRules(p95_increase_ratio=0.3),
    )
    assert warned.status == "WARNING"
    assert _metric(warned, "p95_latency_ms").status == "warn"
    assert any("P95" in item for item in warned.reasons)


def test_slo_failure_is_fail_and_empty_report_does_not_crash() -> None:
    failed = _report(recall=0.9, correctness=0.9, slo="SLO FAILED")
    card = build_health_card(failed, regression=_passed())
    assert card.status == "FAIL"
    assert any("可靠性未通过" in item for item in card.reasons)
    empty = build_health_card(_report(recall=None, correctness=None, questions=[], count=0))
    assert empty.status == "WARNING"
    assert _metric(empty, "judge_correctness").comparison.current is None
    assert empty.failed_cases == []
    assert empty.model_dump()["status"] in {"PASS", "WARNING", "FAIL"}


def test_current_gate_and_slo_use_saved_metrics_without_a_new_run() -> None:
    settings = make_settings().model_copy(
        update={
            "quality_gate_recall_at_5": 0.80,
            "quality_gate_citation_validity": 0.98,
            "quality_gate_groundedness": 3.5,
            "quality_gate_p95_ms": 3000,
            "slo_success_rate": 0.99,
            "slo_p95_latency_ms": 3000,
            "slo_citation_validity": 0.98,
        }
    )
    slow = _report(
        recall=1.0,
        correctness=0.8,
        groundedness=0.875,
        p95=6484,
        gate="QUALITY GATE NOT CONFIGURED",
        slo="SLO NOT CONFIGURED",
    )
    gate, slo = gates_for_saved_report(slow, settings)
    assert gate.status == "QUALITY GATE FAILED"
    assert slo.status == "SLO FAILED"
    assert any(item.metric == "groundedness" and item.status == "passed" for item in gate.checks)
    card = build_health_card(slow, regression=_passed(), live_gate=gate, live_slo=slo)
    assert card.status == "FAIL"
    assert card.quality_gate == "FAIL"
    assert card.slo == "FAIL"
    assert any("p95_latency_ms" in item and "3000" in item for item in card.reasons)
    assert "没有重新调用模型" in card.assessment_note

    fast = _report(
        recall=0.80,
        correctness=0.8,
        groundedness=0.875,
        p95=3000,
        gate="QUALITY GATE NOT CONFIGURED",
        slo="SLO NOT CONFIGURED",
    )
    passed_gate, passed_slo = gates_for_saved_report(fast, settings)
    passed = build_health_card(
        fast,
        regression=_passed(),
        live_gate=passed_gate,
        live_slo=passed_slo,
    )
    assert passed_gate.status == "QUALITY GATE PASSED"
    assert passed_slo.status == "SLO PASSED"
    assert passed.status == "PASS"


def test_failed_case_keeps_trace_ids_and_drops_chunk_text() -> None:
    question = _question("q8", "multi_document", recall=0.5, judge=2)
    question = question.model_copy(
        update={
            "failure_records": [
                FailureRecord(
                    case_id="q8",
                    failure_type="RETRIEVAL_FAILURE",
                    severity="HIGH",
                    stage="retrieval",
                    description="相关文档没有全部出现。",
                    suggested_action="检查检索。",
                )
            ]
        }
    )
    card = build_health_card(
        _report(recall=0.5, correctness=0.5, questions=[question]),
        regression=_passed(),
    )
    assert [item.id for item in card.failed_cases] == ["q8"]
    assert card.failed_cases[0].failure_types == ["RETRIEVAL_FAILURE"]
    assert _SECRET not in card.model_dump_json()


def test_unconfigured_gate_warns_and_experiment_rows_are_copied() -> None:
    report = _report(recall=1.0, correctness=0.8, gate="QUALITY GATE NOT CONFIGURED")
    experiment = ExperimentReport(
        dataset="questions.json",
        dataset_version="v2",
        question_count=1,
        collection="kb_eval",
        embedding_model="fixture",
        semantic_embedding=True,
        note="只复制已有检索实验。",
        retrievers=[
            RetrieverMetrics(
                retriever="vector",
                question_count=1,
                recall_at_1=0.8,
                recall_at_3=0.9,
                recall_at_5=1.0,
                recall_at_10=1.0,
                precision_at_5=0.2,
                mrr_at_5=0.9,
                average_latency_ms=100,
                estimated_cost_usd=None,
            )
        ],
        created_at="2026-10-06T00:00:00+00:00",
    )
    card = build_health_card(report, regression=_passed(), experiment=experiment)
    assert card.status == "WARNING"
    assert card.quality_gate == "NOT CONFIGURED"
    assert card.experiments[0].recall_at_5 == 1.0
    assert card.experiments[0].estimated_cost_usd is None
    assert "不是 P95" in card.experiment_note


def test_saved_context_reads_existing_files_without_a_model() -> None:
    dataset = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    if not dataset.is_file():
        return
    regression, experiment, note = load_health_context(dataset)
    assert "无法读取" not in note
    if (PROJECT_ROOT / "evaluation_results" / "latest.json").is_file():
        assert experiment is not None
        assert experiment.retrievers
    if (PROJECT_ROOT / "evaluation_results" / "baseline.json").is_file() and experiment is not None:
        assert regression is not None


def test_health_api_reports_a_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def _reader(container: object, **kwargs: object) -> object:
        del container, kwargs

        class _Missing:
            def load_report(self, filename: str) -> EvaluationReport:
                raise EvaluationError(f"评测报告不存在：{filename}", status_code=404)

        return _Missing()

    monkeypatch.setattr(
        "app.api.routes.evaluation.EvaluationRunner.from_container",
        _reader,
    )
    with TestClient(create_app(make_settings())) as client:
        response = client.get(
            "/api/v1/evaluation/health",
            params={"report": "eval_19990101T000000000000Z_k5.json"},
        )
    assert response.status_code == 404
    assert "评测报告不存在" in response.json()["error"]["message"]


def _passed() -> RegressionReport:
    return RegressionReport(passed=True, dataset_match=True, checks=[], note="通过")


def _metric(card: HealthCard, name: str) -> HealthMetric:
    return next(item for item in card.metrics if item.name == name)


def _report(
    *,
    recall: float | None,
    correctness: float | None,
    questions: list[QuestionResult] | None = None,
    groundedness: float | None = 0.9,
    p95: float | None = 1000,
    cost: float | None = None,
    gate: str = "QUALITY GATE PASSED",
    slo: str = "SLO PASSED",
    filename: str = "eval_20261006T000000000000Z_k5.json",
    question_count: int | None = None,
    count: int | None = None,
) -> EvaluationReport:
    rows = questions
    if rows is None:
        rows = [_question("q1", "semantic", recall=1.0, judge=4)]
    if count is not None:
        total = count
    elif question_count is not None:
        total = question_count
    else:
        total = len(rows)
    return EvaluationReport(
        filename=filename,
        created_at="2026-10-06T00:00:00+00:00",
        dataset_version="v2",
        dataset_file="questions.json",
        question_count=total,
        top_k=5,
        embedding_model="fixture",
        llm_model="fixture",
        retrieval_mode="vector",
        prompt_version="v1",
        experiment_id="A",
        collection_name="kb_phase9",
        chunk_size=200,
        chunk_overlap=20,
        corpus_documents=[],
        recall_at_k=recall,
        mrr_at_k=recall,
        precision_at_k=0.2 if recall is not None else None,
        citation_validity=1.0 if recall is not None else None,
        citation_coverage=1.0 if recall is not None else None,
        abstention_quality=1.0 if recall is not None else None,
        judge_correctness=correctness,
        judge_groundedness=groundedness,
        judge_completeness=correctness,
        p50_latency_ms=p95,
        p95_latency_ms=p95,
        p99_latency_ms=p95,
        success_rate=1.0,
        failure_rate=0.0,
        success_count=total,
        failure_count=0,
        quality_gate_status=gate,
        slo_status=slo,
        judge_cost=cost,
        failures=[],
        questions=rows,
    )


def _question(
    case_id: str,
    category: str,
    *,
    recall: float | None,
    judge: int | None,
) -> QuestionResult:
    return QuestionResult(
        id=case_id,
        question="不进入健康卡以外的日志",
        category=category,
        expects_abstention=False,
        status="success",
        recall_at_k=recall,
        reciprocal_rank=recall,
        citation_validity=1.0,
        judge_correctness=judge,
        retrieved=[
            EvaluationHit(
                rank=1,
                document_id="doc_0123456789abcdef",
                filename="note.md",
                chunk_id="chunk-1",
                text=_SECRET,
                score=0.1,
                score_kind="distance",
            )
        ],
    )


def test_missing_context_files_stay_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.evaluation.health.latest_experiment_path",
        lambda: tmp_path / "latest.json",
    )
    monkeypatch.setattr("app.evaluation.health.PROJECT_ROOT", tmp_path)
    regression, experiment, note = load_health_context(tmp_path / "questions.json")
    assert regression is None
    assert experiment is None
    assert note == ""
