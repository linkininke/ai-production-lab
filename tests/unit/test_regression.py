"""回归。跌幅用真实阈值判断，不把缺失的 Judge 分数当成 0。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.evaluation.experiment import ExperimentReport, RetrieverMetrics
from app.evaluation.regression import (
    CaseSnapshot,
    freeze_baseline,
    regress_cases,
    regress_experiment,
    render_regression,
)

_LATEST = PROJECT_ROOT / "evaluation_results" / "latest.json"


def _dataset(path: Path, version: str = "v2") -> None:
    path.write_text(
        json.dumps(
            {
                "version": version,
                "questions": [{"id": "q001", "question": "事务", "expected_document_ids": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _experiment(**metrics: float) -> ExperimentReport:
    row = RetrieverMetrics(
        retriever="vector",
        question_count=2,
        recall_at_1=metrics.get("recall_at_1", 1.0),
        recall_at_3=metrics.get("recall_at_3", 1.0),
        recall_at_5=metrics["recall_at_5"],
        recall_at_10=metrics.get("recall_at_10", 1.0),
        precision_at_5=metrics.get("precision_at_5", 0.2),
        mrr_at_5=metrics["mrr_at_5"],
        average_latency_ms=10.0,
        estimated_cost_usd=None,
    )
    return ExperimentReport(
        dataset="questions.json",
        dataset_version="v2",
        question_count=2,
        collection="kb_eval",
        embedding_model="fake",
        semantic_embedding=False,
        note="测试",
        retrievers=[row],
        created_at="2026-10-06T03:22:21.994875+00:00",
    )


def test_drop_beyond_the_limit_fails_and_missing_judge_scores_stay_absent(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.json"
    _dataset(dataset)
    baseline = freeze_baseline(
        _experiment(recall_at_5=0.84, mrr_at_5=0.72),
        dataset,
        source="fixture",
    )
    assert "correctness" not in baseline.retrievers[0].metrics
    assert "estimated_cost_usd" not in baseline.retrievers[0].metrics
    current = _experiment(recall_at_5=0.79, mrr_at_5=0.70)
    report = regress_experiment(baseline, current, dataset)
    assert report.passed is False
    recall = next(item for item in report.checks if item.metric == "recall_at_5")
    assert recall.status == "failed"
    assert recall.drop == pytest.approx(0.05)
    judge = next(item for item in report.checks if item.metric == "correctness")
    assert judge.status == "skipped"
    assert "REGRESSION FAILED" in render_regression(report)


def test_equal_result_passes(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.json"
    _dataset(dataset)
    experiment = _experiment(recall_at_5=1.0, mrr_at_5=0.97)
    baseline = freeze_baseline(experiment, dataset, source="fixture")
    report = regress_experiment(baseline, experiment, dataset)
    assert report.passed is True
    assert report.dataset_match is True


def test_changed_golden_dataset_fails_before_metrics(tmp_path: Path) -> None:
    original = tmp_path / "questions.json"
    _dataset(original)
    baseline = freeze_baseline(
        _experiment(recall_at_5=1.0, mrr_at_5=1.0),
        original,
        source="fixture",
    )
    _dataset(original, version="v3")
    current = _experiment(recall_at_5=1.0, mrr_at_5=1.0)
    current.dataset_version = "v3"
    report = regress_experiment(baseline, current, original)
    assert report.dataset_match is False
    assert report.passed is False


def test_average_can_pass_while_one_case_fails() -> None:
    baseline = [
        CaseSnapshot(id="q001", category="semantic", recall_at_k=1.0),
        CaseSnapshot(id="q002", category="semantic", recall_at_k=1.0),
        CaseSnapshot(id="q003", category="multi_document", recall_at_k=1.0),
    ]
    current = [
        CaseSnapshot(id="q001", category="semantic", recall_at_k=1.0),
        CaseSnapshot(id="q002", category="semantic", recall_at_k=1.0),
        CaseSnapshot(id="q003", category="multi_document", recall_at_k=0.0),
    ]
    drops, categories = regress_cases(baseline, current, max_drop=0.03)
    assert [item.case_id for item in drops] == ["q003"]
    by_category = {item.category: item for item in categories}
    assert by_category["semantic"].failed is False
    assert by_category["multi_document"].failed is True


def test_saved_retrieval_experiment_matches_its_baseline() -> None:
    if not _LATEST.is_file():
        pytest.skip("没有已保存的检索实验")
    experiment = ExperimentReport.model_validate_json(_LATEST.read_text(encoding="utf-8"))
    dataset = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    baseline = freeze_baseline(experiment, dataset, source="evaluation_results/latest.json")
    vector = next(item for item in baseline.retrievers if item.retriever == "vector")
    assert vector.metrics["recall_at_5"] == pytest.approx(1.0)
    assert vector.metrics["mrr_at_5"] == pytest.approx(0.9711538461538461)
    assert "correctness" not in vector.metrics
    report = regress_experiment(baseline, experiment, dataset)
    assert report.passed is True
    assert report.dataset_match is True
