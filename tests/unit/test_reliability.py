"""预算、质量门、SLO 和延迟分位。不调用模型。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.evaluation.reliability import (
    Threshold,
    evaluate_slo,
    latency_percentiles,
    plan_budget,
    predicted_case_cost_usd,
    quality_gate,
)
from app.evaluation.retrieval_cli import main
from app.llm.openai_compatible import OpenAICompatibleLLMProvider
from tests.helpers import make_settings


def test_percentiles_use_nearest_rank_and_keep_the_tail() -> None:
    values = [10, 20, 30, 40, 100]
    p50, p95, p99 = latency_percentiles(values)
    assert p50 == 30
    assert p95 == 100
    assert p99 == 100
    assert sum(values) / len(values) == 40
    assert latency_percentiles([]) == (None, None, None)
    assert latency_percentiles([7]) == (7, 7, 7)


def test_quality_gate_fails_closed_and_treats_the_limit_as_passing() -> None:
    missing = quality_gate(
        {"recall_at_5": None},
        [Threshold(metric="recall_at_5", minimum=0.80)],
    )
    assert missing.status == "QUALITY GATE FAILED"
    assert missing.checks[0].reason == "没有这项实测"
    assert missing.checks[0].actual is None

    below = quality_gate(
        {"recall_at_5": 0.79},
        [Threshold(metric="recall_at_5", minimum=0.80)],
    )
    assert below.status == "QUALITY GATE FAILED"
    assert below.checks[0].reason == "低于下限"

    equal = quality_gate(
        {"recall_at_5": 0.80},
        [Threshold(metric="recall_at_5", minimum=0.80)],
    )
    assert equal.status == "QUALITY GATE PASSED"

    slow = quality_gate(
        {"p95_latency_ms": 4000},
        [Threshold(metric="p95_latency_ms", maximum=3000)],
    )
    assert slow.status == "QUALITY GATE FAILED"
    assert slow.checks[0].reason == "高于上限"
    assert quality_gate({}, []).status == "QUALITY GATE NOT CONFIGURED"


def test_slo_is_separate_from_the_quality_gate() -> None:
    failed = evaluate_slo(
        {"success_rate": 0.90},
        [Threshold(metric="success_rate", minimum=0.99)],
    )
    assert failed.status == "SLO FAILED"
    assert evaluate_slo({}, []).status == "SLO NOT CONFIGURED"


def test_budget_stops_when_cost_cannot_be_predicted_or_exceeds_the_limit() -> None:
    unknown = plan_budget(
        question_count=30,
        max_cases=None,
        max_cost=1.0,
        predicted_cost_per_case=None,
    )
    assert unknown.aborted is True
    assert unknown.selected_cases == 0
    assert "已停止" in unknown.reason
    assert "没有继续调用模型" in unknown.reason

    over = plan_budget(
        question_count=30,
        max_cases=None,
        max_cost=1.0,
        predicted_cost_per_case=0.1,
    )
    assert over.aborted is True
    assert over.predicted_cost_usd == pytest.approx(3.0)

    allowed = plan_budget(
        question_count=30,
        max_cases=5,
        max_cost=1.0,
        predicted_cost_per_case=0.1,
    )
    assert allowed.aborted is False
    assert allowed.selected_cases == 5
    assert allowed.predicted_cost_usd == pytest.approx(0.5)

    assert plan_budget(
        question_count=30,
        max_cases=0,
        max_cost=None,
        predicted_cost_per_case=None,
    ).aborted is True
    unlimited = plan_budget(
        question_count=30,
        max_cases=None,
        max_cost=None,
        predicted_cost_per_case=None,
    )
    assert unlimited.aborted is False
    assert unlimited.selected_cases == 30
    assert predicted_case_cost_usd(
        prompt_tokens=None,
        completion_tokens=None,
        embedding_tokens=None,
        llm_input_price_per_1m=2,
        llm_output_price_per_1m=4,
        embedding_price_per_1m=0.1,
    ) is None


def test_provider_retries_stay_between_zero_and_three() -> None:
    with pytest.raises(ValidationError):
        Settings(provider_max_retries=4)
    settings = make_settings(
        enable_external_models=True,
        llm_base_url="https://example.invalid/v1",
        llm_api_key="sk-test",
        llm_model="demo",
        embedding_base_url="https://example.invalid/v1",
        embedding_api_key="sk-test",
        embedding_model="embed",
        embedding_dimension=8,
        provider_max_retries=0,
    )
    provider = OpenAICompatibleLLMProvider.from_settings(settings)
    try:
        assert provider._max_retries == 0
    finally:
        provider.close()


def test_retrieval_cli_stops_before_the_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "version": "v1",
                "questions": [
                    {
                        "id": "q001",
                        "question": "缓存击穿怎么处理？",
                        "expected_keywords": [],
                        "expected_document_ids": ["doc_0123456789abcdef"],
                        "expects_abstention": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = make_settings(evaluation_dataset_path=str(questions))
    monkeypatch.setattr("app.evaluation.retrieval_cli.get_settings", lambda: settings)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("预算不足时不应开始实验")

    monkeypatch.setattr("app.evaluation.retrieval_cli.run_retrieval_experiment", fail_if_called)
    assert main(["--max-cost", "1"]) == 1
