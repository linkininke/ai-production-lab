"""Judge。非法 JSON 会有限重试。MockJudge 不产生可以当成质量的分数。"""

from __future__ import annotations

import logging

import pytest

from app.core.exceptions import EvaluationError
from app.evaluation.core.case import EvaluationCase
from app.evaluation.judge import LLMJudge, MockJudge
from app.llm.models import LLMResult
from app.retrieval.models import RetrievalResult
from tests.helpers import make_settings

_JSON = (
    '{"correctness": 4, "groundedness": 3, "completeness": 2, "overall": 3, '
    '"reasoning": "上下文提到了代理。"}'
)


class SequenceLLM:
    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-judge"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        self.calls.append((system_prompt, user_prompt))
        return LLMResult(
            text=self._answers.pop(0),
            model="fake-judge",
            prompt_tokens=1_000,
            completion_tokens=500,
        )


def _case() -> EvaluationCase:
    return EvaluationCase(
        id="q013",
        question="事务为什么会失效",
        expected_keywords=["代理"],
        expected_answer_points=["自调用绕过代理"],
        expected_document_ids=["doc_0123456789abcdef"],
    )


def _hit() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_0123456789abcdef0123",
        document_id="doc_0123456789abcdef",
        text="自调用绕过了代理。",
        score=0.1,
        metadata={"filename": "spring.md"},
    )


def test_llm_judge_sees_context_and_scores_with_cost() -> None:
    llm = SequenceLLM([_JSON])
    judge = LLMJudge(llm, input_price_per_1m=2.0, output_price_per_1m=4.0)
    result = judge.evaluate("事务为什么会失效", "因为代理。", [_hit()], _case())
    assert result.source == "llm"
    assert result.correctness == 4
    assert result.groundedness == 3
    assert result.completeness == 2
    assert result.overall == 3
    assert result.score == pytest.approx(0.75)
    assert result.criteria == {"correctness": 4, "completeness": 2, "groundedness": 3}
    assert result.judge_cost == pytest.approx(0.004)
    system_prompt, user_prompt = llm.calls[0]
    assert "Return JSON only." in system_prompt
    assert "自调用绕过了代理。" in user_prompt
    assert "答案要点：自调用绕过代理" in user_prompt
    assert "事务为什么会失效" in user_prompt


def test_invalid_json_retries_then_stops(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    llm = SequenceLLM(["不是 JSON", "仍然不是"])
    judge = LLMJudge(llm, max_retries=1)
    with pytest.raises(EvaluationError, match="无法解析"):
        judge.evaluate("机密问题唯一标记", "回答", [_hit()], _case())
    assert len(llm.calls) == 2
    assert "上次输出不是合法 JSON" in llm.calls[1][1]
    assert "机密问题唯一标记" not in caplog.text
    assert "自调用绕过了代理。" not in caplog.text


def test_missing_price_keeps_judge_cost_empty() -> None:
    judge = LLMJudge(SequenceLLM([_JSON]))
    result = judge.evaluate("事务为什么会失效", "因为代理。", [_hit()], _case())
    assert result.judge_prompt_tokens == 1000
    assert result.judge_cost is None


def test_mock_judge_is_not_a_score() -> None:
    result = MockJudge().evaluate("事务为什么会失效", "因为代理。", [_hit()], _case())
    assert result.source == "mock"
    assert result.score is None
    assert result.criteria == {"correctness": None, "completeness": None, "groundedness": None}
    assert result.judge_cost is None
    assert "不是真实评测" in result.reasoning


def test_llm_judge_mode_requires_external_models() -> None:
    with pytest.raises(ValueError, match="JUDGE_MODE"):
        make_settings(judge_mode="llm")
