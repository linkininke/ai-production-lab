"""LLM-as-a-Judge。

Judge 是评测模型，不是标准答案。
MockJudge 不打分。它只说明自己不是真实评测，避免把占位结果当成质量。
日志不写问题、上下文、回答和 Judge 的推理。
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.container import AppContainer
from app.core.exceptions import ConfigurationError, EvaluationError
from app.evaluation.core.case import EvaluationCase
from app.evaluation.cost import UsageCost, estimate_cost_usd
from app.llm.base import LLMProvider
from app.retrieval.models import RetrievalResult

JudgeSource = Literal["mock", "llm"]
_SCORE_FIELDS = ("correctness", "groundedness", "completeness", "overall")

RUBRIC = """Return JSON only.
你是评测模型，不是标准答案。只返回一个 JSON 对象，不要写别的文字。
correctness、groundedness、completeness、overall 都必须是 0 到 4 的整数。
correctness：0 完全错误，1 大部分错误，2 部分正确，3 基本正确，4 完全正确。
groundedness：0 基本没有证据支持，1 少量支持，2 部分支持，3 大部分支持，4 完全有依据。
completeness：0 没有回答问题，1 覆盖很少，2 覆盖部分，3 基本完整，4 完整。
overall 是综合分，同样使用 0 到 4。
reasoning 用简短中文说明原因。
JSON 字段只能是 correctness、groundedness、completeness、overall、reasoning。
"""


class JudgeResult(BaseModel):
    """一次 Judge。score 是 overall 除以 4；MockJudge 的分数保持为空。"""

    model_config = ConfigDict(extra="forbid")

    score: float | None = None
    reasoning: str
    correctness: int | None = None
    groundedness: int | None = None
    completeness: int | None = None
    overall: int | None = None
    source: JudgeSource
    judge_latency_ms: float = Field(ge=0)
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    judge_cost: float | None = None

    @property
    def criteria(self) -> dict[str, int | None]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "groundedness": self.groundedness,
        }


class Judge(Protocol):
    @property
    def source(self) -> JudgeSource:
        """mock 不是真实评测。llm 是评测模型，仍然不是标准答案。"""

    def evaluate(
        self,
        question: str,
        answer: str,
        context: list[RetrievalResult],
        expected: EvaluationCase,
    ) -> JudgeResult:
        """根据问题、检索上下文、预期属性和回答打分。"""


class MockJudge:
    """API 不可用时的占位。不产生正确性、依据或完整性分数。"""

    @property
    def source(self) -> JudgeSource:
        return "mock"

    def evaluate(
        self,
        question: str,
        answer: str,
        context: list[RetrievalResult],
        expected: EvaluationCase,
    ) -> JudgeResult:
        del question, answer, context, expected
        return JudgeResult(
            reasoning="MockJudge 不是真实评测。",
            source="mock",
            judge_latency_ms=0.0,
        )


class LLMJudge:
    """调用对话模型按量规打分。非法 JSON 会重试，次数有上限。"""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_retries: int = 1,
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
    ) -> None:
        if max_retries < 0:
            raise ConfigurationError("JUDGE_MAX_RETRIES 不能小于 0")
        self._llm = llm
        self._max_retries = max_retries
        self._input_price = input_price_per_1m
        self._output_price = output_price_per_1m

    @property
    def source(self) -> JudgeSource:
        return "llm"

    def evaluate(
        self,
        question: str,
        answer: str,
        context: list[RetrievalResult],
        expected: EvaluationCase,
    ) -> JudgeResult:
        prompt = _user_prompt(question, answer, context, expected)
        started = perf_counter()
        prompt_tokens = 0
        completion_tokens = 0
        saw_prompt = False
        saw_completion = False
        for attempt in range(self._max_retries + 1):
            reminder = "" if attempt == 0 else "\n上次输出不是合法 JSON。Return JSON only."
            generated = self._llm.generate(RUBRIC, prompt + reminder)
            if generated.prompt_tokens is not None:
                prompt_tokens += generated.prompt_tokens
                saw_prompt = True
            if generated.completion_tokens is not None:
                completion_tokens += generated.completion_tokens
                saw_completion = True
            try:
                parsed = _parse_judge_output(generated.text)
            except (ValueError, ValidationError, json.JSONDecodeError):
                continue
            latency_ms = (perf_counter() - started) * 1000
            used_prompt = prompt_tokens if saw_prompt else None
            used_completion = completion_tokens if saw_completion else None
            return JudgeResult(
                score=parsed.overall / 4,
                reasoning=parsed.reasoning,
                correctness=parsed.correctness,
                groundedness=parsed.groundedness,
                completeness=parsed.completeness,
                overall=parsed.overall,
                source="llm",
                judge_latency_ms=latency_ms,
                judge_prompt_tokens=used_prompt,
                judge_completion_tokens=used_completion,
                judge_cost=estimate_cost_usd(
                    UsageCost(used_prompt, self._input_price, 1_000_000),
                    UsageCost(used_completion, self._output_price, 1_000_000),
                ),
            )
        raise EvaluationError("Judge 输出无法解析")


def build_judge(container: AppContainer) -> Judge:
    """mock 不调用模型。llm 使用已经配置的对话模型，不另建一套密钥。"""
    if container.settings.judge_mode == "mock":
        return MockJudge()
    if container.llm is None:
        raise ConfigurationError("JUDGE_MODE=llm 时没有可用的对话模型")
    return LLMJudge(
        container.llm,
        max_retries=container.settings.judge_max_retries,
        input_price_per_1m=container.settings.llm_input_price_per_1m,
        output_price_per_1m=container.settings.llm_output_price_per_1m,
    )


class _JudgePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: int
    groundedness: int
    completeness: int
    overall: int
    reasoning: str

    @field_validator(*_SCORE_FIELDS, mode="before")
    @classmethod
    def score_is_zero_to_four(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("分数必须是 0 到 4 的整数")
        if value < 0 or value > 4:
            raise ValueError("分数必须是 0 到 4 的整数")
        return value

    @field_validator("reasoning")
    @classmethod
    def reasoning_is_short(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reasoning 不能为空")
        return cleaned[:400]


def _parse_judge_output(text: str) -> _JudgePayload:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("不是 JSON")
    payload = json.loads(text[start : end + 1])
    return _JudgePayload.model_validate(payload)


def _user_prompt(
    question: str,
    answer: str,
    context: list[RetrievalResult],
    expected: EvaluationCase,
) -> str:
    keywords = "、".join(expected.expected_keywords) or "无"
    points = "、".join(expected.expected_answer_points) or "无"
    abstention = "是" if expected.expects_abstention else "否"
    blocks = [_context_block(index, item) for index, item in enumerate(context, start=1)]
    context_text = "\n\n".join(blocks) if blocks else "没有检索到上下文。"
    return (
        f"问题：\n{question}\n\n"
        f"预期属性：\n关键词：{keywords}\n答案要点：{points}\n应拒答：{abstention}\n"
        f"相关文档数：{len(expected.expected_document_ids)}\n\n"
        f"检索到的上下文：\n{context_text}\n\n"
        f"回答：\n{answer}"
    )


def _context_block(index: int, item: RetrievalResult) -> str:
    filename = item.metadata.get("filename", "")
    if not isinstance(filename, str):
        filename = str(filename)
    return f"[{index}] {item.chunk_id} {filename}\n{item.text}"
