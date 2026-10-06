"""可靠性。

重试次数由调用方限制。这里负责预算、质量门、SLO 和延迟分位。
没有实测的指标保持为空，不用 0 冒充通过。
无法估算成本时，设置了预算就停止，不继续调用模型。
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.cost import UsageCost, estimate_cost_usd

CheckStatus = Literal["passed", "failed", "skipped"]


class Threshold(BaseModel):
    """一项下限或上限。相等算通过。"""

    model_config = ConfigDict(extra="forbid")

    metric: str
    minimum: float | None = None
    maximum: float | None = None


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    status: CheckStatus
    actual: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    reason: str = ""


class GateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    checks: list[Check] = Field(default_factory=list)


class BudgetDecision(BaseModel):
    """调用模型之前的预算结论。aborted 为真时不能继续请求。"""

    model_config = ConfigDict(extra="forbid")

    aborted: bool
    selected_cases: int = Field(ge=0)
    predicted_cost_usd: float | None = None
    reason: str = ""


def percentile(values: list[float], percent: float) -> float | None:
    """最近秩分位。空列表返回 None，不拿平均值代替尾部。"""
    if not values:
        return None
    if percent <= 0 or percent > 100:
        raise ValueError("分位必须在 0 到 100 之间")
    ordered = sorted(values)
    rank = math.ceil(percent / 100 * len(ordered))
    index = min(len(ordered), max(rank, 1)) - 1
    return ordered[index]


def latency_percentiles(values: list[float]) -> tuple[float | None, float | None, float | None]:
    return percentile(values, 50), percentile(values, 95), percentile(values, 99)


def plan_budget(
    *,
    question_count: int,
    max_cases: int | None,
    max_cost: float | None,
    predicted_cost_per_case: float | None,
) -> BudgetDecision:
    """在调用模型前决定能跑多少题。预测成本超过预算，或无法预测，都停止。"""
    if question_count < 1:
        return BudgetDecision(aborted=True, selected_cases=0, reason="评测集没有题目，已停止。")
    if max_cases is not None and max_cases < 1:
        return BudgetDecision(
            aborted=True,
            selected_cases=0,
            reason="max_cases 必须大于 0，已停止。",
        )
    if max_cost is not None and max_cost < 0:
        return BudgetDecision(
            aborted=True,
            selected_cases=0,
            reason="max_cost 不能小于 0，已停止。",
        )
    selected = question_count if max_cases is None else min(question_count, max_cases)
    if max_cost is None:
        return BudgetDecision(aborted=False, selected_cases=selected)
    if predicted_cost_per_case is None:
        return BudgetDecision(
            aborted=True,
            selected_cases=0,
            reason="设置了预算，但没有可用于估算的 Token 和单价，已停止，没有继续调用模型。",
        )
    predicted = predicted_cost_per_case * selected
    if predicted > max_cost:
        return BudgetDecision(
            aborted=True,
            selected_cases=0,
            predicted_cost_usd=predicted,
            reason=(
                f"预测成本 {predicted:.4f} 美元超过预算 {max_cost:.4f} 美元，"
                "已停止，没有继续调用模型。"
            ),
        )
    return BudgetDecision(
        aborted=False,
        selected_cases=selected,
        predicted_cost_usd=predicted,
    )


def predicted_case_cost_usd(
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    embedding_tokens: int | None,
    llm_input_price_per_1m: float | None,
    llm_output_price_per_1m: float | None,
    embedding_price_per_1m: float | None,
) -> float | None:
    """只按已经填写的用量和单价估算一题。缺任何已填用量的单价时返回 None。"""
    return estimate_cost_usd(
        UsageCost(prompt_tokens, llm_input_price_per_1m, 1_000_000),
        UsageCost(completion_tokens, llm_output_price_per_1m, 1_000_000),
        UsageCost(embedding_tokens, embedding_price_per_1m, 1_000_000),
    )


def quality_gate_rules(
    *,
    recall_at_5: float | None,
    citation_validity: float | None,
    groundedness: float | None,
    p95_ms: float | None,
) -> list[Threshold]:
    """把已经填写的质量门配置收成规则。留空的项不参加检查。"""
    rules: list[Threshold] = []
    if recall_at_5 is not None:
        rules.append(Threshold(metric="recall_at_5", minimum=recall_at_5))
    if citation_validity is not None:
        rules.append(Threshold(metric="citation_validity", minimum=citation_validity))
    if groundedness is not None:
        rules.append(Threshold(metric="groundedness", minimum=groundedness))
    if p95_ms is not None:
        rules.append(Threshold(metric="p95_latency_ms", maximum=p95_ms))
    return rules


def slo_rules(
    *,
    success_rate: float | None,
    p95_ms: float | None,
    citation_validity: float | None,
) -> list[Threshold]:
    """把已经填写的 SLO 配置收成规则。留空的项不参加检查。"""
    rules: list[Threshold] = []
    if success_rate is not None:
        rules.append(Threshold(metric="success_rate", minimum=success_rate))
    if p95_ms is not None:
        rules.append(Threshold(metric="p95_latency_ms", maximum=p95_ms))
    if citation_validity is not None:
        rules.append(Threshold(metric="citation_validity", minimum=citation_validity))
    return rules


def quality_gate(metrics: dict[str, float | None], rules: list[Threshold]) -> GateReport:
    """评测结束后的发布检查。没有配置规则时不假装通过。"""
    return _assess(metrics, rules, passed="QUALITY GATE PASSED", failed="QUALITY GATE FAILED")


def evaluate_slo(metrics: dict[str, float | None], rules: list[Threshold]) -> GateReport:
    """对照可配置的 SLO。示例数值不是写死的生产目标。"""
    return _assess(metrics, rules, passed="SLO PASSED", failed="SLO FAILED")


def _assess(
    metrics: dict[str, float | None],
    rules: list[Threshold],
    *,
    passed: str,
    failed: str,
) -> GateReport:
    if not rules:
        if passed.startswith("QUALITY"):
            label = "QUALITY GATE NOT CONFIGURED"
        else:
            label = "SLO NOT CONFIGURED"
        return GateReport(status=label)
    checks = [_check(metrics, rule) for rule in rules]
    status = failed if any(item.status == "failed" for item in checks) else passed
    return GateReport(status=status, checks=checks)


def _check(metrics: dict[str, float | None], rule: Threshold) -> Check:
    actual = metrics.get(rule.metric)
    if actual is None:
        return Check(
            metric=rule.metric,
            status="failed",
            actual=None,
            minimum=rule.minimum,
            maximum=rule.maximum,
            reason="没有这项实测",
        )
    if rule.minimum is not None and actual < rule.minimum:
        return Check(
            metric=rule.metric,
            status="failed",
            actual=actual,
            minimum=rule.minimum,
            maximum=rule.maximum,
            reason="低于下限",
        )
    if rule.maximum is not None and actual > rule.maximum:
        return Check(
            metric=rule.metric,
            status="failed",
            actual=actual,
            minimum=rule.minimum,
            maximum=rule.maximum,
            reason="高于上限",
        )
    return Check(
        metric=rule.metric,
        status="passed",
        actual=actual,
        minimum=rule.minimum,
        maximum=rule.maximum,
    )
