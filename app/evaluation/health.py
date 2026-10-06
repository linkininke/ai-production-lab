"""系统健康卡。

只读取已经写好的评测、回归和检索实验，不重新计算 Recall 或引用指标。
没有实测的数字保持为空。总状态只有通过、警告和失败，不合成一个总分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import PROJECT_ROOT, Settings
from app.evaluation.experiment import ExperimentReport, latest_experiment_path
from app.evaluation.metrics import mean
from app.evaluation.models import CaseTrace, EvaluationReport, QuestionResult
from app.evaluation.regression import RegressionBaseline, RegressionReport, regress_experiment
from app.evaluation.reliability import (
    Check,
    GateReport,
    evaluate_slo,
    quality_gate,
    quality_gate_rules,
    slo_rules,
)

HealthStatus = Literal["PASS", "WARNING", "FAIL"]
MetricStatus = Literal["ok", "warn", "na"]
DeltaDisplay = Literal["points", "relative"]
MetricUnit = Literal["ratio", "ms", "usd"]
GateLabel = Literal["PASS", "FAIL", "NOT CONFIGURED"]

_CATEGORY_ORDER = (
    "semantic",
    "exact_keyword",
    "technical_identifier",
    "multi_document",
    "unanswerable",
    "未分类",
)
_RATIO_METRICS: tuple[tuple[str, str, str], ...] = (
    ("judge_correctness", "Correctness", "answer"),
    ("judge_groundedness", "Groundedness", "answer"),
    ("judge_completeness", "Completeness", "answer"),
    ("citation_validity", "Citation Validity", "citation"),
    ("citation_coverage", "Citation Coverage", "citation"),
    ("abstention_quality", "Abstention Accuracy", "abstention"),
    ("success_rate", "Success Rate", "reliability"),
    ("failure_rate", "Failure Rate", "reliability"),
)
_LATENCY_METRICS: tuple[tuple[str, str], ...] = (
    ("p50_latency_ms", "P50 Latency"),
    ("p95_latency_ms", "P95 Latency"),
    ("p99_latency_ms", "P99 Latency"),
)


class HealthWarningRules(BaseModel):
    """警告阈值。留空表示这项不触发警告，不在页面里写死比例。"""

    model_config = ConfigDict(extra="forbid")

    p95_increase_ratio: float | None = None
    cost_increase_ratio: float | None = None
    quality_drop: float | None = None
    failure_rate_maximum: float | None = None


class HealthComparison(BaseModel):
    """当前值相对基线。delta 是当前减基线；比例指标用百分点，延迟和成本用相对变化。"""

    model_config = ConfigDict(extra="forbid")

    current: float | None = None
    baseline: float | None = None
    delta: float | None = None
    relative: float | None = None
    delta_text: str = ""


class HealthMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    group: str
    unit: MetricUnit
    higher_is_better: bool = True
    comparison: HealthComparison
    status: MetricStatus = "na"
    note: str = ""


class CategoryScore(BaseModel):
    """某一类题目上、已经算好的逐题分数的平均。空值不参与，也不写成 0。"""

    model_config = ConfigDict(extra="forbid")

    category: str
    metric: str
    value: float | None = None
    sample_count: int = Field(ge=0)


class HealthCase(BaseModel):
    """失败题的摘要。检索片段正文不放进来。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    category: str | None = None
    status: str
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    citation_validity: float | None = None
    abstention_correct: bool | None = None
    judge_correctness: float | None = None
    total_latency_ms: float | None = None
    failure_types: list[str] = Field(default_factory=list)
    trace: CaseTrace | None = None


class ExperimentSnapshot(BaseModel):
    """检索实验里已经汇总的一行。平均耗时不是 P95。"""

    model_config = ConfigDict(extra="forbid")

    retriever: str
    recall_at_1: float | None = None
    recall_at_3: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    precision_at_5: float | None = None
    mrr_at_5: float | None = None
    average_latency_ms: float | None = None
    estimated_cost_usd: float | None = None


class HealthCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: HealthStatus
    reasons: list[str] = Field(default_factory=list)
    quality_gate: GateLabel
    quality_gate_status: str
    regression: Literal["PASS", "FAIL", "NOT RUN"]
    regression_failed_cases: int | None = None
    regression_case_total: int | None = None
    regression_failed_checks: int = 0
    slo: GateLabel
    slo_status: str
    source_filename: str
    experiment_id: str = ""
    retrieval_mode: str = ""
    prompt_version: str = ""
    baseline_filename: str = ""
    top_k: int
    metrics: list[HealthMetric] = Field(default_factory=list)
    categories: list[CategoryScore] = Field(default_factory=list)
    failed_cases: list[HealthCase] = Field(default_factory=list)
    experiments: list[ExperimentSnapshot] = Field(default_factory=list)
    experiment_note: str = ""
    assessment_note: str = ""
    skipped_retrievers: list[str] = Field(default_factory=list)


def build_health_card(
    report: EvaluationReport,
    *,
    baseline: EvaluationReport | None = None,
    regression: RegressionReport | None = None,
    experiment: ExperimentReport | None = None,
    warning: HealthWarningRules | None = None,
    context_note: str = "",
    live_gate: GateReport | None = None,
    live_slo: GateReport | None = None,
) -> HealthCard:
    """把一份回答评测收成健康卡。基线、回归和检索实验都可以缺。"""
    rules = warning or HealthWarningRules()
    metrics = _metrics(report, baseline, rules)
    gate_status, gate_checks = _live_status(report.quality_gate_status, live_gate)
    slo_status, slo_checks = _live_status(report.slo_status, live_slo)
    fail_reasons, warn_reasons, regression_label, failed_cases, case_total, failed_checks = (
        _status_parts(
            report,
            regression,
            metrics,
            gate_status=gate_status,
            slo_status=slo_status,
            gate_checks=gate_checks,
            slo_checks=slo_checks,
        )
    )
    if context_note:
        warn_reasons.append(context_note)
    if report.question_count < 1:
        warn_reasons.append("这份报告没有题目。")
    if fail_reasons:
        status: HealthStatus = "FAIL"
        reasons = fail_reasons + warn_reasons
    elif warn_reasons:
        status = "WARNING"
        reasons = warn_reasons
    else:
        status = "PASS"
        reasons = []
    return HealthCard(
        status=status,
        reasons=reasons,
        quality_gate=_gate_label(gate_status),
        quality_gate_status=gate_status,
        regression=regression_label,
        regression_failed_cases=failed_cases,
        regression_case_total=case_total,
        regression_failed_checks=failed_checks,
        slo=_gate_label(slo_status),
        slo_status=slo_status,
        source_filename=report.filename,
        experiment_id=report.experiment_id,
        retrieval_mode=report.retrieval_mode,
        prompt_version=report.prompt_version,
        baseline_filename="" if baseline is None else baseline.filename,
        top_k=report.top_k,
        metrics=metrics,
        categories=_categories(report.questions),
        failed_cases=_failed_cases(report.questions),
        experiments=_experiments(experiment),
        experiment_note=_experiment_note(experiment),
        assessment_note=_assessment_note(live_gate, live_slo),
        skipped_retrievers=[] if experiment is None else list(experiment.skipped),
    )


def gates_for_saved_report(
    report: EvaluationReport,
    settings: Settings,
) -> tuple[GateReport, GateReport]:
    """用当前配置对照报告里已经测到的数字。不重新计算指标，也不调用模型。"""
    groundedness = report.judge_groundedness
    metrics = {
        "recall_at_5": report.recall_at_k if report.top_k == 5 else None,
        "citation_validity": report.citation_validity,
        "groundedness": None if groundedness is None else groundedness * 4,
        "p95_latency_ms": report.p95_latency_ms,
        "success_rate": report.success_rate,
    }
    gate = quality_gate(
        metrics,
        quality_gate_rules(
            recall_at_5=settings.quality_gate_recall_at_5,
            citation_validity=settings.quality_gate_citation_validity,
            groundedness=settings.quality_gate_groundedness,
            p95_ms=settings.quality_gate_p95_ms,
        ),
    )
    slo = evaluate_slo(
        metrics,
        slo_rules(
            success_rate=settings.slo_success_rate,
            p95_ms=settings.slo_p95_latency_ms,
            citation_validity=settings.slo_citation_validity,
        ),
    )
    return gate, slo


def _live_status(stored: str, live: GateReport | None) -> tuple[str, list[Check]]:
    if live is None or not live.checks:
        return stored, []
    return live.status, live.checks


def _assessment_note(gate: GateReport | None, slo: GateReport | None) -> str:
    configured = (gate is not None and gate.checks) or (slo is not None and slo.checks)
    if not configured:
        return ""
    return "质量门和 SLO 使用当前配置，对照的是这份报告里已经测到的数字，没有重新调用模型。"


def load_health_context(
    dataset_path: Path,
) -> tuple[RegressionReport | None, ExperimentReport | None, str]:
    """读取已有基线和检索实验。文件不存在时返回空，不调用模型。"""
    notes: list[str] = []
    experiment = _read_experiment(notes)
    regression = _read_regression(dataset_path, experiment, notes)
    return regression, experiment, " ".join(notes)


def _metrics(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    rules: HealthWarningRules,
) -> list[HealthMetric]:
    metrics = [_recall_metric(report, baseline, rules), _rank_metric(report, baseline, rules)]
    metrics.append(_precision_metric(report, baseline, rules))
    for name, label, group in _RATIO_METRICS:
        higher = name != "failure_rate"
        metrics.append(
            _from_report(
                report,
                baseline,
                name=name,
                label=label,
                group=group,
                unit="ratio",
                higher_is_better=higher,
                display="points",
                rules=rules,
            )
        )
    for name, label in _LATENCY_METRICS:
        metrics.append(
            _from_report(
                report,
                baseline,
                name=name,
                label=label,
                group="performance",
                unit="ms",
                higher_is_better=False,
                display="relative",
                rules=rules,
            )
        )
    metrics.append(
        _from_report(
            report,
            baseline,
            name="judge_cost",
            label="Cost",
            group="cost",
            unit="usd",
            higher_is_better=False,
            display="relative",
            rules=rules,
            note="成本来自报告里的 Judge 费用。为空表示没有单价或没有可计费调用，不是 0。",
        )
    )
    return metrics


def _recall_metric(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    rules: HealthWarningRules,
) -> HealthMetric:
    at_five = report.top_k == 5
    return _from_report(
        report,
        baseline,
        name="recall_at_k",
        label="Recall@5" if at_five else f"Recall@{report.top_k}",
        group="retrieval",
        unit="ratio",
        higher_is_better=True,
        display="points",
        rules=rules,
        note="" if at_five else "这次 Top-K 不是 5，所以这不是 Recall@5。",
        metric_name="recall_at_5" if at_five else "recall_at_k",
    )


def _rank_metric(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    rules: HealthWarningRules,
) -> HealthMetric:
    note = "" if report.top_k == 5 else "这次 Top-K 不是 5，下面的 MRR 不是 @5。"
    return _from_report(
        report,
        baseline,
        name="mrr_at_k",
        label="MRR@5" if report.top_k == 5 else f"MRR@{report.top_k}",
        group="retrieval",
        unit="ratio",
        higher_is_better=True,
        display="points",
        rules=rules,
        note=note,
        metric_name="mrr_at_5" if report.top_k == 5 else "mrr_at_k",
    )


def _precision_metric(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    rules: HealthWarningRules,
) -> HealthMetric:
    note = "" if report.top_k == 5 else "这次 Top-K 不是 5，下面的 Precision 不是 @5。"
    return _from_report(
        report,
        baseline,
        name="precision_at_k",
        label="Precision@5" if report.top_k == 5 else f"Precision@{report.top_k}",
        group="retrieval",
        unit="ratio",
        higher_is_better=True,
        display="points",
        rules=rules,
        note=note,
        metric_name="precision_at_5" if report.top_k == 5 else "precision_at_k",
    )


def _from_report(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    *,
    name: str,
    label: str,
    group: str,
    unit: MetricUnit,
    higher_is_better: bool,
    display: DeltaDisplay,
    rules: HealthWarningRules,
    note: str = "",
    metric_name: str | None = None,
) -> HealthMetric:
    current = _field(report, name)
    previous = None if baseline is None else _field(baseline, name)
    comparison = _comparison(current, previous, display=display, unit=unit)
    status, warning = _metric_warning(
        metric_name or name,
        label,
        comparison,
        unit=unit,
        higher_is_better=higher_is_better,
        rules=rules,
    )
    if current is None:
        status = "na"
        warning = ""
        if not note:
            note = "成本为空，不是 0。" if name == "judge_cost" else "没有这项实测。"
    shown = warning or note
    return HealthMetric(
        name=metric_name or name,
        label=label,
        group=group,
        unit=unit,
        higher_is_better=higher_is_better,
        comparison=comparison,
        status=status,
        note=shown,
    )


def _field(report: EvaluationReport, name: str) -> float | None:
    value = getattr(report, name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _comparison(
    current: float | None,
    baseline: float | None,
    *,
    display: DeltaDisplay,
    unit: MetricUnit,
) -> HealthComparison:
    delta = None if current is None or baseline is None else round(current - baseline, 6)
    relative = None
    if delta is not None and baseline not in (None, 0):
        relative = round(delta / baseline, 6)
    return HealthComparison(
        current=current,
        baseline=baseline,
        delta=delta,
        relative=relative if display == "relative" else None,
        delta_text=_delta_text(delta, relative if display == "relative" else None, display, unit),
    )


def _delta_text(
    delta: float | None,
    relative: float | None,
    display: DeltaDisplay,
    unit: MetricUnit,
) -> str:
    if delta is None:
        return ""
    if display == "points":
        return f"{delta * 100:+.1f} pp"
    if relative is not None:
        return f"{relative * 100:+.0f}%"
    if unit == "ms":
        return f"{delta:+.0f} ms"
    if unit == "usd":
        return f"{delta:+.4f}"
    return ""


def _metric_warning(
    name: str,
    label: str,
    comparison: HealthComparison,
    *,
    unit: MetricUnit,
    higher_is_better: bool,
    rules: HealthWarningRules,
) -> tuple[MetricStatus, str]:
    if comparison.current is None:
        return "na", ""
    if (
        name == "p95_latency_ms"
        and rules.p95_increase_ratio is not None
        and comparison.relative is not None
        and comparison.relative > rules.p95_increase_ratio
    ):
        return "warn", _ratio_reason("P95 延迟", comparison.relative, rules.p95_increase_ratio)
    if (
        name == "judge_cost"
        and rules.cost_increase_ratio is not None
        and comparison.relative is not None
        and comparison.relative > rules.cost_increase_ratio
    ):
        return "warn", _ratio_reason("成本", comparison.relative, rules.cost_increase_ratio)
    if (
        name == "failure_rate"
        and rules.failure_rate_maximum is not None
        and comparison.current > rules.failure_rate_maximum
    ):
        limit = rules.failure_rate_maximum * 100
        actual = comparison.current * 100
        return "warn", f"失败率 {actual:.1f}% 高于配置上限 {limit:.1f}%。"
    drop = None if comparison.delta is None else -comparison.delta
    if (
        higher_is_better
        and unit == "ratio"
        and rules.quality_drop is not None
        and drop is not None
        and drop > rules.quality_drop
    ):
        return (
            "warn",
            (
                f"{label} 比基线下降 {drop * 100:.1f} 个百分点，"
                f"配置的警告阈值是 {rules.quality_drop * 100:.1f}。"
            ),
        )
    return "ok", ""


def _ratio_reason(label: str, relative: float, limit: float) -> str:
    return f"{label} 相对基线上升 {relative * 100:.0f}%，配置的警告阈值是 {limit * 100:.0f}%。"


def _status_parts(
    report: EvaluationReport,
    regression: RegressionReport | None,
    metrics: list[HealthMetric],
    *,
    gate_status: str,
    slo_status: str,
    gate_checks: list[Check],
    slo_checks: list[Check],
) -> tuple[list[str], list[str], Literal["PASS", "FAIL", "NOT RUN"], int | None, int | None, int]:
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    if gate_status.endswith("FAILED"):
        fail_reasons.append(f"质量门未通过：{gate_status}")
        fail_reasons.extend(_failed_checks("质量门", gate_checks))
    elif not gate_status.endswith("PASSED"):
        warn_reasons.append("质量门未配置，不能视为通过。")
    if slo_status.endswith("FAILED"):
        fail_reasons.append(f"可靠性未通过：{slo_status}")
        fail_reasons.extend(_failed_checks("SLO", slo_checks))
    if report.failure_count > 0 and not slo_status.endswith("FAILED"):
        warn_reasons.append(f"有 {report.failure_count} 次请求失败。")
    regression_label: Literal["PASS", "FAIL", "NOT RUN"]
    failed_cases: int | None
    case_total: int | None
    failed_checks = 0
    if regression is None:
        regression_label = "NOT RUN"
        failed_cases = None
        case_total = None
        warn_reasons.append("还没有回归结果，不能确认相比基线有没有退步。")
    else:
        failed_checks = sum(1 for item in regression.checks if item.status == "failed")
        failed_cases = len(regression.case_failures)
        case_total = report.question_count
        if regression.passed:
            regression_label = "PASS"
        else:
            regression_label = "FAIL"
            fail_reasons.append("回归未通过。")
            if regression.case_failures:
                fail_reasons.append(
                    f"回归失败题 {len(regression.case_failures)} / {report.question_count}。"
                )
            elif failed_checks:
                fail_reasons.append(f"未通过的回归检查：{failed_checks} 项。")
    for item in metrics:
        if item.status == "warn" and item.note and item.note not in warn_reasons:
            warn_reasons.append(item.note)
    return fail_reasons, warn_reasons, regression_label, failed_cases, case_total, failed_checks


def _failed_checks(label: str, checks: list[Check]) -> list[str]:
    lines: list[str] = []
    for item in checks:
        if item.status != "failed":
            continue
        detail = item.reason or "未通过"
        if item.actual is not None:
            detail = f"{detail}，实测 {_format_check_value(item.actual)}"
        if item.minimum is not None:
            detail = f"{detail}，下限 {_format_check_value(item.minimum)}"
        if item.maximum is not None:
            detail = f"{detail}，上限 {_format_check_value(item.maximum)}"
        lines.append(f"{label} {item.metric}：{detail}。")
    return lines


def _format_check_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def _categories(questions: list[QuestionResult]) -> list[CategoryScore]:
    buckets: dict[tuple[str, str], list[float]] = {}
    for item in questions:
        category = item.category or "未分类"
        judge = item.judge_correctness
        if isinstance(judge, int) and not isinstance(judge, bool):
            buckets.setdefault((category, "judge_correctness"), []).append(judge / 4)
        if item.recall_at_k is not None:
            buckets.setdefault((category, "recall_at_k"), []).append(item.recall_at_k)
        if item.abstention_correct is not None:
            buckets.setdefault((category, "abstention_quality"), []).append(
                1.0 if item.abstention_correct else 0.0
            )
    scores: list[CategoryScore] = []
    for (category, metric), values in buckets.items():
        scores.append(
            CategoryScore(
                category=category,
                metric=metric,
                value=mean(values),
                sample_count=len(values),
            )
        )
    scores.sort(key=lambda item: (_category_rank(item.category), item.category, item.metric))
    return scores


def _category_rank(category: str) -> int:
    try:
        return _CATEGORY_ORDER.index(category)
    except ValueError:
        return len(_CATEGORY_ORDER)


def _failed_cases(questions: list[QuestionResult]) -> list[HealthCase]:
    cases = [_case(item) for item in questions if _failed(item)]
    cases.sort(key=lambda item: item.id)
    return cases


def _failed(item: QuestionResult) -> bool:
    if item.status == "error":
        return True
    if item.failure_records:
        return True
    if item.abstention_correct is False:
        return True
    return item.recall_at_k is not None and item.recall_at_k < 1


def _case(item: QuestionResult) -> HealthCase:
    judge = item.judge_correctness
    score = None if not isinstance(judge, int) or isinstance(judge, bool) else judge / 4
    return HealthCase(
        id=item.id,
        question=item.question,
        category=item.category,
        status=item.status,
        recall_at_k=item.recall_at_k,
        reciprocal_rank=item.reciprocal_rank,
        citation_validity=item.citation_validity,
        abstention_correct=item.abstention_correct,
        judge_correctness=score,
        total_latency_ms=item.total_latency_ms,
        failure_types=[record.failure_type for record in item.failure_records],
        trace=item.trace,
    )


def _experiments(experiment: ExperimentReport | None) -> list[ExperimentSnapshot]:
    if experiment is None:
        return []
    return [
        ExperimentSnapshot(
            retriever=row.retriever,
            recall_at_1=row.recall_at_1,
            recall_at_3=row.recall_at_3,
            recall_at_5=row.recall_at_5,
            recall_at_10=row.recall_at_10,
            precision_at_5=row.precision_at_5,
            mrr_at_5=row.mrr_at_5,
            average_latency_ms=row.average_latency_ms,
            estimated_cost_usd=row.estimated_cost_usd,
        )
        for row in experiment.retrievers
    ]


def _experiment_note(experiment: ExperimentReport | None) -> str:
    if experiment is None:
        return "还没有检索实验报告。"
    notes = ["检索实验的平均耗时不是 P95。成本为空不是 0。"]
    if experiment.semantic_embedding is not True:
        notes.append("这次不是语义向量，不能当成生产检索结果。")
    if experiment.note:
        notes.append(experiment.note)
    return " ".join(notes)


def _gate_label(status: str) -> GateLabel:
    if status.endswith("FAILED"):
        return "FAIL"
    if status.endswith("PASSED"):
        return "PASS"
    return "NOT CONFIGURED"


def _read_experiment(notes: list[str]) -> ExperimentReport | None:
    path = latest_experiment_path()
    if not path.is_file():
        return None
    try:
        return ExperimentReport.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        notes.append("检索实验报告无法读取。")
        return None


def _read_regression(
    dataset_path: Path,
    experiment: ExperimentReport | None,
    notes: list[str],
) -> RegressionReport | None:
    baseline_path = PROJECT_ROOT / "evaluation_results" / "baseline.json"
    if not baseline_path.is_file():
        return None
    if experiment is None:
        notes.append("有回归基线，但没有可用的检索实验，无法对照。")
        return None
    if not dataset_path.is_file():
        notes.append("有回归基线，但评测集文件不存在，无法对照。")
        return None
    try:
        baseline = RegressionBaseline.model_validate_json(
            baseline_path.read_text(encoding="utf-8")
        )
    except ValidationError:
        notes.append("回归基线无法读取。")
        return None
    return regress_experiment(baseline, experiment, dataset_path)
