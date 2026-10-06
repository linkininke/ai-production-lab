"""回归检查。

基线里的数字必须来自已经跑过的结果。没有测过的正确性和依据分保持为空，不补造。
平均指标通过时，单题跌幅超过允许值仍然算失败。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import PROJECT_ROOT
from app.evaluation.experiment import ExperimentReport
from app.evaluation.metrics import mean

_REQUIRED_METRICS = ("recall_at_5", "mrr_at_5")
_OPTIONAL_METRICS = ("correctness", "groundedness")
_MEASURED_FIELDS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mrr_at_5",
    "average_latency_ms",
)


class RegressionRule(BaseModel):
    """一项指标允许比基线低多少。相等仍算通过。"""

    model_config = ConfigDict(extra="forbid")

    metric: str
    max_drop: float = Field(ge=0)


class GoldenDataset(BaseModel):
    """回归所绑定的题集。内容一变，就不能再和旧基线比质量。"""

    model_config = ConfigDict(extra="forbid")

    version: str
    question_count: int
    sha256: str
    path: str


class RetrieverBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retriever: str
    metrics: dict[str, float]


class RegressionBaseline(BaseModel):
    """一份冻结的检索基线。只保存实测过的数字。"""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    system_version: str
    source: str
    created_at: str
    golden: GoldenDataset
    retrievers: list[RetrieverBaseline]
    rules: list[RegressionRule]


class CaseSnapshot(BaseModel):
    """一道题的回归输入。不保存问题原文。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str | None = None
    recall_at_k: float | None = None
    abstention_correct: bool | None = None


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retriever: str
    metric: str
    status: Literal["passed", "failed", "skipped"]
    max_drop: float
    baseline: float | None = None
    current: float | None = None
    drop: float | None = None
    reason: str = ""


class CaseDrop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str | None = None
    metric: str
    baseline: float
    current: float | None = None
    drop: float | None = None
    max_drop: float
    failed: bool


class CategoryCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    metric: str
    baseline: float | None = None
    current: float | None = None
    drop: float | None = None
    max_drop: float
    failed: bool


class RegressionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    dataset_match: bool
    checks: list[CheckResult]
    case_failures: list[CaseDrop] = Field(default_factory=list)
    category_checks: list[CategoryCheck] = Field(default_factory=list)
    note: str


DEFAULT_RULES: tuple[RegressionRule, ...] = (
    RegressionRule(metric="recall_at_5", max_drop=0.03),
    RegressionRule(metric="mrr_at_5", max_drop=0.05),
    RegressionRule(metric="correctness", max_drop=0.20),
    RegressionRule(metric="groundedness", max_drop=0.20),
)


def fingerprint_dataset(path: Path) -> GoldenDataset:
    """用版本、题数和文件摘要标识黄金题集。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    count = len(questions) if isinstance(questions, list) else 0
    version = payload.get("version", "")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative = path.name
    return GoldenDataset(
        version=str(version),
        question_count=count,
        sha256=digest,
        path=relative,
    )


def freeze_baseline(
    experiment: ExperimentReport,
    dataset_path: Path,
    *,
    source: str,
    system_version: str = "v0.3",
) -> RegressionBaseline:
    """把一次真实检索实验收成基线。空成本和没有测过的 Judge 分都不写入。"""
    retrievers: list[RetrieverBaseline] = []
    for row in experiment.retrievers:
        raw = row.model_dump()
        metrics = {
            name: float(raw[name])
            for name in _MEASURED_FIELDS
            if isinstance(raw.get(name), int | float)
        }
        retrievers.append(RetrieverBaseline(retriever=row.retriever, metrics=metrics))
    return RegressionBaseline(
        dataset_version=experiment.dataset_version,
        system_version=system_version,
        source=source,
        created_at=experiment.created_at,
        golden=fingerprint_dataset(dataset_path),
        retrievers=retrievers,
        rules=list(DEFAULT_RULES),
    )


def regress_experiment(
    baseline: RegressionBaseline,
    current: ExperimentReport,
    dataset_path: Path,
) -> RegressionReport:
    """用当前检索结果对照基线。Judge 分没有实测时跳过，不算通过也不算失败。"""
    golden = fingerprint_dataset(dataset_path)
    dataset_match = (
        golden.sha256 == baseline.golden.sha256
        and golden.version == baseline.golden.version
        and current.dataset_version == baseline.dataset_version
    )
    current_rows = {item.retriever: item for item in current.retrievers}
    checks: list[CheckResult] = []
    for row in baseline.retrievers:
        measured = current_rows.get(row.retriever)
        current_metrics = {} if measured is None else _metrics_from_experiment_row(measured)
        for rule in baseline.rules:
            checks.append(_check(row.retriever, rule, row.metrics, current_metrics))
    passed = dataset_match and all(item.status != "failed" for item in checks)
    note = _experiment_note(dataset_match, checks)
    note += " 这份检索实验没有逐题分数。单题回归使用带有每题 Recall 的报告。"
    return RegressionReport(
        passed=passed,
        dataset_match=dataset_match,
        checks=checks,
        note=note,
    )


def regress_cases(
    baseline: list[CaseSnapshot],
    current: list[CaseSnapshot],
    *,
    max_drop: float = 0.03,
) -> tuple[list[CaseDrop], list[CategoryCheck]]:
    """平均分合格时，仍标出跌破允许值的单题和类别。"""
    current_by_id = {item.id: item for item in current}
    drops: list[CaseDrop] = []
    for item in baseline:
        if item.recall_at_k is None:
            continue
        found = current_by_id.get(item.id)
        current_recall = None if found is None else found.recall_at_k
        drop = None if current_recall is None else item.recall_at_k - current_recall
        failed = current_recall is None or (drop is not None and drop > max_drop)
        if not failed:
            continue
        drops.append(
            CaseDrop(
                case_id=item.id,
                category=item.category,
                metric="recall_at_k",
                baseline=item.recall_at_k,
                current=current_recall,
                drop=drop,
                max_drop=max_drop,
                failed=True,
            )
        )
    return drops, _category_checks(baseline, current, max_drop)


def render_regression(report: RegressionReport) -> str:
    """生成回归结论。没有实测的指标会写明跳过。"""
    title = "REGRESSION PASSED" if report.passed else "REGRESSION FAILED"
    lines = [title, "", report.note, ""]
    for item in report.checks:
        lines.append(_check_line(item))
    if report.case_failures:
        lines.extend(["", "单题失败："])
        lines.extend(
            f"- {item.case_id} {item.metric} { _number(item.baseline) } → { _number(item.current) }"
            for item in report.case_failures
        )
    if report.category_checks:
        lines.extend(["", "分类："])
        lines.extend(
            (
                f"- {item.category} {item.metric} "
                f"{_number(item.baseline)} → {_number(item.current)}"
                f"{' 失败' if item.failed else ''}"
            )
            for item in report.category_checks
        )
    return "\n".join(lines) + "\n"


def _metrics_from_experiment_row(row: object) -> dict[str, float]:
    raw = row.model_dump() if isinstance(row, BaseModel) else {}
    return {
        name: float(raw[name])
        for name in _MEASURED_FIELDS
        if isinstance(raw.get(name), int | float)
    }


def _check(
    retriever: str,
    rule: RegressionRule,
    baseline: dict[str, float],
    current: dict[str, float],
) -> CheckResult:
    before = baseline.get(rule.metric)
    after = current.get(rule.metric)
    if before is None and after is None:
        return CheckResult(
            retriever=retriever,
            metric=rule.metric,
            status="skipped",
            max_drop=rule.max_drop,
            reason="基线和当前结果都没有这项实测指标",
        )
    if before is None or after is None:
        status: Literal["failed", "skipped"] = (
            "skipped" if rule.metric in _OPTIONAL_METRICS else "failed"
        )
        reason = "当前结果缺少这项指标" if before is not None else "基线没有这项指标"
        if status == "skipped":
            reason = "没有实测的 Judge 分数，不参加回归"
        return CheckResult(
            retriever=retriever,
            metric=rule.metric,
            status=status,
            max_drop=rule.max_drop,
            baseline=before,
            current=after,
            reason=reason,
        )
    drop = before - after
    failed = drop > rule.max_drop
    return CheckResult(
        retriever=retriever,
        metric=rule.metric,
        status="failed" if failed else "passed",
        max_drop=rule.max_drop,
        baseline=before,
        current=after,
        drop=drop,
        reason="跌幅超过允许值" if failed else "",
    )


def _category_checks(
    baseline: list[CaseSnapshot],
    current: list[CaseSnapshot],
    max_drop: float,
) -> list[CategoryCheck]:
    before = _category_means(baseline)
    after = _category_means(current)
    checks: list[CategoryCheck] = []
    for category in sorted(set(before) | set(after)):
        left = before.get(category)
        right = after.get(category)
        drop = None if left is None or right is None else left - right
        failed = left is None or right is None or (drop is not None and drop > max_drop)
        checks.append(
            CategoryCheck(
                category=category,
                metric="recall_at_k",
                baseline=left,
                current=right,
                drop=drop,
                max_drop=max_drop,
                failed=failed,
            )
        )
    return checks


def _category_means(cases: list[CaseSnapshot]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in cases:
        if item.category is None or item.recall_at_k is None:
            continue
        grouped.setdefault(item.category, []).append(item.recall_at_k)
    means: dict[str, float] = {}
    for name, values in grouped.items():
        value = mean(values)
        if value is not None:
            means[name] = value
    return means


def _experiment_note(dataset_match: bool, checks: list[CheckResult]) -> str:
    if not dataset_match:
        return "黄金题集或数据集版本和基线不一致，回归失败。"
    skipped = [item.metric for item in checks if item.status == "skipped"]
    unique = sorted(set(skipped))
    if unique:
        names = "、".join(unique)
        return f"检索指标已对照基线。{names} 没有实测，已跳过。"
    return "检索指标已对照基线。"


def _check_line(item: CheckResult) -> str:
    if item.status == "skipped":
        return f"- {item.retriever} {item.metric}：跳过。{item.reason}"
    drop = "-" if item.drop is None else f"{item.drop:.4f}"
    return (
        f"- {item.retriever} {item.metric}："
        f"{_number(item.baseline)} → {_number(item.current)}，"
        f"跌幅 {drop}，允许 {item.max_drop:.2f}，{item.status}"
    )


def _number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> int:
    """用已保存的检索实验生成基线，并立刻和当前结果对照。不调用模型。"""
    results = PROJECT_ROOT / "evaluation_results"
    latest_path = results / "latest.json"
    dataset_path = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
    if not latest_path.is_file():
        print("没有 evaluation_results/latest.json，不能建立基线。")
        return 1
    experiment = ExperimentReport.model_validate_json(latest_path.read_text(encoding="utf-8"))
    baseline = freeze_baseline(experiment, dataset_path, source="evaluation_results/latest.json")
    baseline_path = results / "baseline.json"
    baseline_path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")
    report = regress_experiment(baseline, experiment, dataset_path)
    text = render_regression(report)
    (results / "regression.md").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
