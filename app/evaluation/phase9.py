"""四组对照实验。

A 向量，B 混合，C 混合加重排，D 在 C 上改用提示词 v2。
同一份评测集。Judge 使用对话模型，分数不是标准答案。
没有单价时成本保持为空。
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

from app.config import PROJECT_ROOT, get_settings
from app.core.container import build_container
from app.core.exceptions import AppError
from app.evaluation.models import EvaluationReport, QuestionResult
from app.evaluation.reliability import Threshold, quality_gate
from app.evaluation.runner import EvaluationRunner

logger = logging.getLogger(__name__)

_EXAMPLE_RULES = (
    Threshold(metric="recall_at_5", minimum=0.80),
    Threshold(metric="citation_validity", minimum=0.98),
    Threshold(metric="groundedness", minimum=3.5),
    Threshold(metric="p95_latency_ms", maximum=3000),
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    retrieval_mode: str
    prompt_version: str
    title: str


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("A", "vector", "v1", "Vector"),
    ExperimentSpec("B", "hybrid", "v1", "Hybrid"),
    ExperimentSpec("C", "hybrid_rerank", "v1", "Hybrid + Reranker"),
    ExperimentSpec("D", "hybrid_rerank", "v2", "Hybrid + Reranker + Prompt v2"),
)


def render_phase9(reports: list[EvaluationReport]) -> str:
    """用已经跑完的报告回答对照问题。缺哪一组就写缺，不补数字。"""
    by_id = {item.experiment_id: item for item in reports if item.experiment_id}
    lines = [
        "# Phase 9 实验报告",
        "",
        "四组使用同一份评测集。Judge 是评测模型，不是标准答案。",
        "提示词 v2 的假设：明确拒答原句和引用标记后，拒答判断、引用有效率和有依据程度会上升。",
        "成本为空表示没有填写单价，不是 0。",
        "",
        "## 总表",
        "",
        _table(reports),
        "",
    ]
    vector = by_id.get("A")
    hybrid = by_id.get("B")
    rerank = by_id.get("C")
    prompted = by_id.get("D")
    lines.extend(_retrieval_section("Hybrid 相对 Vector", vector, hybrid))
    lines.extend(_still_failing(hybrid))
    lines.extend(_retrieval_section("Reranker 相对 Hybrid", hybrid, rerank))
    lines.extend(_latency_and_cost(hybrid, rerank))
    lines.extend(_answer_follows_retrieval(reports))
    lines.extend(_failure_split(reports))
    lines.extend(_unanswerable(reports))
    lines.extend(_prompt_effect(rerank, prompted))
    lines.extend(_category_regression(rerank, prompted))
    lines.extend(_gates(reports))
    missing = [item.experiment_id for item in EXPERIMENTS if item.experiment_id not in by_id]
    if missing:
        lines.extend(["", f"未完成的实验：{'、'.join(missing)}。下面没有这些组的结论。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    os.environ["JUDGE_MODE"] = "llm"
    os.environ["CHROMA_COLLECTION"] = "kb_phase9"
    get_settings.cache_clear()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    container = build_container(settings)
    finished: list[EvaluationReport] = []
    code = 0
    try:
        for spec in EXPERIMENTS:
            print(f"开始实验 {spec.experiment_id} {spec.title}", flush=True)
            report = EvaluationRunner.from_container(
                container,
                retrieval_mode=spec.retrieval_mode,
                prompt_version=spec.prompt_version,
                experiment_id=spec.experiment_id,
            ).run(top_k=5)
            finished.append(report)
            print(
                f"完成 {spec.experiment_id} 文件={report.filename} "
                f"Recall={_fmt(report.recall_at_k)} 失败率={_fmt(report.failure_rate)}",
                flush=True,
            )
    except AppError as exc:
        print(exc.message, file=sys.stderr)
        code = 1
    finally:
        container.close()
    if finished:
        text = render_phase9(finished)
        directory = PROJECT_ROOT / "evaluation_results"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "phase9.md"
        path.write_text(text, encoding="utf-8")
        print(text)
        print(path)
    return code


def _table(reports: list[EvaluationReport]) -> str:
    header = (
        "| 实验 | 检索 | 提示词 | Recall@5 | MRR@5 | Precision@5 | "
        "正确性 | 有依据 | 完整性 | 引用有效 | 拒答判断 | P50 | P95 | P99 | 成本 | 失败率 |"
    )
    sep = (
        "| --- | --- | --- | --- | --- | --- | --- "
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = [header, sep]
    for item in reports:
        rows.append(
            "| "
            + " | ".join(
                [
                    item.experiment_id or "-",
                    item.retrieval_mode or "-",
                    item.prompt_version or "-",
                    _fmt(item.recall_at_k if item.top_k == 5 else None),
                    _fmt(item.mrr_at_k if item.top_k == 5 else None),
                    _fmt(item.precision_at_k if item.top_k == 5 else None),
                    _fmt(item.judge_correctness),
                    _fmt(item.judge_groundedness),
                    _fmt(item.judge_completeness),
                    _fmt(item.citation_validity),
                    _fmt(item.abstention_quality),
                    _ms(item.p50_latency_ms),
                    _ms(item.p95_latency_ms),
                    _ms(item.p99_latency_ms),
                    _fmt(item.judge_cost),
                    _fmt(item.failure_rate),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _retrieval_section(
    title: str,
    before: EvaluationReport | None,
    after: EvaluationReport | None,
) -> list[str]:
    lines = ["", f"## {title}", ""]
    if before is None or after is None:
        lines.append("缺少对照报告，不能比较。")
        return lines
    lines.append(
        f"平均 Recall {_fmt(before.recall_at_k)} → {_fmt(after.recall_at_k)}，"
        f"MRR {_fmt(before.mrr_at_k)} → {_fmt(after.mrr_at_k)}。"
    )
    improved, declined = _recall_changes(before, after)
    lines.append("Recall 上升的题：" + (_ids(improved) or "没有。"))
    lines.append("Recall 下降的题：" + (_ids(declined) or "没有。"))
    return lines


def _still_failing(report: EvaluationReport | None) -> list[str]:
    lines = ["", "## Hybrid 仍然失败的题", ""]
    if report is None:
        lines.append("没有 Hybrid 报告。")
        return lines
    failed = [
        item
        for item in report.questions
        if _recall_miss(item) or _has_type(item, "RETRIEVAL_FAILURE")
    ]
    if not failed:
        lines.append("没有 Recall 不满、也没有检索失败记录的题。")
        return lines
    for item in failed:
        lines.append(f"- {item.id} · {item.category or '未分类'} · Recall {_fmt(item.recall_at_k)}")
    return lines


def _latency_and_cost(
    hybrid: EvaluationReport | None,
    rerank: EvaluationReport | None,
) -> list[str]:
    lines = ["", "## Reranker 的耗时和成本", ""]
    if hybrid is None or rerank is None:
        lines.append("缺少对照报告。")
        return lines
    lines.append(
        f"平均总耗时 {_ms(hybrid.mean_total_latency_ms)} → {_ms(rerank.mean_total_latency_ms)}。"
    )
    lines.append(f"P95 {_ms(hybrid.p95_latency_ms)} → {_ms(rerank.p95_latency_ms)}。")
    lines.append(
        "成本 "
        f"{_fmt(hybrid.judge_cost)} → {_fmt(rerank.judge_cost)}。"
        "空表示没有单价，不能写成增加了 0。"
    )
    return lines


def _answer_follows_retrieval(reports: list[EvaluationReport]) -> list[str]:
    lines = ["", "## 答案质量是否跟着检索一起变化", ""]
    if not reports:
        lines.append("没有报告。")
        return lines
    for item in reports:
        lines.append(
            f"- {item.experiment_id or '-'} Recall {_fmt(item.recall_at_k)}，"
            f"关键词 {_fmt(item.keyword_coverage)}，"
            f"引用有效 {_fmt(item.citation_validity)}，"
            f"Judge 正确性 {_fmt(item.judge_correctness)}，"
            f"有依据 {_fmt(item.judge_groundedness)}。"
        )
    if all(item.judge_correctness is None for item in reports):
        lines.append("Judge 正确性没有实测，不能说答案质量提高了。")
    return lines


def _failure_split(reports: list[EvaluationReport]) -> list[str]:
    lines = ["", "## 失败属于哪一层", ""]
    for item in reports:
        counts = "、".join(f"{row.name} {row.count}" for row in item.failure_by_type)
        lines.append(f"- {item.experiment_id or '-'}：{counts or '没有失败记录'}")
    return lines


def _unanswerable(reports: list[EvaluationReport]) -> list[str]:
    lines = ["", "## 无法回答的题", ""]
    for item in reports:
        cases = [question for question in item.questions if question.expects_abstention]
        if not cases:
            lines.append(f"- {item.experiment_id or '-'}：没有拒答题。")
            continue
        correct = sum(question.abstention_correct is True for question in cases)
        wrong = [question.id for question in cases if question.abstention_correct is False]
        lines.append(
            f"- {item.experiment_id or '-'}：{correct}/{len(cases)} 题拒答判断正确。"
            + (f" 不正确：{'、'.join(wrong)}。" if wrong else "")
        )
    return lines


def _prompt_effect(
    before: EvaluationReport | None,
    after: EvaluationReport | None,
) -> list[str]:
    lines = ["", "## 提示词 v2 相对同一检索", ""]
    if before is None or after is None:
        lines.append("缺少 C 或 D，不能判断提示词有没有改善。")
        return lines
    lines.append(
        "拒答判断 "
        f"{_fmt(before.abstention_quality)} → {_fmt(after.abstention_quality)}，"
        f"引用有效 {_fmt(before.citation_validity)} → {_fmt(after.citation_validity)}，"
        f"有依据 {_fmt(before.judge_groundedness)} → {_fmt(after.judge_groundedness)}，"
        f"关键词 {_fmt(before.keyword_coverage)} → {_fmt(after.keyword_coverage)}，"
        f"失败率 {_fmt(before.failure_rate)} → {_fmt(after.failure_rate)}。"
    )
    lines.append("Judge 分数是 0 到 1 的均值。没有实测时保持为空。")
    return lines


def _category_regression(
    before: EvaluationReport | None,
    after: EvaluationReport | None,
) -> list[str]:
    lines = ["", "## 平均上升时有没有某一类下降", ""]
    if before is None or after is None:
        lines.append("缺少对照。")
        return lines
    categories = sorted(
        {
            item.category or "未分类"
            for report in (before, after)
            for item in report.questions
            if item.recall_at_k is not None
        }
    )
    dropped: list[str] = []
    for name in categories:
        left = _category_recall(before, name)
        right = _category_recall(after, name)
        lines.append(f"- {name}：{_fmt(left)} → {_fmt(right)}")
        if left is not None and right is not None and right < left:
            dropped.append(name)
    if dropped:
        lines.append("下降的分类：" + "、".join(dropped) + "。")
    else:
        lines.append("没有分类的平均 Recall 下降。")
    return lines


def _gates(reports: list[EvaluationReport]) -> list[str]:
    lines = [
        "",
        "## 质量门",
        "",
        "配置里的质量门没填时，状态是 NOT CONFIGURED，不能当成已经达标。",
    ]
    for item in reports:
        lines.append(f"- {item.experiment_id or '-'} 配置结果：{item.quality_gate_status}")
        recall = item.recall_at_k if item.top_k == 5 else None
        example = quality_gate(
            {
                "recall_at_5": recall,
                "citation_validity": item.citation_validity,
                "groundedness": _raw_groundedness(item),
                "p95_latency_ms": item.p95_latency_ms,
            },
            list(_EXAMPLE_RULES),
        )
        failed = "、".join(
            f"{check.metric} {check.reason}" for check in example.checks if check.status == "failed"
        )
        lines.append(f"  示例门槛（不是已保存的配置）：{example.status}。{failed}")
    return lines


def _recall_changes(
    before: EvaluationReport,
    after: EvaluationReport,
) -> tuple[list[str], list[str]]:
    current = {item.id: item for item in after.questions}
    improved: list[str] = []
    declined: list[str] = []
    for item in before.questions:
        other = current.get(item.id)
        if other is None or item.recall_at_k is None or other.recall_at_k is None:
            continue
        if other.recall_at_k > item.recall_at_k:
            improved.append(item.id)
        elif other.recall_at_k < item.recall_at_k:
            declined.append(item.id)
    return improved, declined


def _recall_miss(item: QuestionResult) -> bool:
    return item.recall_at_k is not None and item.recall_at_k < 1


def _has_type(item: QuestionResult, failure_type: str) -> bool:
    return any(record.failure_type == failure_type for record in item.failure_records)


def _ids(values: list[str]) -> str:
    return "、".join(values)


def _category_recall(report: EvaluationReport, category: str) -> float | None:
    values = [
        item.recall_at_k
        for item in report.questions
        if (item.category or "未分类") == category and item.recall_at_k is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _raw_groundedness(report: EvaluationReport) -> float | None:
    values = [
        item.judge_groundedness
        for item in report.questions
        if isinstance(item.judge_groundedness, int) and not isinstance(
            item.judge_groundedness, bool
        )
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} ms"


if __name__ == "__main__":
    raise SystemExit(main())
