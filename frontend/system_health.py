"""系统健康页。只展示健康卡 API 返回的数字，缺的显示 N/A。"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlencode

import streamlit as st

RequestJson = Callable[..., tuple[object, str | None]]

_GROUPS = (
    ("answer", "Answer Quality"),
    ("retrieval", "Retrieval Quality"),
    ("citation", "Citation / Safety"),
    ("abstention", "Abstention"),
    ("reliability", "Reliability"),
    ("performance", "Performance"),
    ("cost", "Cost"),
)
_CATEGORY_METRIC = {
    "recall_at_5": "recall_at_k",
    "recall_at_k": "recall_at_k",
    "judge_correctness": "judge_correctness",
    "abstention_quality": "abstention_quality",
}
_STAGE_LABELS = {
    "vector": "Vector Results",
    "bm25": "BM25 Results",
    "rrf": "RRF Results",
    "reranker": "Reranker Results",
}


def render_system_health(api_base: str, request_json: RequestJson) -> None:
    st.subheader("System Health")
    st.caption("数字来自已保存的评测、质量门、回归和检索实验。没有实测的项是 N/A，不是 0。")
    names = _report_names(api_base, request_json)
    if not names:
        st.info("还没有评测报告。先在「评测」页运行一次。")
        return
    selected = st.selectbox("当前评测", names, key="health-report")
    baseline_choice = st.selectbox("基线评测", ["不对比", *names], key="health-baseline")
    baseline = "" if baseline_choice in {"不对比", selected} else baseline_choice
    if baseline_choice == selected:
        st.caption("基线和当前是同一份报告，不对比。")
    card = _load_card(api_base, request_json, selected, baseline)
    if card is None:
        return
    _render_banner(card)
    _render_groups(card)
    _render_categories(card)
    _render_experiments(card)
    _render_cases(card)


def _report_names(api_base: str, request_json: RequestJson) -> list[str]:
    body, error = request_json("GET", f"{api_base}/api/v1/evaluation/reports")
    if error is not None:
        st.caption(error)
        return []
    if not isinstance(body, list):
        return []
    return [
        str(item.get("filename", ""))
        for item in body
        if isinstance(item, dict) and item.get("filename")
    ]


def _load_card(
    api_base: str,
    request_json: RequestJson,
    report: str,
    baseline: str,
) -> dict[str, object] | None:
    query = {"report": report}
    if baseline:
        query["baseline"] = baseline
    url = f"{api_base}/api/v1/evaluation/health?{urlencode(query)}"
    body, error = request_json("GET", url)
    if error is not None:
        st.error(error)
        return None
    if not isinstance(body, dict):
        st.error("健康卡无法读取。")
        return None
    return body


def _render_banner(card: dict[str, object]) -> None:
    status = str(card.get("status") or "")
    title = f"Overall Status: {status or 'N/A'}"
    if status == "PASS":
        st.success(title)
    elif status == "FAIL":
        st.error(title)
    else:
        st.warning(title)
    gate = f"Quality Gate: {card.get('quality_gate', 'N/A')}"
    regression = f"Regression: {card.get('regression', 'N/A')}"
    left, right = st.columns(2)
    _status_box(left, str(card.get("quality_gate") or ""), gate)
    _status_box(right, str(card.get("regression") or ""), regression)
    if card.get("regression") == "FAIL":
        failed = card.get("regression_failed_cases")
        total = card.get("regression_case_total")
        checks = card.get("regression_failed_checks")
        if isinstance(failed, int) and failed > 0 and isinstance(total, int):
            st.error(f"Failed Cases: {failed} / {total}")
        elif isinstance(checks, int) and checks > 0:
            st.error(f"未通过的回归检查：{checks} 项")
    meta = (
        f"实验 {card.get('experiment_id') or '-'} · "
        f"检索 {card.get('retrieval_mode') or '-'} · "
        f"提示词 {card.get('prompt_version') or '-'}"
    )
    if card.get("baseline_filename"):
        meta += f" · 基线 {card.get('baseline_filename')}"
    st.caption(meta)
    note = card.get("assessment_note")
    if isinstance(note, str) and note:
        st.caption(note)
    reasons = card.get("reasons")
    if isinstance(reasons, list) and reasons:
        for item in reasons:
            st.caption(str(item))
    elif status == "PASS":
        st.caption("质量门和回归都已通过，没有触发已配置的警告。")


def _render_groups(card: dict[str, object]) -> None:
    metrics = _metric_list(card)
    for group, title in _GROUPS:
        rows = [item for item in metrics if item.get("group") == group]
        if group == "citation":
            grounded = _metric_named(metrics, "judge_groundedness")
            if grounded is not None and grounded not in rows:
                rows = [grounded, *rows]
        st.markdown(f"**{title}**")
        if not rows:
            st.caption("N/A")
            continue
        columns = st.columns(min(len(rows), 3))
        for index, item in enumerate(rows):
            column = columns[index % len(columns)]
            comparison = item.get("comparison")
            delta = ""
            baseline = None
            if isinstance(comparison, dict):
                delta = str(comparison.get("delta_text") or "")
                baseline = comparison.get("baseline")
            worse = item.get("higher_is_better") is False
            column.metric(
                str(item.get("label") or item.get("name")),
                _value(item),
                delta=delta or None,
                delta_color="inverse" if worse and delta else "off" if not delta else "normal",
            )
            if baseline is not None:
                column.caption(f"Baseline {_format_number(baseline, str(item.get('unit') or ''))}")
            note = item.get("note")
            if isinstance(note, str) and note:
                column.caption(note)
        if group == "performance":
            st.caption("P95、P99 来自报告里的分位。平均值不能代替尾部延迟。")
        if group == "cost":
            st.caption("成本为空时显示 N/A，不是 0 美元。")


def _render_categories(card: dict[str, object]) -> None:
    st.markdown("**Category**")
    st.caption("分类均值来自逐题里已经算好的分数。总览数字仍用报告上的汇总值。空值不参加平均。")
    categories = card.get("categories")
    if not isinstance(categories, list) or not categories:
        st.caption("这份报告没有可按类型汇总的逐题分数。")
        return
    metrics = _metric_list(card)
    labels = [str(item.get("label") or item.get("name")) for item in metrics]
    chosen = st.selectbox("查看哪个指标的分类", labels, key="health-category-metric")
    metric = next(
        (item for item in metrics if str(item.get("label") or item.get("name")) == chosen),
        None,
    )
    metric_name = "" if metric is None else str(metric.get("name") or "")
    category_metric = _CATEGORY_METRIC.get(metric_name, "")
    rows = [
        item
        for item in categories
        if isinstance(item, dict) and item.get("metric") == category_metric
    ]
    if not rows:
        st.caption("这个指标没有分类明细。")
        return
    for item in rows:
        value = item.get("value")
        shown = "N/A" if not isinstance(value, int | float) else f"{float(value) * 100:.1f}%"
        st.write(f"{item.get('category', '')}  {shown}  （{item.get('sample_count', 0)} 题）")


def _render_experiments(card: dict[str, object]) -> None:
    st.markdown("**Current vs Baseline · Retrieval Experiments**")
    note = card.get("experiment_note")
    if isinstance(note, str) and note:
        st.caption(note)
    rows = card.get("experiments")
    if not isinstance(rows, list) or not rows:
        st.caption("没有可对比的检索实验。")
        return
    columns = st.columns(len(rows))
    for column, item in zip(columns, rows, strict=True):
        if not isinstance(item, dict):
            continue
        column.metric(str(item.get("retriever") or ""), _ratio(item.get("recall_at_5")))
        column.caption("Recall@5")
        column.caption(f"Recall@1 {_ratio(item.get('recall_at_1'))}")
        column.caption(f"MRR@5 {_ratio(item.get('mrr_at_5'))}")
        column.caption(f"Precision@5 {_ratio(item.get('precision_at_5'))}")
        column.caption(f"平均耗时 {_milliseconds(item.get('average_latency_ms'))}")
        column.caption(f"成本 {_cost(item.get('estimated_cost_usd'))}")
    skipped = card.get("skipped_retrievers")
    if isinstance(skipped, list) and skipped:
        st.caption("未运行：" + "、".join(str(item) for item in skipped))


def _render_cases(card: dict[str, object]) -> None:
    st.markdown("**Failed Cases**")
    st.caption("从失败题可以看到 Recall、失败类型和 Trace。Trace 不含片段正文。")
    cases = card.get("failed_cases")
    if not isinstance(cases, list) or not cases:
        st.info("这份报告里没有失败题。")
        return
    labels = [str(item.get("id", "")) for item in cases if isinstance(item, dict)]
    chosen = st.selectbox("失败题", labels, key="health-case")
    case = next(
        (item for item in cases if isinstance(item, dict) and item.get("id") == chosen),
        None,
    )
    if not isinstance(case, dict):
        return
    st.markdown("**Question**")
    st.write(str(case.get("question") or ""))
    st.caption(
        f"{case.get('category') or '未分类'} · {case.get('status') or ''} · "
        f"Recall {_ratio(case.get('recall_at_k'))} · "
        f"MRR {_ratio(case.get('reciprocal_rank'))} · "
        f"引用 {_ratio(case.get('citation_validity'))} · "
        f"拒答 {_yes_no(case.get('abstention_correct'))} · "
        f"Correctness {_ratio(case.get('judge_correctness'))}"
    )
    types = case.get("failure_types")
    if isinstance(types, list) and types:
        st.write("失败类型：" + "、".join(str(item) for item in types))
    else:
        st.caption("这道题没有失败类型记录。")
    trace = case.get("trace")
    if not isinstance(trace, dict):
        st.caption("这道题没有保存 Trace。")
        return
    st.markdown("**Trace**")
    st.caption(
        f"trace {trace.get('trace_id', '')} · {trace.get('status', '')} · "
        f"{trace.get('retrieval_mode', '')}"
    )
    stages = trace.get("stages")
    if isinstance(stages, list) and stages:
        for stage in stages:
            if isinstance(stage, dict):
                _render_stage(stage)
    else:
        st.caption("没有分步检索。")
    spans = trace.get("spans")
    if isinstance(spans, list) and spans:
        for item in spans:
            if not isinstance(item, dict):
                continue
            st.caption(
                f"{item.get('name', '')} · {_milliseconds(item.get('duration_ms'))} · "
                f"{item.get('status', '')}"
            )


def _render_stage(stage: dict[str, object]) -> None:
    name = str(stage.get("stage", ""))
    hits = stage.get("hits")
    count = len(hits) if isinstance(hits, list) else 0
    with st.expander(f"{_STAGE_LABELS.get(name, name)} · {count}", expanded=False):
        if not isinstance(hits, list) or not hits:
            st.caption("这一步跑过，但没有命中。")
            return
        for item in hits:
            if not isinstance(item, dict):
                continue
            st.caption(
                f"#{item.get('rank', '')} {item.get('chunk_id', '')} "
                f"{item.get('document_id', '')}"
            )


def _status_box(column: object, status: str, text: str) -> None:
    if status in {"PASS", "PASSED"} or status.endswith("PASSED"):
        column.success(text)  # type: ignore[attr-defined]
    elif status in {"FAIL", "FAILED"} or status.endswith("FAILED"):
        column.error(text)  # type: ignore[attr-defined]
    else:
        column.info(text)  # type: ignore[attr-defined]


def _metric_list(card: dict[str, object]) -> list[dict[str, object]]:
    metrics = card.get("metrics")
    if not isinstance(metrics, list):
        return []
    return [item for item in metrics if isinstance(item, dict)]


def _metric_named(metrics: list[dict[str, object]], name: str) -> dict[str, object] | None:
    return next((item for item in metrics if item.get("name") == name), None)


def _value(metric: dict[str, object]) -> str:
    comparison = metric.get("comparison")
    if not isinstance(comparison, dict):
        return "N/A"
    return _format_number(comparison.get("current"), str(metric.get("unit") or ""))


def _format_number(value: object, unit: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "N/A"
    if unit == "ratio":
        return f"{float(value) * 100:.1f}%"
    if unit == "ms":
        return f"{float(value):.0f} ms"
    if unit == "usd":
        return f"${float(value):.4f}"
    return f"{float(value):.4f}"


def _ratio(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _milliseconds(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "N/A"
    return f"{float(value):.0f} ms"


def _cost(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "N/A"
    return f"${float(value):.4f}"


def _yes_no(value: object) -> str:
    if value is True:
        return "对"
    if value is False:
        return "错"
    return "N/A"
