"""评测仪表盘。只展示 API 已经返回的数字，缺的显示为空。"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

RequestJson = Callable[..., tuple[object, str | None]]

_STAGE_LABELS = {
    "vector": "Vector Results",
    "bm25": "BM25 Results",
    "rrf": "RRF Results",
    "reranker": "Reranker Results",
}
_RATIO_NAMES = {
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "keyword_coverage",
    "citation_validity",
    "citation_coverage",
    "answer_completeness",
    "abstention_rate",
    "abstention_quality",
    "judge_correctness",
    "judge_groundedness",
    "judge_completeness",
    "judge_overall",
    "success_rate",
    "failure_rate",
    "timeout_rate",
}


def render_dashboard(api_base: str, request_json: RequestJson) -> None:
    st.subheader("Evaluation Dashboard")
    st.caption("数字来自已保存的回答评测。没有实测的项显示为 -，不写成 0。")
    names = _report_names(api_base, request_json)
    if not names:
        st.info("还没有评测报告。先在「评测」页运行一次。")
    else:
        selected = st.selectbox("选择一次评测", names, key="dashboard-report")
        report, error = request_json("GET", f"{api_base}/api/v1/evaluation/reports/{selected}")
        if error is not None:
            st.error(error)
        elif isinstance(report, dict):
            _render_evaluation(report)
            _render_failures(report)
            _render_trace(report)
    st.subheader("Experiment Comparison")
    _render_retrieval_experiment(api_base, request_json)


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


def _render_evaluation(report: dict[str, object]) -> None:
    top_k = report.get("top_k")
    k_text = str(top_k) if isinstance(top_k, int) else "K"
    st.caption(
        f"{report.get('question_count', 0)} 道题 · Top-K {k_text} · "
        f"{report.get('judge_note', '')}"
    )
    if top_k != 5:
        st.caption(f"这次 Top-K 是 {k_text}，下面的检索指标不是 @5。")
    experiment = str(report.get("experiment_id") or "")
    mode = str(report.get("retrieval_mode") or "")
    prompt = str(report.get("prompt_version") or "")
    if experiment or mode or prompt:
        st.caption(f"实验 {experiment or '-'} · 检索 {mode or '-'} · 提示词 {prompt or '-'}")
    recall, rank, precision, abstention = st.columns(4)
    recall.metric(f"Recall@{k_text}", _ratio(report.get("recall_at_k")))
    rank.metric(f"MRR@{k_text}", _ratio(report.get("mrr_at_k")))
    precision.metric(f"Precision@{k_text}", _ratio(report.get("precision_at_k")))
    abstention.metric("拒答判断", _ratio(report.get("abstention_quality")))
    correct, grounded, complete, citation = st.columns(4)
    correct.metric("Correctness", _ratio(report.get("judge_correctness")))
    grounded.metric("Groundedness", _ratio(report.get("judge_groundedness")))
    complete.metric("Completeness", _ratio(report.get("judge_completeness")))
    citation.metric("引用有效率", _ratio(report.get("citation_validity")))
    if report.get("judge_source") == "mock":
        st.caption("Judge 分数为空。MockJudge 不是真实评测。")
    p50, p95, p99, average = st.columns(4)
    p50.metric("P50", _milliseconds(report.get("p50_latency_ms")))
    p95.metric("P95", _milliseconds(report.get("p95_latency_ms")))
    p99.metric("P99", _milliseconds(report.get("p99_latency_ms")))
    average.metric("平均耗时", _milliseconds(report.get("mean_total_latency_ms")))
    st.caption("P95、P99 用每题总耗时计算。平均值不能代表尾部延迟。")
    cost = report.get("judge_cost")
    st.metric("Cost", _cost(cost))
    if cost is None:
        st.caption("成本为空。没有单价，或这次没有可计费的 Judge 调用。空不是 0 美元。")
    _status_line("质量门", report.get("quality_gate_status"))
    _status_line("SLO", report.get("slo_status"))
    note = report.get("budget_note")
    if isinstance(note, str) and note:
        st.caption(note)
    rates = (
        f"成功率 {_ratio(report.get('success_rate'))} · "
        f"失败率 {_ratio(report.get('failure_rate'))} · "
        f"超时率 {_ratio(report.get('timeout_rate'))}"
    )
    st.caption(rates)


def _render_failures(report: dict[str, object]) -> None:
    st.subheader("Failure Dashboard")
    st.caption("按失败记录统计类型、严重程度、阶段和题目分类。没有记录的类型不列出来。")
    groups = (
        ("Failure Type", report.get("failure_by_type")),
        ("Severity", report.get("failure_by_severity")),
        ("Stage", report.get("failure_by_stage")),
        ("Category", report.get("failure_by_category")),
    )
    if all(_empty_counts(item[1]) for item in groups):
        st.info("这份报告里没有失败记录。")
        return
    columns = st.columns(4)
    for column, (title, counts) in zip(columns, groups, strict=True):
        column.markdown(f"**{title}**")
        if _empty_counts(counts):
            column.caption("无")
            continue
        assert isinstance(counts, list)
        for item in counts:
            if isinstance(item, dict):
                column.write(f"{item.get('name', '')}  {item.get('count', 0)}")


def _render_trace(report: dict[str, object]) -> None:
    st.subheader("Trace Viewer")
    st.caption("评测 → 题目 → Trace。Trace 只有阶段、编号和分数，不包含提示词和片段正文。")
    questions = report.get("questions")
    if not isinstance(questions, list) or not questions:
        st.caption("这份报告没有逐题结果。")
        return
    labels = [str(item.get("id", "")) for item in questions if isinstance(item, dict)]
    chosen = st.selectbox("题目", labels, key="dashboard-case")
    case = next(
        (item for item in questions if isinstance(item, dict) and item.get("id") == chosen),
        None,
    )
    if not isinstance(case, dict):
        return
    st.markdown("**Question**")
    st.write(str(case.get("question", "")))
    st.caption("题目来自评测集。请求 Trace 只保存了问题长度，没有再存一份原文。")
    trace = case.get("trace")
    if not isinstance(trace, dict):
        st.caption("这道题没有保存 Trace。更早的报告只有答案和检索结果。")
        _render_saved_hits(case)
        _render_case_tail(case)
        return
    st.caption(
        f"trace {trace.get('trace_id', '')} · {trace.get('status', '')} · "
        f"{trace.get('retrieval_mode', '')} · 问题长度 {trace.get('question_length', '-')}"
    )
    st.markdown("**Retrieval**")
    stages = trace.get("stages")
    if isinstance(stages, list) and stages:
        for stage in stages:
            if isinstance(stage, dict):
                _render_stage(stage)
    else:
        st.caption("没有分步检索。未执行的阶段不会显示成空结果。")
    st.markdown("**Context**")
    citation_ids = trace.get("context_citation_ids")
    if isinstance(citation_ids, list) and citation_ids:
        st.caption("进入上下文的引用编号：" + " ".join(str(item) for item in citation_ids))
    else:
        st.caption("没有进入上下文的引用编号。上下文正文不保存。")
    st.markdown("**Prompt**")
    st.caption(f"提示词不保存。输入 Token：{_optional(case.get('prompt_tokens'))}")
    st.markdown("**LLM**")
    st.caption(
        f"生成耗时 {_milliseconds(case.get('generation_latency_ms'))} · "
        f"输出 Token {_optional(case.get('completion_tokens'))}"
    )
    _render_case_tail(case)
    st.markdown("**Spans**")
    spans = trace.get("spans")
    if not isinstance(spans, list) or not spans:
        st.caption("没有阶段耗时。")
        return
    for item in spans:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("output_summary") or "")
        suffix = f" · {summary}" if summary else ""
        st.caption(
            f"{item.get('name', '')} · {_milliseconds(item.get('duration_ms'))} · "
            f"{item.get('status', '')}{suffix}"
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
            score = item.get("score", 0)
            number = f"{float(score):.4f}" if isinstance(score, int | float) else "-"
            st.caption(
                f"#{item.get('rank', '')} {item.get('chunk_id', '')} "
                f"score={number} {item.get('score_kind', '')}"
            )


def _render_saved_hits(case: dict[str, object]) -> None:
    retrieved = case.get("retrieved")
    if not isinstance(retrieved, list) or not retrieved:
        st.caption("没有最终检索结果。")
        return
    st.markdown("**Retrieval**")
    for item in retrieved:
        if isinstance(item, dict):
            st.caption(f"#{item.get('rank', '')} {item.get('chunk_id', '')}")


def _render_case_tail(case: dict[str, object]) -> None:
    st.markdown("**Answer**")
    if case.get("status") == "error":
        st.error(str(case.get("error_message") or "这道题失败了。"))
    else:
        st.markdown(str(case.get("answer") or ""))
    st.markdown("**Citations**")
    invalid = case.get("invalid_citations")
    if isinstance(invalid, list) and invalid:
        st.caption("无效引用：" + " ".join(str(item) for item in invalid))
    else:
        st.caption(f"引用有效率 {_ratio(case.get('citation_validity'))}")
    st.markdown("**Evaluation**")
    st.caption(
        f"Recall {_ratio(case.get('recall_at_k'))} · "
        f"关键词 {_ratio(case.get('keyword_coverage'))} · "
        f"拒答判断 {_yes_no(case.get('abstention_correct'))} · "
        f"Judge {_ratio(case.get('judge_score'))}"
    )
    st.markdown("**Failures**")
    records = case.get("failure_records")
    if not isinstance(records, list) or not records:
        st.caption("这道题没有失败记录。")
        return
    for item in records:
        if not isinstance(item, dict):
            continue
        st.write(
            f"{item.get('failure_type', '')} · {item.get('severity', '')} · {item.get('stage', '')}"
        )
        st.caption(str(item.get("description") or ""))
        st.caption(str(item.get("suggested_action") or ""))


def _render_retrieval_experiment(api_base: str, request_json: RequestJson) -> None:
    st.caption("同一次检索实验里的几种检索器。这份文件没有逐题 Trace，也没有 P95。")
    body, error = request_json("GET", f"{api_base}/api/v1/evaluation/retrieval-experiment")
    if error is not None:
        st.info(error)
        return
    if not isinstance(body, dict):
        st.info("检索实验报告无法读取。")
        return
    note = body.get("note")
    if isinstance(note, str) and note:
        st.caption(note)
    if body.get("semantic_embedding") is not True:
        st.caption("这次不是语义向量，不能当成生产检索结果。")
    retrievers = body.get("retrievers")
    if not isinstance(retrievers, list) or not retrievers:
        st.caption("报告里没有检索器结果。")
        return
    columns = st.columns(len(retrievers))
    for column, item in zip(columns, retrievers, strict=True):
        if not isinstance(item, dict):
            continue
        column.metric(str(item.get("retriever", "")), _ratio(item.get("recall_at_5")))
        column.caption("Recall@5")
        column.caption(f"MRR@5 {_ratio(item.get('mrr_at_5'))}")
        column.caption(f"Precision@5 {_ratio(item.get('precision_at_5'))}")
        column.caption(f"平均耗时 {_milliseconds(item.get('average_latency_ms'))}")
        column.caption(f"成本 {_cost(item.get('estimated_cost_usd'))}")
    skipped = body.get("skipped")
    if isinstance(skipped, list) and skipped:
        st.caption("未运行：" + "、".join(str(item) for item in skipped))
    st.caption("成本为空表示没有单价或没有 Token 计数，不是 0。平均耗时不能代表尾部延迟。")


def _status_line(label: str, status: object) -> None:
    text = str(status or "")
    if text.endswith("PASSED"):
        st.success(f"{label}：{text}")
    elif text.endswith("FAILED"):
        st.error(f"{label}：{text}")
    else:
        st.info(f"{label}：{text or '未配置'}")


def _empty_counts(value: object) -> bool:
    return not isinstance(value, list) or not value


def _ratio(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return f"{value * 100:.1f}%"


def _milliseconds(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return f"{value:.0f} ms"


def _cost(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "空"
    return f"${value:.4f}"


def _optional(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return f"{value:.0f}"


def _yes_no(value: object) -> str:
    if value is True:
        return "对"
    if value is False:
        return "错"
    return "-"


def is_ratio_metric(name: str) -> bool:
    return name in _RATIO_NAMES
