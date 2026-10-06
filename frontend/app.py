"""个人知识库页面。

这个进程只调用 API，不直接打开向量库或模型。
API 地址用环境变量 RAG_API_BASE_URL，默认是本机 8000 端口。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

_UI_DIR = str(Path(__file__).resolve().parent)
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)
from dashboard import is_ratio_metric, render_dashboard
from system_health import render_system_health

_DEFAULT_API = "http://127.0.0.1:8000"
_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_MODE_LABELS = {
    "vector": "Vector",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + Reranker",
}
_SCORE_NOTES = {
    "distance": "余弦距离，越小越近",
    "bm25": "BM25 分数，越大越相关",
    "rrf": "RRF 分数，越大越靠前",
    "rerank": "重排相关度，越大越靠前",
}
_STAGE_LABELS = {
    "vector": "Vector Results",
    "bm25": "BM25 Results",
    "rrf": "RRF Results",
    "reranker": "Reranker Results",
}


def main() -> None:
    st.set_page_config(page_title="AI Production Lab", layout="wide")
    st.title("个人技术知识库")
    st.caption("导入 Markdown 或 TXT。提问时可切换检索模式，并展开每一步候选。")

    api_base = _sidebar_api_base()
    documents_tab, chat_tab, evaluation_tab, dashboard_tab, health_tab = st.tabs(
        ["文档", "问答", "评测", "仪表盘", "系统健康"]
    )
    with documents_tab:
        _render_documents(api_base)
    with chat_tab:
        _render_chat(api_base)
    with evaluation_tab:
        _render_evaluation(api_base)
    with dashboard_tab:
        render_dashboard(api_base, _request_json)
    with health_tab:
        render_system_health(api_base, _request_json)


def _sidebar_api_base() -> str:
    default = os.environ.get("RAG_API_BASE_URL", _DEFAULT_API).rstrip("/")
    api_base = st.sidebar.text_input("API 地址", value=default).rstrip("/")
    health, error = _request("GET", f"{api_base}/api/v1/health")
    if error is not None:
        st.sidebar.warning(error)
    else:
        status = (health or {}).get("status", "unknown")
        services = (health or {}).get("services", {})
        if isinstance(services, dict):
            store_status = services.get("vector_store", "unknown")
        else:
            store_status = "unknown"
        st.sidebar.success(f"API {status}")
        st.sidebar.caption(f"向量库 {store_status}")
    return api_base


def _render_documents(api_base: str) -> None:
    st.subheader("导入文档")
    st.caption("同名文件会更新同一份资料。文件不会按上传名写到磁盘。")
    uploaded = st.file_uploader("选择 Markdown 或 TXT", type=["md", "markdown", "txt"])
    if uploaded is not None and st.button("导入", type="primary"):
        payload = {"file": (uploaded.name, uploaded.getvalue())}
        result = _show(_request("POST", f"{api_base}/api/v1/documents/upload", files=payload))
        if result is not None:
            st.success(
                f"已导入 {result['filename']}，"
                f"{result['chunk_count']} 个片段，状态 {result['status']}"
            )

    st.subheader("当前文档")
    listing = _show(_request("GET", f"{api_base}/api/v1/documents"))
    if listing is None:
        return
    documents = listing.get("documents", [])
    if not documents:
        st.info("知识库里还没有文档。")
        return
    for item in documents:
        columns = st.columns([3, 3, 2, 1, 1])
        columns[0].write(item.get("filename", ""))
        columns[1].caption(item.get("document_id", ""))
        columns[2].write(f"{item.get('chunk_count', 0)} 个片段")
        columns[3].caption(item.get("created_at", ""))
        if columns[4].button("删除", key=f"delete-{item.get('document_id', '')}"):
            deleted = _show(
                _request(
                    "DELETE",
                    f"{api_base}/api/v1/documents/{item['document_id']}",
                )
            )
            if deleted is not None:
                st.rerun()


def _render_chat(api_base: str) -> None:
    st.subheader("提问")
    with st.form("ask"):
        question = st.text_area("问题", placeholder="例如：Spring 事务为什么会失效？")
        top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5, step=1)
        retrieval_mode = st.selectbox(
            "检索模式",
            options=list(_MODE_LABELS),
            format_func=lambda item: _MODE_LABELS[item],
        )
        debug = st.checkbox("展开检索过程")
        submitted = st.form_submit_button("提问", type="primary")
    if submitted:
        result = _show(
            _request(
                "POST",
                f"{api_base}/api/v1/chat",
                json={
                    "question": question,
                    "top_k": int(top_k),
                    "retrieval_mode": retrieval_mode,
                    "debug": debug,
                },
            )
        )
        if result is not None:
            st.session_state["last_answer"] = result
    result = st.session_state.get("last_answer")
    if not isinstance(result, dict):
        return
    st.markdown(result.get("answer", ""))
    _render_retrieval_details(result.get("metrics", {}))
    _render_metrics(result.get("metrics", {}))
    _render_citations(result.get("citations", []))
    _render_retrieved(result.get("retrieved_chunks", []))
    _render_debug(result.get("debug"))


def _render_retrieval_details(metrics: object) -> None:
    if not isinstance(metrics, dict):
        return
    st.subheader("Retrieval Details")
    mode = str(metrics.get("retrieval_mode") or "vector")
    st.write(f"Mode: {_MODE_LABELS.get(mode, mode)}")
    st.write(f"Candidates: {_candidate_text(metrics)}")
    st.write(f"Final: {metrics.get('final_result_count')}")
    reranker_enabled = metrics.get("reranker_enabled") is True
    reranker_name = metrics.get("reranker_name") or "未调用"
    if reranker_enabled:
        st.write(f"Reranker: {reranker_name}，候选 {metrics.get('reranker_candidate_count')}")
    else:
        st.write("Reranker: 未启用。这次没有调用重排服务。")
    st.caption(_latency_text(metrics))


def _candidate_text(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    labels = (
        ("vector_candidate_count", "Vector"),
        ("bm25_candidate_count", "BM25"),
        ("hybrid_candidate_count", "RRF"),
        ("reranker_candidate_count", "Reranker"),
    )
    for key, label in labels:
        value = metrics.get(key)
        if isinstance(value, int):
            parts.append(f"{label} {value}")
    return " · ".join(parts) if parts else "-"


def _latency_text(metrics: dict[str, object]) -> str:
    parts: list[str] = []
    labels = (
        ("embedding_latency_ms", "Embedding"),
        ("vector_latency_ms", "Vector"),
        ("bm25_latency_ms", "BM25"),
        ("hybrid_latency_ms", "Hybrid"),
        ("rrf_latency_ms", "RRF"),
        ("reranker_latency_ms", "Reranker"),
        ("retrieval_latency_ms", "Retrieval"),
        ("context_build_latency_ms", "Context"),
        ("generation_latency_ms", "LLM"),
        ("total_latency_ms", "Total"),
    )
    for key, label in labels:
        value = metrics.get(key)
        if isinstance(value, int | float):
            parts.append(f"{label} {value:.0f} ms")
    return "Latency: " + (" · ".join(parts) if parts else "-")


def _render_debug(debug: object) -> None:
    if not isinstance(debug, dict):
        return
    spans = debug.get("spans")
    if isinstance(spans, list) and spans:
        st.subheader("请求 Trace")
        status = debug.get("trace_status") or "-"
        st.caption(f"状态 {status}")
        for item in spans:
            if not isinstance(item, dict):
                continue
            duration = item.get("duration_ms", 0)
            milliseconds = f"{float(duration):.1f}" if isinstance(duration, int | float) else "-"
            summary = str(item.get("output_summary") or "")
            suffix = f" · {summary}" if summary else ""
            name = item.get("name", "")
            status_text = item.get("status", "")
            st.caption(f"{name} · {milliseconds} ms · {status_text}{suffix}")
    stages = debug.get("stages")
    if not isinstance(stages, list):
        return
    st.subheader("检索过程")
    if not stages:
        st.caption("这次没有分步结果。")
        return
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("stage", ""))
        hits = stage.get("hits")
        count = len(hits) if isinstance(hits, list) else 0
        with st.expander(f"{_STAGE_LABELS.get(name, name)} · {count}", expanded=False):
            if not isinstance(hits, list) or not hits:
                st.caption("这一步没有命中。")
                continue
            for index, item in enumerate(hits, start=1):
                if isinstance(item, dict):
                    st.write(_hit_line(index, item))


def _hit_line(index: int, item: dict[str, object]) -> str:
    score = item.get("score", 0)
    number = float(score) if isinstance(score, int | float) else 0.0
    return (
        f"#{index} {item.get('chunk_id', '')} "
        f"score={number:.4f} {_SCORE_NOTES.get(str(item.get('score_kind', '')), '')}"
    )


def _render_metrics(metrics: dict[str, object]) -> None:
    retrieval, generation, total = st.columns(3)
    retrieval.metric("检索耗时", _milliseconds(metrics.get("retrieval_latency_ms")))
    generation.metric("生成耗时", _milliseconds(metrics.get("generation_latency_ms")))
    total.metric("总耗时", _milliseconds(metrics.get("total_latency_ms")))
    prompt_tokens = metrics.get("prompt_tokens")
    completion_tokens = metrics.get("completion_tokens")
    st.caption(f"Token：输入 {prompt_tokens}，输出 {completion_tokens}")


def _render_citations(citations: list[dict[str, object]]) -> None:
    st.subheader("引用")
    if not citations:
        st.caption("这次回答没有对应到知识库片段。")
        return
    for item in citations:
        title = f"[{item.get('citation_id', '')}] {item.get('filename', '')}"
        with st.expander(title, expanded=True):
            st.caption(str(item.get("chunk_id", "")))
            st.text(str(item.get("text", "")))


def _render_retrieved(chunks: list[dict[str, object]]) -> None:
    st.subheader("检索结果")
    if not chunks:
        st.caption("没有检索到片段。")
        return
    kind = str(chunks[0].get("score_kind", "distance"))
    st.caption(_SCORE_NOTES.get(kind, ""))
    for index, item in enumerate(chunks, start=1):
        score = item.get("score", 0)
        number = float(score) if isinstance(score, int | float) else 0.0
        title = f"{index}. {item.get('filename', '')} · {number:.4f}"
        with st.expander(title):
            st.caption(str(item.get("chunk_id", "")))
            st.text(str(item.get("text", "")))


def _render_evaluation(api_base: str) -> None:
    st.subheader("运行评测")
    st.caption(
        "样例文档会写入当前向量库，其他文档仍会参与检索。"
        "拒答、引用和答案要点都是规则检查，不是语义评分。"
    )
    with st.form("run-eval"):
        top_k = st.number_input("评测 Top-K", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("运行评测", type="primary")
    if submitted:
        result = _show(
            _request(
                "POST",
                f"{api_base}/api/v1/evaluation/run",
                json={"top_k": int(top_k)},
                timeout=httpx.Timeout(600.0, connect=5.0),
            )
        )
        if result is not None:
            st.session_state["last_evaluation"] = result
    report = st.session_state.get("last_evaluation")
    if isinstance(report, dict):
        _render_evaluation_report(report)
    _render_evaluation_compare(api_base)


def _render_evaluation_report(report: dict[str, object]) -> None:
    st.subheader("本次结果")
    st.caption(
        f"{report.get('question_count', 0)} 道题 · "
        f"{report.get('filename', '')} · "
        f"Top-K {report.get('top_k', '')}"
    )
    recall, rank, keywords, abstention = st.columns(4)
    recall.metric("Recall@K", _ratio(report.get("recall_at_k")))
    rank.metric("MRR@K", _ratio(report.get("mrr_at_k")))
    keywords.metric("关键词覆盖率", _ratio(report.get("keyword_coverage")))
    abstention.metric("拒答率", _ratio(report.get("abstention_rate")))
    validity, coverage, completeness, decision = st.columns(4)
    validity.metric("引用有效率", _ratio(report.get("citation_validity")))
    coverage.metric("引用覆盖率", _ratio(report.get("citation_coverage")))
    completeness.metric("答案完整性", _ratio(report.get("answer_completeness")))
    decision.metric("拒答判断", _ratio(report.get("abstention_quality")))
    judged_correct, judged_grounded, judged_complete, judged_overall = st.columns(4)
    judged_correct.metric("Judge 正确性", _ratio(report.get("judge_correctness")))
    judged_grounded.metric("Judge 有依据", _ratio(report.get("judge_groundedness")))
    judged_complete.metric("Judge 完整性", _ratio(report.get("judge_completeness")))
    judged_overall.metric("Judge 综合", _ratio(report.get("judge_overall")))
    st.caption(str(report.get("judge_note", "")))
    retrieval, generation, total = st.columns(3)
    retrieval.metric("平均检索耗时", _milliseconds(report.get("mean_retrieval_latency_ms")))
    generation.metric("平均生成耗时", _milliseconds(report.get("mean_generation_latency_ms")))
    total.metric("平均总耗时", _milliseconds(report.get("mean_total_latency_ms")))
    st.caption(
        "Token：输入 "
        f"{_optional_number(report.get('mean_prompt_tokens'))}，输出 "
        f"{_optional_number(report.get('mean_completion_tokens'))}"
    )
    st.caption(str(report.get("abstention_note", "")))
    _render_failures(report.get("failures", []))
    questions = report.get("questions", [])
    if not isinstance(questions, list):
        return
    st.subheader("每道题")
    for item in questions:
        if isinstance(item, dict):
            _render_question(item)


def _render_failures(failures: object) -> None:
    st.subheader("失败")
    if not isinstance(failures, list) or not failures:
        st.caption("没有检索、生成或接口异常。指标偏低会显示在每道题上，不算进这个列表。")
        return
    for item in failures:
        if not isinstance(item, dict):
            continue
        st.error(
            f"{item.get('id', '')} · {item.get('failure_stage', '')} · "
            f"{item.get('error_type', '')}：{item.get('error_message', '')}"
        )


def _render_question(item: dict[str, object]) -> None:
    if item.get("status") == "error":
        title = f"{item.get('id', '')} · 失败"
    elif item.get("expects_abstention"):
        title = f"{item.get('id', '')} · 拒答 {_abstained(item.get('abstained'))}"
    else:
        title = f"{item.get('id', '')} · Recall {_ratio(item.get('recall_at_k'))}"
    with st.expander(title):
        st.write(str(item.get("question", "")))
        if item.get("status") == "error":
            st.error(str(item.get("error_message", "")))
            return
        st.markdown(str(item.get("answer", "")))
        st.caption(
            "MRR "
            f"{_ratio(item.get('reciprocal_rank'))} · 关键词 "
            f"{_ratio(item.get('keyword_coverage'))} · 引用有效 "
            f"{_ratio(item.get('citation_validity'))} · 引用覆盖 "
            f"{_ratio(item.get('citation_coverage'))} · 完整性 "
            f"{_ratio(item.get('answer_completeness'))} · 拒答 "
            f"{_abstained(item.get('abstained'))} · Judge "
            f"{_ratio(item.get('judge_score'))}"
        )
        retrieved = item.get("retrieved", [])
        if not isinstance(retrieved, list) or not retrieved:
            st.caption("没有检索到片段。")
            return
        for hit in retrieved:
            if not isinstance(hit, dict):
                continue
            score = hit.get("score", 0)
            distance = f"{float(score):.4f}" if isinstance(score, int | float) else "-"
            st.markdown(f"{hit.get('rank', '')}. {hit.get('filename', '')} · 距离 {distance}")
            st.text(str(hit.get("text", "")))


def _render_evaluation_compare(api_base: str) -> None:
    st.subheader("比较两次评测")
    body, list_error = _request_json("GET", f"{api_base}/api/v1/evaluation/reports")
    if list_error is not None:
        st.caption(list_error)
        return
    if not isinstance(body, list) or len(body) < 2:
        st.caption("至少保存两份报告后才能比较。")
        return
    names = [str(item.get("filename", "")) for item in body if isinstance(item, dict)]
    names = [name for name in names if name]
    if len(names) < 2:
        st.caption("至少保存两份报告后才能比较。")
        return
    with st.form("compare-eval"):
        left = st.selectbox("左侧报告", names, index=1)
        right = st.selectbox("右侧报告", names, index=0)
        submitted = st.form_submit_button("比较")
    if submitted:
        compared = _show(
            _request(
                "POST",
                f"{api_base}/api/v1/evaluation/compare",
                json={"left": left, "right": right},
            )
        )
        if compared is not None:
            st.session_state["evaluation_comparison"] = compared
    comparison = st.session_state.get("evaluation_comparison")
    if not isinstance(comparison, dict):
        return
    differences = comparison.get("config_differences", [])
    if isinstance(differences, list) and differences:
        for item in differences:
            st.caption(str(item))
    else:
        st.caption("两次配置相同。差值是右侧减去左侧。")
    metrics = comparison.get("metrics", [])
    if not isinstance(metrics, list):
        return
    for item in metrics:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        st.metric(
            _METRIC_LABELS.get(name, name),
            _ratio(item.get("right")) if _is_ratio(name) else _plain(item.get("right")),
            delta=_signed(item.get("delta"), ratio=_is_ratio(name)),
        )


_METRIC_LABELS = {
    "recall_at_k": "Recall@K",
    "mrr_at_k": "MRR@K",
    "precision_at_k": "Precision@K",
    "keyword_coverage": "关键词覆盖率",
    "citation_validity": "引用有效率",
    "citation_coverage": "引用覆盖率",
    "answer_completeness": "答案完整性",
    "abstention_rate": "拒答率",
    "abstention_quality": "拒答判断",
    "judge_correctness": "Judge 正确性",
    "judge_groundedness": "Judge 有依据",
    "judge_completeness": "Judge 完整性",
    "judge_overall": "Judge 综合",
    "mean_retrieval_latency_ms": "平均检索耗时",
    "mean_generation_latency_ms": "平均生成耗时",
    "mean_total_latency_ms": "平均总耗时",
    "p50_latency_ms": "P50 耗时",
    "p95_latency_ms": "P95 耗时",
    "p99_latency_ms": "P99 耗时",
    "success_rate": "成功率",
    "failure_rate": "失败率",
    "timeout_rate": "超时率",
    "judge_cost": "Judge 成本",
    "mean_prompt_tokens": "平均输入 Token",
    "mean_completion_tokens": "平均输出 Token",
}


def _is_ratio(name: str) -> bool:
    return is_ratio_metric(name)


def _ratio(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value * 100:.1f}%"
    return "-"


def _plain(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.1f}"
    return "-"


def _signed(value: object, *, ratio: bool) -> str | None:
    if not isinstance(value, int | float):
        return None
    if ratio:
        return f"{value * 100:.1f}%"
    return f"{value:.1f}"


def _abstained(value: object) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "-"


def _optional_number(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.1f}"
    return "-"


def _milliseconds(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.0f} ms"
    return "-"


def _show(result: tuple[dict[str, object] | None, str | None]) -> dict[str, object] | None:
    body, error = result
    if error is not None:
        st.error(error)
        return None
    return body


def _request(
    method: str,
    url: str,
    *,
    json: dict[str, object] | None = None,
    files: dict[str, tuple[str, bytes]] | None = None,
    timeout: httpx.Timeout | None = None,
) -> tuple[dict[str, object] | None, str | None]:
    body, error = _request_json(method, url, json=json, files=files, timeout=timeout)
    if error is not None:
        return None, error
    if not isinstance(body, dict):
        return None, "API 返回了无法识别的内容。"
    return body, None


def _request_json(
    method: str,
    url: str,
    *,
    json: dict[str, object] | None = None,
    files: dict[str, tuple[str, bytes]] | None = None,
    timeout: httpx.Timeout | None = None,
) -> tuple[object, str | None]:
    try:
        response = httpx.request(
            method,
            url,
            json=json,
            files=files,
            timeout=timeout or _TIMEOUT,
        )
    except httpx.HTTPError:
        return None, "无法连接 API。请确认服务已启动，并且地址和端口一致。"
    if response.status_code >= 400:
        return None, _error_message(response)
    try:
        return response.json(), None
    except ValueError:
        return None, "API 返回了无法识别的内容。"


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return "请求失败。"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return "请求失败。"


main()
