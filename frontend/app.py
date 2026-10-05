"""个人知识库页面。

这个进程只调用 API，不直接打开向量库或模型。
API 地址用环境变量 RAG_API_BASE_URL，默认是本机 8000 端口。
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

_DEFAULT_API = "http://127.0.0.1:8000"
_TIMEOUT = httpx.Timeout(120.0, connect=5.0)


def main() -> None:
    st.set_page_config(page_title="AI Production Lab", layout="wide")
    st.title("个人技术知识库")
    st.caption("导入 Markdown 或 TXT，提问时查看引用和本次耗时，也可以运行评测集。")

    api_base = _sidebar_api_base()
    documents_tab, chat_tab, evaluation_tab = st.tabs(["文档", "问答", "评测"])
    with documents_tab:
        _render_documents(api_base)
    with chat_tab:
        _render_chat(api_base)
    with evaluation_tab:
        _render_evaluation(api_base)


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
        submitted = st.form_submit_button("提问", type="primary")
    if submitted:
        result = _show(
            _request(
                "POST",
                f"{api_base}/api/v1/chat",
                json={"question": question, "top_k": int(top_k)},
            )
        )
        if result is not None:
            st.session_state["last_answer"] = result
    result = st.session_state.get("last_answer")
    if not isinstance(result, dict):
        return
    st.markdown(result.get("answer", ""))
    _render_metrics(result.get("metrics", {}))
    _render_citations(result.get("citations", []))
    _render_retrieved(result.get("retrieved_chunks", []))


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
    st.caption("score 是余弦距离，越小越近。")
    if not chunks:
        st.caption("没有检索到片段。")
        return
    for index, item in enumerate(chunks, start=1):
        score = item.get("score", 0)
        title = f"{index}. {item.get('filename', '')} · 距离 {float(score):.4f}"
        with st.expander(title):
            st.text(str(item.get("text", "")))


def _render_evaluation(api_base: str) -> None:
    st.subheader("运行评测")
    st.caption(
        "样例文档会写入当前向量库，其他文档仍会参与检索。"
        "拒答检查是短语规则，不是语义评分。"
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
            f"{_ratio(item.get('keyword_coverage'))} · 拒答 "
            f"{_abstained(item.get('abstained'))}"
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
    "keyword_coverage": "关键词覆盖率",
    "abstention_rate": "拒答率",
    "mean_retrieval_latency_ms": "平均检索耗时",
    "mean_generation_latency_ms": "平均生成耗时",
    "mean_total_latency_ms": "平均总耗时",
    "mean_prompt_tokens": "平均输入 Token",
    "mean_completion_tokens": "平均输出 Token",
}
_RATIO_METRICS = {"recall_at_k", "mrr_at_k", "keyword_coverage", "abstention_rate"}


def _is_ratio(name: str) -> bool:
    return name in _RATIO_METRICS


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
