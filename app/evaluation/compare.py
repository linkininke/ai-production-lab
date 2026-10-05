"""比较两份评测报告的配置和指标。差值是右侧减去左侧。"""

from __future__ import annotations

from app.evaluation.models import EvaluationReport, MetricComparison, ReportComparison

_METRICS: tuple[tuple[str, str], ...] = (
    ("recall_at_k", "Recall@K"),
    ("mrr_at_k", "MRR@K"),
    ("keyword_coverage", "关键词覆盖率"),
    ("abstention_rate", "拒答率"),
    ("mean_retrieval_latency_ms", "平均检索耗时"),
    ("mean_generation_latency_ms", "平均生成耗时"),
    ("mean_total_latency_ms", "平均总耗时"),
    ("mean_prompt_tokens", "平均输入 Token"),
    ("mean_completion_tokens", "平均输出 Token"),
)

_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
    ("dataset_version", "数据集版本"),
    ("top_k", "Top-K"),
    ("embedding_model", "Embedding 模型"),
    ("llm_model", "LLM 模型"),
    ("collection_name", "集合"),
    ("chunk_size", "切分长度"),
    ("chunk_overlap", "重叠长度"),
    ("retrieval_max_distance", "距离上限"),
)


def compare_reports(left: EvaluationReport, right: EvaluationReport) -> ReportComparison:
    return ReportComparison(
        left=left.filename,
        right=right.filename,
        config_differences=_config_differences(left, right),
        metrics=[_metric(name, left, right) for name, _label in _METRICS],
    )


def metric_labels() -> dict[str, str]:
    return dict(_METRICS)


def _metric(name: str, left: EvaluationReport, right: EvaluationReport) -> MetricComparison:
    before = getattr(left, name)
    after = getattr(right, name)
    delta = None
    if isinstance(before, int | float) and isinstance(after, int | float):
        delta = float(after) - float(before)
    return MetricComparison(name=name, left=before, right=after, delta=delta)


def _config_differences(left: EvaluationReport, right: EvaluationReport) -> list[str]:
    changes: list[str] = []
    for field_name, label in _CONFIG_FIELDS:
        before = getattr(left, field_name)
        after = getattr(right, field_name)
        if before != after:
            changes.append(f"{label}：{before} → {after}")
    return changes
