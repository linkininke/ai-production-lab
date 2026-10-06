"""一道题的检索指标。

问答评测和检索实验都通过这里调用同一套指标函数。
两边仍各自保存报告，不把字段合并进同一个文件。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.core.case import EvaluationCase
from app.evaluation.metrics import mrr_at_k, precision_at_k, recall_at_k


class EvaluationResult(BaseModel):
    """单题检索结果。没有标注的题不计算召回、精确率和 MRR。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    expects_abstention: bool
    status: Literal["success", "error"]
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    reciprocal_rank: float | None = None
    recall_by_cutoff: dict[int, float] = Field(default_factory=dict)
    latency_ms: float | None = None


def measure_retrieval(
    case: EvaluationCase,
    retrieved_ids: list[str],
    relevant_ids: list[str],
    *,
    precision_k: int,
    mrr_k: int,
    cutoffs: tuple[int, ...] = (),
    latency_ms: float | None = None,
) -> EvaluationResult:
    """按给定标注计算这一题的检索指标。调用方决定用文档 ID 还是片段 ID。"""
    if not relevant_ids:
        raise ValueError("计算检索指标时必须有标注")
    return EvaluationResult(
        id=case.id,
        expects_abstention=case.expects_abstention,
        status="success",
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids),
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, precision_k),
        reciprocal_rank=mrr_at_k(retrieved_ids, relevant_ids, mrr_k),
        recall_by_cutoff={
            cutoff: recall_at_k(retrieved_ids[:cutoff], relevant_ids) for cutoff in cutoffs
        },
        latency_ms=latency_ms,
    )
