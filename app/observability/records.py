"""一次问答的阶段耗时和错误分类。

这是后续接入监控时的稳定记录，不是某一次 HTTP 响应的副本。
日志里只有长度、数量、耗时、Token 和错误类别，没有问题原文、资料正文和密钥。
超时、上游 HTTP 失败记为接口问题；检索逻辑和空回答仍记在各自阶段。
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.core.exceptions import (
    AppError,
    ConfigurationError,
    ContextOverflowError,
    EmbeddingError,
    LLMError,
    QuestionValidationError,
    RerankerError,
    RetrievalError,
    VectorStoreError,
)
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

ErrorStage = Literal["retrieval", "generation", "interface", "validation"]
ErrorCategory = Literal[
    "timeout",
    "upstream",
    "retrieval",
    "generation",
    "validation",
    "configuration",
    "unexpected",
]


class ObservedSpan(BaseModel):
    """日志里的阶段摘要。只有名称、耗时和状态。"""

    name: str
    duration_ms: float
    status: Literal["success", "error", "warning"]


class QueryObservation(BaseModel):
    """一次问答结束时能确定的指标。失败时尚未发生的阶段留空。"""

    request_id: str = ""
    question_length: int = Field(ge=0)
    retrieval_count: int | None = None
    retrieval_latency_ms: float | None = None
    context_build_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: Literal["success", "error"]
    error_stage: ErrorStage | None = None
    error_type: str | None = None
    error_category: ErrorCategory | None = None
    retrieval_mode: str | None = None
    vector_candidate_count: int | None = None
    bm25_candidate_count: int | None = None
    hybrid_candidate_count: int | None = None
    reranker_enabled: bool | None = None
    reranker_candidate_count: int | None = None
    final_result_count: int | None = None
    embedding_latency_ms: float | None = None
    vector_latency_ms: float | None = None
    bm25_latency_ms: float | None = None
    hybrid_latency_ms: float | None = None
    rrf_latency_ms: float | None = None
    reranker_latency_ms: float | None = None
    trace_id: str | None = None
    trace_status: Literal["success", "error", "warning"] | None = None
    span_count: int | None = None
    spans: list[ObservedSpan] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)


def classify_failure(exc: BaseException) -> tuple[ErrorStage, ErrorCategory]:
    """把异常分成检索、生成或接口问题。类别用来区分超时和上游错误。"""
    if isinstance(exc, QuestionValidationError):
        return "validation", "validation"
    if isinstance(exc, ConfigurationError):
        return "interface", "configuration"
    if isinstance(exc, EmbeddingError | RerankerError):
        category = _transport_category(exc.message)
        if category is not None or "响应" in exc.message:
            return "interface", category or "upstream"
        return "retrieval", "retrieval"
    if isinstance(exc, RetrievalError | VectorStoreError):
        return "retrieval", "retrieval"
    if isinstance(exc, LLMError):
        category = _transport_category(exc.message)
        if category is not None or "响应" in exc.message:
            return "interface", category or "upstream"
        return "generation", "generation"
    if isinstance(exc, ContextOverflowError):
        return "generation", "generation"
    if isinstance(exc, AppError):
        return "interface", "unexpected"
    return "interface", "unexpected"


def log_query_observation(record: QueryObservation) -> None:
    """写入一条结构化日志。调用方不要把问题或正文放进这条记录。"""
    request_id = record.request_id or request_id_var.get()
    payload = record.model_copy(update={"request_id": request_id})
    logger.info("query_observation %s", payload.model_dump_json(exclude_none=True))


def _transport_category(message: str) -> ErrorCategory | None:
    if "超时" in message:
        return "timeout"
    if any(token in message for token in ("网络错误", "服务返回", "请求失败")):
        return "upstream"
    return None
