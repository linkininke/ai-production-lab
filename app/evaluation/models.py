"""一次评测的报告。配置和指标写在一起，方便以后比较不同 Top-K 或模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.metrics import ABSTENTION_METHOD
from app.observability.records import ErrorCategory, ErrorStage


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    document_id: str
    status: Literal["completed", "skipped"]


class EvaluationHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    document_id: str
    filename: str
    chunk_id: str
    text: str
    score: float = Field(description="余弦距离，越小越近")
    score_kind: Literal["distance"] = "distance"


class QuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expects_abstention: bool
    status: Literal["success", "error"]
    failure_stage: ErrorStage | None = None
    error_category: ErrorCategory | None = None
    error_type: str | None = None
    error_message: str | None = None
    answer: str | None = None
    retrieved: list[EvaluationHit] = Field(default_factory=list)
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    keyword_coverage: float | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    abstained: bool | None = None
    abstention_phrase: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class FailureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    failure_stage: ErrorStage
    error_category: ErrorCategory
    error_type: str
    error_message: str


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    created_at: str
    dataset_version: str
    dataset_file: str
    question_count: int
    top_k: int
    embedding_model: str
    llm_model: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_max_distance: float | None = None
    corpus_documents: list[CorpusDocument]
    recall_at_k: float | None = None
    mrr_at_k: float | None = None
    keyword_coverage: float | None = None
    abstention_rate: float | None = None
    abstention_method: str = ABSTENTION_METHOD
    abstention_note: str = "短语规则只判断回答有没有声明缺少依据，不能代替人工检查。"
    mean_retrieval_latency_ms: float | None = None
    mean_generation_latency_ms: float | None = None
    mean_total_latency_ms: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    success_count: int
    failure_count: int
    failures: list[FailureItem]
    questions: list[QuestionResult]


class MetricComparison(BaseModel):
    name: str
    left: float | None = None
    right: float | None = None
    delta: float | None = None


class ReportComparison(BaseModel):
    left: str
    right: str
    config_differences: list[str]
    metrics: list[MetricComparison]


class ReportSummary(BaseModel):
    filename: str
    created_at: str
    dataset_version: str
    top_k: int
    embedding_model: str
    llm_model: str
    question_count: int
    recall_at_k: float | None = None
    mrr_at_k: float | None = None
    keyword_coverage: float | None = None
    abstention_rate: float | None = None
    failure_count: int
