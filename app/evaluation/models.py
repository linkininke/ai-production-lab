"""一次评测的报告。配置和指标写在一起，方便以后比较不同 Top-K 或模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.metrics import ABSTENTION_METHOD
from app.observability.failure import FailureRecord
from app.observability.records import ErrorCategory, ErrorStage
from app.retrieval.models import ScoreKind


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
    score: float = Field(description="原始分数。distance 越小越近；bm25、rrf 和 rerank 越大越靠前")
    score_kind: ScoreKind = "distance"


class TraceHitView(BaseModel):
    """检索阶段的一条候选。只保留编号和分数，不保留片段正文。"""

    model_config = ConfigDict(extra="forbid")

    rank: int
    chunk_id: str
    document_id: str
    score: float
    score_kind: str


class TraceStageView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    hits: list[TraceHitView] = Field(default_factory=list)


class TraceSpanView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    duration_ms: float
    status: str
    output_summary: str = ""


class CaseTrace(BaseModel):
    """一题的请求轨迹。不保存问题原文、提示词和资料正文。"""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = ""
    status: str = ""
    retrieval_mode: str = ""
    question_length: int | None = None
    spans: list[TraceSpanView] = Field(default_factory=list)
    stages: list[TraceStageView] = Field(default_factory=list)
    context_citation_ids: list[str] = Field(default_factory=list)


class NamedCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int = Field(ge=0)


class QuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    category: str | None = None
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
    precision_at_k: float | None = None
    invalid_citations: list[str] = Field(default_factory=list)
    keyword_coverage: float | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    citation_validity: float | None = None
    citation_coverage: float | None = None
    answer_completeness: float | None = None
    matched_answer_points: list[str] = Field(default_factory=list)
    abstained: bool | None = None
    abstention_phrase: str | None = None
    abstention_correct: bool | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    failure_records: list[FailureRecord] = Field(default_factory=list)
    judge_correctness: int | None = None
    judge_groundedness: int | None = None
    judge_completeness: int | None = None
    judge_overall: int | None = None
    judge_score: float | None = None
    judge_reasoning: str = ""
    judge_source: Literal["mock", "llm"] | None = None
    judge_latency_ms: float | None = None
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    judge_cost: float | None = None
    trace: CaseTrace | None = None


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
    retrieval_mode: str = ""
    prompt_version: str = ""
    experiment_id: str = ""
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_max_distance: float | None = None
    corpus_documents: list[CorpusDocument]
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr_at_k: float | None = None
    invalid_citation_count: int = 0
    keyword_coverage: float | None = None
    citation_validity: float | None = None
    citation_coverage: float | None = None
    answer_completeness: float | None = None
    abstention_rate: float | None = None
    abstention_quality: float | None = None
    abstention_method: str = ABSTENTION_METHOD
    abstention_note: str = (
        "拒答检查只匹配事先写明的短语。引用覆盖率只统计已经检索到的相关文档。"
        "答案完整性只统计已填写的答案要点，没有要点时为 null。这些都不是答案正确率。"
    )
    judge_correctness: float | None = None
    judge_groundedness: float | None = None
    judge_completeness: float | None = None
    judge_overall: float | None = None
    judge_source: Literal["mock", "llm"] = "mock"
    judge_note: str = "MockJudge 不是真实评测。LLM Judge 也不是标准答案。"
    mean_judge_latency_ms: float | None = None
    mean_judge_prompt_tokens: float | None = None
    mean_judge_completion_tokens: float | None = None
    judge_cost: float | None = None
    success_rate: float | None = None
    failure_rate: float | None = None
    timeout_rate: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    quality_gate_status: str = "QUALITY GATE NOT CONFIGURED"
    slo_status: str = "SLO NOT CONFIGURED"
    budget_note: str = ""
    mean_retrieval_latency_ms: float | None = None
    mean_generation_latency_ms: float | None = None
    mean_total_latency_ms: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    success_count: int
    failure_count: int
    failure_by_type: list[NamedCount] = Field(default_factory=list)
    failure_by_severity: list[NamedCount] = Field(default_factory=list)
    failure_by_stage: list[NamedCount] = Field(default_factory=list)
    failure_by_category: list[NamedCount] = Field(default_factory=list)
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
