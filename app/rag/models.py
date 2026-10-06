"""问答结果。引用只描述真正放进上下文的片段，不由模型编造来源。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.observability.failure import FailureRecord
from app.observability.trace import RequestTrace
from app.retrieval.models import RetrievalResult
from app.retrieval.trace import RetrievalTrace


class Citation(BaseModel):
    citation_id: str
    document_id: str
    filename: str
    chunk_id: str
    text: str


class BuiltContext(BaseModel):
    """格式化后的资料，以及这些资料对应的引用。"""

    text: str
    citations: list[Citation]
    used_results: list[RetrievalResult]


class RAGMetrics(BaseModel):
    retrieval_latency_ms: float
    context_build_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retrieval_mode: str | None = None
    vector_candidate_count: int | None = None
    bm25_candidate_count: int | None = None
    hybrid_candidate_count: int | None = None
    reranker_enabled: bool = False
    reranker_candidate_count: int | None = None
    reranker_name: str | None = None
    final_result_count: int | None = None
    embedding_latency_ms: float | None = None
    vector_latency_ms: float | None = None
    bm25_latency_ms: float | None = None
    hybrid_latency_ms: float | None = None
    rrf_latency_ms: float | None = None
    reranker_latency_ms: float | None = None


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    context_citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    request_trace: RequestTrace | None = None
    failure_records: list[FailureRecord] = Field(default_factory=list)
    metrics: RAGMetrics
