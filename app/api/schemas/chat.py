"""问答请求和响应。

向量检索的 score 是余弦距离，越小越近。
BM25 的 score 是关键词相关度，越大越相关。
不传 retrieval_mode 时仍使用向量检索。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.retrieval.models import RerankerName, RetrieverName, ScoreKind

RetrievalMode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)
    retrieval_mode: RetrievalMode | None = None
    debug: bool = False


class CitationResponse(BaseModel):
    citation_id: str
    document_id: str
    filename: str
    chunk_id: str
    text: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float = Field(description="原始分数。distance 越小越近；bm25、rrf 和 rerank 越大越靠前")
    score_kind: ScoreKind = "distance"
    retriever: RetrieverName = "vector"
    reranker: RerankerName | None = None


class ChatMetricsResponse(BaseModel):
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


class DebugStageResponse(BaseModel):
    stage: Literal["vector", "bm25", "rrf", "reranker"]
    hits: list[RetrievedChunkResponse]


class SpanResponse(BaseModel):
    name: str
    duration_ms: float
    status: Literal["success", "error", "warning"]
    input_summary: str = ""
    output_summary: str = ""


class RetrievalDebugResponse(BaseModel):
    stages: list[DebugStageResponse]
    trace_status: Literal["success", "error", "warning"] | None = None
    spans: list[SpanResponse] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]
    metrics: ChatMetricsResponse
    debug: RetrievalDebugResponse | None = None
