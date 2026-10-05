"""问答结果。引用只描述真正放进上下文的片段，不由模型编造来源。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.retrieval.models import RetrievalResult


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


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    metrics: RAGMetrics
