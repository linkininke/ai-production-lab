"""问答请求和响应。retrieved_chunks 里的 score 是余弦距离。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)


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
    score: float = Field(description="余弦距离，越小越近")
    score_kind: Literal["distance"] = "distance"


class ChatMetricsResponse(BaseModel):
    retrieval_latency_ms: float
    context_build_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]
    metrics: ChatMetricsResponse
