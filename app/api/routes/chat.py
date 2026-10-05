"""问答接口。检索和生成都在 RAGPipeline 中完成，这里只做参数转发。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.chat import (
    ChatMetricsResponse,
    ChatRequest,
    ChatResponse,
    CitationResponse,
    RetrievedChunkResponse,
)
from app.core.container import AppContainer
from app.core.dependencies import get_container
from app.retrieval.models import RetrievalResult

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def answer_question(
    body: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    result = container.require_rag().query(body.question, top_k=body.top_k)
    metrics = result.metrics
    return ChatResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_id=item.citation_id,
                document_id=item.document_id,
                filename=item.filename,
                chunk_id=item.chunk_id,
                text=item.text,
            )
            for item in result.citations
        ],
        retrieved_chunks=[_chunk_response(item) for item in result.retrieved_chunks],
        metrics=ChatMetricsResponse(
            retrieval_latency_ms=metrics.retrieval_latency_ms,
            context_build_latency_ms=metrics.context_build_latency_ms,
            generation_latency_ms=metrics.generation_latency_ms,
            total_latency_ms=metrics.total_latency_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
        ),
    )


def _chunk_response(hit: RetrievalResult) -> RetrievedChunkResponse:
    filename = hit.metadata.get("filename", "")
    if not isinstance(filename, str):
        filename = str(filename)
    return RetrievedChunkResponse(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=filename,
        text=hit.text,
        score=hit.score,
        score_kind=hit.score_kind,
    )
