"""问答接口。检索和生成都在 RAGPipeline 中完成，这里只做参数转发。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas.chat import (
    ChatMetricsResponse,
    ChatRequest,
    ChatResponse,
    CitationResponse,
    DebugStageResponse,
    RetrievalDebugResponse,
    RetrievedChunkResponse,
    SpanResponse,
)
from app.core.container import AppContainer
from app.core.dependencies import get_container
from app.observability.trace import RequestTrace
from app.rag.models import RAGMetrics
from app.retrieval.models import RetrievalResult
from app.retrieval.trace import RetrievalTrace

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def answer_question(
    body: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    result = container.require_rag(body.retrieval_mode or "vector").query(
        body.question,
        top_k=body.top_k,
    )
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
        metrics=_metrics_response(result.metrics),
        debug=_debug_response(result.retrieval_trace, result.request_trace) if body.debug else None,
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
        retriever=hit.retriever,
        reranker=hit.reranker,
    )


def _metrics_response(metrics: RAGMetrics) -> ChatMetricsResponse:
    return ChatMetricsResponse.model_validate(metrics.model_dump())


def _debug_response(
    trace: RetrievalTrace | None,
    request_trace: RequestTrace | None,
) -> RetrievalDebugResponse:
    if trace is None:
        return RetrievalDebugResponse(stages=[])
    stages: list[DebugStageResponse] = []
    named = (
        ("vector", trace.vector_hits),
        ("bm25", trace.bm25_hits),
        ("rrf", trace.rrf_hits),
        ("reranker", trace.reranker_hits),
    )
    for stage, hits in named:
        if hits is None:
            continue
        stages.append(
            DebugStageResponse(
                stage=stage,
                hits=[_chunk_response(item) for item in hits],
            )
        )
    return RetrievalDebugResponse(
        stages=stages,
        trace_status=None if request_trace is None else request_trace.status,
        spans=[
            SpanResponse(
                name=item.name,
                duration_ms=item.duration_ms,
                status=item.status,
                input_summary=item.input_summary,
                output_summary=item.output_summary,
            )
            for item in ([] if request_trace is None else request_trace.spans)
        ],
    )
