"""把检索、上下文、提示词、生成和引用串成一次问答。

检索失败和生成失败保持各自的异常，不包成同一种错误。
没有检索结果时仍然调用模型，由提示词要求它说明知识库缺少依据。
日志只记长度、数量和耗时，不记问题原文、资料正文和完整提示词。
"""

from __future__ import annotations

from contextvars import Token
from time import perf_counter
from typing import Literal

from app.core.exceptions import AppError, QuestionValidationError
from app.llm.base import LLMProvider
from app.observability.failure import FailureRecord, FailureSignals, classify_failures
from app.observability.records import (
    ObservedSpan,
    QueryObservation,
    classify_failure,
    log_query_observation,
)
from app.observability.trace import (
    RequestTrace,
    begin_request_trace,
    bind_error_trace,
    current_request_trace,
    end_request_trace,
    record_retrieval_spans,
)
from app.rag.citation import audit_citations, map_citations
from app.rag.context_builder import ContextBuilder
from app.rag.models import RAGMetrics, RAGResponse
from app.rag.prompt import PromptBuilder
from app.retrieval.base import Retriever
from app.retrieval.models import RetrievalResult
from app.retrieval.trace import RetrievalTrace, begin_retrieval_trace, end_retrieval_trace


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
        default_top_k: int,
        max_top_k: int,
        max_question_chars: int,
    ) -> None:
        if default_top_k < 1 or default_top_k > max_top_k:
            raise QuestionValidationError("DEFAULT_TOP_K 超出 MAX_TOP_K 的范围")
        if max_question_chars < 1:
            raise QuestionValidationError("MAX_QUESTION_CHARS 必须大于 0")
        self._retriever = retriever
        self._context = context_builder
        self._prompts = prompt_builder
        self._llm = llm
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k
        self._max_question_chars = max_question_chars

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        cleaned = question.strip()
        selected = self._default_top_k if top_k is None else top_k
        mode = str(getattr(self._retriever, "name", "vector"))
        request_token = begin_request_trace(retrieval_mode=mode, question_length=len(cleaned))
        context = current_request_trace()
        if context is None:
            end_request_trace(request_token)
            raise RuntimeError("请求 Trace 没有开始")
        started = perf_counter()
        retrieval_ms: float | None = None
        context_ms: float | None = None
        generation_ms: float | None = None
        retrieved: list[RetrievalResult] = []
        retrieval_trace: RetrievalTrace | None = None
        retrieval_token: Token[RetrievalTrace | None] | None = None
        invalid_citations: list[str] = []
        citation_validity: float | None = None
        context_citation_ids: list[str] = []
        try:
            with context.span("query_validation", input_summary=f"chars={len(cleaned)}"):
                self._validate(cleaned, selected)

            retrieval_token = begin_retrieval_trace(mode)
            retrieval_started = perf_counter()
            try:
                retrieved = self._retriever.retrieve(cleaned, selected)
            except AppError:
                retrieval_ms = (perf_counter() - retrieval_started) * 1000
                context.add_measured(
                    "retrieval",
                    retrieval_ms,
                    status="error",
                    output_summary="failed",
                )
                raise
            retrieval_ms = (perf_counter() - retrieval_started) * 1000
            retrieval_trace = _finish_trace(retrieval_token)
            retrieval_token = None
            if retrieval_trace is not None:
                retrieval_trace.final_result_count = len(retrieved)
            record_retrieval_spans(context, retrieval_trace, retrieval_ms, len(retrieved))

            context_started = perf_counter()
            with context.span(
                "context_builder",
                input_summary=f"retrieved={len(retrieved)}",
            ) as span:
                try:
                    built = self._context.build(cleaned, retrieved)
                except AppError:
                    context_ms = (perf_counter() - context_started) * 1000
                    raise
                context_ms = (perf_counter() - context_started) * 1000
                span.output_summary = f"citations={len(built.citations)} chars={len(built.text)}"
                context_citation_ids = [item.citation_id for item in built.citations]

            with context.span(
                "prompt_builder",
                input_summary=f"context_chars={len(built.text)}",
            ) as span:
                system_prompt, user_prompt = self._prompts.build(cleaned, built.text)
                span.output_summary = f"prompt_chars={len(user_prompt)}"

            generation_started = perf_counter()
            with context.span("llm_generation") as span:
                try:
                    generated = self._llm.generate(system_prompt, user_prompt)
                except AppError:
                    generation_ms = (perf_counter() - generation_started) * 1000
                    raise
                generation_ms = (perf_counter() - generation_started) * 1000
                span.output_summary = f"answer_chars={len(generated.text)}"
                if generated.prompt_tokens is not None:
                    span.metadata["prompt_tokens"] = generated.prompt_tokens
                if generated.completion_tokens is not None:
                    span.metadata["completion_tokens"] = generated.completion_tokens

            citations = map_citations(generated.text, built.citations)
            with context.span(
                "citation_validation",
                input_summary=f"available={len(built.citations)}",
            ) as span:
                valid, invalid = audit_citations(
                    generated.text,
                    [item.citation_id for item in built.citations],
                )
                invalid_citations = invalid
                marker_count = len(valid) + len(invalid)
                citation_validity = None if marker_count == 0 else len(valid) / marker_count
                span.output_summary = f"valid={len(valid)} invalid={len(invalid)}"
                if invalid:
                    span.status = "warning"
                    span.metadata["invalid_count"] = len(invalid)
        except AppError as exc:
            retrieval_trace = _close_retrieval(retrieval_token, retrieval_trace)
            request_trace = end_request_trace(request_token)
            self._observe(
                question_length=len(cleaned),
                status="error",
                retrieval_count=len(retrieved) if retrieval_ms is not None else None,
                retrieval_latency_ms=retrieval_ms,
                context_build_latency_ms=context_ms,
                generation_latency_ms=generation_ms,
                total_latency_ms=(perf_counter() - started) * 1000,
                error=exc,
                trace=retrieval_trace,
                request_trace=request_trace,
                failure_types=_failure_types(
                    exc,
                    request_trace,
                    invalid_citations,
                    citation_validity,
                ),
            )
            bind_error_trace(exc, request_trace, retrieval_trace, context_citation_ids)
            raise
        except Exception as exc:
            retrieval_trace = _close_retrieval(retrieval_token, retrieval_trace)
            request_trace = end_request_trace(request_token)
            self._observe(
                question_length=len(cleaned),
                status="error",
                retrieval_count=len(retrieved) if retrieval_ms is not None else None,
                retrieval_latency_ms=retrieval_ms,
                context_build_latency_ms=context_ms,
                generation_latency_ms=generation_ms,
                total_latency_ms=(perf_counter() - started) * 1000,
                error=exc,
                trace=retrieval_trace,
                request_trace=request_trace,
                failure_types=_failure_types(
                    exc,
                    request_trace,
                    invalid_citations,
                    citation_validity,
                ),
            )
            bind_error_trace(exc, request_trace, retrieval_trace, context_citation_ids)
            raise

        total_ms = (perf_counter() - started) * 1000
        if retrieval_ms is None or context_ms is None or generation_ms is None:
            end_request_trace(request_token)
            raise RuntimeError("问答耗时没有记全")
        request_trace = end_request_trace(request_token)
        failures = _request_failures(None, request_trace, invalid_citations, citation_validity)
        self._observe(
            question_length=len(cleaned),
            status="success",
            retrieval_count=len(retrieved),
            retrieval_latency_ms=retrieval_ms,
            context_build_latency_ms=context_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=total_ms,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            trace=retrieval_trace,
            request_trace=request_trace,
            failure_types=[item.failure_type for item in failures],
        )
        return RAGResponse(
            answer=generated.text,
            citations=citations,
            context_citations=built.citations,
            retrieved_chunks=retrieved,
            retrieval_trace=retrieval_trace,
            request_trace=request_trace,
            failure_records=failures,
            metrics=_metrics(
                retrieval_ms,
                context_ms,
                generation_ms,
                total_ms,
                generated.prompt_tokens,
                generated.completion_tokens,
                retrieval_trace,
            ),
        )

    def _observe(
        self,
        *,
        question_length: int,
        status: Literal["success", "error"],
        retrieval_count: int | None = None,
        retrieval_latency_ms: float | None = None,
        context_build_latency_ms: float | None = None,
        generation_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        error: BaseException | None = None,
        trace: RetrievalTrace | None = None,
        request_trace: RequestTrace | None = None,
        failure_types: list[str] | None = None,
    ) -> None:
        error_stage = None
        error_category = None
        error_type = None
        if error is not None:
            error_stage, error_category = classify_failure(error)
            error_type = type(error).__name__
        log_query_observation(
            QueryObservation(
                question_length=question_length,
                retrieval_count=retrieval_count,
                retrieval_latency_ms=retrieval_latency_ms,
                context_build_latency_ms=context_build_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status=status,
                error_stage=error_stage,
                error_type=error_type,
                error_category=error_category,
                retrieval_mode=None if trace is None else trace.retrieval_mode,
                vector_candidate_count=None if trace is None else trace.vector_candidate_count,
                bm25_candidate_count=None if trace is None else trace.bm25_candidate_count,
                hybrid_candidate_count=None if trace is None else trace.hybrid_candidate_count,
                reranker_enabled=None if trace is None else trace.reranker_enabled,
                reranker_candidate_count=None if trace is None else trace.reranker_candidate_count,
                final_result_count=None if trace is None else trace.final_result_count,
                embedding_latency_ms=None if trace is None else trace.embedding_latency_ms,
                vector_latency_ms=None if trace is None else trace.vector_latency_ms,
                bm25_latency_ms=None if trace is None else trace.bm25_latency_ms,
                hybrid_latency_ms=None if trace is None else trace.hybrid_latency_ms,
                rrf_latency_ms=None if trace is None else trace.rrf_latency_ms,
                reranker_latency_ms=None if trace is None else trace.reranker_latency_ms,
                trace_id=None if request_trace is None else request_trace.trace_id,
                trace_status=None if request_trace is None else request_trace.status,
                span_count=None if request_trace is None else len(request_trace.spans),
                spans=_observed_spans(request_trace),
                failure_types=failure_types or [],
            )
        )

    def _validate(self, question: str, top_k: int) -> None:
        if not question:
            raise QuestionValidationError("问题不能为空")
        if len(question) > self._max_question_chars:
            raise QuestionValidationError(
                f"问题长度为 {len(question)}，超过 MAX_QUESTION_CHARS={self._max_question_chars}"
            )
        if top_k < 1 or top_k > self._max_top_k:
            raise QuestionValidationError(
                f"top_k 必须在 1 到 {self._max_top_k} 之间，当前是 {top_k}"
            )


def _failure_types(
    error: BaseException,
    request_trace: RequestTrace | None,
    invalid_citations: list[str],
    citation_validity: float | None,
) -> list[str]:
    records = _request_failures(error, request_trace, invalid_citations, citation_validity)
    return [item.failure_type for item in records]


def _request_failures(
    error: BaseException | None,
    request_trace: RequestTrace | None,
    invalid_citations: list[str],
    citation_validity: float | None,
) -> list[FailureRecord]:
    error_stage = None
    error_category = None
    error_type = None
    if error is not None:
        error_stage, error_category = classify_failure(error)
        error_type = type(error).__name__
    traced = "" if request_trace is None else request_trace.trace_id
    return classify_failures(
        FailureSignals(
            trace_id=traced,
            error_type=error_type,
            error_stage=error_stage,
            error_category=error_category,
            invalid_citations=invalid_citations,
            citation_validity=citation_validity,
        )
    )


def _observed_spans(trace: RequestTrace | None) -> list[ObservedSpan]:
    if trace is None:
        return []
    return [
        ObservedSpan(name=item.name, duration_ms=item.duration_ms, status=item.status)
        for item in trace.spans
    ]


def _close_retrieval(
    token: Token[RetrievalTrace | None] | None,
    trace: RetrievalTrace | None,
) -> RetrievalTrace | None:
    if token is None:
        return trace
    return end_retrieval_trace(token)


def _finish_trace(token: Token[RetrievalTrace | None] | None) -> RetrievalTrace | None:
    if token is None:
        return None
    return end_retrieval_trace(token)


def _metrics(
    retrieval_ms: float,
    context_ms: float,
    generation_ms: float,
    total_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    trace: RetrievalTrace | None,
) -> RAGMetrics:
    return RAGMetrics(
        retrieval_latency_ms=retrieval_ms,
        context_build_latency_ms=context_ms,
        generation_latency_ms=generation_ms,
        total_latency_ms=total_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        retrieval_mode=None if trace is None else trace.retrieval_mode,
        vector_candidate_count=None if trace is None else trace.vector_candidate_count,
        bm25_candidate_count=None if trace is None else trace.bm25_candidate_count,
        hybrid_candidate_count=None if trace is None else trace.hybrid_candidate_count,
        reranker_enabled=False if trace is None else trace.reranker_enabled,
        reranker_candidate_count=None if trace is None else trace.reranker_candidate_count,
        reranker_name=None if trace is None else trace.reranker_name,
        final_result_count=None if trace is None else trace.final_result_count,
        embedding_latency_ms=None if trace is None else trace.embedding_latency_ms,
        vector_latency_ms=None if trace is None else trace.vector_latency_ms,
        bm25_latency_ms=None if trace is None else trace.bm25_latency_ms,
        hybrid_latency_ms=None if trace is None else trace.hybrid_latency_ms,
        rrf_latency_ms=None if trace is None else trace.rrf_latency_ms,
        reranker_latency_ms=None if trace is None else trace.reranker_latency_ms,
    )
