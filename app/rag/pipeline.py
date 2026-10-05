"""把检索、上下文、提示词、生成和引用串成一次问答。

检索失败和生成失败保持各自的异常，不包成同一种错误。
没有检索结果时仍然调用模型，由提示词要求它说明知识库缺少依据。
日志只记长度、数量和耗时，不记问题原文、资料正文和完整提示词。
"""

from __future__ import annotations

from time import perf_counter
from typing import Literal

from app.core.exceptions import AppError, QuestionValidationError
from app.llm.base import LLMProvider
from app.observability.records import QueryObservation, classify_failure, log_query_observation
from app.rag.citation import map_citations
from app.rag.context_builder import ContextBuilder
from app.rag.models import RAGMetrics, RAGResponse
from app.rag.prompt import PromptBuilder
from app.retrieval.base import Retriever
from app.retrieval.models import RetrievalResult


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
        try:
            self._validate(cleaned, selected)
        except AppError as exc:
            self._observe(question_length=len(cleaned), status="error", error=exc)
            raise

        started = perf_counter()
        retrieval_ms: float | None = None
        context_ms: float | None = None
        generation_ms: float | None = None
        retrieved: list[RetrievalResult] = []
        try:
            retrieval_started = perf_counter()
            try:
                retrieved = self._retriever.retrieve(cleaned, selected)
            except AppError:
                retrieval_ms = (perf_counter() - retrieval_started) * 1000
                raise
            retrieval_ms = (perf_counter() - retrieval_started) * 1000

            context_started = perf_counter()
            try:
                built = self._context.build(cleaned, retrieved)
            except AppError:
                context_ms = (perf_counter() - context_started) * 1000
                raise
            context_ms = (perf_counter() - context_started) * 1000

            system_prompt, user_prompt = self._prompts.build(cleaned, built.text)
            generation_started = perf_counter()
            try:
                generated = self._llm.generate(system_prompt, user_prompt)
            except AppError:
                generation_ms = (perf_counter() - generation_started) * 1000
                raise
            generation_ms = (perf_counter() - generation_started) * 1000
        except AppError as exc:
            self._observe(
                question_length=len(cleaned),
                status="error",
                retrieval_count=len(retrieved) if retrieval_ms is not None else None,
                retrieval_latency_ms=retrieval_ms,
                context_build_latency_ms=context_ms,
                generation_latency_ms=generation_ms,
                total_latency_ms=(perf_counter() - started) * 1000,
                error=exc,
            )
            raise
        except Exception as exc:
            self._observe(
                question_length=len(cleaned),
                status="error",
                retrieval_count=len(retrieved) if retrieval_ms is not None else None,
                retrieval_latency_ms=retrieval_ms,
                context_build_latency_ms=context_ms,
                generation_latency_ms=generation_ms,
                total_latency_ms=(perf_counter() - started) * 1000,
                error=exc,
            )
            raise

        citations = map_citations(generated.text, built.citations)
        total_ms = (perf_counter() - started) * 1000
        if retrieval_ms is None or context_ms is None or generation_ms is None:
            raise RuntimeError("问答耗时没有记全")
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
        )
        return RAGResponse(
            answer=generated.text,
            citations=citations,
            retrieved_chunks=retrieved,
            metrics=RAGMetrics(
                retrieval_latency_ms=retrieval_ms,
                context_build_latency_ms=context_ms,
                generation_latency_ms=generation_ms,
                total_latency_ms=total_ms,
                prompt_tokens=generated.prompt_tokens,
                completion_tokens=generated.completion_tokens,
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
