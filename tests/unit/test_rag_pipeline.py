"""RAG 编排。检索、生成和引用各自失败时不混成同一种结果。"""

from __future__ import annotations

import logging

import pytest

from app.core.exceptions import (
    ContextOverflowError,
    EmbeddingError,
    LLMError,
    QuestionValidationError,
)
from app.rag.context_builder import ContextBuilder
from app.rag.pipeline import RAGPipeline
from app.rag.prompt import PromptBuilder
from app.retrieval.models import RetrievalResult
from tests.fake_llm import ScriptedLLM


class FixedRetriever:
    def __init__(self, hits: list[RetrievalResult], fail: Exception | None = None) -> None:
        self.hits = hits
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        if self.fail is not None:
            raise self.fail
        return self.hits


def _hit(text: str = "事务通过代理生效。") -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_a",
        document_id="doc_spring",
        text=text,
        score=0.0,
        metadata={
            "filename": "spring.md",
            "file_type": "markdown",
            "chunk_index": 0,
            "document_id": "doc_spring",
        },
    )


def _pipeline(
    retriever: FixedRetriever,
    llm: ScriptedLLM,
    *,
    max_chars: int = 200,
    max_question_chars: int = 20,
    max_top_k: int = 5,
) -> RAGPipeline:
    return RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(max_chars),
        prompt_builder=PromptBuilder(),
        llm=llm,
        default_top_k=2,
        max_top_k=max_top_k,
        max_question_chars=max_question_chars,
    )


def test_answer_citations_come_from_retrieved_chunks(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    retriever = FixedRetriever([_hit("机密片段唯一标记")])
    llm = ScriptedLLM("结论见 [C1]，编造来源 [C9]。")
    response = _pipeline(retriever, llm).query("  机密问题唯一标记  ", top_k=1)
    assert response.answer.startswith("结论见")
    assert [item.citation_id for item in response.citations] == ["C1"]
    assert response.citations[0].chunk_id == "chunk_a"
    assert response.citations[0].filename == "spring.md"
    assert response.citations[0].text == "机密片段唯一标记"
    assert response.retrieved_chunks[0].chunk_id == "chunk_a"
    assert response.metrics.prompt_tokens == 11
    assert response.metrics.retrieval_latency_ms >= 0
    assert llm.calls[0][0].count("机密") == 0
    assert "机密片段唯一标记" in llm.calls[0][1]
    assert "机密问题唯一标记" not in caplog.text
    assert "机密片段唯一标记" not in caplog.text
    assert '"status":"success"' in caplog.text
    assert '"question_length":8' in caplog.text
    assert retriever.calls == [("机密问题唯一标记", 1)]


def test_no_hits_still_asks_the_model_to_abstain() -> None:
    llm = ScriptedLLM("这条不会被用到。")
    response = _pipeline(FixedRetriever([]), llm).query("知识库里没有的问题")
    assert response.answer == "知识库中缺少相关信息。"
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert len(llm.calls) == 1


def test_generation_failure_is_distinct_from_retrieval_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    llm = ScriptedLLM()
    llm.fail = True
    with pytest.raises(LLMError, match="模拟生成失败"):
        _pipeline(FixedRetriever([_hit()]), llm).query("事务为什么失效")
    assert '"error_type":"LLMError"' in caplog.text
    assert '"error_stage":"generation"' in caplog.text
    assert "事务为什么失效" not in caplog.text

    retriever = FixedRetriever([_hit()], fail=EmbeddingError("模拟向量化失败"))
    fresh_llm = ScriptedLLM()
    with pytest.raises(EmbeddingError, match="模拟向量化失败"):
        _pipeline(retriever, fresh_llm).query("事务为什么失效")
    assert fresh_llm.calls == []
    assert '"error_stage":"retrieval"' in caplog.text
    assert "事务为什么失效" not in caplog.text


def test_oversized_chunk_does_not_call_the_model() -> None:
    llm = ScriptedLLM()
    with pytest.raises(ContextOverflowError):
        _pipeline(FixedRetriever([_hit("一二三四五六七八")]), llm, max_chars=4).query("事务")
    assert llm.calls == []


def test_question_and_top_k_are_validated_before_retrieval() -> None:
    retriever = FixedRetriever([_hit()])
    pipeline = _pipeline(retriever, ScriptedLLM())
    with pytest.raises(QuestionValidationError, match="不能为空"):
        pipeline.query("   ")
    with pytest.raises(QuestionValidationError, match="MAX_QUESTION_CHARS"):
        pipeline.query("超" * 21)
    with pytest.raises(QuestionValidationError, match="top_k"):
        pipeline.query("事务", top_k=9)
    assert retriever.calls == []
