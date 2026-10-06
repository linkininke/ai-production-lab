"""失败分类。异常和规则信号分开判断，记录里不出现问题原文。"""

from __future__ import annotations

from app.observability.failure import FailureSignals, classify_failures


def _signals(**overrides: object) -> FailureSignals:
    return FailureSignals.model_validate({"case_id": "q013", **overrides})


def test_missed_documents_are_retrieval_failure_not_generation() -> None:
    records = classify_failures(
        _signals(
            expected_document_count=2,
            retrieved_expected_document_count=0,
            has_expected_keywords=True,
            keyword_coverage=0.0,
            reranker_executed=False,
            invalid_citations=["C99"],
            citation_validity=0.0,
        )
    )
    assert [item.failure_type for item in records] == ["RETRIEVAL_FAILURE", "CITATION_FAILURE"]
    retrieval = records[0]
    assert retrieval.severity == "HIGH"
    assert retrieval.evidence["reranker_executed"] is False
    assert retrieval.evidence["retrieved_expected_document_count"] == 0
    assert "相关文档没有全部出现在检索结果里。" == retrieval.description


def test_wrong_answer_with_context_is_generation_failure() -> None:
    records = classify_failures(
        _signals(
            expected_document_count=1,
            retrieved_expected_document_count=1,
            has_expected_keywords=True,
            keyword_coverage=0.0,
            has_expected_points=True,
            answer_completeness=0.0,
        )
    )
    assert [item.failure_type for item in records] == ["GENERATION_FAILURE"]
    assert records[0].severity == "MEDIUM"
    assert records[0].stage == "llm_generation"


def test_partial_keyword_coverage_is_not_a_failure() -> None:
    records = classify_failures(
        _signals(
            expected_document_count=1,
            retrieved_expected_document_count=1,
            has_expected_keywords=True,
            keyword_coverage=0.5,
        )
    )
    assert records == []


def test_provider_timeout_and_rerank_are_distinct() -> None:
    provider = classify_failures(
        _signals(error_type="LLMError", error_stage="generation", error_category="upstream")
    )
    timeout = classify_failures(
        _signals(error_type="EmbeddingError", error_stage="retrieval", error_category="timeout")
    )
    rerank = classify_failures(
        _signals(error_type="RerankerError", error_stage="retrieval", error_category="retrieval")
    )
    config = classify_failures(
        _signals(
            error_type="ConfigurationError",
            error_stage="interface",
            error_category="configuration",
        )
    )
    assert provider[0].failure_type == "PROVIDER_ERROR"
    assert timeout[0].failure_type == "TIMEOUT"
    assert rerank[0].failure_type == "RERANK_FAILURE"
    assert config[0].failure_type == "CONFIGURATION_ERROR"
    assert config[0].severity == "CRITICAL"
    secret = "问题原文不应进入失败记录"
    dumped = provider[0].model_dump_json() + timeout[0].model_dump_json()
    assert secret not in dumped
    assert "error_type" in provider[0].evidence


def test_context_overflow_is_not_labeled_as_generation() -> None:
    records = classify_failures(
        _signals(
            error_type="ContextOverflowError",
            error_stage="generation",
            error_category="generation",
        )
    )
    assert records[0].failure_type == "CONTEXT_FAILURE"
    assert records[0].stage == "context_builder"


def test_missed_chunk_blocks_generation_failure() -> None:
    records = classify_failures(
        _signals(
            expected_document_count=1,
            retrieved_expected_document_count=1,
            expected_chunk_count=1,
            retrieved_expected_chunk_count=0,
            has_expected_keywords=True,
            keyword_coverage=0.0,
        )
    )
    assert [item.failure_type for item in records] == ["RETRIEVAL_FAILURE"]
    assert records[0].severity == "HIGH"
