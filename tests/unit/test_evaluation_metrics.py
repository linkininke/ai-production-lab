"""Recall、MRR、关键词覆盖率和拒答短语。这些函数不访问模型。"""

from __future__ import annotations

import pytest

from app.evaluation.compare import compare_reports
from app.evaluation.metrics import (
    abstention_phrase,
    document_recall_at_k,
    keyword_coverage,
    matched_keywords,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from app.evaluation.models import (
    EvaluationReport,
)


def test_recall_counts_each_expected_document_once() -> None:
    recall = document_recall_at_k(
        ["doc_a", "doc_a", "doc_c"],
        ["doc_a", "doc_b"],
    )
    assert recall == 0.5


def test_mrr_uses_the_first_relevant_rank() -> None:
    assert reciprocal_rank_at_k(["doc_x", "doc_a", "doc_a"], ["doc_a"]) == 0.5
    assert reciprocal_rank_at_k(["doc_x"], ["doc_a"]) == 0.0
    assert reciprocal_rank_at_k(["doc_a"], ["doc_a", "doc_b"]) == 1.0


def test_recall_and_mrr_require_labels() -> None:
    with pytest.raises(ValueError, match="标注文档"):
        document_recall_at_k(["doc_a"], [])
    with pytest.raises(ValueError, match="标注文档"):
        reciprocal_rank_at_k([], [])


def test_recall_counts_one_of_two_labels_as_half() -> None:
    assert recall_at_k(["doc_a", "doc_c"], ["doc_a", "doc_b"]) == 0.5


def test_precision_divides_by_k_not_by_the_result_count() -> None:
    assert precision_at_k(["doc_a", "doc_x"], ["doc_a"], 5) == 0.2
    assert mrr_at_k(["doc_x", "doc_a"], ["doc_a"], 5) == 0.5
    assert mrr_at_k(["doc_x"], ["doc_a"], 1) == 0.0


def test_keyword_coverage_is_casefold_and_skips_empty_lists() -> None:
    assert keyword_coverage("Spring 代理", ["spring", "没有"]) == 0.5
    assert matched_keywords("Spring 代理", ["spring", "没有"]) == ["spring"]
    assert keyword_coverage("任何回答", []) is None


def test_abstention_phrase_does_not_match_a_bare_negative() -> None:
    assert abstention_phrase("知识库中缺少相关信息。") == "知识库中缺少相关信息"
    assert abstention_phrase("这里没有提到 Kubernetes。") is None
    assert abstention_phrase("缺少") is None


def test_compare_reports_subtracts_the_left_from_the_right() -> None:
    left = _report("eval_20261005T010000000000Z_k5.json", recall=0.25, top_k=5)
    right = _report("eval_20261005T020000000000Z_k8.json", recall=0.5, top_k=8)
    compared = compare_reports(left, right)
    recall = next(item for item in compared.metrics if item.name == "recall_at_k")
    assert recall.left == 0.25
    assert recall.right == 0.5
    assert recall.delta == pytest.approx(0.25)
    assert any(item.startswith("Top-K：") for item in compared.config_differences)


def _report(filename: str, *, recall: float, top_k: int) -> EvaluationReport:
    return EvaluationReport(
        filename=filename,
        created_at="2026-10-05T00:00:00+00:00",
        dataset_version="v1",
        dataset_file="questions.json",
        question_count=1,
        top_k=top_k,
        embedding_model="fake-hash",
        llm_model="fake-llm",
        collection_name="kb_local",
        chunk_size=200,
        chunk_overlap=20,
        corpus_documents=[],
        recall_at_k=recall,
        mrr_at_k=recall,
        success_count=1,
        failure_count=0,
        failures=[],
        questions=[],
    )
