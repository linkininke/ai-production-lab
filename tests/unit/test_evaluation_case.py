"""评测核心类型兼容现有 v2 题集，并给新旧调用方同一套检索指标。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.core.exceptions import EvaluationError
from app.evaluation.core.case import EvaluationCase
from app.evaluation.core.result import measure_retrieval
from app.evaluation.dataset import EvalQuestion, load_dataset

_QUESTIONS = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
_DOC = "doc_a3b81248fcd23568"


def test_shipped_cases_keep_v2_labels() -> None:
    dataset = load_dataset(_QUESTIONS)
    assert all(isinstance(item, EvaluationCase) for item in dataset.questions)
    assert all(item.expected_answer_points == [] for item in dataset.questions)
    for item in dataset.questions:
        assert item.answerable is not item.expects_abstention
        assert item.relevant_chunk_ids == item.expected_chunk_ids
        assert isinstance(item, EvalQuestion)


def test_optional_answer_points_and_new_category(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "version": "v2",
                "questions": [
                    {
                        "id": "q001",
                        "category": "reasoning",
                        "question": "为什么自调用不会开启事务？",
                        "expected_keywords": ["自调用"],
                        "expected_answer_points": ["同类内部调用绕过代理"],
                        "expected_document_ids": [_DOC],
                        "expects_abstention": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    case = load_dataset(path).questions[0]
    assert case.category == "reasoning"
    assert case.expected_answer_points == ["同类内部调用绕过代理"]
    assert case.answerable is True


def test_blank_answer_point_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "version": "v2",
                "questions": [
                    {
                        "id": "q001",
                        "question": "示例",
                        "expected_answer_points": ["  "],
                        "expected_document_ids": [_DOC],
                        "expects_abstention": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="答案要点不能为空"):
        load_dataset(path)


def test_measure_retrieval_uses_the_same_metric_functions() -> None:
    case = EvaluationCase(
        id="q001",
        question="示例",
        expected_document_ids=[_DOC, "doc_71eaf1f9cf0da067"],
    )
    result = measure_retrieval(
        case,
        [_DOC, "doc_0000000000000000"],
        case.expected_document_ids,
        precision_k=5,
        mrr_k=5,
        cutoffs=(1, 5),
        latency_ms=12.5,
    )
    assert result.recall_at_k == 0.5
    assert result.recall_by_cutoff[1] == 0.5
    assert result.recall_by_cutoff[5] == 0.5
    assert result.precision_at_k == 0.2
    assert result.reciprocal_rank == 1.0
    assert result.latency_ms == 12.5
