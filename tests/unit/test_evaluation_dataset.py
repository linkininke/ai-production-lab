"""随仓库发布的评测集必须对得上样例语料，不能使用占位文档 ID。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.core.exceptions import EvaluationError
from app.evaluation.dataset import load_dataset
from app.ingestion.identity import document_id_for

_CORPUS = PROJECT_ROOT / "data" / "evaluation" / "corpus"
_QUESTIONS = PROJECT_ROOT / "data" / "evaluation" / "questions.json"


def test_shipped_questions_match_the_corpus() -> None:
    dataset = load_dataset(_QUESTIONS)
    produced = {
        path.name: document_id_for(f"eval/{path.name}")
        for path in _CORPUS.iterdir()
        if path.suffix.lower() in {".md", ".markdown", ".txt"}
    }
    assert dataset.version == "v2"
    assert len(dataset.questions) >= 30
    assert len(produced) >= 6
    abstention = [item for item in dataset.questions if item.expects_abstention]
    multi = [item for item in dataset.questions if len(item.expected_document_ids) >= 2]
    labeled = [item for item in dataset.questions if item.expected_document_ids]
    categories = Counter(item.category for item in dataset.questions)
    assert len(abstention) >= 2
    assert len(multi) >= 2
    assert categories["semantic"] >= 6
    assert categories["exact_keyword"] >= 4
    assert categories["technical_identifier"] >= 4
    assert categories["multi_document"] >= 4
    assert categories["unanswerable"] >= 4
    assert all(item.expected_keywords for item in labeled)
    assert all(not item.expected_keywords for item in abstention)
    used: set[str] = set()
    for item in dataset.questions:
        for document_id in item.expected_document_ids:
            assert document_id in set(produced.values())
            used.add(document_id)
    assert used == set(produced.values())


def test_placeholder_document_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "version": "v1",
                "questions": [
                    {
                        "id": "q001",
                        "question": "示例",
                        "expected_keywords": ["示例"],
                        "expected_document_ids": ["doc_spring"],
                        "expects_abstention": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="评测集格式无效"):
        load_dataset(path)


def test_abstention_cannot_also_expect_a_document(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "version": "v1",
                "questions": [
                    {
                        "id": "q001",
                        "question": "示例",
                        "expected_keywords": [],
                        "expected_document_ids": ["doc_a3b81248fcd23568"],
                        "expects_abstention": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="拒答"):
        load_dataset(path)
