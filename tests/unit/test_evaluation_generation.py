"""生成侧规则指标。不调用模型，也不把空标注算成零分。"""

from __future__ import annotations

from app.evaluation.core.case import EvaluationCase
from app.evaluation.generation import evaluate_generation
from app.rag.models import Citation

_DOC = "doc_a3b81248fcd23568"
_OTHER = "doc_71eaf1f9cf0da067"


def _citation(citation_id: str, document_id: str) -> Citation:
    return Citation(
        citation_id=citation_id,
        document_id=document_id,
        filename="note.md",
        chunk_id="chunk_" + "a" * 20,
        text="正文",
    )


def _case(**overrides: object) -> EvaluationCase:
    payload: dict[str, object] = {
        "id": "q001",
        "question": "为什么事务会失效？",
        "expected_keywords": ["代理"],
        "expected_document_ids": [_DOC],
        "expects_abstention": False,
    }
    payload.update(overrides)
    return EvaluationCase.model_validate(payload)


def test_citation_validity_ignores_answers_without_markers() -> None:
    score = evaluate_generation(_case(), "只写了结论。", [_citation("C1", _DOC)])
    assert score.citation_validity is None
    assert score.citation_coverage == 0.0
    assert score.keyword_coverage == 0.0


def test_citation_coverage_skips_documents_that_were_not_retrieved() -> None:
    score = evaluate_generation(
        _case(expected_document_ids=[_DOC, _OTHER]),
        "见 [C1]。",
        [_citation("C1", _DOC)],
    )
    assert score.citation_validity == 1.0
    assert score.citation_coverage == 1.0
    assert score.valid_citations == ["C1"]


def test_invalid_marker_lowers_validity_only() -> None:
    score = evaluate_generation(
        _case(),
        "见 [C1] 和 [C9]。",
        [_citation("C1", _DOC)],
    )
    assert score.citation_validity == 0.5
    assert score.invalid_citations == ["C9"]
    assert score.citation_coverage == 1.0


def test_completeness_is_null_until_answer_points_exist() -> None:
    missing = evaluate_generation(_case(), "代理。", [])
    assert missing.answer_completeness is None
    present = evaluate_generation(
        _case(expected_answer_points=["代理机制", "自调用"]),
        "原因是代理机制。",
        [],
    )
    assert present.answer_completeness == 0.5
    assert present.matched_answer_points == ["代理机制"]


def test_abstention_quality_checks_both_directions() -> None:
    refused = evaluate_generation(
        _case(expects_abstention=True, expected_document_ids=[], expected_keywords=[]),
        "知识库中缺少相关信息。",
        [],
    )
    assert refused.abstained is True
    assert refused.abstention_correct is True
    assert refused.citation_coverage is None
    answered = evaluate_generation(_case(), "知识库中缺少相关信息。", [])
    assert answered.abstention_correct is False
