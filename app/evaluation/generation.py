"""生成侧的规则评测。

关键词、引用和拒答都按事先写好的规则计算，不调用模型。
答案完整性只看已填写的答案要点有没有出现在回答里。没有要点时记为 null，
不把空列表当成 0 分。正确性留给后面的 Judge，这里不估计。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.core.case import EvaluationCase
from app.evaluation.metrics import abstention_phrase, keyword_coverage, matched_keywords
from app.rag.citation import audit_citations
from app.rag.models import Citation


class GenerationScore(BaseModel):
    """一道题的生成指标。无法判断的项目保持为空。"""

    model_config = ConfigDict(extra="forbid")

    keyword_coverage: float | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    citation_validity: float | None = None
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citation_coverage: float | None = None
    answer_completeness: float | None = None
    matched_answer_points: list[str] = Field(default_factory=list)
    abstained: bool
    abstention_phrase: str | None = None
    abstention_correct: bool


def evaluate_generation(
    case: EvaluationCase,
    answer: str,
    citations: list[Citation],
) -> GenerationScore:
    """根据回答、上下文引用和题目标注计算生成指标。"""
    valid, invalid = audit_citations(answer, [item.citation_id for item in citations])
    phrase = abstention_phrase(answer)
    abstained = phrase is not None
    points = matched_keywords(answer, case.expected_answer_points)
    return GenerationScore(
        keyword_coverage=keyword_coverage(answer, case.expected_keywords),
        matched_keywords=matched_keywords(answer, case.expected_keywords),
        citation_validity=_ratio(len(valid), len(valid) + len(invalid)),
        valid_citations=valid,
        invalid_citations=invalid,
        citation_coverage=_citation_coverage(valid, citations, case.expected_document_ids),
        answer_completeness=_ratio(len(points), len(case.expected_answer_points)),
        matched_answer_points=points,
        abstained=abstained,
        abstention_phrase=phrase,
        abstention_correct=abstained == case.expects_abstention,
    )


def _citation_coverage(
    valid_citations: list[str],
    citations: list[Citation],
    expected_document_ids: list[str],
) -> float | None:
    """相关文档里，已经进入上下文、并且回答引用了的比例。

    没检索到相关文档时返回 None。那是检索问题，不记成生成失败。
    拒答题没有相关文档，同样不计算引用覆盖。
    """
    expected = set(expected_document_ids)
    if not expected:
        return None
    available: list[str] = []
    seen: set[str] = set()
    for item in citations:
        if item.document_id not in expected or item.document_id in seen:
            continue
        seen.add(item.document_id)
        available.append(item.document_id)
    if not available:
        return None
    cited = {
        item.document_id
        for item in citations
        if item.citation_id.upper() in {marker.upper() for marker in valid_citations}
    }
    hits = sum(1 for document_id in available if document_id in cited)
    return hits / len(available)


def _ratio(hits: int, total: int) -> float | None:
    if total < 1:
        return None
    return hits / total
