"""Phase 9 的对照说明只使用报告里的数字。"""

from __future__ import annotations

from app.evaluation.models import EvaluationReport, QuestionResult
from app.evaluation.phase9 import render_phase9
from app.observability.failure import FailureRecord


def test_summary_keeps_missing_judge_scores_empty_and_names_recall_changes() -> None:
    vector = _report(
        "A",
        "vector",
        "v1",
        recall=0.5,
        questions=[
            _question("q1", "semantic", recall=0.0),
            _question("q2", "semantic", recall=1.0),
            _question("q9", "unanswerable", recall=None, abstention=False, expects=True),
        ],
    )
    hybrid = _report(
        "B",
        "hybrid",
        "v1",
        recall=1.0,
        questions=[
            _question("q1", "semantic", recall=1.0),
            _question("q2", "semantic", recall=1.0),
            _question("q9", "unanswerable", recall=None, abstention=True, expects=True),
        ],
    )
    text = render_phase9([vector, hybrid])
    assert "q1" in text
    assert "Judge 正确性没有实测" in text
    assert "未完成的实验：C、D" in text
    assert "0.000" not in text.split("成本", maxsplit=1)[0] or "成本为空" in text
    assert "不正确：q9" in text
    assert "- B：1/1 题拒答判断正确。" in text


def test_prompt_section_states_when_the_pair_is_missing() -> None:
    text = render_phase9([])
    assert "缺少 C 或 D" in text
    assert "不能判断提示词有没有改善" in text


def _report(
    experiment_id: str,
    mode: str,
    prompt: str,
    *,
    recall: float,
    questions: list[QuestionResult],
) -> EvaluationReport:
    return EvaluationReport(
        filename=f"eval_20261006T000000000000Z_k5_{experiment_id}.json",
        created_at="2026-10-06T00:00:00+00:00",
        dataset_version="v2",
        dataset_file="questions.json",
        question_count=len(questions),
        top_k=5,
        embedding_model="fixture",
        llm_model="fixture",
        retrieval_mode=mode,
        prompt_version=prompt,
        experiment_id=experiment_id,
        collection_name="kb_phase9",
        chunk_size=200,
        chunk_overlap=20,
        corpus_documents=[],
        recall_at_k=recall,
        mrr_at_k=recall,
        precision_at_k=0.2,
        citation_validity=1.0,
        abstention_quality=1.0,
        keyword_coverage=1.0,
        judge_correctness=None,
        judge_groundedness=None,
        p50_latency_ms=10,
        p95_latency_ms=20,
        p99_latency_ms=20,
        mean_total_latency_ms=12,
        judge_cost=None,
        failure_rate=0.0,
        success_count=len(questions),
        failure_count=0,
        failures=[],
        questions=questions,
    )


def _question(
    case_id: str,
    category: str,
    *,
    recall: float | None,
    abstention: bool | None = None,
    expects: bool = False,
) -> QuestionResult:
    records: list[FailureRecord] = []
    if recall == 0:
        records.append(
            FailureRecord(
                case_id=case_id,
                failure_type="RETRIEVAL_FAILURE",
                severity="HIGH",
                stage="retrieval",
                description="相关文档没有全部出现在检索结果里。",
                suggested_action="检查检索。",
            )
        )
    return QuestionResult(
        id=case_id,
        question="不写进总表",
        category=category,
        expects_abstention=expects,
        status="success",
        recall_at_k=recall,
        abstention_correct=abstention,
        failure_records=records,
    )
