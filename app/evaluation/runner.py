"""按评测集逐题运行当前问答链路，并保存 JSON 报告。

样例语料以 eval/{文件名} 作为身份写入当前向量库。已有的其他文档不会被删除，
它们仍会参与检索，所以指标描述的是当前库，不只是一份孤立样本。
单题失败记入报告后继续。标注对不上语料时整次停止，并且不改预期答案。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.core.container import AppContainer
from app.core.exceptions import AppError, EvaluationError, QuestionValidationError
from app.core.logging import request_id_var
from app.evaluation.compare import compare_reports
from app.evaluation.core.case import EvaluationCase
from app.evaluation.core.result import measure_retrieval
from app.evaluation.dashboard import build_case_trace, with_failure_counts
from app.evaluation.dataset import EvaluationDataset, load_dataset
from app.evaluation.generation import GenerationScore, evaluate_generation
from app.evaluation.judge import Judge, JudgeResult, MockJudge, build_judge
from app.evaluation.metrics import mean
from app.evaluation.models import (
    CorpusDocument,
    EvaluationHit,
    EvaluationReport,
    FailureItem,
    QuestionResult,
    ReportComparison,
    ReportSummary,
)
from app.evaluation.reliability import (
    Threshold,
    evaluate_slo,
    latency_percentiles,
    plan_budget,
    predicted_case_cost_usd,
    quality_gate,
    quality_gate_rules,
    slo_rules,
)
from app.ingestion.pipeline import IngestionPipeline
from app.observability.failure import FailureRecord, FailureSignals, classify_failures
from app.observability.records import classify_failure
from app.observability.trace import bound_error_trace
from app.rag.models import RAGResponse
from app.rag.pipeline import RAGPipeline
from app.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

_REPORT_NAME = re.compile(r"^eval_[0-9]{8}T[0-9]{12}Z_k[0-9]+\.json$")
_CORPUS_SUFFIXES = {".md", ".markdown", ".txt"}


class EvaluationRunner:
    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        ingestion: IngestionPipeline,
        settings: Settings,
        embedding_model: str,
        llm_model: str,
        judge: Judge | None = None,
        retrieval_mode: str = "vector",
        prompt_version: str = "v1",
        experiment_id: str = "",
    ) -> None:
        self._pipeline = pipeline
        self._ingestion = ingestion
        self._settings = settings
        self._embedding_model = embedding_model or "unknown"
        self._llm_model = llm_model or "unknown"
        self._judge = judge if judge is not None else MockJudge()
        self._retrieval_mode = retrieval_mode
        self._prompt_version = prompt_version
        self._experiment_id = experiment_id

    @classmethod
    def from_container(
        cls,
        container: AppContainer,
        *,
        retrieval_mode: str = "vector",
        prompt_version: str = "v1",
        experiment_id: str = "",
    ) -> EvaluationRunner:
        pipeline = container.require_rag(retrieval_mode, prompt_version=prompt_version)
        ingestion = container.require_ingestion()
        embedder = container.embedder
        llm = container.llm
        embedding_model = str(getattr(embedder, "model_name", "") or "")
        llm_model = str(getattr(llm, "model_name", "") or "")
        return cls(
            pipeline=pipeline,
            ingestion=ingestion,
            settings=container.settings,
            embedding_model=embedding_model,
            llm_model=llm_model,
            judge=build_judge(container),
            retrieval_mode=retrieval_mode,
            prompt_version=prompt_version,
            experiment_id=experiment_id,
        )

    def run(
        self,
        top_k: int | None = None,
        max_cases: int | None = None,
        max_cost: float | None = None,
    ) -> EvaluationReport:
        selected = self._settings.default_top_k if top_k is None else top_k
        self._validate_top_k(selected)
        dataset = load_dataset(self._settings.evaluation_dataset)
        case_limit = self._settings.evaluation_max_cases if max_cases is None else max_cases
        cost_limit = self._settings.evaluation_max_cost if max_cost is None else max_cost
        decision = plan_budget(
            question_count=len(dataset.questions),
            max_cases=case_limit,
            max_cost=cost_limit,
            predicted_cost_per_case=_predicted_case_cost(self._settings),
        )
        if decision.aborted:
            raise EvaluationError(decision.reason)
        full_count = len(dataset.questions)
        if decision.selected_cases < full_count:
            dataset = dataset.model_copy(
                update={"questions": dataset.questions[: decision.selected_cases]}
            )
        corpus = self._index_corpus(dataset)
        questions = [self._run_question(item, selected) for item in dataset.questions]
        report = with_failure_counts(
            self._aggregate(
                dataset,
                selected,
                corpus,
                questions,
                budget_note=_budget_note(full_count, decision.selected_cases),
            )
        )
        report.filename = _reserve_filename(
            self._settings.evaluation_reports,
            report.created_at,
            selected,
        )
        self._write(report)
        logger.info(
            "evaluation_completed dataset_version=%s question_count=%s failure_count=%s "
            "recall_at_k=%s mrr_at_k=%s top_k=%s",
            report.dataset_version,
            report.question_count,
            report.failure_count,
            _number(report.recall_at_k),
            _number(report.mrr_at_k),
            report.top_k,
        )
        return report

    def list_reports(self) -> list[ReportSummary]:
        directory = self._settings.evaluation_reports
        if not directory.is_dir():
            return []
        summaries: list[ReportSummary] = []
        for path in sorted(directory.glob("eval_*.json"), reverse=True):
            if not _REPORT_NAME.fullmatch(path.name):
                continue
            try:
                report = _read_report(path)
            except EvaluationError:
                logger.warning("evaluation_report_unreadable filename=%s", path.name)
                continue
            summaries.append(_summary(report))
        return summaries

    def load_report(self, filename: str) -> EvaluationReport:
        return _read_report(self._report_path(filename))

    def compare(self, left: str, right: str) -> ReportComparison:
        return compare_reports(self.load_report(left), self.load_report(right))

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1 or top_k > self._settings.max_top_k:
            raise QuestionValidationError(
                f"top_k 必须在 1 到 {self._settings.max_top_k} 之间，当前是 {top_k}"
            )

    def _index_corpus(self, dataset: EvaluationDataset) -> list[CorpusDocument]:
        directory = self._settings.evaluation_corpus
        if not directory.is_dir():
            raise EvaluationError(f"评测语料目录不存在：{directory.name}")
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _CORPUS_SUFFIXES
        )
        if not paths:
            raise EvaluationError("评测语料目录里没有 Markdown 或 TXT")
        indexed: list[CorpusDocument] = []
        for path in paths:
            result = self._ingestion.ingest_path(path, source_key=f"eval/{path.name}")
            indexed.append(
                CorpusDocument(
                    filename=result.document.filename,
                    document_id=result.document.document_id,
                    status=result.status,
                )
            )
        produced = {item.document_id for item in indexed}
        missing = [
            f"{question.id}:{document_id}"
            for question in dataset.questions
            for document_id in question.expected_document_ids
            if document_id not in produced
        ]
        if missing:
            raise EvaluationError("评测标注的文档不在样例语料中：" + ", ".join(missing))
        logger.info("evaluation_corpus_ready file_count=%s", len(indexed))
        return indexed

    def _run_question(self, item: EvaluationCase, top_k: int) -> QuestionResult:
        token = request_id_var.set(f"eval_{item.id}")
        try:
            try:
                response = self._pipeline.query(item.question, top_k=top_k)
            except AppError as exc:
                result = _failed_question(item, exc)
            except Exception as exc:
                logger.exception(
                    "evaluation_question_failed question_id=%s error_type=%s",
                    item.id,
                    type(exc).__name__,
                )
                result = _failed_question(item, exc)
            else:
                result = _successful_question(item, response, top_k, self._judge)
        finally:
            request_id_var.reset(token)
        logger.info(
            "evaluation_question_completed question_id=%s status=%s failure_stage=%s "
            "failure_types=%s",
            item.id,
            result.status,
            result.failure_stage or "-",
            ",".join(record.failure_type for record in result.failure_records) or "-",
        )
        return result

    def _aggregate(
        self,
        dataset: EvaluationDataset,
        top_k: int,
        corpus: list[CorpusDocument],
        questions: list[QuestionResult],
        *,
        budget_note: str,
    ) -> EvaluationReport:
        created_at = datetime.now(UTC).isoformat()
        failures = [
            FailureItem(
                id=item.id,
                failure_stage=item.failure_stage or "interface",
                error_category=item.error_category or "unexpected",
                error_type=item.error_type or "Exception",
                error_message=item.error_message or "评测题目失败",
            )
            for item in questions
            if item.status == "error"
        ]
        labeled = [item for item in questions if item.recall_at_k is not None]
        covered = [item.keyword_coverage for item in questions if item.keyword_coverage is not None]
        citation_validity = [
            item.citation_validity for item in questions if item.citation_validity is not None
        ]
        citation_coverage = [
            item.citation_coverage for item in questions if item.citation_coverage is not None
        ]
        completeness = [
            item.answer_completeness for item in questions if item.answer_completeness is not None
        ]
        abstained = [item.abstained for item in questions if item.abstained is not None]
        abstention_quality = [
            item.abstention_correct for item in questions if item.abstention_correct is not None
        ]
        latencies = [
            item.total_latency_ms for item in questions if item.total_latency_ms is not None
        ]
        p50, p95, p99 = latency_percentiles(latencies)
        total = len(questions)
        successes = sum(item.status == "success" for item in questions)
        success_rate = None if total == 0 else successes / total
        timeout_count = sum(item.error_category == "timeout" for item in questions)
        timeout_rate = None if total == 0 else timeout_count / total
        groundedness = _judge_mean(questions, "judge_groundedness")
        recall_at_5 = None
        if top_k == 5:
            recall_at_5 = mean(
                [item.recall_at_k for item in labeled if item.recall_at_k is not None]
            )
        reliability_metrics = {
            "recall_at_5": recall_at_5,
            "citation_validity": mean(citation_validity) if citation_validity else None,
            "groundedness": None if groundedness is None else groundedness * 4,
            "p95_latency_ms": p95,
            "success_rate": success_rate,
        }
        gate = quality_gate(reliability_metrics, _gate_rules(self._settings))
        slo = evaluate_slo(reliability_metrics, _slo_rules(self._settings))
        return EvaluationReport(
            filename=_report_filename(created_at, top_k),
            created_at=created_at,
            dataset_version=dataset.version,
            dataset_file=self._settings.evaluation_dataset_path,
            question_count=len(questions),
            top_k=top_k,
            embedding_model=self._embedding_model,
            llm_model=self._llm_model,
            retrieval_mode=self._retrieval_mode,
            prompt_version=self._prompt_version,
            experiment_id=self._experiment_id,
            collection_name=self._settings.chroma_collection,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            retrieval_max_distance=self._settings.retrieval_max_distance,
            corpus_documents=corpus,
            recall_at_k=mean(
                [item.recall_at_k for item in labeled if item.recall_at_k is not None]
            ),
            mrr_at_k=mean(
                [item.reciprocal_rank for item in labeled if item.reciprocal_rank is not None]
            ),
            precision_at_k=mean(
                [item.precision_at_k for item in labeled if item.precision_at_k is not None]
            ),
            invalid_citation_count=sum(len(item.invalid_citations) for item in questions),
            keyword_coverage=mean([value for value in covered if value is not None]),
            citation_validity=mean(citation_validity),
            citation_coverage=mean(citation_coverage),
            answer_completeness=mean(completeness),
            abstention_rate=mean([1.0 if value else 0.0 for value in abstained]),
            abstention_quality=mean([1.0 if value else 0.0 for value in abstention_quality]),
            judge_correctness=_judge_mean(questions, "judge_correctness"),
            judge_groundedness=_judge_mean(questions, "judge_groundedness"),
            judge_completeness=_judge_mean(questions, "judge_completeness"),
            judge_overall=_judge_mean(questions, "judge_overall"),
            judge_source=self._judge.source,
            mean_judge_latency_ms=_mean_present([item.judge_latency_ms for item in questions]),
            mean_judge_prompt_tokens=_mean_present(
                _optional_floats(questions, "judge_prompt_tokens")
            ),
            mean_judge_completion_tokens=_mean_present(
                _optional_floats(questions, "judge_completion_tokens")
            ),
            judge_cost=_judge_cost(questions),
            success_rate=success_rate,
            failure_rate=None if success_rate is None else 1 - success_rate,
            timeout_rate=timeout_rate,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            quality_gate_status=gate.status,
            slo_status=slo.status,
            budget_note=budget_note,
            mean_retrieval_latency_ms=_mean_present(
                [item.retrieval_latency_ms for item in questions]
            ),
            mean_generation_latency_ms=_mean_present(
                [item.generation_latency_ms for item in questions]
            ),
            mean_total_latency_ms=_mean_present([item.total_latency_ms for item in questions]),
            mean_prompt_tokens=_mean_present(_optional_floats(questions, "prompt_tokens")),
            mean_completion_tokens=_mean_present(
                _optional_floats(questions, "completion_tokens")
            ),
            success_count=sum(1 for item in questions if item.status == "success"),
            failure_count=len(failures),
            failures=failures,
            questions=questions,
        )

    def _write(self, report: EvaluationReport) -> None:
        directory = self._settings.evaluation_reports
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / report.filename
        path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _report_path(self, filename: str) -> Path:
        if not _REPORT_NAME.fullmatch(filename):
            raise EvaluationError("评测报告文件名不合法")
        path = self._settings.evaluation_reports / filename
        if not path.is_file():
            raise EvaluationError(f"评测报告不存在：{filename}", status_code=404)
        return path


def _successful_question(
    item: EvaluationCase,
    response: RAGResponse,
    top_k: int,
    judge: Judge,
) -> QuestionResult:
    retrieved = [_hit(rank, hit) for rank, hit in enumerate(response.retrieved_chunks, start=1)]
    document_ids = [hit.document_id for hit in retrieved]
    chunk_ids = [hit.chunk_id for hit in retrieved]
    recall = None
    rank_score = None
    precision = None
    if item.expected_document_ids:
        scored = measure_retrieval(
            item,
            document_ids,
            item.expected_document_ids,
            precision_k=top_k,
            mrr_k=top_k,
        )
        recall = scored.recall_at_k
        rank_score = scored.reciprocal_rank
        precision = scored.precision_at_k
    generated = evaluate_generation(item, response.answer, response.context_citations)
    judged, judge_failures = _judge_question(judge, item, response)
    return QuestionResult(
        id=item.id,
        question=item.question,
        category=item.category,
        expects_abstention=item.expects_abstention,
        status="success",
        answer=response.answer,
        retrieved=retrieved,
        recall_at_k=recall,
        reciprocal_rank=rank_score,
        precision_at_k=precision,
        invalid_citations=generated.invalid_citations,
        keyword_coverage=generated.keyword_coverage,
        matched_keywords=generated.matched_keywords,
        citation_validity=generated.citation_validity,
        citation_coverage=generated.citation_coverage,
        answer_completeness=generated.answer_completeness,
        matched_answer_points=generated.matched_answer_points,
        abstained=generated.abstained if item.expects_abstention else None,
        abstention_phrase=generated.abstention_phrase if item.expects_abstention else None,
        abstention_correct=generated.abstention_correct,
        retrieval_latency_ms=response.metrics.retrieval_latency_ms,
        generation_latency_ms=response.metrics.generation_latency_ms,
        total_latency_ms=response.metrics.total_latency_ms,
        prompt_tokens=response.metrics.prompt_tokens,
        completion_tokens=response.metrics.completion_tokens,
        failure_records=[
            *_quality_failures(item, response, generated, document_ids, chunk_ids),
            *judge_failures,
        ],
        judge_correctness=None if judged is None else judged.correctness,
        judge_groundedness=None if judged is None else judged.groundedness,
        judge_completeness=None if judged is None else judged.completeness,
        judge_overall=None if judged is None else judged.overall,
        judge_score=None if judged is None else judged.score,
        judge_reasoning="" if judged is None else judged.reasoning,
        judge_source=None if judged is None else judged.source,
        judge_latency_ms=None if judged is None else judged.judge_latency_ms,
        judge_prompt_tokens=None if judged is None else judged.judge_prompt_tokens,
        judge_completion_tokens=None if judged is None else judged.judge_completion_tokens,
        judge_cost=None if judged is None else judged.judge_cost,
        trace=build_case_trace(
            request_trace=response.request_trace,
            retrieval_trace=response.retrieval_trace,
            context_citation_ids=[item.citation_id for item in response.context_citations],
        ),
    )


def _failed_question(item: EvaluationCase, exc: BaseException) -> QuestionResult:
    stage, category = classify_failure(exc)
    message = exc.message if isinstance(exc, AppError) else "评测题目失败"
    request_trace, retrieval_trace, citation_ids = bound_error_trace(exc)
    trace_id = "" if request_trace is None else request_trace.trace_id
    return QuestionResult(
        id=item.id,
        question=item.question,
        category=item.category,
        expects_abstention=item.expects_abstention,
        status="error",
        failure_stage=stage,
        error_category=category,
        error_type=type(exc).__name__,
        error_message=message,
        failure_records=classify_failures(
            FailureSignals(
                case_id=item.id,
                trace_id=trace_id,
                error_type=type(exc).__name__,
                error_stage=stage,
                error_category=category,
            )
        ),
        trace=build_case_trace(
            request_trace=request_trace,
            retrieval_trace=retrieval_trace,
            context_citation_ids=citation_ids,
        ),
    )


def _predicted_case_cost(settings: Settings) -> float | None:
    return predicted_case_cost_usd(
        prompt_tokens=settings.budget_prompt_tokens,
        completion_tokens=settings.budget_completion_tokens,
        embedding_tokens=settings.budget_embedding_tokens,
        llm_input_price_per_1m=settings.llm_input_price_per_1m,
        llm_output_price_per_1m=settings.llm_output_price_per_1m,
        embedding_price_per_1m=settings.embedding_price_per_1m,
    )


def _budget_note(full_count: int, selected: int) -> str:
    if selected < full_count:
        return f"只运行了前 {selected} 题，共 {full_count} 题。这不是全量评测。"
    return ""


def _gate_rules(settings: Settings) -> list[Threshold]:
    return quality_gate_rules(
        recall_at_5=settings.quality_gate_recall_at_5,
        citation_validity=settings.quality_gate_citation_validity,
        groundedness=settings.quality_gate_groundedness,
        p95_ms=settings.quality_gate_p95_ms,
    )


def _slo_rules(settings: Settings) -> list[Threshold]:
    return slo_rules(
        success_rate=settings.slo_success_rate,
        p95_ms=settings.slo_p95_latency_ms,
        citation_validity=settings.slo_citation_validity,
    )


def _judge_question(
    judge: Judge,
    item: EvaluationCase,
    response: RAGResponse,
) -> tuple[JudgeResult | None, list[FailureRecord]]:
    try:
        return judge.evaluate(item.question, response.answer, response.retrieved_chunks, item), []
    except AppError as exc:
        trace_id = "" if response.request_trace is None else response.request_trace.trace_id
        return None, classify_failures(
            FailureSignals(
                case_id=item.id,
                trace_id=trace_id,
                error_type=type(exc).__name__,
                error_stage="interface",
                error_category="unexpected",
            )
        )


def _judge_mean(questions: list[QuestionResult], name: str) -> float | None:
    values: list[float] = []
    for item in questions:
        raw = getattr(item, name)
        if isinstance(raw, int) and not isinstance(raw, bool):
            values.append(raw / 4)
    return mean(values)


def _judge_cost(questions: list[QuestionResult]) -> float | None:
    judged = [item.judge_cost for item in questions if item.judge_source == "llm"]
    if not judged or any(item is None for item in judged):
        return None
    return float(sum(item for item in judged if item is not None))


def _quality_failures(
    item: EvaluationCase,
    response: RAGResponse,
    generated: GenerationScore,
    document_ids: list[str],
    chunk_ids: list[str],
) -> list[FailureRecord]:
    expected = set(item.expected_document_ids)
    trace_id = "" if response.request_trace is None else response.request_trace.trace_id
    return classify_failures(
        FailureSignals(
            case_id=item.id,
            trace_id=trace_id,
            expects_abstention=item.expects_abstention,
            expected_document_count=len(expected),
            retrieved_expected_document_count=len(expected.intersection(document_ids)),
            expected_chunk_count=len(item.expected_chunk_ids),
            retrieved_expected_chunk_count=len(
                set(item.expected_chunk_ids).intersection(chunk_ids)
            ),
            reranker_executed=response.metrics.reranker_enabled,
            has_expected_keywords=bool(item.expected_keywords),
            keyword_coverage=generated.keyword_coverage,
            has_expected_points=bool(item.expected_answer_points),
            answer_completeness=generated.answer_completeness,
            invalid_citations=generated.invalid_citations,
            citation_validity=generated.citation_validity,
            abstention_correct=generated.abstention_correct,
        )
    )


def _hit(rank: int, hit: RetrievalResult) -> EvaluationHit:
    filename = hit.metadata.get("filename", "")
    if not isinstance(filename, str):
        filename = str(filename)
    return EvaluationHit(
        rank=rank,
        document_id=hit.document_id,
        filename=filename,
        chunk_id=hit.chunk_id,
        text=hit.text,
        score=hit.score,
        score_kind=hit.score_kind,
    )


def _read_report(path: Path) -> EvaluationReport:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return with_failure_counts(EvaluationReport.model_validate(payload))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise EvaluationError(f"评测报告无法读取：{path.name}") from exc


def _summary(report: EvaluationReport) -> ReportSummary:
    return ReportSummary(
        filename=report.filename,
        created_at=report.created_at,
        dataset_version=report.dataset_version,
        top_k=report.top_k,
        embedding_model=report.embedding_model,
        llm_model=report.llm_model,
        question_count=report.question_count,
        recall_at_k=report.recall_at_k,
        mrr_at_k=report.mrr_at_k,
        keyword_coverage=report.keyword_coverage,
        abstention_rate=report.abstention_rate,
        failure_count=report.failure_count,
    )


def _report_filename(created_at: str, top_k: int, *, extra_microseconds: int = 0) -> str:
    parsed = datetime.fromisoformat(created_at) + timedelta(microseconds=extra_microseconds)
    stamp = parsed.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"eval_{stamp}_k{top_k}.json"


def _reserve_filename(directory: Path, created_at: str, top_k: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    for extra in range(1000):
        filename = _report_filename(created_at, top_k, extra_microseconds=extra)
        if _REPORT_NAME.fullmatch(filename) and not (directory / filename).exists():
            return filename
    raise EvaluationError("无法分配评测报告文件名")


def _mean_present(values: list[float | None]) -> float | None:
    return mean([value for value in values if value is not None])


def _optional_floats(questions: list[QuestionResult], field_name: str) -> list[float | None]:
    values: list[float | None] = []
    for item in questions:
        value = getattr(item, field_name)
        values.append(float(value) if isinstance(value, int) else None)
    return values


def _number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
