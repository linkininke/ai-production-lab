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
from app.evaluation.dataset import EvalQuestion, EvaluationDataset, load_dataset
from app.evaluation.metrics import (
    abstention_phrase,
    document_recall_at_k,
    keyword_coverage,
    matched_keywords,
    mean,
    reciprocal_rank_at_k,
)
from app.evaluation.models import (
    CorpusDocument,
    EvaluationHit,
    EvaluationReport,
    FailureItem,
    QuestionResult,
    ReportComparison,
    ReportSummary,
)
from app.ingestion.pipeline import IngestionPipeline
from app.observability.records import classify_failure
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
    ) -> None:
        self._pipeline = pipeline
        self._ingestion = ingestion
        self._settings = settings
        self._embedding_model = embedding_model or "unknown"
        self._llm_model = llm_model or "unknown"

    @classmethod
    def from_container(cls, container: AppContainer) -> EvaluationRunner:
        pipeline = container.require_rag()
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
        )

    def run(self, top_k: int | None = None) -> EvaluationReport:
        selected = self._settings.default_top_k if top_k is None else top_k
        self._validate_top_k(selected)
        dataset = load_dataset(self._settings.evaluation_dataset)
        corpus = self._index_corpus(dataset)
        questions = [self._run_question(item, selected) for item in dataset.questions]
        report = self._aggregate(dataset, selected, corpus, questions)
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

    def _run_question(self, item: EvalQuestion, top_k: int) -> QuestionResult:
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
                result = _successful_question(item, response)
        finally:
            request_id_var.reset(token)
        logger.info(
            "evaluation_question_completed question_id=%s status=%s failure_stage=%s",
            item.id,
            result.status,
            result.failure_stage or "-",
        )
        return result

    def _aggregate(
        self,
        dataset: EvaluationDataset,
        top_k: int,
        corpus: list[CorpusDocument],
        questions: list[QuestionResult],
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
        abstained = [item.abstained for item in questions if item.abstained is not None]
        return EvaluationReport(
            filename=_report_filename(created_at, top_k),
            created_at=created_at,
            dataset_version=dataset.version,
            dataset_file=self._settings.evaluation_dataset_path,
            question_count=len(questions),
            top_k=top_k,
            embedding_model=self._embedding_model,
            llm_model=self._llm_model,
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
            keyword_coverage=mean([value for value in covered if value is not None]),
            abstention_rate=mean([1.0 if value else 0.0 for value in abstained]),
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


def _successful_question(item: EvalQuestion, response: RAGResponse) -> QuestionResult:
    retrieved = [_hit(rank, hit) for rank, hit in enumerate(response.retrieved_chunks, start=1)]
    document_ids = [hit.document_id for hit in retrieved]
    recall = None
    rank_score = None
    if item.expected_document_ids:
        recall = document_recall_at_k(document_ids, item.expected_document_ids)
        rank_score = reciprocal_rank_at_k(document_ids, item.expected_document_ids)
    phrase = None
    abstained = None
    if item.expects_abstention:
        phrase = abstention_phrase(response.answer)
        abstained = phrase is not None
    return QuestionResult(
        id=item.id,
        question=item.question,
        expects_abstention=item.expects_abstention,
        status="success",
        answer=response.answer,
        retrieved=retrieved,
        recall_at_k=recall,
        reciprocal_rank=rank_score,
        keyword_coverage=keyword_coverage(response.answer, item.expected_keywords),
        matched_keywords=matched_keywords(response.answer, item.expected_keywords),
        abstained=abstained,
        abstention_phrase=phrase,
        retrieval_latency_ms=response.metrics.retrieval_latency_ms,
        generation_latency_ms=response.metrics.generation_latency_ms,
        total_latency_ms=response.metrics.total_latency_ms,
        prompt_tokens=response.metrics.prompt_tokens,
        completion_tokens=response.metrics.completion_tokens,
    )


def _failed_question(item: EvalQuestion, exc: BaseException) -> QuestionResult:
    stage, category = classify_failure(exc)
    message = exc.message if isinstance(exc, AppError) else "评测题目失败"
    return QuestionResult(
        id=item.id,
        question=item.question,
        expects_abstention=item.expects_abstention,
        status="error",
        failure_stage=stage,
        error_category=category,
        error_type=type(exc).__name__,
        error_message=message,
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
        return EvaluationReport.model_validate(payload)
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
