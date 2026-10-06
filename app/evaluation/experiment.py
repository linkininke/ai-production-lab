"""在隔离集合上比较 vector、bm25、hybrid 和可选的 hybrid_rerank。

这次实验只测检索，不调用生成模型。没有重排配置时跳过 hybrid_rerank。
没有 Token 计数时 estimated_cost_usd 保持 null。
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.config import PROJECT_ROOT, Settings
from app.core.exceptions import AppError, EvaluationError
from app.embeddings.base import EmbeddingProvider
from app.evaluation.core.case import EvaluationCase
from app.evaluation.core.result import EvaluationResult, measure_retrieval
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.metrics import mean
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.base import Retriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import RerankingRetriever, require_production_reranker
from app.retrieval.tokenizer import JiebaTokenizer
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

_TOP_K = 10
_CUTOFFS = (1, 3, 5, 10)
_PRECISION_K = 5
_MRR_K = 5


class RetrieverMetrics(BaseModel):
    """一种检索器在整份数据集上的平均指标。数字只来自这次运行。"""

    retriever: str
    question_count: int
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    precision_at_5: float | None
    mrr_at_5: float | None
    average_latency_ms: float | None
    estimated_cost_usd: float | None = None
    embedding_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ExperimentReport(BaseModel):
    """一次检索对比的完整结果。"""

    dataset: str
    dataset_version: str
    question_count: int
    collection: str
    embedding_model: str
    semantic_embedding: bool
    note: str
    retrievers: list[RetrieverMetrics]
    skipped: list[str] = Field(default_factory=list)
    created_at: str


class _RerankerLike(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """按问题重排候选。"""


def run_retrieval_experiment(
    settings: Settings,
    dataset: EvaluationDataset,
    embedder: EmbeddingProvider,
    *,
    output_dir: Path,
    semantic_embedding: bool,
    note: str,
    reranker: _RerankerLike | None = None,
) -> ExperimentReport:
    """把评测语料写入临时集合，跑各检索器，并把报告写到 output_dir。"""
    with TemporaryDirectory(prefix="kb-eval-") as directory:
        store = ChromaVectorStore(
            persist_dir=Path(directory) / "chroma",
            collection_name="kb_eval",
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.dimension,
        )
        try:
            _index_corpus(settings, dataset, embedder, store)
            retrievers = _retrievers(settings, embedder, store)
            rows = [
                _measure(name, retriever, dataset.questions)
                for name, retriever in retrievers.items()
            ]
            skipped = _rerank_row(settings, retrievers["hybrid"], reranker, dataset.questions, rows)
        finally:
            store.close()
    report = ExperimentReport(
        dataset=str(settings.evaluation_dataset),
        dataset_version=dataset.version,
        question_count=len(dataset.questions),
        collection="kb_eval",
        embedding_model=embedder.model_name,
        semantic_embedding=semantic_embedding,
        note=note,
        retrievers=rows,
        skipped=skipped,
        created_at=datetime.now(UTC).isoformat(),
    )
    write_experiment(report, output_dir)
    return report


def latest_experiment_path() -> Path:
    return PROJECT_ROOT / "evaluation_results" / "latest.json"


def load_saved_experiment(path: Path) -> ExperimentReport:
    """读取已经写好的检索实验。文件不存在时停止，不编造指标。"""
    if not path.is_file():
        raise EvaluationError(
            "还没有检索实验报告。先运行 python scripts/evaluate.py。",
            status_code=404,
        )
    try:
        return ExperimentReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise EvaluationError("检索实验报告无法读取。") from exc


def write_experiment(report: ExperimentReport, directory: Path) -> Path:
    """写出 latest.json、带时间的 json 和 summary.md。"""
    directory.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump_json(indent=2)
    latest = directory / "latest.json"
    latest.write_text(payload + "\n", encoding="utf-8")
    stamp = report.created_at.replace(":", "").replace("+", "")
    (directory / f"experiment_{stamp}.json").write_text(payload + "\n", encoding="utf-8")
    (directory / "summary.md").write_text(render_summary(report), encoding="utf-8")
    return latest


def render_summary(report: ExperimentReport) -> str:
    """用这次报告里的数字生成摘要。摘要不预填指标。"""
    lines = [
        "# 检索评测摘要",
        "",
        report.note,
        "",
        f"- 数据集版本：{report.dataset_version}",
        f"- 问题数：{report.question_count}",
        f"- 向量模型：{report.embedding_model}",
        f"- 语义向量：{'是' if report.semantic_embedding else '否'}",
        _cost_line(report),
        "",
        (
            "| 检索器 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | "
            "Precision@5 | MRR@5 | 平均延迟(ms) | 成本(USD) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.retrievers:
        lines.append(
            f"| {item.retriever} | {_cell(item.recall_at_1)} | {_cell(item.recall_at_3)} | "
            f"{_cell(item.recall_at_5)} | {_cell(item.recall_at_10)} | "
            f"{_cell(item.precision_at_5)} | {_cell(item.mrr_at_5)} | "
            f"{_cell(item.average_latency_ms)} | {_cell(item.estimated_cost_usd)} |"
        )
    lines.extend(["", "## 跳过", ""])
    if report.skipped:
        lines.extend(f"- {item}" for item in report.skipped)
    else:
        lines.append("- 无")
    lines.append("")
    return "\n".join(lines)


def _index_corpus(
    settings: Settings,
    dataset: EvaluationDataset,
    embedder: EmbeddingProvider,
    store: ChromaVectorStore,
) -> None:
    directory = settings.evaluation_corpus
    if not directory.is_dir():
        raise EvaluationError(f"评测语料目录不存在：{directory.name}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt"}
    )
    if not paths:
        raise EvaluationError("评测语料目录里没有 Markdown 或 TXT")
    pipeline = IngestionPipeline(settings, embedder=embedder, vector_store=store)
    documents: set[str] = set()
    chunks: set[str] = set()
    for path in paths:
        result = pipeline.ingest_path(path, source_key=f"eval/{path.name}")
        documents.add(result.document.document_id)
        chunks.update(chunk.chunk_id for chunk in result.chunks)
    for question in dataset.questions:
        missing_documents = [
            item for item in question.expected_document_ids if item not in documents
        ]
        missing_chunks = [item for item in question.expected_chunk_ids if item not in chunks]
        if missing_documents or missing_chunks:
            raise EvaluationError(f"题目 {question.id} 的标注和语料不一致")


def _retrievers(
    settings: Settings,
    embedder: EmbeddingProvider,
    store: ChromaVectorStore,
) -> dict[str, Retriever]:
    vector = VectorRetriever(embedder, store, max_distance=settings.retrieval_max_distance)
    bm25 = BM25Retriever(store, JiebaTokenizer())
    hybrid = HybridRetriever(
        vector,
        bm25,
        vector_top_k=settings.hybrid_vector_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        final_top_k=_TOP_K,
        rrf_k=settings.rrf_k,
    )
    return {"vector": vector, "bm25": bm25, "hybrid": hybrid}


def _measure(name: str, retriever: Retriever, questions: list[EvaluationCase]) -> RetrieverMetrics:
    quality: list[EvaluationResult] = []
    latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        try:
            hits = retriever.retrieve(question.question, top_k=_TOP_K)
        except AppError:
            logger.warning(
                "retrieval_question_failed retriever=%s question_id=%s",
                name,
                question.id,
            )
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        if question.expects_abstention:
            continue
        quality.append(
            measure_retrieval(
                question,
                _ranked(hits, question),
                _labels(question),
                precision_k=_PRECISION_K,
                mrr_k=_MRR_K,
                cutoffs=_CUTOFFS,
                latency_ms=latency_ms,
            )
        )
    return RetrieverMetrics(
        retriever=name,
        question_count=len(questions),
        recall_at_1=_cutoff_mean(quality, 1),
        recall_at_3=_cutoff_mean(quality, 3),
        recall_at_5=_cutoff_mean(quality, 5),
        recall_at_10=_cutoff_mean(quality, 10),
        precision_at_5=mean(
            [item.precision_at_k for item in quality if item.precision_at_k is not None]
        ),
        mrr_at_5=mean(
            [item.reciprocal_rank for item in quality if item.reciprocal_rank is not None]
        ),
        average_latency_ms=mean(latencies),
    )


def _rerank_row(
    settings: Settings,
    hybrid: Retriever,
    reranker: _RerankerLike | None,
    questions: list[EvaluationCase],
    rows: list[RetrieverMetrics],
) -> list[str]:
    if not settings.reranker_enabled:
        return ["hybrid_rerank：RERANKER_ENABLED=false，未调用重排服务"]
    try:
        checked = require_production_reranker(reranker)  # type: ignore[arg-type]
    except AppError as exc:
        return [f"hybrid_rerank：{exc.message}"]
    wrapped = RerankingRetriever(
        hybrid,
        checked,
        candidate_top_k=settings.reranker_candidate_top_k,
        final_top_k=min(settings.reranker_final_top_k, _TOP_K),
    )
    rows.append(_measure("hybrid_rerank", wrapped, questions))
    return []


def _cutoff_mean(results: list[EvaluationResult], cutoff: int) -> float | None:
    return mean([item.recall_by_cutoff[cutoff] for item in results])


def _labels(question: EvaluationCase) -> list[str]:
    if question.expected_chunk_ids:
        return question.expected_chunk_ids
    return question.expected_document_ids


def _ranked(hits: list[RetrievalResult], question: EvaluationCase) -> list[str]:
    if question.expected_chunk_ids:
        return [hit.chunk_id for hit in hits]
    return [hit.document_id for hit in hits]


def _cost_line(report: ExperimentReport) -> str:
    if any(item.estimated_cost_usd is not None for item in report.retrievers):
        return "- estimated_cost_usd：见下表。缺 Token 或价格的检索器仍是 null。"
    return "- estimated_cost_usd：null（这次检索实验没有 Token 用量，不估算费用）"


def _cell(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"
