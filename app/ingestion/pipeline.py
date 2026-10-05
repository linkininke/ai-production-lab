"""把读取、清洗、切分和可选的向量写入串成一次导入。

未提供 Embedding 与向量库时，只在内存中返回片段。
提供两者时：内容与片段 ID 都没变就跳过向量化；变了就先写入新片段，再删除旧片段。
向量化失败时不会改动已有索引。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from app.chunking.models import Chunk
from app.chunking.splitter import CharacterTextSplitter, build_chunks
from app.config import Settings
from app.core.exceptions import (
    AppError,
    ConfigurationError,
    DocumentIngestionError,
    DocumentValidationError,
)
from app.embeddings.base import EmbeddingProvider
from app.ingestion.identity import content_hash, document_id_for
from app.ingestion.indexer import index_document
from app.ingestion.loader import LoadedFile, load_bytes, load_path
from app.ingestion.models import Document
from app.ingestion.normalizer import normalize_text
from app.ingestion.parser import parse_loaded
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    document: Document
    chunks: list[Chunk] = Field(default_factory=list)
    chunk_count: int
    elapsed_ms: float
    status: Literal["completed", "skipped"]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], datetime] | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        if (embedder is None) != (vector_store is None):
            raise ConfigurationError("写入向量库时必须同时提供 Embedding 和 VectorStore")
        self._settings = settings
        self._splitter = CharacterTextSplitter(settings.chunk_size, settings.chunk_overlap)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._embedder = embedder
        self._store = vector_store

    def ingest_path(self, path: Path, *, source_key: str | None = None) -> IngestionResult:
        started = perf_counter()
        try:
            loaded = load_path(
                path,
                max_bytes=self._max_bytes(),
                source_key=source_key,
            )
            return self._complete(loaded.filename, loaded, started)
        except AppError:
            logger.warning("ingestion_failed filename=%s", path.name)
            raise
        except OSError as exc:
            logger.warning("ingestion_failed filename=%s", path.name)
            raise DocumentIngestionError("读取文档失败") from exc

    def ingest_bytes(self, filename: str, data: bytes, *, source_key: str) -> IngestionResult:
        started = perf_counter()
        try:
            loaded = load_bytes(
                filename,
                data,
                max_bytes=self._max_bytes(),
                source_key=source_key,
            )
            return self._complete(loaded.filename, loaded, started)
        except AppError:
            logger.warning("ingestion_failed filename=%s", Path(filename).name)
            raise

    def _complete(self, filename: str, loaded: LoadedFile, started: float) -> IngestionResult:
        parsed = parse_loaded(loaded)
        normalized = normalize_text(parsed.text)
        if not normalized:
            raise DocumentValidationError("文档去掉空白后没有内容")
        document = Document(
            document_id=document_id_for(parsed.source_key),
            filename=parsed.filename,
            content=normalized,
            file_type=parsed.file_type,
            content_hash=content_hash(normalized),
            created_at=self._created_at(),
        )
        try:
            chunks = build_chunks(document, self._splitter.split(normalized))
        except ValueError as exc:
            raise DocumentIngestionError("生成文档片段失败") from exc
        status: Literal["completed", "skipped"] = "completed"
        if self._embedder is not None and self._store is not None:
            chunks, status = index_document(document, chunks, self._embedder, self._store)
        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "ingestion_completed filename=%s document_id=%s chunk_count=%s "
            "status=%s elapsed_ms=%.1f",
            filename,
            document.document_id,
            len(chunks),
            status,
            elapsed_ms,
        )
        return IngestionResult(
            document=document,
            chunks=chunks,
            chunk_count=len(chunks),
            elapsed_ms=elapsed_ms,
            status=status,
        )

    def _max_bytes(self) -> int:
        return self._settings.max_upload_size_mb * 1024 * 1024

    def _created_at(self) -> str:
        current = self._clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return current.isoformat()
