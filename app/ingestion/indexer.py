"""把已经切好的片段写入向量库。

先计算向量，再 upsert 新片段，最后删除不再使用的旧片段。
向量化或写入失败时，旧索引还在。
如果进程在删除旧片段之前退出，同一文档会暂时留下两版片段；
下一次导入会发现 chunk_id 对不上，再次写入并清掉旧 ID。
"""

from __future__ import annotations

from typing import Literal

from app.chunking.models import Chunk
from app.core.exceptions import ConfigurationError
from app.embeddings.base import EmbeddingProvider
from app.ingestion.models import Document
from app.vectorstore.base import VectorStore

IndexStatus = Literal["completed", "skipped"]


def index_document(
    document: Document,
    chunks: list[Chunk],
    embedder: EmbeddingProvider,
    store: VectorStore,
) -> tuple[list[Chunk], IndexStatus]:
    if embedder.dimension != store.dimension:
        raise ConfigurationError(
            f"Embedding 维度与向量库不一致：{embedder.dimension} 对 {store.dimension}"
        )
    if embedder.model_name != store.embedding_model:
        raise ConfigurationError(
            f"Embedding 模型与向量库不一致：{embedder.model_name} 对 {store.embedding_model}"
        )
    existing = store.get_by_document_id(document.document_id)
    if _is_current(existing, chunks, document.content_hash):
        existing.sort(key=lambda chunk: chunk.chunk_index)
        return existing, "skipped"
    stored = _prepare_stored_chunks(chunks, document)
    vectors = embedder.embed_documents([chunk.text for chunk in stored])
    store.add_chunks(stored, vectors)
    fresh_ids = {chunk.chunk_id for chunk in stored}
    stale = [chunk.chunk_id for chunk in existing if chunk.chunk_id not in fresh_ids]
    if stale:
        store.delete_by_ids(stale)
    return stored, "completed"


def _is_current(existing: list[Chunk], chunks: list[Chunk], content_hash: str) -> bool:
    if not existing or len(existing) != len(chunks):
        return False
    if {chunk.chunk_id for chunk in existing} != {chunk.chunk_id for chunk in chunks}:
        return False
    return all(chunk.metadata.get("content_hash") == content_hash for chunk in existing)


def _prepare_stored_chunks(chunks: list[Chunk], document: Document) -> list[Chunk]:
    stored: list[Chunk] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["content_hash"] = document.content_hash
        metadata["created_at"] = document.created_at
        stored.append(chunk.model_copy(update={"metadata": metadata}))
    return stored
