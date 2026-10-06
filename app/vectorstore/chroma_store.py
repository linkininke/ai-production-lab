"""ChromaDB 持久化适配器。

集合元数据记下 embedding_model、embedding_dimension 和 hnsw:space。
打开已有集合时，这三项必须和当前配置一致，否则拒绝写入。
距离使用余弦距离。score 越小越近，不要把它当成相似度。

本模块在导入 Chroma 之前关闭匿名遥测，避免健康检查访问外网。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.api import ClientAPI  # noqa: E402
from chromadb.api.models.Collection import Collection  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from chromadb.errors import NotFoundError  # noqa: E402

from app.chunking.models import Chunk
from app.core.exceptions import VectorStoreError
from app.retrieval.models import RetrievalResult
from app.vectorstore.models import IndexedDocument

logger = logging.getLogger(__name__)

_MODEL_KEY = "embedding_model"
_DIMENSION_KEY = "embedding_dimension"
_SPACE = "cosine"
_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")


class ChromaVectorStore:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        if not _COLLECTION_NAME.fullmatch(collection_name.strip()):
            raise VectorStoreError(
                "集合名须为 3 到 512 个字符，只能包含字母、数字、点、下划线和短横线，"
                "并且以字母或数字开头和结尾"
            )
        if embedding_dimension < 1:
            raise VectorStoreError("向量维度必须是正整数")
        if not embedding_model.strip():
            raise VectorStoreError("Embedding 模型名不能为空")
        self._persist_dir = persist_dir
        self._collection_name = collection_name.strip()
        self._model = embedding_model.strip()
        self._dimension = embedding_dimension
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None
        try:
            self._client = _open_client(persist_dir)
            self._collection = self._open_collection(self._client)
        except VectorStoreError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise VectorStoreError(f"打开向量库失败：{exc}") from exc

    @property
    def embedding_model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("片段数量与向量数量不一致")
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise VectorStoreError("同一批写入中存在重复的 chunk_id")
        for embedding in embeddings:
            _check_dimension(embedding, self._dimension)
        collection = self._require_collection()
        batch_size = _batch_size(self._client)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embeddings[start : start + batch_size]
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=vectors,
                documents=[chunk.text for chunk in batch],
                metadatas=[dict(chunk.metadata) for chunk in batch],
            )
        logger.info(
            "vector_upsert collection=%s count=%s",
            self._collection_name,
            len(chunks),
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        if top_k < 1:
            raise VectorStoreError("top_k 必须大于 0")
        _check_dimension(query_embedding, self._dimension)
        collection = self._require_collection()
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        if not ids or not ids[0]:
            return []
        rows = _aligned_rows(ids[0], documents[0], metadatas[0], distances[0])
        hits: list[RetrievalResult] = []
        for chunk_id, text, metadata, distance in rows:
            if text is None or distance is None or not isinstance(metadata, dict):
                raise VectorStoreError("检索结果缺少文本或距离")
            stored = _metadata_map(metadata)
            hits.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    document_id=str(stored.get("document_id", "")),
                    text=str(text),
                    score=float(distance),
                    score_kind="distance",
                    metadata=stored,
                )
            )
        return hits

    def delete_by_document_id(self, document_id: str) -> int:
        if not document_id:
            raise VectorStoreError("document_id 不能为空")
        deleted = self._require_collection().delete(where={"document_id": document_id})
        count = int(deleted["deleted"])
        logger.info(
            "vector_delete_document collection=%s document_id=%s deleted=%s",
            self._collection_name,
            document_id,
            count,
        )
        return count

    def delete_by_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        deleted = self._require_collection().delete(ids=chunk_ids)
        return int(deleted["deleted"])

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        if not document_id:
            raise VectorStoreError("document_id 不能为空")
        result = self._require_collection().get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        return _chunks_from_get(result)

    def list_chunks(self) -> list[Chunk]:
        collection = self._require_collection()
        if collection.count() == 0:
            return []
        result = collection.get(include=["documents", "metadatas"])
        chunks = _chunks_from_get(result)
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.chunk_index, chunk.chunk_id))
        return chunks

    def list_documents(self) -> list[IndexedDocument]:
        collection = self._require_collection()
        if collection.count() == 0:
            return []
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        grouped: dict[str, list[dict[str, object]]] = {}
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                raise VectorStoreError("向量记录缺少元数据")
            document_id = str(metadata.get("document_id", "")).strip()
            if not document_id:
                raise VectorStoreError("向量记录缺少 document_id")
            grouped.setdefault(document_id, []).append(metadata)
        documents = [_summarize(document_id, rows) for document_id, rows in grouped.items()]
        documents.sort(key=lambda item: item.created_at, reverse=True)
        return documents

    def close(self) -> None:
        client = self._client
        self._client = None
        self._collection = None
        if client is not None:
            client.close()

    def _require_collection(self) -> Collection:
        if self._collection is None:
            raise VectorStoreError("向量库已经关闭")
        return self._collection

    def _open_collection(self, client: ClientAPI) -> Collection:
        try:
            collection = client.get_collection(
                self._collection_name,
                embedding_function=None,
            )
        except NotFoundError:
            return client.create_collection(
                self._collection_name,
                metadata={
                    "hnsw:space": _SPACE,
                    _MODEL_KEY: self._model,
                    _DIMENSION_KEY: self._dimension,
                },
                embedding_function=None,
            )
        self._ensure_compatible(collection.metadata or {})
        return collection

    def _ensure_compatible(self, metadata: dict[str, object]) -> None:
        stored_model = str(metadata.get(_MODEL_KEY, ""))
        stored_dimension = _optional_int(metadata.get(_DIMENSION_KEY))
        stored_space = str(metadata.get("hnsw:space", ""))
        if (
            stored_model != self._model
            or stored_dimension != self._dimension
            or stored_space not in {"", _SPACE}
        ):
            raise VectorStoreError(
                "索引与当前 Embedding 配置不兼容。"
                f"索引中是 model={stored_model or '未知'}，"
                f"dimension={stored_dimension}，space={stored_space or '未知'}；"
                f"当前是 model={self._model}，dimension={self._dimension}，space={_SPACE}。"
                "请更换集合名，或删除该集合后重建索引。"
            )


def probe_vector_store(
    persist_dir: Path,
    *,
    collection_name: str,
    embedding_model: str = "",
    embedding_dimension: int | None = None,
) -> str:
    """打开本地库并核对已有集合的模型签名。不计算 Embedding，也不创建集合。"""
    if persist_dir.exists() and not persist_dir.is_dir():
        logger.warning("vector_store_path_not_directory path=%s", persist_dir)
        return "error"
    client: ClientAPI | None = None
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = _open_client(persist_dir)
        try:
            collection = client.get_collection(collection_name, embedding_function=None)
        except NotFoundError:
            return "ok"
        metadata = collection.metadata or {}
        if embedding_model and str(metadata.get(_MODEL_KEY, "")) not in {"", embedding_model}:
            return "error"
        stored_dimension = metadata.get(_DIMENSION_KEY)
        if (
            embedding_dimension is not None
            and stored_dimension is not None
            and int(stored_dimension) != embedding_dimension
        ):
            return "error"
        return "ok"
    except Exception:
        logger.warning("vector_store_unavailable path=%s", persist_dir)
        return "error"
    finally:
        if client is not None:
            client.close()


def _batch_size(client: ClientAPI | None) -> int:
    if client is None:
        return 100
    try:
        return max(1, int(client.get_max_batch_size()))
    except Exception:
        return 100


def _chunks_from_get(result: dict[str, object]) -> list[Chunk]:
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    columns = (ids, documents, metadatas)
    if not all(isinstance(column, list) for column in columns):
        raise VectorStoreError("向量记录字段数量不一致")
    if not (len(ids) == len(documents) == len(metadatas)):
        raise VectorStoreError("向量记录字段数量不一致")
    chunks: list[Chunk] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
        if text is None or not isinstance(metadata, dict):
            raise VectorStoreError("向量记录缺少文本或元数据")
        stored = _metadata_map(metadata)
        missing = [
            key
            for key in ("filename", "file_type", "chunk_index", "document_id")
            if key not in stored
        ]
        if missing:
            raise VectorStoreError("向量记录缺少元数据 " + ", ".join(missing))
        chunks.append(
            Chunk(
                chunk_id=str(chunk_id),
                document_id=str(stored["document_id"]),
                text=str(text),
                chunk_index=int(stored["chunk_index"]),
                metadata=stored,
            )
        )
    chunks.sort(key=lambda chunk: chunk.chunk_index)
    return chunks


def _aligned_rows(
    ids: list[object],
    documents: list[object],
    metadatas: list[object],
    distances: list[object],
) -> list[tuple[object, object, object, object]]:
    width = len(ids)
    if not all(len(column) == width for column in (documents, metadatas, distances)):
        raise VectorStoreError("检索结果字段数量不一致")
    return list(zip(ids, documents, metadatas, distances, strict=True))


def _summarize(document_id: str, rows: list[dict[str, object]]) -> IndexedDocument:
    primary = min(rows, key=_chunk_index)
    filename = primary.get("filename", "")
    created_at = primary.get("created_at", "")
    return IndexedDocument(
        document_id=document_id,
        filename=filename if isinstance(filename, str) else str(filename),
        created_at=created_at if isinstance(created_at, str) else "",
        chunk_count=len(rows),
    )


def _chunk_index(metadata: dict[str, object]) -> int:
    value = _optional_int(metadata.get("chunk_index"))
    return 0 if value is None else value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _open_client(persist_dir: Path) -> ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _check_dimension(vector: list[float], expected: int) -> None:
    if len(vector) != expected:
        raise VectorStoreError(f"向量维度是 {len(vector)}，索引要求 {expected}")


def _metadata_map(metadata: dict[str, object]) -> dict[str, str | int]:
    mapped: dict[str, str | int] = {}
    for key, value in metadata.items():
        mapped[str(key)] = _metadata_value(value)
    return mapped


def _metadata_value(value: object) -> str | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return str(value)
