"""向量库接口。业务代码只依赖这组方法，不直接调用 Chroma。"""

from __future__ import annotations

from typing import Protocol

from app.chunking.models import Chunk
from app.retrieval.models import RetrievalResult
from app.vectorstore.models import IndexedDocument


class VectorStore(Protocol):
    @property
    def embedding_model(self) -> str:
        """建索引时使用的 Embedding 模型名。"""

    @property
    def dimension(self) -> int:
        """索引接受的向量维度。"""

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """写入或覆盖同一批 chunk_id。"""

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """按距离从近到远返回。空库返回空列表。"""

    def delete_by_document_id(self, document_id: str) -> int:
        """删除一份文档的全部片段，返回删除数量。"""

    def delete_by_ids(self, chunk_ids: list[str]) -> int:
        """按片段 ID 删除。更新文档时用来清掉不再使用的旧片段。"""

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        """读取一份文档的全部片段，按 chunk_index 排序。"""

    def list_chunks(self) -> list[Chunk]:
        """读取全部片段。BM25 用它重建关键词索引，不另存一份正文。"""

    def list_documents(self) -> list[IndexedDocument]:
        """按文档汇总索引。只返回身份和片段数量，不返回正文。"""

    def close(self) -> None:
        """释放本地数据库连接。"""
