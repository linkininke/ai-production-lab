"""向量库里能直接汇总出来的文档信息。"""

from __future__ import annotations

from pydantic import BaseModel


class IndexedDocument(BaseModel):
    document_id: str
    filename: str
    created_at: str
    chunk_count: int
