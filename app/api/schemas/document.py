"""文档上传、列表和删除的响应。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: Literal["completed", "skipped"]
    chunk_count: int


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    created_at: str = Field(description="首次写入索引的时间，ISO 8601")
    chunk_count: int
    status: Literal["completed"] = "completed"


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted_chunks: int
