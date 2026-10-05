"""文档上传、列表和删除。

上传的字节只在内存中进入导入流程，不按用户提供的文件名落盘。
文档身份使用 upload/{文件名}。同名文件再次上传会更新同一份文档。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.schemas.document import (
    DeleteDocumentResponse,
    DocumentItem,
    DocumentListResponse,
    UploadResponse,
)
from app.core.container import AppContainer
from app.core.dependencies import get_container
from app.core.exceptions import DocumentNotFoundError, DocumentValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_BLOCK = 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    container: AppContainer = Depends(get_container),
) -> UploadResponse:
    filename = _bare_filename(file.filename)
    data = _read_limited(file, container.settings.max_upload_size_mb * 1024 * 1024)
    result = container.require_ingestion().ingest_bytes(
        filename,
        data,
        source_key=f"upload/{filename}",
    )
    logger.info(
        "document_uploaded filename=%s document_id=%s status=%s chunk_count=%s",
        result.document.filename,
        result.document.document_id,
        result.status,
        result.chunk_count,
    )
    return UploadResponse(
        document_id=result.document.document_id,
        filename=result.document.filename,
        status=result.status,
        chunk_count=result.chunk_count,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    container: AppContainer = Depends(get_container),
) -> DocumentListResponse:
    documents = container.require_store().list_documents()
    return DocumentListResponse(
        documents=[
            DocumentItem(
                document_id=item.document_id,
                filename=item.filename,
                created_at=item.created_at,
                chunk_count=item.chunk_count,
            )
            for item in documents
        ]
    )


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    container: AppContainer = Depends(get_container),
) -> DeleteDocumentResponse:
    deleted = container.require_store().delete_by_document_id(document_id)
    if deleted == 0:
        raise DocumentNotFoundError("文档不存在或已经删除")
    logger.info("document_deleted document_id=%s deleted_chunks=%s", document_id, deleted)
    return DeleteDocumentResponse(document_id=document_id, deleted_chunks=deleted)


def _bare_filename(filename: str | None) -> str:
    if filename is None:
        raise DocumentValidationError("文件名不合法")
    name = filename.strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name or Path(name).name != name:
        raise DocumentValidationError("文件名不能包含路径")
    return name


def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    buffer = bytearray()
    while True:
        block = file.file.read(_READ_BLOCK)
        if not block:
            break
        if len(buffer) + len(block) > max_bytes:
            raise DocumentValidationError(f"文件超过大小限制（最大 {max_bytes} 字节）")
        buffer.extend(block)
    return bytes(buffer)
