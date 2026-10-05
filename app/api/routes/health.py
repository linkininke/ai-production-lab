"""健康检查。

这里确认 API 进程，并打开本地 Chroma 目录。
已有集合时，会核对 Embedding 模型名和维度。
不调用大模型，也不计算向量。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.schemas.common import HealthResponse, HealthServices
from app.config import Settings
from app.core.dependencies import get_settings_from_app
from app.vectorstore.chroma_store import probe_vector_store

router = APIRouter(tags=["health"])


def vector_store_status(persist_dir: Path, *, collection_name: str = "knowledge_base") -> str:
    """打开本地向量库。集合还不存在时也视为可写，不会因此创建集合。"""
    return probe_vector_store(persist_dir, collection_name=collection_name)


@router.get("/health", response_model=HealthResponse)
def read_health(
    settings: Settings = Depends(get_settings_from_app),
) -> HealthResponse:
    store_status = probe_vector_store(
        settings.chroma_path,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
    )
    status = "ok" if store_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        services=HealthServices(api="ok", vector_store=store_status),
    )
