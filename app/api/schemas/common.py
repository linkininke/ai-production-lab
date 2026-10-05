"""跨接口共用的响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthServices(BaseModel):
    api: str = Field(description="API 进程状态")
    vector_store: str = Field(
        description="本地向量库能否打开，已有索引是否与当前 Embedding 配置一致"
    )


class HealthResponse(BaseModel):
    status: str = Field(
        description="ok 表示依赖可用；degraded 表示进程存活但向量库不可用或索引配置不兼容"
    )
    services: HealthServices
