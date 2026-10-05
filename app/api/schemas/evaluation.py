"""评测请求。报告正文使用评测模块里的模型，避免两套字段。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationRunRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1)


class CompareRequest(BaseModel):
    left: str
    right: str
