"""向量检索命中。

Chroma 适配器把 score 定义为余弦距离：0 表示方向相同，数值越大越远。
这不是相似度。相似度通常是 1 减去距离，方向和距离相反。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MetadataValue = str | int | float


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float = Field(description="余弦距离，越小越近")
    score_kind: Literal["distance"] = "distance"
    metadata: dict[str, MetadataValue]
