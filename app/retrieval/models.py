"""检索命中。

向量检索的 score 是余弦距离：0 表示方向相同，数值越大越远。这不是相似度。
BM25 的 score 是关键词相关度，数值越大越相关。两种分数方向相反，不能相加。
混合检索的 score 是 RRF 名次分，只由各路名次算出，不使用上面两种原始分数。
重排的 score 是重排服务返回的相关度，越大越靠前。MockReranker 不写这个分数。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MetadataValue = str | int | float
ScoreKind = Literal["distance", "bm25", "rrf", "rerank"]
RetrieverName = Literal["vector", "bm25", "hybrid"]
RerankerName = Literal["mock", "openai_compatible"]


class RetrievalSource(BaseModel):
    """某个检索器给出的原始名次。rank 从 1 开始，1 是该路的第一名。"""

    retriever: RetrieverName
    rank: int = Field(ge=1)
    score: float
    score_kind: ScoreKind


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float = Field(description="原始分数。distance 越小越近；bm25、rrf 和 rerank 越大越靠前")
    score_kind: ScoreKind = "distance"
    retriever: RetrieverName = "vector"
    reranker: RerankerName | None = None
    sources: list[RetrievalSource] = Field(default_factory=list)
    metadata: dict[str, MetadataValue]
