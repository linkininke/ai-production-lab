"""检索器接口。

向量检索、关键词检索和混合检索实现同一组方法。
混合检索按名次融合，调用方不要比较或相加 distance 和 bm25。
"""

from __future__ import annotations

from typing import Protocol

from app.retrieval.models import RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """返回最相关的若干片段，相关的排在前面。"""
