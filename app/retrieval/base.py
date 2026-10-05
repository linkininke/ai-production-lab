"""检索器接口。

向量检索是第一版的唯一实现。以后的混合检索实现同一组方法即可，
调用方不用知道分数是怎么算出来的。
"""

from __future__ import annotations

from typing import Protocol

from app.retrieval.models import RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """返回最接近问题的若干片段。顺序是由近到远。"""
