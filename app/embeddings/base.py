"""向量化接口。

查询和文档必须使用同一个模型和同一种向量维度。
更换模型后要重建索引，不能把两套向量放进同一个集合。
本地模型以后实现同一组方法即可，业务代码不用改。
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str:
        """当前向量模型的名字，会写进索引元数据。"""

    @property
    def dimension(self) -> int:
        """每条向量的长度。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批文档片段变成向量。返回顺序与 texts 一致。"""

    def embed_query(self, text: str) -> list[float]:
        """把用户问题变成向量。必须和 embed_documents 使用同一模型。"""
