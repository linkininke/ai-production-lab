"""测试用的确定性向量。

它只保证同一段文本得到同一条向量，没有语义。
生产环境的向量来自 EmbeddingProvider 的真实实现。
"""

from __future__ import annotations

import hashlib

from app.core.exceptions import EmbeddingError


class HashEmbedding:
    def __init__(self, dimension: int = 8, model_name: str = "fake-hash") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.document_calls = 0
        self.fail = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise EmbeddingError("模拟向量化失败")
        if any(not text.strip() for text in texts):
            raise EmbeddingError("存在空文本，无法向量化")
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def _vector(self, text: str) -> list[float]:
        material = hashlib.sha256(text.encode()).digest()
        values: list[float] = []
        while len(values) < self._dimension:
            material = hashlib.sha256(material).digest()
            for byte in material:
                values.append(byte / 127.5 - 1.0)
                if len(values) == self._dimension:
                    break
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]
