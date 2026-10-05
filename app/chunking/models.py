"""切分后的文本片段。"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator

MetadataValue = str | int


class Chunk(BaseModel):
    """一个可单独检索的片段。

    metadata 只放定位信息，不放整篇原文。整篇原文留在 Document.content。
    """

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, MetadataValue]

    @model_validator(mode="after")
    def metadata_matches_chunk(self) -> Self:
        required = ("filename", "file_type", "chunk_index", "document_id")
        missing = [key for key in required if key not in self.metadata]
        if missing:
            raise ValueError("chunk metadata 缺少 " + ", ".join(missing))
        if self.metadata["document_id"] != self.document_id:
            raise ValueError("metadata.document_id 与 chunk.document_id 不一致")
        if self.metadata["chunk_index"] != self.chunk_index:
            raise ValueError("metadata.chunk_index 与 chunk.chunk_index 不一致")
        for value in self.metadata.values():
            if isinstance(value, str) and len(value) > 500:
                raise ValueError("metadata 不能写入大段原文")
        return self
