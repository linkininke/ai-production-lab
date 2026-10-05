"""一份已规范化的原始文档。

document_id 是业务身份，由导入时的 source_key 决定，不直接使用文件名。
同名文件只要 source_key 不同，就会得到不同的 document_id。
content_hash 只反映规范化之后的正文，用来判断内容有没有变化。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator


class Document(BaseModel):
    document_id: str
    filename: str
    content: str
    file_type: Literal["markdown", "text"]
    content_hash: str
    created_at: str

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_iso8601(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value
