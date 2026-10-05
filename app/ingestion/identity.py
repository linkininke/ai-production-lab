"""文档身份与内容指纹。

document_id 来自 source_key，同一份资料重复导入时身份不变。
content_hash 来自规范化后的正文，正文变了，哈希才会变。
chunk_id 同时绑定文档、序号和片段文本：同一输入反复切分，得到相同 ID。
"""

from __future__ import annotations

import hashlib

from app.core.exceptions import DocumentValidationError


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id_for(source_key: str) -> str:
    if not source_key.strip():
        raise DocumentValidationError("文档身份标识不能为空")
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def chunk_id_for(document_id: str, chunk_index: int, text: str) -> str:
    raw = f"{document_id}:{chunk_index}:{text}".encode()
    return "chunk_" + hashlib.sha256(raw).hexdigest()[:20]
