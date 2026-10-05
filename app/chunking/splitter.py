"""按字符数切分，并尽量避开句子中间。

长度使用 Python 的 len()，也就是 Unicode 码位，不是模型 Token。
「事务」是 2 个字符，在某个 Embedding 模型里可能是 1 个或更多 Token。
CHUNK_SIZE=800 只表示大约 800 个字符，不能当成 800 Token 的预算。

相邻块共享一段原文。共享长度由 CHUNK_OVERLAP 决定。
这样句子刚好落在边界上时，前后两块都还看得到这句话的一部分。

Markdown 标题写入 metadata["heading"]，不拼进正文。
正文保持原文切片，重叠字符才能和原文逐字对照。
Phase 4 组装上下文时再把标题加回去。
代码围栏里的 # 行不当成标题。
"""

from __future__ import annotations

import re

from app.chunking.models import Chunk
from app.chunking.strategies import TextSpan
from app.core.exceptions import ConfigurationError, DocumentIngestionError
from app.ingestion.identity import chunk_id_for
from app.ingestion.models import Document

_HEADING_LINE = re.compile(r"(#{1,6})[ \t]+(\S.*)$")
_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "!", "?", "；", ";", "，", ",", "、", " ")


class CharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size < 1:
            raise ConfigurationError("CHUNK_SIZE 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP 必须大于等于 0，并且小于 CHUNK_SIZE")
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    def split(self, text: str) -> list[TextSpan]:
        if text == "":
            return []
        spans: list[TextSpan] = []
        start = 0
        length = len(text)
        while start < length:
            end = self._window_end(text, start, length)
            if end <= start:
                raise DocumentIngestionError("切分没有前进")
            spans.append(
                TextSpan(
                    text=text[start:end],
                    start=start,
                    end=end,
                    heading=active_heading(text, start),
                )
            )
            if end >= length:
                break
            next_start = end - self._overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return spans

    def _window_end(self, text: str, start: int, length: int) -> int:
        window_end = min(start + self._chunk_size, length)
        if window_end >= length:
            return length
        earliest = start + max(1, (window_end - start) // 2)
        for separator in _SEPARATORS:
            found = text.rfind(separator, earliest, window_end)
            if found != -1:
                return found + len(separator)
        return window_end


def active_heading(text: str, position: int) -> str | None:
    """返回 position 之前最近的 ATX 标题。围栏代码里的 # 忽略。"""
    found: str | None = None
    for offset, heading in _iter_headings(text):
        if offset > position:
            break
        found = heading
    return found


def build_chunks(document: Document, spans: list[TextSpan]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, span in enumerate(spans):
        metadata: dict[str, str | int] = {
            "filename": document.filename,
            "file_type": document.file_type,
            "chunk_index": index,
            "document_id": document.document_id,
            "char_start": span.start,
            "char_end": span.end,
        }
        if span.heading:
            metadata["heading"] = span.heading[:200]
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(document.document_id, index, span.text),
                document_id=document.document_id,
                text=span.text,
                chunk_index=index,
                metadata=metadata,
            )
        )
    return chunks


def _iter_headings(text: str):
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_LINE.match(line.rstrip("\r\n"))
            if match:
                yield offset, line.rstrip("\r\n").strip()
        offset += len(line)
