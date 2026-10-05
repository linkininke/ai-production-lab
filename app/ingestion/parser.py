"""把字节解码成文本。

Markdown 保持原文，不转成 HTML。检索需要看到标题符号和代码块本身。
编码先按 UTF-8（可带 BOM），失败再试 GB18030。GB18030 能覆盖常见的 GBK 文本。
两种都失败就拒绝，不用 latin-1 硬解，否则坏字节也会变成看起来正常的乱码。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.exceptions import DocumentValidationError
from app.ingestion.loader import FileType, LoadedFile

_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class ParsedText:
    filename: str
    file_type: FileType
    source_key: str
    text: str


def parse_loaded(loaded: LoadedFile) -> ParsedText:
    return ParsedText(
        filename=loaded.filename,
        file_type=loaded.file_type,
        source_key=loaded.source_key,
        text=decode_text(loaded.data),
    )


def decode_text(data: bytes) -> str:
    payload = data
    if payload.startswith(_UTF8_BOM):
        payload = payload[len(_UTF8_BOM) :]
        return _decode(payload, "utf-8", "文件不是有效的 UTF-8 文本")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return _decode(payload, "gb18030", "无法识别文件编码，请使用 UTF-8 或 GB18030")


def _decode(payload: bytes, encoding: str, message: str) -> str:
    try:
        return payload.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DocumentValidationError(message) from exc
