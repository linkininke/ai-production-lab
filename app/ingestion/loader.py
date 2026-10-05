"""读取本地文件或上传字节，并检查类型和大小。

这里只接受调用方已经确定的路径或原始字节。
文件名只作为展示名，不能带目录，避免以后的上传接口把用户输入拼进文件系统路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.exceptions import DocumentIngestionError, DocumentValidationError

FileType = Literal["markdown", "text"]

_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_TEXT_SUFFIXES = {".txt"}


@dataclass(frozen=True)
class LoadedFile:
    filename: str
    file_type: FileType
    data: bytes
    source_key: str


def load_path(path: Path, *, max_bytes: int, source_key: str | None = None) -> LoadedFile:
    if not path.is_file():
        raise DocumentValidationError("路径不是可读文件")
    filename = path.name
    file_type = _file_type_for(filename)
    size = path.stat().st_size
    _ensure_size(size, max_bytes)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DocumentIngestionError("读取文件失败") from exc
    _ensure_size(len(data), max_bytes)
    key = source_key if source_key is not None else path.resolve().as_posix()
    return LoadedFile(filename=filename, file_type=file_type, data=data, source_key=key)


def load_bytes(
    filename: str,
    data: bytes,
    *,
    max_bytes: int,
    source_key: str,
) -> LoadedFile:
    safe_name = _display_filename(filename)
    file_type = _file_type_for(safe_name)
    _ensure_size(len(data), max_bytes)
    return LoadedFile(
        filename=safe_name,
        file_type=file_type,
        data=data,
        source_key=source_key,
    )


def _display_filename(filename: str) -> str:
    if not filename or filename in {".", ".."}:
        raise DocumentValidationError("文件名不合法")
    if "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise DocumentValidationError("文件名不能包含路径")
    return filename


def _file_type_for(filename: str) -> FileType:
    suffix = Path(filename).suffix.lower()
    if suffix in _MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in _TEXT_SUFFIXES:
        return "text"
    raise DocumentValidationError("只支持 Markdown 和 TXT 文件")


def _ensure_size(size: int, max_bytes: int) -> None:
    if size <= 0:
        raise DocumentValidationError("文件内容为空")
    if size > max_bytes:
        raise DocumentValidationError(f"文件超过大小限制（最大 {max_bytes} 字节）")
