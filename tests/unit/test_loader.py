"""文件读取与类型、大小校验。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import DocumentValidationError
from app.ingestion.loader import load_bytes, load_path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_reads_markdown_fixture() -> None:
    loaded = load_path(
        FIXTURES / "spring_transaction.md",
        max_bytes=1024 * 1024,
        source_key="fixtures/spring_transaction.md",
    )
    assert loaded.filename == "spring_transaction.md"
    assert loaded.file_type == "markdown"
    assert "事务".encode() in loaded.data


def test_reads_txt_fixture() -> None:
    loaded = load_path(FIXTURES / "notes.txt", max_bytes=1024 * 1024, source_key="notes")
    assert loaded.file_type == "text"
    assert loaded.filename == "notes.txt"


def test_rejects_unsupported_type(tmp_path: Path) -> None:
    target = tmp_path / "notes.pdf"
    target.write_text("pdf", encoding="utf-8")
    with pytest.raises(DocumentValidationError, match="Markdown"):
        load_path(target, max_bytes=1000)


def test_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(DocumentValidationError, match="不是可读文件"):
        load_path(tmp_path, max_bytes=1000)


def test_rejects_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.md"
    target.write_bytes(b"")
    with pytest.raises(DocumentValidationError, match="为空"):
        load_path(target, max_bytes=1000)


def test_rejects_oversize_bytes() -> None:
    with pytest.raises(DocumentValidationError, match="大小限制"):
        load_bytes("a.txt", b"abcdef", max_bytes=4, source_key="a")


def test_rejects_filename_with_path() -> None:
    with pytest.raises(DocumentValidationError, match="不能包含路径"):
        load_bytes(r"..\secret.md", b"abc", max_bytes=100, source_key="secret")


def test_same_filename_can_use_different_source_keys() -> None:
    left = load_bytes("note.md", b"left", max_bytes=100, source_key="kb/a")
    right = load_bytes("note.md", b"right", max_bytes=100, source_key="kb/b")
    assert left.filename == right.filename
    assert left.source_key != right.source_key
