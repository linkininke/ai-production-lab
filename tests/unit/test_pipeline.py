"""导入流程：读取、清洗、切分和稳定身份。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.exceptions import DocumentValidationError
from app.ingestion.identity import content_hash, document_id_for
from app.ingestion.pipeline import IngestionPipeline
from tests.helpers import make_settings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _pipeline() -> IngestionPipeline:
    moment = datetime(2026, 10, 5, 10, 0, tzinfo=UTC)
    return IngestionPipeline(
        make_settings(chunk_size=40, chunk_overlap=8),
        clock=lambda: moment,
    )


def test_ingests_fixture_markdown() -> None:
    result = _pipeline().ingest_path(
        FIXTURES / "spring_transaction.md",
        source_key="fixtures/spring_transaction.md",
    )
    assert result.status == "completed"
    assert result.chunk_count == len(result.chunks)
    assert result.chunk_count > 1
    assert result.document.file_type == "markdown"
    assert result.document.filename == "spring_transaction.md"
    assert result.document.document_id == document_id_for("fixtures/spring_transaction.md")
    assert result.document.content_hash == content_hash(result.document.content)
    datetime.fromisoformat(result.document.created_at)
    assert result.elapsed_ms >= 0
    for chunk in result.chunks:
        assert chunk.text
        assert (
            chunk.text
            == result.document.content[chunk.metadata["char_start"] : chunk.metadata["char_end"]]
        )
        assert chunk.metadata["filename"] == "spring_transaction.md"
        assert len(chunk.text) <= 40


def test_short_note_is_a_single_chunk() -> None:
    result = IngestionPipeline(make_settings()).ingest_path(
        FIXTURES / "notes.txt",
        source_key="fixtures/notes.txt",
    )
    assert result.document.file_type == "text"
    assert result.chunk_count == 1
    assert "Token" in result.chunks[0].text


def test_repeated_import_keeps_ids() -> None:
    pipeline = _pipeline()
    data = "事务通过代理生效。自调用会绕过代理。".encode()
    first = pipeline.ingest_bytes("spring.md", data, source_key="kb/spring")
    second = pipeline.ingest_bytes("spring.md", data, source_key="kb/spring")
    assert first.document.document_id == second.document.document_id
    assert first.document.content_hash == second.document.content_hash
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]


def test_content_change_keeps_document_id_and_changes_hash() -> None:
    pipeline = _pipeline()
    first = pipeline.ingest_bytes(
        "spring.md",
        "版本一的事务说明。".encode(),
        source_key="kb/spring",
    )
    second = pipeline.ingest_bytes(
        "spring.md",
        "版本二改写了传播行为。".encode(),
        source_key="kb/spring",
    )
    assert first.document.document_id == second.document.document_id
    assert first.document.content_hash != second.document.content_hash
    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


def test_same_filename_with_different_source_keys() -> None:
    pipeline = _pipeline()
    left = pipeline.ingest_bytes("note.md", "左边的正文。".encode(), source_key="team/a")
    right = pipeline.ingest_bytes("note.md", "左边的正文。".encode(), source_key="team/b")
    assert left.document.filename == right.document.filename
    assert left.document.document_id != right.document.document_id


def test_newline_style_does_not_change_hash() -> None:
    pipeline = _pipeline()
    crlf = pipeline.ingest_bytes("a.txt", "第一行\r\n第二行\r\n".encode(), source_key="same")
    lf = pipeline.ingest_bytes("a.txt", "第一行\n第二行\n".encode(), source_key="same")
    assert crlf.document.content == "第一行\n第二行"
    assert crlf.document.content_hash == lf.document.content_hash


def test_rejects_blank_document() -> None:
    pipeline = _pipeline()
    with pytest.raises(DocumentValidationError, match="没有内容"):
        pipeline.ingest_bytes("blank.md", b" \r\n\t\r\n ", source_key="blank")


def test_rejects_unsupported_file(tmp_path: Path) -> None:
    target = tmp_path / "slide.pdf"
    target.write_text("not a pdf really", encoding="utf-8")
    with pytest.raises(DocumentValidationError, match="Markdown"):
        _pipeline().ingest_path(target)


def test_pipeline_does_not_log_document_body(caplog: pytest.LogCaptureFixture) -> None:
    pipeline = _pipeline()
    with caplog.at_level(logging.INFO):
        pipeline.ingest_bytes(
            "body.md",
            "这段正文带有标记UNIQTOKEN123。".encode(),
            source_key="kb/body",
        )
    assert "UNIQTOKEN123" not in caplog.text
    assert "document_id=" in caplog.text
