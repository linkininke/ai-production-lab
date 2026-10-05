"""字符切分、重叠和标题。"""

from __future__ import annotations

import pytest

from app.chunking.splitter import CharacterTextSplitter, active_heading, build_chunks
from app.core.exceptions import ConfigurationError
from app.ingestion.models import Document


def _document(content: str) -> Document:
    return Document(
        document_id="doc_test",
        filename="demo.md",
        content=content,
        file_type="markdown",
        content_hash="abc",
        created_at="2026-10-05T10:00:00+00:00",
    )


def test_short_text_is_one_chunk() -> None:
    spans = CharacterTextSplitter(80, 10).split("只有一句。")
    assert len(spans) == 1
    assert spans[0].text == "只有一句。"
    assert spans[0].start == 0
    assert spans[0].end == len("只有一句。")


def test_empty_text_has_no_chunks() -> None:
    assert CharacterTextSplitter(80, 10).split("") == []


def test_hard_split_overlap_is_exact() -> None:
    text = "字" * 100
    spans = CharacterTextSplitter(20, 5).split(text)
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    assert all(len(span.text) <= 20 for span in spans)
    assert all(span.text == text[span.start : span.end] for span in spans)
    for previous, following in zip(spans, spans[1:]):
        assert following.start == previous.end - 5
        assert text[following.start : previous.end] == previous.text[-5:]
        assert following.start > previous.start


def test_zero_overlap_has_no_shared_characters() -> None:
    text = "甲" * 50
    spans = CharacterTextSplitter(20, 0).split(text)
    for previous, following in zip(spans, spans[1:]):
        assert following.start == previous.end


def test_prefers_sentence_boundary() -> None:
    text = "啊" * 30 + "。" + "呀" * 30
    spans = CharacterTextSplitter(40, 5).split(text)
    assert spans[0].end == 31
    assert spans[0].text.endswith("。")
    assert spans[1].start == 26
    assert spans[0].text[-5:] == text[26:31]


def test_ignores_boundary_that_would_make_chunk_too_short() -> None:
    text = "A" * 10 + "。" + "B" * 50
    spans = CharacterTextSplitter(40, 4).split(text)
    assert len(spans[0].text) == 40


def test_prefers_paragraph_break() -> None:
    text = "A" * 30 + "\n\n" + "B" * 30
    spans = CharacterTextSplitter(50, 5).split(text)
    assert spans[0].text.endswith("\n\n")
    assert len(spans[0].text) == 32
    assert spans[1].start == 27


def test_large_overlap_still_terminates() -> None:
    spans = CharacterTextSplitter(10, 9).split("a" * 100)
    assert spans[-1].end == 100
    assert spans[0].start == 0
    assert all(span.end > span.start for span in spans)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
        CharacterTextSplitter(10, 10)


def test_heading_skips_fenced_code() -> None:
    text = "# 真标题\n\n```\n# 假标题\n```\n\n" + ("甲" * 200)
    body_at = text.index("甲")
    assert active_heading(text, body_at) == "# 真标题"
    spans = CharacterTextSplitter(40, 8).split(text)
    assert spans
    assert all(span.heading == "# 真标题" for span in spans)


def test_heading_changes_along_the_document() -> None:
    text = "# 第一节\n\n" + ("甲" * 50) + "\n\n## 第二节\n\n" + ("乙" * 50)
    spans = CharacterTextSplitter(30, 5).split(text)
    assert any(span.heading == "# 第一节" for span in spans)
    assert any(span.heading == "## 第二节" for span in spans)


def test_chunk_ids_are_stable_and_unique_per_index() -> None:
    text = "甲" * 40
    document = _document(text)
    spans = CharacterTextSplitter(20, 0).split(text)
    first = build_chunks(document, spans)
    second = build_chunks(document, spans)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].text == first[1].text
    assert first[0].chunk_id != first[1].chunk_id
    assert first[0].metadata["document_id"] == document.document_id
    assert first[0].metadata["chunk_index"] == 0
    assert "content" not in first[0].metadata
