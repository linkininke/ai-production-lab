"""文本清洗。"""

from __future__ import annotations

from app.ingestion.normalizer import normalize_text


def test_normalizes_newlines_and_trailing_space() -> None:
    raw = "第一行  \r\n第二行\t\r第三行"
    assert normalize_text(raw) == "第一行\n第二行\n第三行"


def test_collapses_excess_blank_lines() -> None:
    assert normalize_text("上一段\n\n\n\n下一段") == "上一段\n\n下一段"


def test_removes_control_characters_and_keeps_words() -> None:
    assert normalize_text("事务\x00失效\x07原因") == "事务失效原因"


def test_preserves_code_indentation() -> None:
    raw = "说明\n\n    code = 1  \n\n结束"
    assert normalize_text(raw) == "说明\n\n    code = 1\n\n结束"


def test_strips_outer_blank_lines() -> None:
    assert normalize_text("\n\n正文\n\n") == "正文"
