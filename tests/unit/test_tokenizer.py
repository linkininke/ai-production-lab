"""分词：英文标识符保持完整，中文不能整句当成一个词。"""

from __future__ import annotations

from app.retrieval.tokenizer import JiebaTokenizer


def test_technical_identifier_is_kept_and_split() -> None:
    tokens = JiebaTokenizer().tokenize("参数 max_overflow 已设置")
    assert tokens[:3] == ["max_overflow", "max", "overflow"]


def test_camel_case_identifier_is_split() -> None:
    tokens = JiebaTokenizer().tokenize("SpringTransaction")
    assert "springtransaction" in tokens
    assert "spring" in tokens
    assert "transaction" in tokens


def test_chinese_sentence_is_not_one_token() -> None:
    tokens = JiebaTokenizer().tokenize("Spring事务失效原因")
    assert "spring" in tokens
    assert "事务失效原因" not in tokens
    assert "事务" in tokens
    assert "失效" in tokens


def test_blank_text_has_no_tokens() -> None:
    assert JiebaTokenizer().tokenize("  \n\t?? ") == []
