"""把正文切成 BM25 用的词。

英文和技术标识符按整词保留，蛇形和驼峰再拆开，方便精确关键词命中。
连续中文交给 jieba，避免把「事务失效」当成一个不可分的长串。
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]:
        """返回小写词序列。空文本返回空列表。"""


class JiebaTokenizer:
    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for match in _ASCII_WORD.finditer(text):
            tokens.extend(_ascii_tokens(match.group(0)))
        cutter = _jieba()
        for match in _CJK_RUN.finditer(text):
            tokens.extend(token.strip() for token in cutter.lcut(match.group(0)) if token.strip())
        return tokens


def _jieba():
    import jieba

    jieba.setLogLevel(logging.WARNING)
    return jieba


def _ascii_tokens(word: str) -> list[str]:
    lowered = word.lower()
    tokens = [lowered]
    snake_parts = [part.lower() for part in word.split("_") if part]
    if len(snake_parts) > 1:
        tokens.extend(snake_parts)
    camel = _CAMEL_BOUNDARY.sub(r"\1 \2", word)
    camel_parts = [part.lower() for part in camel.split() if part]
    if len(camel_parts) > 1:
        tokens.extend(camel_parts)
    return tokens
