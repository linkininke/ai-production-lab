"""切分策略。

Phase 2 只实现按字符窗口切分。Token 切分和按标题切分以后实现同一个 split 接口即可。
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class TextSpan(BaseModel):
    """原文上的一个连续区间。text 必须等于原文的 text[start:end]。"""

    text: str
    start: int
    end: int
    heading: str | None = None


class TextSplitStrategy(Protocol):
    def split(self, text: str) -> list[TextSpan]:
        """把正文切成有序片段。空文本返回空列表。"""
