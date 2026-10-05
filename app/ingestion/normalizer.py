"""清理空白和不可见控制字符。

只处理排版噪音：换行、行尾空格、多余空行、空字节。
不改大小写，不删标点，不压掉行首缩进。代码块和 Markdown 结构靠这些字符保持原意。
"""

from __future__ import annotations

import re

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHARACTERS.sub("", normalized)
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = _EXCESS_BLANK_LINES.sub("\n\n", normalized)
    return normalized.strip()
