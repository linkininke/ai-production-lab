"""大模型一次生成的结果。"""

from __future__ import annotations

from pydantic import BaseModel


class LLMResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
