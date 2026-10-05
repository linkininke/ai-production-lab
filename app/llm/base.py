"""大模型接口。对话模型和 Embedding 分开实现，避免两套配置混用。"""

from __future__ import annotations

from typing import Protocol

from app.llm.models import LLMResult


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """根据系统规则和用户消息生成回答。第一版不使用流式输出。"""
