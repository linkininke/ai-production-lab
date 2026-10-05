"""测试用的固定回答。不访问外网，也不把假模型当成正式能力。"""

from __future__ import annotations

from app.core.exceptions import LLMError
from app.llm.models import LLMResult


class ScriptedLLM:
    def __init__(
        self,
        answer: str = "根据资料回答。[C1]",
        *,
        prompt_tokens: int | None = 11,
        completion_tokens: int | None = 7,
    ) -> None:
        self.answer = answer
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.fail = False
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResult:
        self.calls.append((system_prompt, user_prompt))
        if self.fail:
            raise LLMError("模拟生成失败")
        text = self.answer
        if "没有检索到参考资料" in user_prompt:
            text = "知识库中缺少相关信息。"
        return LLMResult(
            text=text,
            model="fake-llm",
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
