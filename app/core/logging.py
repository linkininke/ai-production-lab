"""日志配置。

每条日志带上 request_id，方便把一次请求的各阶段串起来。
密钥只允许出现在 SecretStr 中；如果日志文本里意外包含密钥，过滤器会把它抹掉。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from app.config import Settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_HANDLER_NAME = "ai-production-lab"
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
_MIN_SECRET_LENGTH = 8


class RequestIdFilter(logging.Filter):
    """把当前请求 ID 写进日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class SecretRedactionFilter(logging.Filter):
    """从日志消息中移除已知密钥。过短的字符串不处理，避免误伤普通数字。"""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self._secrets = [secret for secret in (secrets or []) if len(secret) >= _MIN_SECRET_LENGTH]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            redacted = redacted.replace(secret, "***")
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(settings: Settings) -> None:
    """安装应用自己的日志处理器。重复调用只刷新过滤规则，不叠加处理器。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = next(
        (item for item in root.handlers if getattr(item, "name", None) == _HANDLER_NAME),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.name = _HANDLER_NAME
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)

    handler.filters.clear()
    handler.addFilter(RequestIdFilter())
    handler.addFilter(
        SecretRedactionFilter(
            [
                settings.llm_api_key.get_secret_value(),
                settings.embedding_api_key.get_secret_value(),
            ]
        )
    )
