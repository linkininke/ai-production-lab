"""日志不泄露密钥，并带上请求 ID。"""

from __future__ import annotations

import logging

from app.core.logging import (
    RequestIdFilter,
    SecretRedactionFilter,
    request_id_var,
    setup_logging,
)
from tests.helpers import make_settings


def _record(msg: str, args: tuple[object, ...] = ()) -> logging.LogRecord:
    return logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_redacts_api_key() -> None:
    record = _record("calling with key sk-test-secret-value")
    assert SecretRedactionFilter(["sk-test-secret-value"]).filter(record) is True
    assert record.getMessage() == "calling with key ***"


def test_ignores_short_secrets() -> None:
    record = _record("code 1234")
    assert SecretRedactionFilter(["1234"]).filter(record) is True
    assert "1234" in record.getMessage()


def test_request_id_filter_reads_context() -> None:
    token = request_id_var.set("req_abc12345")
    try:
        record = _record("hello")
        assert RequestIdFilter().filter(record) is True
        assert record.request_id == "req_abc12345"
    finally:
        request_id_var.reset(token)


def test_setup_logging_installs_redaction() -> None:
    settings = make_settings(llm_api_key="sk-live-secret-key")
    setup_logging(settings)
    handler = next(
        item
        for item in logging.getLogger().handlers
        if getattr(item, "name", None) == "ai-production-lab"
    )
    redactors = [item for item in handler.filters if isinstance(item, SecretRedactionFilter)]
    assert redactors
    record = _record("key=%s", ("sk-live-secret-key",))
    assert redactors[0].filter(record) is True
    assert "sk-live-secret-key" not in record.getMessage()
