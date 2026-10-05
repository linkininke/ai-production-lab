"""编码识别。"""

from __future__ import annotations

import pytest

from app.core.exceptions import DocumentValidationError
from app.ingestion.parser import decode_text


def test_decodes_utf8_bom() -> None:
    assert decode_text(b"\xef\xbb\xbf\xe4\xba\x8b\xe5\x8a\xa1") == "事务"


def test_decodes_gb18030() -> None:
    assert decode_text("事务失效".encode("gbk")) == "事务失效"


def test_rejects_bytes_that_are_not_utf8_or_gb18030() -> None:
    with pytest.raises(DocumentValidationError, match="GB18030"):
        decode_text(b"\xff")
