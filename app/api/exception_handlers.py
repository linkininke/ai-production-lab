"""把内部异常转换成统一的 HTTP 错误响应。

响应体只包含错误码和可读说明，不包含堆栈。
堆栈留在服务端日志里，供排查使用。

404 由 Starlette 抛出，必须注册 Starlette 的 HTTPException。
只注册 FastAPI 子类会漏掉框架自己产生的 404。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


def _error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _validation_message(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", []))
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    detail = "；".join(parts) if parts else "请求参数无效"
    return f"请求参数无效。{detail}"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "application_error code=%s path=%s",
            exc.code,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.info("validation_error path=%s", request.url.path)
        return JSONResponse(
            status_code=422,
            content=_error_body("validation_error", _validation_message(exc)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        message = exc.detail if isinstance(exc.detail, str) else "请求无法处理"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(code, message),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_error path=%s error_type=%s",
            request.url.path,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "服务内部错误"),
        )
