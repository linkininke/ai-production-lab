"""FastAPI 启动入口。

导入本模块时就会读取并校验配置。配置错误在进程启动时暴露，
而不是等到第一次问答才失败。
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from app import __version__
from app.api.exception_handlers import register_exception_handlers
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.config import Settings, get_settings
from app.core.container import AppContainer, build_container
from app.core.logging import request_id_var, setup_logging

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    container = getattr(app.state, "container", None)
    if isinstance(container, AppContainer):
        container.close()


def create_app(
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    resolved = settings if settings is not None else get_settings()
    setup_logging(resolved)
    app = FastAPI(
        title="AI Production Lab",
        version=__version__,
        description="个人技术知识库 RAG API（v0.1）",
        lifespan=_lifespan,
    )
    app.state.settings = resolved
    app.state.container = container if container is not None else build_container(resolved)
    register_exception_handlers(app)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")
    _register_request_logging(app)
    return app


def _register_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else f"req_{uuid4().hex[:12]}"
        token = request_id_var.set(request_id)
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started) * 1000
            logger.exception(
                "request method=%s path=%s status=error duration_ms=%.1f",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        else:
            duration_ms = (perf_counter() - started) * 1000
            logger.info(
                "request method=%s path=%s status=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    main()
