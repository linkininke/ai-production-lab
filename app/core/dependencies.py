"""FastAPI 依赖。

路由从应用状态读取配置和已经装配好的客户端，不在每个请求里重新解析环境变量。
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.core.container import AppContainer
from app.core.exceptions import ConfigurationError


def get_settings_from_app(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise ConfigurationError("应用未初始化配置")
    return settings


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, AppContainer):
        raise ConfigurationError("应用未初始化服务")
    return container
