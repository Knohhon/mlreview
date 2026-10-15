"""MCP-сервер по stdio на официальном SDK `mcp` 2.x (FR-R02).

Пакет `mcp` (extra `mcp`) импортируется лениво: без него остальные команды работают.
Ошибки операций уходят клиенту как ошибки инструмента с тем же сообщением, что в CLI.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from mlreview.cli.common import ERRORS
from mlreview.mcp_server.tools import TOOLS

INSTALL_HINT = 'MCP-сервер требует пакет mcp: pip install "mlreview[mcp]"'
INSTRUCTIONS = (
    "MLReview — ассистент-критик ML-экспериментов. Инструменты работают с прогонами"
    " (каталог или id MLflow) и спецификацией задачи (YAML). Находки детерминированы;"
    " инструмент не применяет изменения сам, а предлагает эксперименты."
)


class MCPUnavailableError(RuntimeError):
    """Пакет mcp не установлен."""


def _wrap(function: Callable[..., Any], tool_error: type[Exception]) -> Callable[..., Any]:
    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except (*ERRORS, ValueError) as exc:
            raise tool_error(str(exc)) from exc

    return call


def build_server() -> Any:
    """Сервер с инструментами `mlreview.mcp_server.tools`."""
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: PLC0415 — extra mcp
        from mcp.server.mcpserver.exceptions import ToolError  # noqa: PLC0415
    except ImportError:
        raise MCPUnavailableError(INSTALL_HINT) from None
    server = MCPServer("mlreview", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.add_tool(
            _wrap(tool.function, ToolError), name=tool.name, description=tool.description
        )
    return server
