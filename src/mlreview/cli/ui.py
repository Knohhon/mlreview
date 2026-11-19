"""Запуск опционального localhost UI без импорта web extra в обычном CLI."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from mlreview.core.workspace import Workspace, WorkspaceError, find_workspace

MAX_PORT = 65535


def add_ui_arguments(parser: argparse.ArgumentParser) -> None:
    """Параметры процесса UI."""
    parser.add_argument("--project", type=Path, help="явный корень проекта")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="только вывести URL")


def resolve_workspace(project: Path | None) -> Workspace:
    """Явный project исключает неявный выбор другого workspace из окружения."""
    workspace = Workspace(project.resolve() / ".mlreview") if project else find_workspace()
    if workspace is None or not workspace.root.is_dir():
        raise WorkspaceError("Рабочее пространство не инициализировано — выполните mlreview init")
    return workspace


def run_ui(args: argparse.Namespace) -> int:
    """Проверяет workspace и порт до запуска сервера/браузера."""
    workspace = resolve_workspace(args.project)
    if not 1 <= args.port <= MAX_PORT:
        raise WorkspaceError("Порт должен быть от 1 до 65535")
    try:
        import uvicorn  # noqa: PLC0415 — опциональный web extra

        from mlreview.web.app import create_app  # noqa: PLC0415
    except ImportError:
        print('Для UI установите web extra: pip install "mlreview[web]"', file=sys.stderr)
        return 1
    family = socket.AF_INET6 if args.host == "::1" else socket.AF_INET
    sock = None
    try:
        sock = socket.socket(family)
        sock.bind((args.host, args.port))
        sock.listen(128)
    except OSError as exc:
        if sock is not None:
            sock.close()
        print(
            f"Не удалось открыть {args.host}:{args.port}: {exc}; укажите другой --port",
            file=sys.stderr,
        )
        return 1
    host = "[::1]" if args.host == "::1" else args.host
    url = f"http://{host}:{args.port}/"
    try:
        app = create_app(workspace)
        server = uvicorn.Server(
            uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
        )
        if not args.no_browser and sys.stdout.isatty() and not os.environ.get("SSH_CONNECTION"):

            def open_when_ready():
                while not server.started and not server.should_exit:
                    threading.Event().wait(0.1)
                if server.started:
                    webbrowser.open(url)

            threading.Thread(target=open_when_ready, daemon=True).start()
        print(f"MLReview UI: {url}", flush=True)
        server.run(sockets=[sock])
    finally:
        sock.close()
    return 0
