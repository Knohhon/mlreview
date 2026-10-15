"""Команды watch (проактивное ревью, FR-R04) и mcp (MCP-сервер, FR-R02)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlreview.cli.common import add_severity_argument, add_spec_argument, require_workspace
from mlreview.core.task_spec import load_task_spec
from mlreview.proactive import DirectorySource, MlflowSource, watch


def add_watch_arguments(parser: argparse.ArgumentParser) -> None:
    add_spec_argument(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-dir", type=Path, help="каталог с каталогами прогонов")
    source.add_argument("--experiment", help="эксперимент MLflow")
    parser.add_argument(
        "--tracking-uri", help="адрес трекера MLflow; по умолчанию — MLFLOW_TRACKING_URI"
    )
    parser.add_argument("--interval", type=float, default=60, help="пауза между просмотрами, с")
    parser.add_argument("--once", action="store_true", help="один просмотр и выход")
    parser.add_argument("--limit", type=int, default=5, help="находок в отчёте (по умолчанию 5)")
    add_severity_argument(parser, "low")


def run_watch(args: argparse.Namespace) -> int:
    workspace = require_workspace()
    spec = load_task_spec(args.spec)
    if args.runs_dir is not None:
        source = DirectorySource(args.runs_dir)
    else:
        source = MlflowSource(args.experiment, args.tracking_uri)
    try:
        watch(
            spec,
            source,
            workspace=workspace,
            interval=args.interval,
            max_scans=1 if args.once else None,
            tracking_uri=args.tracking_uri,
            limit=args.limit,
            min_severity=args.min_severity,
        )
    except KeyboardInterrupt:
        print("наблюдение остановлено")
    return 0


def run_mcp(_args: argparse.Namespace) -> int:
    from mlreview.mcp_server.server import MCPUnavailableError, build_server  # noqa: PLC0415

    try:
        server = build_server()
    except MCPUnavailableError as exc:
        print(exc, file=sys.stderr)
        return 1
    server.run("stdio")
    return 0
