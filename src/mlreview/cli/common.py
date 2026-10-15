"""Общее для команд CLI: ошибки, рабочее пространство, аргументы прогонов (FR-R01)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlreview.adapters.errors import AdapterError
from mlreview.core.findings import SEVERITIES
from mlreview.core.run_contract import RunContractError
from mlreview.core.store import Store, StoreError
from mlreview.core.task_spec import TaskSpecError
from mlreview.core.workspace import INIT_HINT, Workspace, WorkspaceError, find_workspace


class CliError(RuntimeError):
    """Ошибка команды: сообщение печатается в stderr, код возврата 1."""


ERRORS: tuple[type[Exception], ...] = (
    CliError,
    TaskSpecError,
    RunContractError,
    AdapterError,
    StoreError,
    WorkspaceError,
)


def require_workspace() -> Workspace:
    """Рабочее пространство или ошибка с подсказкой `mlreview init` (E4)."""
    workspace = find_workspace()
    if workspace is None:
        raise CliError(INIT_HINT)
    return workspace


def open_store() -> Store:
    return Store(require_workspace().store_path)


def add_spec_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--spec", type=Path, required=required, help="путь к YAML-файлу спецификации задачи"
    )


def add_tracking_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-uri", help="адрес трекера MLflow; по умолчанию — MLFLOW_TRACKING_URI"
    )


def add_severity_argument(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument(
        "--min-severity",
        choices=SEVERITIES,
        default=default,
        help=f"показывать находки не ниже уровня (по умолчанию {default})",
    )


RUN_HELP = "каталог прогона или id прогона MLflow"
