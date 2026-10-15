"""Выбор адаптера по ссылке на прогон: каталог — файлы, иначе — MLflow (FR-C03).

Поведение зафиксировано в `docs/specs/files-adapter.md`.
"""

from __future__ import annotations

from pathlib import Path

from mlreview.adapters.errors import AdapterError
from mlreview.adapters.files_adapter import load_files_contract
from mlreview.adapters.mlflow_adapter import load_mlflow_contract
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec

__all__ = ["AdapterError", "load_run", "run_source"]


def run_source(ref: str, tracking_uri: str | None = None) -> str:
    """Источник прогона для хранилища: `files:<абсолютный путь>` или `mlflow:<адрес>`."""
    path = Path(ref)
    if path.is_dir():
        return f"files:{path.resolve()}"
    return f"mlflow:{tracking_uri or 'default'}"


def load_run(spec: TaskSpec, ref: str, *, tracking_uri: str | None = None) -> RunContract:
    """Контракт прогона по ссылке: существующий каталог или id прогона MLflow."""
    if Path(ref).is_dir():
        return load_files_contract(spec, ref)
    return load_mlflow_contract(spec, ref, tracking_uri=tracking_uri)
