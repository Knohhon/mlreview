"""Проактивное ревью по событию «прогон завершён»: наблюдатель за прогонами (FR-R04).

Новые завершённые прогоны каталога или эксперимента MLflow проходят ревью против референса
из журнала; находки — в хранилище, отчёт — в `.mlreview/reports/`. Поведение зафиксировано
в `docs/specs/proactive.md`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mlreview.adapters.files_adapter import is_finished, run_id_of
from mlreview.adapters.mlflow_adapter import finished_runs
from mlreview.api import LIBRARY_ERRORS, PROACTIVE_LIMIT, PROACTIVE_SEVERITY, proactive_review
from mlreview.core.store import Store
from mlreview.core.task_spec import TaskSpec, load_task_spec
from mlreview.core.workspace import INIT_HINT, Workspace, WorkspaceError, find_workspace
from mlreview.metrics.library import MetricsError

NOTHING_NEW = "новых завершённых прогонов нет"


def load_spec_for_watch(path: str | Path) -> TaskSpec:
    return load_task_spec(path)


@dataclass(frozen=True)
class DirectorySource:
    """Каталог прогонов: подкаталоги с `run.yaml` — завершённые прогоны."""

    root: Path

    def finished(self) -> list[tuple[str, str]]:
        directories = sorted(path for path in Path(self.root).iterdir() if path.is_dir())
        return [(run_id_of(path), str(path)) for path in directories if is_finished(path)]


@dataclass(frozen=True)
class MlflowSource:
    """Эксперимент MLflow: прогоны со статусом FINISHED."""

    experiment: str
    tracking_uri: str | None = None

    def finished(self) -> list[tuple[str, str]]:
        return [
            (run_id, run_id)
            for run_id in finished_runs(self.experiment, tracking_uri=self.tracking_uri)
        ]


def _summary(run_id: str, result, path: Path) -> str:
    entries = [] if result.record is None else result.record.entries
    critical = sum(entry.finding.severity == "critical" for entry in entries)
    high = sum(entry.finding.severity == "high" for entry in entries)
    return f"прогон {run_id}: находок {len(entries)} (critical {critical}, high {high}) → {path}"


def scan(  # noqa: PLR0913 — источник, рабочее пространство и вид отчёта
    spec: TaskSpec,
    source: DirectorySource | MlflowSource,
    workspace: Workspace,
    *,
    tracking_uri: str | None = None,
    limit: int = PROACTIVE_LIMIT,
    min_severity: str = PROACTIVE_SEVERITY,
    failed: set[str] | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    """Один просмотр: ревью новых завершённых прогонов; возвращает число проверенных."""
    failed = set() if failed is None else failed
    store = Store(workspace.store_path)
    new = [
        (run_id, ref)
        for run_id, ref in source.finished()
        if run_id not in failed and store.run(run_id) is None
    ]
    if not new:
        echo(NOTHING_NEW)
        return 0
    reviewed = 0
    for run_id, ref in new:
        try:
            result, path, _ = proactive_review(
                spec,
                ref,
                workspace,
                tracking_uri=tracking_uri,
                limit=limit,
                min_severity=min_severity,
            )
        except (*LIBRARY_ERRORS, MetricsError, OSError) as exc:
            failed.add(run_id)
            echo(f"прогон {run_id}: не проверен — {exc}")
            continue
        reviewed += 1
        echo(_summary(run_id, result, path))
    return reviewed


def watch(  # noqa: PLR0913 — источник, пауза, число просмотров и вид отчёта
    spec: TaskSpec,
    source: DirectorySource | MlflowSource,
    *,
    workspace: Workspace | None = None,
    interval: float = 60,
    sleep: Callable[[float], None] = time.sleep,
    max_scans: int | None = None,
    tracking_uri: str | None = None,
    limit: int = PROACTIVE_LIMIT,
    min_severity: str = PROACTIVE_SEVERITY,
    echo: Callable[[str], None] = print,
) -> None:
    """Просмотры с паузой `interval`, пока не прервут (или `max_scans` раз)."""
    workspace = workspace or find_workspace()
    if workspace is None:
        raise WorkspaceError(INIT_HINT)
    failed: set[str] = set()
    scans = 0
    while True:
        scan(
            spec,
            source,
            workspace,
            tracking_uri=tracking_uri,
            limit=limit,
            min_severity=min_severity,
            failed=failed,
            echo=echo,
        )
        scans += 1
        if max_scans is not None and scans >= max_scans:
            return
        sleep(interval)
