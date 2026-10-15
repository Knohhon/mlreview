"""Python API: те же операции, что CLI, с объектами вместо текста (FR-R03, FR-R04).

CLI, MCP-сервер и проактивный режим построены на этих функциях. Поведение зафиксировано
в `docs/specs/api.md`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlreview.adapters.base import AdapterError, load_run, run_source
from mlreview.analyzers.review import GROUPS, review
from mlreview.core.findings import Report
from mlreview.core.run_contract import RunContract, RunContractError
from mlreview.core.store import ReviewRecord, RunRecord, Store, StoreError
from mlreview.core.task_spec import TaskSpec, TaskSpecError, load_task_spec
from mlreview.core.workspace import Workspace, WorkspaceError, find_workspace
from mlreview.metrics.library import MetricsError
from mlreview.reporting import (
    NOT_CONNECTED,
    NOT_USED,
    ReportView,
    build_view,
    render_report,
    report_payload,
)

__all__ = [
    "ReviewResult",
    "load_run",
    "load_spec",
    "on_run_end",
    "reference_ref",
    "review_runs",
]

PROACTIVE_LIMIT = 5
PROACTIVE_SEVERITY = "low"
LIBRARY_ERRORS = (TaskSpecError, RunContractError, AdapterError, StoreError, WorkspaceError)


def load_spec(path: str | Path) -> TaskSpec:
    """Спецификация задачи из YAML-файла (FR-C01)."""
    return load_task_spec(path)


@dataclass
class ReviewResult:
    """Ревью прогона: отчёт, запись в хранилище, контракт кандидата."""

    report: Report
    record: ReviewRecord | None
    store_state: str
    contract: RunContract
    spec: TaskSpec
    title: str = "ревью прогона"

    @property
    def store(self) -> str | None:
        """Путь базы хранилища или `None`, если запись не велась."""
        return None if self.store_state in (NOT_CONNECTED, NOT_USED) else self.store_state

    def view(self, *, min_severity: str = "info", limit: int = 0) -> ReportView:
        return build_view(
            self.report, self.record, min_severity=min_severity, limit=limit, store=self.store_state
        )

    def to_dict(self, *, min_severity: str = "info", limit: int = 0) -> dict[str, Any]:
        """Как `--format json` в CLI."""
        payload = report_payload(self.report, self.view(min_severity=min_severity, limit=limit))
        payload["llm"] = None
        return payload

    def to_text(
        self, *, min_severity: str = "info", limit: int = 0, title: str | None = None
    ) -> str:
        """Как текстовый отчёт CLI."""
        view = self.view(min_severity=min_severity, limit=limit)
        return render_report(self.report, title or self.title, view)

    def _repr_markdown_(self) -> str:
        report = self.report
        version = report.data_version or "нет"
        lines = [f"**{self.title} {report.run_id}** · версия данных: {version}"]
        if report.baseline_id:
            lines[0] += f" · базовый прогон: {report.baseline_id}"
        lines += ["", "| важность | id | проверка | сообщение |", "| --- | --- | --- | --- |"]
        for finding, entry in self.view().items:
            finding_id = entry.id if entry is not None else ""
            message = finding.message.replace("|", "\\|")
            lines.append(f"| {finding.severity} | {finding_id} | {finding.check} | {message} |")
        lines += ["", f"не проверено: {len(report.skipped)}"]
        return "\n".join(lines)


def _refs(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _open_store(store: bool, workspace: Workspace | None) -> tuple[Store | None, str]:
    if not store:
        return None, NOT_USED
    workspace = workspace or find_workspace()
    if workspace is None:
        return None, NOT_CONNECTED
    db = Store(workspace.store_path)
    return db, str(db.path)


def review_runs(  # noqa: PLR0913 — прогоны, сиды, группы, трекер, хранилище
    spec: TaskSpec,
    run: str,
    baseline: str | None = None,
    *,
    seeds: Sequence[str] = (),
    baseline_seeds: Sequence[str] = (),
    groups: Sequence[str] = GROUPS,
    tracking_uri: str | None = None,
    store: bool = True,
    workspace: Workspace | None = None,
    title: str = "ревью прогона",
) -> ReviewResult:
    """Ревью прогона (каталог или id MLflow), при `store` — запись в хранилище (B1, B4)."""
    candidates = [(load_run(spec, ref, tracking_uri=tracking_uri), ref) for ref in [run, *seeds]]
    baselines = [
        (load_run(spec, ref, tracking_uri=tracking_uri), ref)
        for ref in [*_refs(baseline), *baseline_seeds]
    ]
    if baselines and baseline is None:
        raise ValueError("сиды базового прогона без базового прогона")
    contract = candidates[0][0]
    db, state = _open_store(store, workspace)
    prior = None if db is None else db.count_runs(contract.data_hash, exclude=[contract.run_id])
    report = review(
        spec,
        contract,
        baselines[0][0] if baselines else None,
        groups=groups,
        candidate_seeds=[item for item, _ in candidates[1:]],
        baseline_seeds=[item for item, _ in baselines[1:]],
        prior_runs=prior,
    )
    record = None
    if db is not None:
        runs = [
            RunRecord.from_contract(item, run_source(ref, tracking_uri))
            for item, ref in [*candidates, *baselines]
        ]
        record = db.record_review(report, runs=runs)
    return ReviewResult(report, record, state, contract, spec, title)


def reference_ref(store: Store) -> str | None:
    """Ссылка на референс: первый прогон последнего принятого решения журнала (S02)."""
    run_id = store.reference_run()
    if run_id is None:
        return None
    known = store.run(run_id)
    if known is not None and known.source.startswith("files:"):
        return known.source.removeprefix("files:")
    return run_id


def proactive_review(  # noqa: PLR0913 — прогон, рабочее пространство и вид отчёта
    spec: TaskSpec,
    run: str,
    workspace: Workspace,
    *,
    tracking_uri: str | None = None,
    limit: int = PROACTIVE_LIMIT,
    min_severity: str = PROACTIVE_SEVERITY,
) -> tuple[ReviewResult, Path, str]:
    """Ревью против референса, запись и отчёт в `.mlreview/reports/` (FR-R04)."""
    store = Store(workspace.store_path)
    reference = reference_ref(store)
    if reference is not None:
        candidate_id = load_run(spec, run, tracking_uri=tracking_uri).run_id
        if reference in (run, candidate_id) or Path(reference) == Path(run).resolve():
            reference = None
    result = review_runs(spec, run, reference, tracking_uri=tracking_uri, workspace=workspace)
    text = result.to_text(min_severity=min_severity, limit=limit)
    workspace.reports_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in result.report.run_id
    )
    path = workspace.reports_dir / f"{safe}.txt"
    path.write_text(text + "\n", encoding="utf-8")
    return result, path, text


def on_run_end(  # noqa: PLR0913 — прогон, спецификация и вид отчёта
    run: str,
    *,
    spec: str | Path | TaskSpec,
    tracking_uri: str | None = None,
    limit: int = PROACTIVE_LIMIT,
    min_severity: str = PROACTIVE_SEVERITY,
    echo: bool = True,
) -> ReviewResult | None:
    """Проактивное ревью в конце скрипта обучения; ошибки не роняют скрипт (B5, E2)."""
    try:
        task = spec if isinstance(spec, TaskSpec) else load_spec(spec)
        workspace = find_workspace()
        if workspace is None:
            result = review_runs(task, run, tracking_uri=tracking_uri)
            text = result.to_text(min_severity=min_severity, limit=limit)
        else:
            result, _, text = proactive_review(
                task,
                run,
                workspace,
                tracking_uri=tracking_uri,
                limit=limit,
                min_severity=min_severity,
            )
    except (*LIBRARY_ERRORS, MetricsError, OSError, ValueError) as exc:
        print(f"mlreview: ревью не выполнено: {exc}", file=sys.stderr)
        return None
    if echo:
        print(text)
    return result
