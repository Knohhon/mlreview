"""Что уходит в LLM: агрегаты и явно выбранные примеры, без сырых данных (NFR-04, X6).

Находки передаются без доказательств (id примеров), таблиц, признаков и конфигов прогонов.
Правила зафиксированы в `docs/specs/llm.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from mlreview.core.findings import Finding, Report
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec


def finding_dict(finding: Finding, finding_id: str) -> dict[str, Any]:
    """Находка для LLM: всё, кроме доказательств-примеров."""
    return {
        "id": finding_id,
        "check": finding.check,
        "severity": finding.severity,
        "message": finding.message,
        "requirement": finding.requirement,
        "facts": dict(finding.facts),
        "suggestion": finding.suggestion,
    }


def task_dict(spec: TaskSpec) -> dict[str, Any]:
    return {
        "type": spec.task.type,
        "output": spec.task.output,
        "decision": spec.task.decision,
        "segments": list(spec.task.segments),
        "data_version": spec.data.version if spec.data else None,
        "known_issues": list(spec.data.known_issues) if spec.data else [],
    }


def run_dict(report: Report) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "baseline_id": report.baseline_id,
        "data_version": report.data_version,
        "baseline_data_version": report.baseline_data_version,
        "compared_examples": report.compared_examples,
        "model": None if report.model is None else dict(report.model),
    }


def _plain(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value)
    return str(value)


def example_rows(contract: RunContract, ids: Sequence[str]) -> tuple[list[dict], list[str]]:
    """Строки явно выбранных примеров (examples + label, prediction) и не найденные id."""
    predictions = contract.predictions.assign(
        example_id=contract.predictions["example_id"].astype(str)
    ).set_index("example_id")
    examples = None
    if contract.examples is not None:
        examples = contract.examples.assign(
            example_id=contract.examples["example_id"].astype(str)
        ).set_index("example_id")
    rows, missing = [], []
    for example_id in ids:
        row: dict[str, Any] = {"example_id": example_id}
        found = False
        for table in (examples, predictions):
            if table is not None and example_id in table.index:
                found = True
                row.update({str(k): _plain(v) for k, v in table.loc[example_id].items()})
        (rows if found else missing).append(row if found else example_id)
    return rows, missing


def describe_rows(rows: Sequence[dict[str, Any]]) -> str:
    """Что именно уйдёт: id примеров и колонки (X6)."""
    if not rows:
        return "нет"
    columns = sorted({key for row in rows for key in row if key != "example_id"})
    ids = ", ".join(str(row["example_id"]) for row in rows)
    return f"{ids} (колонки: {', '.join(columns)})"


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _plain(v) for k, v in row.items()} for row in frame.to_dict("records")]
