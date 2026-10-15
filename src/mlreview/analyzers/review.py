"""Ревью прогона: группы проверок data, segments, metrics → отчёт (FR-R01, FR-D01–D03).

Отчёт детерминированный и работает без LLM (NFR-03); что не проверено — в `skipped` (X1).
Поведение зафиксировано в `docs/specs/review.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from mlreview.analyzers.comparison import compare_runs, run_metrics
from mlreview.analyzers.data_checks import data_findings
from mlreview.analyzers.slices import segment_findings
from mlreview.core.findings import CheckResults, Report, sort_findings
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec

GROUPS: tuple[str, ...] = ("data", "segments", "metrics")


def _metrics(spec: TaskSpec, contract: RunContract, baseline: RunContract | None) -> CheckResults:
    return (
        run_metrics(spec, contract) if baseline is None else compare_runs(spec, contract, baseline)
    )


_RUNNERS: dict[str, Callable[[TaskSpec, RunContract, RunContract | None], CheckResults]] = {
    "data": lambda spec, contract, _baseline: data_findings(spec, contract),
    "segments": segment_findings,
    "metrics": _metrics,
}


def _common_examples(candidate: RunContract, baseline: RunContract) -> int:
    ids = set(candidate.predictions["example_id"].astype(str))
    return len(ids & set(baseline.predictions["example_id"].astype(str)))


def review(
    spec: TaskSpec,
    contract: RunContract,
    baseline: RunContract | None = None,
    groups: Sequence[str] = GROUPS,
) -> Report:
    """Запускает выбранные группы проверок и собирает отчёт."""
    unknown = [group for group in groups if group not in GROUPS]
    if unknown:
        raise ValueError(f"неизвестные группы проверок: {', '.join(unknown)}")
    results = CheckResults()
    for group in GROUPS:
        if group in groups:
            results.extend(_RUNNERS[group](spec, contract, baseline))
    return Report(
        run_id=contract.run_id,
        findings=sort_findings(results.findings),
        skipped=results.skipped,
        baseline_id=None if baseline is None else baseline.run_id,
        data_version=contract.data_version,
        baseline_data_version=None if baseline is None else baseline.data_version,
        compared_examples=None if baseline is None else _common_examples(contract, baseline),
    )
