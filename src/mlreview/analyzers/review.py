"""Ревью прогона: анализаторы выбранных групп → отчёт (FR-R01, FR-D01–D08, FR-E01).

Отчёт детерминированный и работает без LLM (NFR-03); что не проверено — в `skipped` (X1).
Поведение зафиксировано в `docs/specs/review.md` и `docs/specs/analyzers.md`.
"""

from __future__ import annotations

from collections.abc import Sequence

from mlreview.analyzers import builtin  # noqa: F401 — регистрирует встроенные анализаторы
from mlreview.analyzers.registry import GROUPS, ReviewContext, run_analyzers
from mlreview.core.findings import Report, sort_findings
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec

__all__ = ["GROUPS", "review"]


def _common_examples(candidate: RunContract, baseline: RunContract) -> int:
    ids = set(candidate.predictions["example_id"].astype(str))
    return len(ids & set(baseline.predictions["example_id"].astype(str)))


def review(  # noqa: PLR0913 — прогоны, сиды и история ревью
    spec: TaskSpec,
    contract: RunContract,
    baseline: RunContract | None = None,
    groups: Sequence[str] = GROUPS,
    *,
    candidate_seeds: Sequence[RunContract] = (),
    baseline_seeds: Sequence[RunContract] = (),
    prior_runs: int | None = None,
) -> Report:
    """Запускает анализаторы выбранных групп и собирает отчёт."""
    ctx = ReviewContext(
        spec,
        contract,
        baseline,
        tuple(candidate_seeds),
        tuple(baseline_seeds) if baseline is not None else (),
        prior_runs,
    )
    results = run_analyzers(ctx, groups)
    return Report(
        run_id=contract.run_id,
        findings=sort_findings(results.findings),
        skipped=results.skipped,
        baseline_id=None if baseline is None else baseline.run_id,
        data_version=contract.data_version,
        baseline_data_version=None if baseline is None else baseline.data_version,
        compared_examples=None if baseline is None else _common_examples(contract, baseline),
        data_hash=contract.data_hash,
        baseline_data_hash=None if baseline is None else baseline.data_hash,
        candidate_seeds=tuple(run.run_id for run in ctx.candidate_seeds),
        baseline_seeds=tuple(run.run_id for run in ctx.baseline_seeds),
    )
