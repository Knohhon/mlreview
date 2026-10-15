"""Срезы: сегменты хуже остальных и регрессии сегментов между прогонами (FR-D02, B12).

Сегменты — колонки `examples` из `task.segments`; метрика — основная метрика задачи.
Поведение зафиксировано в `docs/specs/slices.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlreview.analyzers.evaluation import (
    Estimate,
    Metric,
    Outputs,
    example_errors,
    holm,
    independent_delta,
    is_worse,
    pair_runs,
    paired_delta,
    prepare,
    rounded,
    top_ids,
)
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec
from mlreview.metrics.library import primary

WORSE_CHECK = "slices.segment_worse"
REGRESSION_CHECK = "slices.segment_regression"
MIN_SEGMENT_SIZE = 30
ALPHA = 0.05
MISSING_VALUE = "<пропуск>"
WORSE_SUGGESTION = (
    "посмотреть примеры сегмента с наибольшей ошибкой (в доказательствах) и сравнить признаки"
    " сегмента с остальными; если сегмент важен — следить за его метрикой при отборе моделей"
)
REGRESSION_SUGGESTION = (
    "сравнить прогоны на примерах сегмента с наибольшим ростом ошибки (в доказательствах);"
    " различающий эксперимент — переносить отличия кандидата от базового прогона по одному"
    " и смотреть метрику сегмента"
)


@dataclass(frozen=True)
class _Test:
    """Проверенное значение сегмента до поправки на множественность."""

    segment: str
    value: str
    mask: np.ndarray
    estimate: Estimate
    group_metric: float
    other_metric: float


def _segment_values(examples: pd.DataFrame, segment: str, ids: np.ndarray) -> np.ndarray:
    """Значение сегмента для каждого id (строкой); пропуск — `<пропуск>`."""
    column = examples.set_index(examples["example_id"].astype(str))[segment]
    values = column.reindex(pd.Index(ids.astype(str)))
    return np.array(
        [MISSING_VALUE if pd.isna(value) else str(value) for value in values], dtype=object
    )


def _checked_values(
    results: CheckResults, check: str, segment: str, values: np.ndarray
) -> list[str]:
    """Значения сегмента с достаточным числом примеров; остальные — в пропуски (E4)."""
    names, counts = np.unique(values.astype(str), return_counts=True)
    small = [
        f"{name} ({count})"
        for name, count in zip(names, counts, strict=True)
        if count < MIN_SEGMENT_SIZE
    ]
    if small:
        results.skip(
            check,
            f"{segment}: значения с менее чем {MIN_SEGMENT_SIZE} примерами не проверены: "
            + ", ".join(small),
        )
    return [
        str(name) for name, count in zip(names, counts, strict=True) if count >= MIN_SEGMENT_SIZE
    ]


def _significant(tests: list[_Test], metric: Metric) -> list[tuple[_Test, float]]:
    """Ухудшения, значимые после поправки Холма."""
    adjusted = holm([test.estimate.p_value for test in tests])
    return [
        (test, p_value)
        for test, p_value in zip(tests, adjusted, strict=True)
        if p_value < ALPHA and is_worse(metric, test.estimate.value)
    ]


def _collect(
    results: CheckResults,
    check: str,
    segments: list[str],
    values_of: Callable[[str], np.ndarray],
    run_test: Callable[[str, str, np.ndarray], _Test | str],
) -> list[_Test]:
    tests = []
    for segment in segments:
        values = values_of(segment)
        for value in _checked_values(results, check, segment, values):
            outcome = run_test(segment, value, values == value)
            if isinstance(outcome, str):
                results.skip(check, f"{segment}={value}: {outcome}")
            else:
                tests.append(outcome)
    return tests


def _worse_findings(
    spec: TaskSpec, outputs: Outputs, segments: list[str], examples: pd.DataFrame
) -> CheckResults:
    results = CheckResults()
    metric = primary(spec)

    def run_test(segment: str, value: str, mask: np.ndarray) -> _Test | str:
        if (~mask).sum() < MIN_SEGMENT_SIZE:
            return f"вне значения меньше {MIN_SEGMENT_SIZE} примеров — сравнивать не с чем"
        group, rest = outputs.take(np.flatnonzero(mask)), outputs.take(np.flatnonzero(~mask))
        estimate = independent_delta(metric, group, rest)
        if isinstance(estimate, str):
            return estimate
        return _Test(segment, value, mask, estimate, metric(group), metric(rest))

    tests = _collect(
        results,
        WORSE_CHECK,
        segments,
        lambda segment: _segment_values(examples, segment, outputs.ids),
        run_test,
    )
    errors = example_errors(spec.task, outputs)
    for test, p_value in _significant(tests, metric):
        group, rest = rounded(test.group_metric), rounded(test.other_metric)
        delta, low, high = (
            rounded(x) for x in (group - rest, test.estimate.low, test.estimate.high)
        )
        n = test.estimate.n
        message = (
            f"сегмент {test.segment}={test.value} хуже остальных: {metric.name} {group:.3f}"
            f" против {rest:.3f} (Δ {delta:+.3f}, 95% CI [{low:+.3f}; {high:+.3f}], n = {n})"
        )
        facts = {
            "segment": test.segment,
            "value": test.value,
            "metric": metric.name,
            "segment_metric": group,
            "rest_metric": rest,
            "delta": delta,
            "ci_low": low,
            "ci_high": high,
            "p_value": rounded(p_value),
            "n": n,
        }
        evidence = top_ids(outputs.ids[test.mask], None if errors is None else errors[test.mask])
        results.findings.append(
            Finding(WORSE_CHECK, "medium", message, "FR-D02", evidence, facts, WORSE_SUGGESTION)
        )
    return results


def _regression_findings(
    spec: TaskSpec,
    candidate: RunContract,
    baseline: RunContract,
    segments: list[str],
    examples: pd.DataFrame,
) -> CheckResults:
    results = CheckResults()
    pair = pair_runs(spec.task, candidate, baseline)
    if isinstance(pair, str):
        results.skip(REGRESSION_CHECK, pair)
        return results
    metric = primary(spec)
    cand, base = pair.candidate, pair.baseline

    def run_test(segment: str, value: str, mask: np.ndarray) -> _Test | str:
        index = np.flatnonzero(mask)
        group_cand, group_base = cand.take(index), base.take(index)
        estimate = paired_delta(metric, group_cand, group_base)
        if isinstance(estimate, str):
            return estimate
        return _Test(segment, value, mask, estimate, metric(group_cand), metric(group_base))

    tests = _collect(
        results,
        REGRESSION_CHECK,
        segments,
        lambda segment: _segment_values(examples, segment, cand.ids),
        run_test,
    )
    overall = rounded(rounded(metric(cand)) - rounded(metric(base)))
    errors_cand, errors_base = example_errors(spec.task, cand), example_errors(spec.task, base)
    growth = None if errors_cand is None else errors_cand - errors_base
    for test, p_value in _significant(tests, metric):
        after, before = rounded(test.group_metric), rounded(test.other_metric)
        delta, low, high = (
            rounded(x) for x in (after - before, test.estimate.low, test.estimate.high)
        )
        n = test.estimate.n
        message = (
            f"регрессия сегмента {test.segment}={test.value}: {metric.name} {before:.3f} →"
            f" {after:.3f} (Δ {delta:+.3f}, 95% CI [{low:+.3f}; {high:+.3f}], n = {n});"
            f" в целом по прогону Δ {overall:+.3f}"
        )
        facts = {
            "segment": test.segment,
            "value": test.value,
            "metric": metric.name,
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "ci_low": low,
            "ci_high": high,
            "p_value": rounded(p_value),
            "n": n,
            "overall_delta": overall,
        }
        evidence = top_ids(cand.ids[test.mask], None if growth is None else growth[test.mask])
        results.findings.append(
            Finding(
                REGRESSION_CHECK,
                "high",
                message,
                "FR-D02, B12",
                evidence,
                facts,
                REGRESSION_SUGGESTION,
            )
        )
    return results


def segment_findings(
    spec: TaskSpec, contract: RunContract, baseline: RunContract | None = None
) -> CheckResults:
    """Срезы прогона; с базовым прогоном — ещё и регрессии сегментов."""
    results = CheckResults()
    checks = [WORSE_CHECK] if baseline is None else [WORSE_CHECK, REGRESSION_CHECK]
    if not spec.task.segments:
        for check in checks:
            results.skip(check, "в спецификации нет task.segments")
        return results
    examples = contract.examples
    if examples is None:
        for check in checks:
            results.skip(check, "нет examples — значения сегментов неизвестны")
        return results

    present = [segment for segment in spec.task.segments if segment in examples.columns]
    outputs = prepare(spec.task, contract.predictions)
    for check in checks:
        for segment in spec.task.segments:
            if segment not in examples.columns:
                results.skip(check, f"нет examples.{segment}")
        if baseline is not None and check == REGRESSION_CHECK:
            results.extend(_regression_findings(spec, contract, baseline, present, examples))
        else:
            results.extend(_worse_findings(spec, outputs, present, examples))
    return results
