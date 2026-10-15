"""Разброс по сидам и «улучшение в пределах шума» (FR-D05, B08).

Метрика — основная метрика задачи по всем предсказаниям каждого прогона; значимость —
t-тест Уэлча. Поведение зафиксировано в `docs/specs/seeds.md`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import norm, ttest_ind

from mlreview.analyzers.comparison import DECLINE_SUGGESTION
from mlreview.analyzers.evaluation import Metric, is_worse, prepare, rounded
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec
from mlreview.metrics.library import primary

ALPHA = 0.05
POWER = 0.8
RECOMMENDED_SEEDS = 3
REQUIREMENT = "FR-D05"


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова для числа: 1 сид, 3 сида, 5 сидов."""
    if n % 10 == 1 and n % 100 != 11:  # noqa: PLR2004 — правила русских окончаний
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:  # noqa: PLR2004
        return few
    return many


def seeds_needed(delta: float, variance: float) -> int | None:
    """Сидов на конфигурацию, чтобы различить эффект `delta` (α = 0,05, мощность 0,8)."""
    if delta == 0:
        return None
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
    return max(2, math.ceil(2 * z * z * variance / (delta * delta)))


def _values(
    metric: Metric, spec: TaskSpec, runs: Sequence[RunContract]
) -> tuple[np.ndarray, str | None]:
    values = []
    for run in runs:
        value = metric(prepare(spec.task, run.predictions))
        if not np.isfinite(value):
            return np.array([]), (
                f"{metric.name} не определена на прогоне {run.run_id}: {metric.undefined}"
            )
        values.append(value)
    return np.array(values, dtype=float), None


def _different_data(runs: Sequence[RunContract]) -> str | None:
    reference = runs[0].data_hash
    other = [f"{run.run_id} ({run.data_hash})" for run in runs[1:] if run.data_hash != reference]
    if not other:
        return None
    return f"сиды оценены на разных данных: {runs[0].run_id} ({reference}), {', '.join(other)}"


def _spread(metric: Metric, values: np.ndarray) -> Finding:
    mean, std = rounded(values.mean()), rounded(values.std(ddof=1))
    low, high = rounded(values.min()), rounded(values.max())
    n = len(values)
    message = (
        f"разброс по {n} {plural(n, 'сиду', 'сидам', 'сидам')}: {metric.name} {mean:.3f} ±"
        f" {std:.3f} (min {low:.3f}, max {high:.3f})"
    )
    facts = {"metric": metric.name, "seeds": n, "mean": mean, "std": std, "min": low, "max": high}
    return Finding("seeds.spread", "info", message, REQUIREMENT, (), facts)


def _not_evaluated(n_candidate: int, n_baseline: int) -> Finding:
    runs = plural(n_candidate, "прогон", "прогона", "прогонов")
    message = (
        f"разброс по сидам не оценён: у кандидата {n_candidate} {runs}, у базового"
        f" {n_baseline} — интервал только по примерам; нужно хотя бы {RECOMMENDED_SEEDS} сида"
        " на конфигурацию"
    )
    facts = {
        "candidate_seeds": n_candidate,
        "baseline_seeds": n_baseline,
        "recommended": RECOMMENDED_SEEDS,
    }
    suggestion = (
        f"повторить обе конфигурации хотя бы на {RECOMMENDED_SEEDS} сидах и передать их в ревью:"
        " mlreview review <сиды кандидата> --baseline <сиды базового>"
    )
    return Finding("seeds.not_evaluated", "low", message, REQUIREMENT, (), facts, suggestion)


def _comparison(metric: Metric, candidate: np.ndarray, baseline: np.ndarray) -> Finding:
    c_mean, b_mean = float(candidate.mean()), float(baseline.mean())
    c_std, b_std = float(candidate.std(ddof=1)), float(baseline.std(ddof=1))
    delta = c_mean - b_mean
    if c_std == 0 and b_std == 0:
        p_value = 0.0 if delta != 0 else 1.0
    else:
        p_value = float(ttest_ind(candidate, baseline, equal_var=False).pvalue)
    variance = (c_std**2 + b_std**2) / 2
    needed = seeds_needed(delta, variance)
    facts = {
        "metric": metric.name,
        "baseline_mean": rounded(b_mean),
        "candidate_mean": rounded(c_mean),
        "delta": rounded(delta),
        "baseline_std": rounded(b_std),
        "candidate_std": rounded(c_std),
        "sigma": rounded(math.sqrt(variance)),
        "p_value": rounded(p_value),
        "baseline_seeds": len(baseline),
        "candidate_seeds": len(candidate),
        "seeds_needed": needed,
    }
    worse = is_worse(metric, delta)
    significant = p_value < ALPHA
    n_text = f"{len(candidate)} и {len(baseline)} {plural(len(baseline), 'сид', 'сида', 'сидов')}"
    body = (
        f"{metric.name} {facts['baseline_mean']:.3f} → {facts['candidate_mean']:.3f}"
        f" (Δ {facts['delta']:+.3f}), σ по сидам {facts['sigma']:.3f},"
        f" p = {facts['p_value']:.2f} (Уэлч, {n_text})"
    )
    suggestion = None
    if significant:
        check = "seeds.significant"
        severity, head = ("high", "ухудшение") if worse else ("info", "улучшение")
        message = f"{head} устойчиво по сидам: {body}"
        suggestion = DECLINE_SUGGESTION if worse else None
    else:
        check = "seeds.within_noise"
        severity, head = ("info", "ухудшение") if worse or delta == 0 else ("high", "улучшение")
        tail = (
            f"для эффекта такого размера нужно ≈ {needed}"
            f" {plural(needed, 'сид', 'сида', 'сидов')} на конфигурацию"
            if needed
            else "разницы нет"
        )
        message = f"{head} в пределах шума по сидам: {body}; {tail}"
        if severity == "high":
            suggestion = (
                f"не считать улучшение доказанным: повторить обе конфигурации на {needed} сидах"
                " или искать больший эффект; различающий эксперимент — сравнить конфигурации"
                " на одних и тех же сидах"
            )
    return Finding(check, severity, message, f"{REQUIREMENT}, B08", (), facts, suggestion)


def _spread_results(metric: Metric, spec: TaskSpec, runs: Sequence[RunContract]) -> CheckResults:
    results = CheckResults()
    if len(runs) < 2:  # noqa: PLR2004 — разброс считается хотя бы по двум сидам
        results.skip("seeds.spread", "один прогон — разброс по сидам не оценён")
        return results
    values, reason = _values(metric, spec, runs)
    reason = _different_data(runs) or reason
    if reason:
        results.skip("seeds.spread", reason)
    else:
        results.findings.append(_spread(metric, values))
    return results


def _comparison_results(
    metric: Metric,
    spec: TaskSpec,
    candidates: Sequence[RunContract],
    baselines: Sequence[RunContract],
) -> CheckResults:
    results = CheckResults()
    if len(candidates) < 2 or len(baselines) < 2:  # noqa: PLR2004
        results.findings.append(_not_evaluated(len(candidates), len(baselines)))
        return results
    candidate, undefined = _values(metric, spec, candidates)
    baseline, undefined_base = _values(metric, spec, baselines)
    reason = _different_data([*candidates, *baselines]) or undefined or undefined_base
    if reason:
        results.skip("seeds.within_noise", reason)
    else:
        results.findings.append(_comparison(metric, candidate, baseline))
    return results


def seed_findings(
    spec: TaskSpec,
    candidates: Sequence[RunContract],
    baselines: Sequence[RunContract] | None,
) -> CheckResults:
    """Разброс по сидам кандидата и (с базовым прогоном) значимость разницы конфигураций."""
    metric = primary(spec)
    if not baselines:
        return _spread_results(metric, spec, candidates)
    return _comparison_results(metric, spec, candidates, baselines)
