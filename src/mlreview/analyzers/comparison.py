"""Метрики прогона и сравнение двух прогонов с интервалами и значимостью (FR-D01).

Поведение зафиксировано в `docs/specs/run-comparison.md`.
"""

from __future__ import annotations

from mlreview.analyzers.evaluation import (
    METRICS,
    Estimate,
    estimate_value,
    is_worse,
    metric_names,
    pair_runs,
    paired_delta,
    prepare,
    rounded,
)
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec

VALUE_CHECK = "metrics.value"
DELTA_CHECK = "comparison.metric_delta"
REQUIREMENT = "FR-D01"
DECLINE_SUGGESTION = (
    "сравнить конфиги прогонов и примеры с наибольшим ростом ошибки; если отличий несколько,"
    " проверить их по одному"
)


def run_metrics(spec: TaskSpec, contract: RunContract) -> CheckResults:
    """Метрики одного прогона с 95% интервалами (находки `metrics.value`)."""
    results = CheckResults()
    outputs = prepare(spec.task, contract.predictions)
    for name in metric_names(spec.task):
        estimate = estimate_value(METRICS[name], outputs)
        if isinstance(estimate, str):
            results.skip(VALUE_CHECK, estimate)
            continue
        value, low, high = (rounded(x) for x in (estimate.value, estimate.low, estimate.high))
        message = f"{name} = {value:.3f}, 95% CI [{low:.3f}; {high:.3f}], n = {estimate.n}"
        facts = {"metric": name, "value": value, "ci_low": low, "ci_high": high, "n": estimate.n}
        results.findings.append(Finding(VALUE_CHECK, "info", message, REQUIREMENT, facts=facts))
    return results


def _delta_finding(name: str, primary: bool, base: float, cand: float, est: Estimate) -> Finding:
    metric = METRICS[name]
    base, cand = rounded(base), rounded(cand)
    delta = rounded(cand - base)
    low, high = rounded(est.low), rounded(est.high)
    significant = low > 0 or high < 0
    suggestion = None
    if not significant:
        severity, verdict = "info", "в пределах шума (CI включает 0)"
    elif is_worse(metric, delta):
        severity, verdict = ("high" if primary else "medium"), "значимо, хуже"
        suggestion = DECLINE_SUGGESTION
    else:
        severity, verdict = "info", "значимо, лучше"
    message = (
        f"{name}: {base:.3f} → {cand:.3f}, Δ {delta:+.3f}, 95% CI [{low:+.3f}; {high:+.3f}]"
        f" — {verdict}"
    )
    facts = {
        "metric": name,
        "baseline": base,
        "candidate": cand,
        "delta": delta,
        "ci_low": low,
        "ci_high": high,
        "p_value": rounded(est.p_value),
        "n": est.n,
    }
    return Finding(DELTA_CHECK, severity, message, REQUIREMENT, facts=facts, suggestion=suggestion)


def compare_runs(spec: TaskSpec, candidate: RunContract, baseline: RunContract) -> CheckResults:
    """Дельты метрик кандидата относительно базового прогона по общим примерам."""
    results = CheckResults()
    pair = pair_runs(spec.task, candidate, baseline)
    if isinstance(pair, str):
        results.skip(DELTA_CHECK, pair)
        return results
    for position, name in enumerate(metric_names(spec.task)):
        metric = METRICS[name]
        estimate = paired_delta(metric, pair.candidate, pair.baseline)
        if isinstance(estimate, str):
            results.skip(DELTA_CHECK, estimate)
            continue
        base, cand = metric(pair.baseline), metric(pair.candidate)
        results.findings.append(_delta_finding(name, position == 0, base, cand, estimate))
    return results
