"""Калибровка вероятностей и примеры расхождения двух моделей (FR-D08).

ECE — 10 корзин равной ширины; интервал дельты — парный бутстрап с фиксированным сидом
(NFR-05). Поведение зафиксировано в `docs/specs/calibration.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlreview.analyzers.evaluation import (
    Metric,
    Outputs,
    pair_runs,
    paired_delta,
    prepare,
    rounded,
    top_ids,
)
from mlreview.core.findings import CheckResults, Finding, evidence_ids
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSection, TaskSpec

BINS = 10
MEDIUM_ECE = 0.1
LOW_ECE = 0.05
CONCENTRATION = 0.15
ECE_CHECK = "calibration.ece"
DELTA_CHECK = "calibration.ece_delta"
DISAGREEMENT_CHECK = "comparison.disagreement"
REQUIREMENT = "FR-D08"
CALIBRATE = (
    "калибровать вероятности на валидации (temperature или Platt scaling, изотоническая"
    " регрессия) и проверить reliability diagram; различающий эксперимент — сравнить ECE"
    " до и после калибровки на test"
)


def _bins(confidence: np.ndarray) -> np.ndarray:
    return np.minimum((confidence * BINS).astype(int), BINS - 1)


def _ece_parts(confidence: np.ndarray, correct: np.ndarray) -> tuple[float, np.ndarray]:
    """ECE и разрыв |точность − уверенность| по корзинам (NaN для пустых)."""
    index = _bins(confidence)
    counts = np.bincount(index, minlength=BINS)
    conf = np.bincount(index, weights=confidence, minlength=BINS)
    hits = np.bincount(index, weights=correct.astype(float), minlength=BINS)
    value = float(np.sum(np.abs(hits - conf)) / len(confidence))
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = np.abs(hits - conf) / counts
    return value, np.where(counts > 0, gaps, np.nan)


def ece(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """ECE бинарной классификации: вероятность класса 1 против доли меток 1."""
    return _ece_parts(np.asarray(probabilities, dtype=float), np.asarray(labels) == 1)[0]


def _confidence(outputs: Outputs) -> tuple[np.ndarray, np.ndarray]:
    if outputs.values.ndim == 1:
        return outputs.values, outputs.labels == 1
    predicted = np.argmax(outputs.values, axis=1)
    return outputs.values.max(axis=1), predicted == outputs.labels


def _ece_metric(outputs: Outputs) -> float:
    if len(outputs) == 0:
        return float("nan")
    return _ece_parts(*_confidence(outputs))[0]


ECE = Metric("ece", _ece_metric, False, "нет примеров")


def _single(outputs: Outputs) -> Finding | None:
    confidence, correct = _confidence(outputs)
    value, gaps = _ece_parts(confidence, correct)
    if value < LOW_ECE:
        return None
    worst = int(np.nanargmax(gaps))
    facts = {
        "ece": rounded(value),
        "bins": BINS,
        "n": len(outputs),
        "worst_bin": f"{worst / BINS:.1f}–{(worst + 1) / BINS:.1f}",
        "worst_gap": rounded(gaps[worst]),
    }
    message = (
        f"калибровка: ECE {facts['ece']:.3f} ({BINS} корзин) — вероятности систематически"
        " расходятся с частотами"
    )
    evidence = evidence_ids(outputs.ids[_bins(confidence) == worst])
    severity = "medium" if value >= MEDIUM_ECE else "low"
    return Finding(ECE_CHECK, severity, message, REQUIREMENT, evidence, facts, CALIBRATE)


def _delta(candidate: Outputs, baseline: Outputs) -> Finding | str | None:
    estimate = paired_delta(ECE, candidate, baseline)
    if isinstance(estimate, str):
        return estimate
    if estimate.low <= 0 <= estimate.high:
        return None
    worse = estimate.low > 0
    base, cand = rounded(ECE(baseline)), rounded(ECE(candidate))
    facts = {
        "baseline": base,
        "candidate": cand,
        "delta": rounded(cand - base),
        "ci_low": rounded(estimate.low),
        "ci_high": rounded(estimate.high),
        "n": estimate.n,
    }
    head = "калибровка ухудшилась" if worse else "калибровка улучшилась"
    message = (
        f"{head}: ECE {base:.3f} → {cand:.3f} (Δ {facts['delta']:+.3f}, 95% CI"
        f" [{facts['ci_low']:+.3f}; {facts['ci_high']:+.3f}])"
    )
    severity = "medium" if worse else "info"
    suggestion = CALIBRATE if worse else None
    return Finding(DELTA_CHECK, severity, message, REQUIREMENT, (), facts, suggestion)


def _classes(task: TaskSection, outputs: Outputs) -> np.ndarray:
    if outputs.values.ndim == 2:  # noqa: PLR2004 — матрица вероятностей по классам
        return np.argmax(outputs.values, axis=1)
    if task.output == "probabilities":
        return (outputs.values > 0.5).astype(int)  # noqa: PLR2004 — порог по умолчанию
    return outputs.values.astype(int)


def _segment_values(examples: pd.DataFrame, segment: str, ids: np.ndarray) -> np.ndarray:
    column = examples.set_index(examples["example_id"].astype(str))[segment]
    values = column.reindex(pd.Index(ids.astype(str)))
    return np.array(["<пропуск>" if pd.isna(v) else str(v) for v in values], dtype=object)


def _concentration(
    spec: TaskSpec, examples: pd.DataFrame | None, ids: np.ndarray, differ: np.ndarray
) -> dict[str, object] | None:
    """Значение сегмента, чья доля среди расхождений больше доли среди всех (B5)."""
    if examples is None:
        return None
    best = None
    for segment in spec.task.segments:
        if segment not in examples.columns:
            continue
        values = _segment_values(examples, segment, ids)
        for value in sorted(set(values[differ])):
            inside = float(np.mean(values[differ] == value))
            overall = float(np.mean(values == value))
            if inside - overall >= CONCENTRATION and (best is None or inside - overall > best[0]):
                best = (inside - overall, segment, value, inside, overall)
    if best is None:
        return None
    _, segment, value, inside, overall = best
    return {
        "segment": segment,
        "segment_value": value,
        "segment_share": rounded(inside),
        "overall_share": rounded(overall),
    }


def _disagreement(
    spec: TaskSpec, candidate: Outputs, baseline: Outputs, examples: pd.DataFrame | None
) -> Finding | None:
    cand, base = _classes(spec.task, candidate), _classes(spec.task, baseline)
    differ = cand != base
    k = int(differ.sum())
    if k == 0:
        return None
    n = len(candidate)
    cand_right = int(np.sum(differ & (cand == candidate.labels)))
    base_right = int(np.sum(differ & (base == baseline.labels)))
    facts: dict[str, object] = {
        "disagree": k,
        "total": n,
        "share": rounded(k / n),
        "candidate_right": cand_right,
        "baseline_right": base_right,
    }
    message = (
        f"модели расходятся на {k} из {n} примеров ({k / n:.1%}): кандидат прав в {cand_right},"
        f" базовый — в {base_right}"
    )
    concentration = _concentration(spec, examples, candidate.ids, differ)
    if concentration is not None:
        facts.update(concentration)
        message += (
            f"; {concentration['segment_share']:.0%} из них —"
            f" {concentration['segment']}={concentration['segment_value']}"
            f" (против {concentration['overall_share']:.0%} среди всех)"
        )
    gap = np.abs(candidate.values - baseline.values)
    if gap.ndim == 2:  # noqa: PLR2004
        gap = gap.max(axis=1)
    evidence = top_ids(candidate.ids[differ], gap[differ].astype(float))
    suggestion = (
        "посмотреть примеры расхождения (в доказательствах): где кандидат ошибается, а базовый"
        " прав, — цена перехода на кандидата"
    )
    return Finding(DISAGREEMENT_CHECK, "info", message, REQUIREMENT, evidence, facts, suggestion)


def _compare(
    spec: TaskSpec,
    contract: RunContract,
    baseline: RunContract,
    *,
    probabilities: bool,
    classification: bool,
) -> CheckResults:
    results = CheckResults()
    task = spec.task
    pair = pair_runs(task, contract, baseline)
    if not probabilities:
        results.skip(DELTA_CHECK, "калибровка — только для output: probabilities")
    elif isinstance(pair, str):
        results.skip(DELTA_CHECK, pair)
    else:
        outcome = _delta(pair.candidate, pair.baseline)
        if isinstance(outcome, str):
            results.skip(DELTA_CHECK, outcome)
        elif outcome is not None:
            results.findings.append(outcome)
    if not classification:
        reason = f"расхождение классов не определено для {task.type} · {task.output}"
        results.skip(DISAGREEMENT_CHECK, reason)
    elif isinstance(pair, str):
        results.skip(DISAGREEMENT_CHECK, pair)
    else:
        finding = _disagreement(spec, pair.candidate, pair.baseline, contract.examples)
        if finding is not None:
            results.findings.append(finding)
    return results


def calibration_findings(
    spec: TaskSpec, contract: RunContract, baseline: RunContract | None = None
) -> CheckResults:
    """Калибровка прогона, а с базовым прогоном — её изменение и расхождение моделей."""
    task = spec.task
    probabilities = task.output == "probabilities" and task.type != "classification/multilabel"
    classification = task.type.startswith("classification/") and not (
        task.type == "classification/binary" and task.output == "scores"
    )
    if baseline is not None:
        return _compare(
            spec, contract, baseline, probabilities=probabilities, classification=classification
        )
    results = CheckResults()
    if not probabilities:
        results.skip(ECE_CHECK, "калибровка — только для output: probabilities")
        return results
    finding = _single(prepare(task, contract.predictions))
    if finding is not None:
        results.findings.append(finding)
    return results
