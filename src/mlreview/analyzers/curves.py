"""Анализ кривых обучения: переобучение, недообучение, нестабильность, расхождение (FR-D04).

Роль кривой (train/val) и направление метрики определяются по имени. Пороги и формулировки
зафиксированы в `docs/specs/learning-curves.md`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlreview.analyzers.evaluation import rounded
from mlreview.core.findings import CheckResults, Finding

REQUIREMENT = "FR-D04"
ROLE_WORDS = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "eval": "val",
    "dev": "val",
    "test": "test",
}
LOWER_IS_BETTER = frozenset(
    {"loss", "logloss", "error", "err", "mae", "mse", "rmse", "nll", "perplexity", "ppl"}
    | {"brier", "ece"}
)
HIGHER_IS_BETTER = frozenset(
    {"acc", "accuracy", "auc", "f1", "precision", "recall", "r2", "map", "ndcg", "iou"}
    | {"dice", "score"}
)
MIN_VAL_POINTS = 4
MIN_POINTS_AFTER_BEST = 2
OVERFIT_LOST = 0.10
OVERFIT_MEDIUM = 0.25
UNDERFIT_TAIL = 0.25
UNDERFIT_SHARE = 0.10
MIN_UNSTABLE_POINTS = 5
SPIKE_SHARE = 0.25
MIN_SPIKES = 2


@dataclass(frozen=True)
class Curve:
    """Кривая: имя, роль, метрика, направление и конечные точки до первой неконечной."""

    name: str
    role: str
    metric: str
    lower_is_better: bool | None
    steps: np.ndarray
    values: np.ndarray
    first_bad_step: int | None
    non_finite: int

    @property
    def goodness(self) -> np.ndarray:
        """Значения так, что больше — лучше."""
        return -self.values if self.lower_is_better else self.values


def parse_name(name: str) -> tuple[str, str]:
    """Роль и метрика по имени кривой: `val/loss` → (`val`, `loss`)."""
    words = [word for word in re.split(r"[^0-9a-z]+", name.lower()) if word]
    roles = [ROLE_WORDS[word] for word in words if word in ROLE_WORDS]
    metric = "_".join(word for word in words if word not in ROLE_WORDS) or name.lower()
    return (roles[0] if roles else "train"), metric


def direction(metric: str) -> bool | None:
    """`True` — меньше лучше, `False` — больше лучше, `None` — неизвестно."""
    words = set(metric.split("_"))
    if words & LOWER_IS_BETTER:
        return True
    if words & HIGHER_IS_BETTER:
        return False
    return None


def _curve(name: str, frame: pd.DataFrame) -> Curve:
    frame = frame.sort_values("step", kind="mergesort")
    steps = frame["step"].to_numpy(dtype=float).astype(int)
    values = frame["value"].to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    first = int(np.argmax(bad)) if bad.any() else len(values)
    role, metric = parse_name(name)
    return Curve(
        name=name,
        role=role,
        metric=metric,
        lower_is_better=direction(metric),
        steps=steps[:first],
        values=values[:first],
        first_bad_step=int(steps[first]) if bad.any() else None,
        non_finite=int(bad.sum()),
    )


def _divergence(curve: Curve) -> Finding:
    facts = {
        "curve": curve.name,
        "first_step": curve.first_bad_step,
        "non_finite": curve.non_finite,
    }
    message = (
        f"кривая {curve.name} содержит NaN/inf с шага {curve.first_bad_step} — обучение разошлось"
    )
    suggestion = (
        "уменьшить шаг обучения, добавить клиппинг градиента, проверить входы на NaN;"
        " различающий эксперимент — повторить с меньшим lr и сравнить шаг, где появляется NaN"
    )
    return Finding("curves.divergence", "high", message, REQUIREMENT, (), facts, suggestion)


def _value_at(curve: Curve, step: int) -> float:
    """Последнее значение кривой на шаге не позже `step` (или первое, если раньше нет)."""
    index = int(np.searchsorted(curve.steps, step, side="right")) - 1
    return float(curve.values[max(index, 0)])


def _overfitting(train: Curve, val: Curve) -> Finding | None:
    goodness = val.goodness
    n = len(goodness)
    if n < MIN_VAL_POINTS or len(train.values) == 0:
        return None
    best = int(np.argmax(goodness))
    last = n - 1
    gain = goodness[best] - goodness[0]
    if last - best < MIN_POINTS_AFTER_BEST or gain <= 0:
        return None
    lost = (goodness[best] - goodness[last]) / gain
    best_step, last_step = int(val.steps[best]), int(val.steps[last])
    train_at_best, train_last = _value_at(train, best_step), float(train.values[-1])
    train_improved = (train_last < train_at_best) == bool(val.lower_is_better)
    if lost < OVERFIT_LOST or train_last == train_at_best or not train_improved:
        return None
    facts = {
        "metric": val.metric,
        "val_curve": val.name,
        "train_curve": train.name,
        "best_step": best_step,
        "best_value": rounded(val.values[best]),
        "last_step": last_step,
        "last_value": rounded(val.values[last]),
        "lost_share": rounded(lost),
        "train_at_best": rounded(train_at_best),
        "train_last": rounded(train_last),
    }
    message = (
        f"переобучение: {val.name} лучший на шаге {best_step} ({facts['best_value']:.3f}),"
        f" к шагу {last_step} — {facts['last_value']:.3f}"
        f" (потеряно {facts['lost_share']:.0%} улучшения), а {train.name} продолжает"
        f" улучшаться ({facts['train_at_best']:.3f} → {facts['train_last']:.3f})"
    )
    suggestion = (
        f"ранняя остановка по {val.name} (лучший шаг {best_step}) или регуляризация;"
        f" различающий эксперимент — оценить на test чекпойнты шагов {best_step} и {last_step}:"
        " если ранний лучше и на test, модель переобучается"
    )
    severity = "medium" if lost >= OVERFIT_MEDIUM else "low"
    return Finding("curves.overfitting", severity, message, REQUIREMENT, (), facts, suggestion)


def _underfitting(curve: Curve) -> Finding | None:
    goodness = curve.goodness
    n = len(goodness)
    if n < MIN_VAL_POINTS:
        return None
    last = n - 1
    total = goodness[last] - goodness[0]
    if goodness[last] < goodness.max() or total <= 0:
        return None
    tail = max(1, math.floor(n * UNDERFIT_TAIL))
    share = (goodness[last] - goodness[last - tail]) / total
    if share < UNDERFIT_SHARE:
        return None
    facts = {
        "curve": curve.name,
        "metric": curve.metric,
        "points": n,
        "tail_points": tail,
        "tail_share": rounded(share),
        "first_value": rounded(curve.values[0]),
        "last_value": rounded(curve.values[last]),
    }
    message = (
        f"недообучение: {curve.name} ещё улучшается в конце обучения (последние {tail} из {n}"
        f" точек дали {facts['tail_share']:.0%} всего улучшения) — вероятно, обучение"
        " остановлено рано"
    )
    suggestion = (
        "обучать дольше или ослабить регуляризацию; различающий эксперимент — продолжить"
        f" обучение и сравнить {curve.name}: если она продолжает улучшаться, модель недообучена"
    )
    return Finding("curves.underfitting", "low", message, REQUIREMENT, (), facts, suggestion)


def _instability(curve: Curve) -> Finding | None:
    goodness = curve.goodness
    if len(goodness) < MIN_UNSTABLE_POINTS:
        return None
    spread = float(goodness.max() - goodness.min())
    if spread <= 0:
        return None
    worsening = goodness[:-1] - goodness[1:]
    spikes = np.flatnonzero(worsening > SPIKE_SHARE * spread) + 1
    if len(spikes) < MIN_SPIKES:
        return None
    facts = {
        "curve": curve.name,
        "spikes": len(spikes),
        "first_step": int(curve.steps[spikes[0]]),
        "largest": rounded(worsening.max()),
        "range": rounded(spread),
    }
    message = (
        f"нестабильность: {curve.name} резко ухудшается {len(spikes)} раз (скачки больше"
        f" {SPIKE_SHARE:.0%} размаха кривой), первый — на шаге {facts['first_step']}"
    )
    suggestion = (
        "уменьшить шаг обучения, добавить клиппинг градиента, проверить батчи с выбросами;"
        " различающий эксперимент — повторить с меньшим lr и сравнить число скачков"
    )
    return Finding("curves.instability", "medium", message, REQUIREMENT, (), facts, suggestion)


def curve_findings(curves: pd.DataFrame | None) -> CheckResults:
    """Проверки кривых обучения контракта (`name`, `step`, `value`)."""
    results = CheckResults()
    if curves is None or curves.empty:
        results.skip("curves", "нет curves — кривые обучения не залогированы")
        return results
    parsed = [_curve(str(name), frame) for name, frame in curves.groupby("name", sort=True)]
    for curve in parsed:
        if curve.non_finite:
            results.findings.append(_divergence(curve))
    unknown = [curve.name for curve in parsed if curve.lower_is_better is None]
    if unknown:
        results.skip("curves", f"направление неизвестно: {', '.join(unknown)}")
    known = [c for c in parsed if c.lower_is_better is not None and c.role != "test"]

    metrics = sorted({curve.metric for curve in known})
    has_pair = False
    for metric in metrics:
        trains = [c for c in known if c.metric == metric and c.role == "train"]
        vals = [c for c in known if c.metric == metric and c.role == "val"]
        if trains and vals:
            has_pair = True
            finding = _overfitting(trains[0], vals[0])
            if finding is not None:
                results.findings.append(finding)
        main = vals[0] if vals else trains[0]
        finding = _underfitting(main)
        if finding is not None:
            results.findings.append(finding)
    for curve in known:
        finding = _instability(curve)
        if finding is not None:
            results.findings.append(finding)
    if not has_pair:
        results.skip("curves.overfitting", "нет пары train/val для одной метрики")
    return results
