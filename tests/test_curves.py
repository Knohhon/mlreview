"""Анализ кривых обучения (FR-D04).

Спецификация: docs/specs/learning-curves.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlreview.analyzers.curves import curve_findings


def _curves(**curves):
    rows = [
        (name, step, value) for name, values in curves.items() for step, value in enumerate(values)
    ]
    return pd.DataFrame(rows, columns=["name", "step", "value"])


def _checks(results):
    return [finding.check for finding in results.findings]


def test_overfitting():
    """B1: val_loss растёт после шага 3 при падающем train_loss."""
    train = [1.0 - 0.08 * step for step in range(10)]
    val = [1.0, 0.8, 0.65, 0.6, 0.62, 0.66, 0.7, 0.73, 0.76, 0.78]
    results = curve_findings(_curves(train_loss=train, val_loss=val))
    (finding,) = results.findings
    assert finding.check == "curves.overfitting"
    assert finding.severity == "medium"
    assert finding.facts == {
        "metric": "loss",
        "val_curve": "val_loss",
        "train_curve": "train_loss",
        "best_step": 3,
        "best_value": 0.6,
        "last_step": 9,
        "last_value": 0.78,
        "lost_share": 0.45,
        "train_at_best": 0.76,
        "train_last": 0.28,
    }
    assert finding.message == (
        "переобучение: val_loss лучший на шаге 3 (0.600), к шагу 9 — 0.780"
        " (потеряно 45% улучшения), а train_loss продолжает улучшаться (0.760 → 0.280)"
    )
    assert finding.requirement == "FR-D04"
    assert finding.suggestion


def test_mild_overfitting_is_low():
    """B1: потеряно от 10% до 25% улучшения — low."""
    train = [1.0 - 0.08 * step for step in range(8)]
    val = [1.0, 0.8, 0.6, 0.62, 0.64, 0.66, 0.67, 0.68]
    (finding,) = curve_findings(_curves(train_loss=train, val_loss=val)).findings
    assert (finding.check, finding.severity) == ("curves.overfitting", "low")


def test_underfitting():
    """B2: val_auc ещё растёт в конце."""
    val = [0.6, 0.7, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86]
    train = [value + 0.05 for value in val]
    results = curve_findings(_curves(train_auc=train, val_auc=val))
    (finding,) = results.findings
    assert (finding.check, finding.severity) == ("curves.underfitting", "low")
    assert finding.message.startswith("недообучение: val_auc ещё улучшается в конце обучения")
    assert finding.facts["curve"] == "val_auc"
    assert finding.facts["tail_share"] == 0.153846


def test_instability():
    """B3: три скачка вверх больше 25% размаха."""
    values = [1.0 - 0.045 * step for step in range(20)]
    for step in (5, 10, 15):
        values[step] += 0.4
    results = curve_findings(_curves(train_loss=values))
    by_check = {finding.check: finding for finding in results.findings}
    finding = by_check["curves.instability"]
    assert finding.severity == "medium"
    assert finding.facts["spikes"] == 3
    assert finding.facts["first_step"] == 5
    assert finding.facts["curve"] == "train_loss"


def test_divergence():
    """B4: NaN с шага 7 — расхождение; остальное — по точкам до него."""
    values = [1.0 - 0.1 * step for step in range(7)] + [np.nan, np.nan, np.inf]
    results = curve_findings(_curves(train_loss=values))
    divergence = [f for f in results.findings if f.check == "curves.divergence"]
    (finding,) = divergence
    assert finding.severity == "high"
    assert finding.message == "кривая train_loss содержит NaN/inf с шага 7 — обучение разошлось"
    assert finding.facts == {"curve": "train_loss", "first_step": 7, "non_finite": 3}


def test_healthy_curves_have_no_findings():
    """B5: обе кривые падают и выходят на плато."""
    train = [1 / (1 + step) for step in range(10)]
    val = [1.1 / (1 + step) + 0.05 for step in range(10)]
    results = curve_findings(_curves(train_loss=train, val_loss=val))
    assert results.findings == []
    assert results.skipped == []


def test_keras_names_form_a_pair():
    """B6: loss и val_loss — пара метрики loss."""
    train = [1.0 - 0.08 * step for step in range(10)]
    val = [1.0, 0.8, 0.65, 0.6, 0.62, 0.66, 0.7, 0.73, 0.76, 0.78]
    (finding,) = curve_findings(_curves(loss=train, val_loss=val)).findings
    assert finding.facts["train_curve"] == "loss"
    assert finding.facts["metric"] == "loss"


def test_no_curves_is_skipped():
    """E1: нет curves."""
    results = curve_findings(None)
    assert [(s.check, s.reason) for s in results.skipped] == [
        ("curves", "нет curves — кривые обучения не залогированы")
    ]


def test_unknown_direction_and_missing_pair():
    """E2, E3: неизвестное направление; нет пары train/val."""
    train = [1 / (1 + step) for step in range(10)]
    results = curve_findings(_curves(lr=[0.1] * 5, train_loss=train))
    reasons = {s.check: s.reason for s in results.skipped}
    assert reasons["curves"] == "направление неизвестно: lr"
    assert reasons["curves.overfitting"] == "нет пары train/val для одной метрики"
