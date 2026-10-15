"""Реализации метрик на numpy против эталона (FR-M02, FR-M04).

Спецификация: docs/specs/metrics.md (пункт B2).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlreview.metrics import kernels

rng = np.random.default_rng(0)
N = 400
BINARY = rng.integers(0, 2, N)
SCORES = np.round(rng.uniform(0, 1, N), 2)  # с повторами: проверка равных значений
PROBS = np.clip(BINARY * 0.3 + rng.uniform(0.05, 0.7, N), 0.01, 0.99)
CLASSES = (PROBS >= 0.5).astype(int)
MULTI = rng.integers(0, 4, N)
MATRIX = rng.dirichlet(np.ones(4), N)
MULTI_PRED = np.argmax(MATRIX, axis=1)
Y = rng.normal(3, 1, N)
Y_HAT = Y + rng.normal(0, 0.5, N)


def test_hand_computed_values():
    """B2: значения, посчитанные вручную."""
    labels = np.array([0, 0, 1, 1])
    assert kernels.roc_auc(labels, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert kernels.precision_at_k(labels, np.array([0.1, 0.4, 0.35, 0.8]), 2) == 0.5
    assert kernels.recall_at_k(labels, np.array([0.1, 0.4, 0.35, 0.8]), 1) == 0.5
    assert kernels.accuracy(labels, np.array([0, 1, 1, 1])) == 0.75
    assert kernels.balanced_accuracy(labels, np.array([0, 1, 1, 1])) == 0.75
    assert kernels.pinball(np.array([1.0, 2.0]), np.array([2.0, 1.0]), 0.9) == pytest.approx(0.5)
    assert kernels.mape(np.array([1.0, 2.0]), np.array([1.5, 1.0])) == pytest.approx(0.5)
    assert np.isnan(kernels.mape(np.array([0.0, 2.0]), np.array([1.0, 1.0])))
    assert np.isnan(kernels.roc_auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_against_scikit_learn():
    """B2: совпадение со scikit-learn до 1e-9."""
    metrics = pytest.importorskip("sklearn.metrics")
    pairs = [
        (kernels.roc_auc(BINARY, SCORES), metrics.roc_auc_score(BINARY, SCORES)),
        (kernels.pr_auc(BINARY, SCORES), metrics.average_precision_score(BINARY, SCORES)),
        (kernels.log_loss(BINARY, PROBS), metrics.log_loss(BINARY, PROBS)),
        (kernels.log_loss(MULTI, MATRIX), metrics.log_loss(MULTI, MATRIX, labels=[0, 1, 2, 3])),
        (kernels.brier(BINARY, PROBS), metrics.brier_score_loss(BINARY, PROBS)),
        (kernels.accuracy(BINARY, CLASSES), metrics.accuracy_score(BINARY, CLASSES)),
        (
            kernels.balanced_accuracy(MULTI, MULTI_PRED),
            metrics.balanced_accuracy_score(MULTI, MULTI_PRED),
        ),
        (kernels.precision(BINARY, CLASSES), metrics.precision_score(BINARY, CLASSES)),
        (kernels.recall(BINARY, CLASSES), metrics.recall_score(BINARY, CLASSES)),
        (kernels.f1(BINARY, CLASSES), metrics.f1_score(BINARY, CLASSES)),
        (kernels.fbeta(BINARY, CLASSES, 2.0), metrics.fbeta_score(BINARY, CLASSES, beta=2.0)),
        (
            kernels.macro_f1(MULTI, MULTI_PRED),
            metrics.f1_score(MULTI, MULTI_PRED, average="macro"),
        ),
        (
            kernels.top_k_accuracy(MULTI, MATRIX, 2),
            metrics.top_k_accuracy_score(MULTI, MATRIX, k=2),
        ),
        (kernels.mae(Y, Y_HAT), metrics.mean_absolute_error(Y, Y_HAT)),
        (kernels.rmse(Y, Y_HAT), np.sqrt(metrics.mean_squared_error(Y, Y_HAT))),
        (kernels.median_ae(Y, Y_HAT), metrics.median_absolute_error(Y, Y_HAT)),
        (kernels.r2(Y, Y_HAT), metrics.r2_score(Y, Y_HAT)),
        (kernels.mape(Y, Y_HAT), metrics.mean_absolute_percentage_error(Y, Y_HAT)),
        (kernels.pinball(Y, Y_HAT, 0.8), metrics.mean_pinball_loss(Y, Y_HAT, alpha=0.8)),
    ]
    for ours, reference in pairs:
        assert ours == pytest.approx(reference, abs=1e-9)


def test_ece_kernel_matches_analyzer():
    """ECE в kernels и в анализаторе калибровки одинаковы."""
    from mlreview.analyzers.calibration import ece  # noqa: PLC0415

    assert kernels.ece(BINARY, PROBS) == pytest.approx(ece(BINARY, PROBS), abs=1e-12)
