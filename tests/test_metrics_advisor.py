"""Рекомендации метрик по ответам диалога (FR-M01, FR-M02).

Спецификация: docs/specs/metrics.md (пункты B4, E3).
"""

from __future__ import annotations

import pytest

from mlreview.core.task_spec import MetricsContext, parse_task_spec
from mlreview.metrics.advisor import recommend


def _spec(task_type, output):
    return parse_task_spec(f"task: {{type: {task_type}, output: {output}}}\n")


def _plan(task_type, output, positive_rate=None, **context):
    return recommend(_spec(task_type, output), MetricsContext(**context), positive_rate)


def test_binary_ranking_with_imbalance():
    """B4: ranking, K=50, доля 0,02 — precision@k, recall@k, pr_auc."""
    plan = _plan("classification/binary", "probabilities", 0.02, decision="ranking", budget_k=50)
    assert plan.primary == "precision_at_k@50"
    assert plan.secondary == ["recall_at_k@50", "pr_auc"]
    assert not plan.limited_support
    for item in plan.items:
        assert item.reason
        assert item.pitfalls
    assert plan.items[0].role == "primary"


@pytest.mark.parametrize(
    ("args", "context", "primary", "secondary"),
    [
        (
            ("classification/binary", "probabilities", 0.3),
            {"decision": "ranking", "budget_k": 30},
            "precision_at_k@30",
            ["recall_at_k@30", "roc_auc"],
        ),
        (
            ("classification/binary", "probabilities", 0.3),
            {"decision": "threshold", "cost_ratio": 4},
            "fbeta@2",
            ["precision", "recall", "roc_auc"],
        ),
        (
            ("classification/binary", "labels", None),
            {"decision": "threshold", "cost_ratio": 1},
            "f1",
            ["precision", "recall"],
        ),
        (
            ("classification/binary", "probabilities", 0.3),
            {"decision": "probability"},
            "log_loss",
            ["ece", "brier", "roc_auc"],
        ),
        (
            ("classification/binary", "probabilities", 0.02),
            {"decision": "probability"},
            "log_loss",
            ["ece", "roc_auc"],
        ),
        (
            ("classification/multiclass", "probabilities", None),
            {"decision": "ranking", "budget_k": 3},
            "top_k_accuracy@3",
            ["macro_f1"],
        ),
        (
            ("classification/multiclass", "labels", None),
            {"decision": "threshold"},
            "macro_f1",
            ["balanced_accuracy", "accuracy"],
        ),
        (
            ("classification/multiclass", "probabilities", None),
            {"decision": "probability"},
            "log_loss",
            ["ece", "macro_f1"],
        ),
        (("regression", "values", None), {"cost_ratio": 3}, "pinball@0.75", ["mae"]),
        (("regression", "values", None), {"large_errors": True}, "rmse", ["mae"]),
        (("regression", "values", None), {}, "mae", ["rmse"]),
    ],
)
def test_rules_table(args, context, primary, secondary):
    """B4: строки таблицы правил."""
    plan = _plan(*args, **context)
    assert plan.primary == primary
    assert plan.secondary == secondary


def test_calibration_adds_ece_and_threshold_imbalance_uses_pr_auc():
    """B4: calibration добавляет ece; дисбаланс — pr_auc вместо roc_auc."""
    plan = _plan(
        "classification/binary",
        "probabilities",
        0.05,
        decision="threshold",
        cost_ratio=1,
        calibration=True,
    )
    assert plan.primary == "f1"
    assert plan.secondary == ["precision", "recall", "pr_auc", "ece"]


def test_ranking_with_labels_falls_back():
    """Ранжирование при output: labels невозможно — правило порога с пояснением."""
    plan = _plan("classification/binary", "labels", None, decision="ranking", budget_k=10)
    assert plan.primary == "f1"
    assert any("labels" in note for note in plan.notes)


def test_multilabel_is_limited():
    """E3: multilabel — ограниченная поддержка."""
    plan = _plan("classification/multilabel", "probabilities", None, decision="threshold")
    assert plan.limited_support
    assert plan.items == []
    assert plan.notes == [
        "ограниченная поддержка: в библиотеке нет метрик для classification/multilabel"
    ]
