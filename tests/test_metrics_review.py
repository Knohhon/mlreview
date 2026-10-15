"""Метрики спецификации в ревью и пригодность метрик (FR-M02, FR-M03, B05).

Спецификация: docs/specs/metrics.md (пункты B3, B11).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _factories import binary_tables
from mlreview.analyzers.review import review
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

TASK = "task: {type: classification/binary, output: probabilities, segments: [size]}\n"


def _contract(spec, run_id="cand", seed=1):
    predictions, examples = binary_tables(seed=seed)
    return build_run_contract(spec, run_id=run_id, predictions=predictions, examples=examples)


def test_review_uses_spec_metrics():
    """B3: metrics.value и сравнение — по метрикам раздела; срезы — по primary."""
    spec = parse_task_spec(TASK + "metrics: {primary: pr_auc, secondary: [roc_auc, f1]}\n")
    single = review(spec, _contract(spec), groups=("metrics",))
    values = [f.facts["metric"] for f in single.findings if f.check == "metrics.value"]
    assert sorted(values) == ["f1", "pr_auc", "roc_auc"]  # отчёт сортирует по сообщению
    compared = review(spec, _contract(spec), _contract(spec, "base", seed=2), groups=("metrics",))
    deltas = [f.facts["metric"] for f in compared.findings if f.check == "comparison.metric_delta"]
    assert sorted(deltas) == ["f1", "pr_auc", "roc_auc"]
    segments = review(spec, _contract(spec), _contract(spec, "base", seed=2), groups=("segments",))
    for skipped in segments.skipped:
        assert "roc_auc" not in skipped.reason


def _imbalanced(spec, n=500):
    rng = np.random.default_rng(0)
    labels = np.zeros(n, dtype=int)
    labels[:: n // 10] = 1  # 10 положительных из 500 — 1:49
    scores = np.clip(labels * 0.4 + rng.uniform(0, 0.6, n), 0, 1)
    predictions = pd.DataFrame(
        {"example_id": [f"e{i:03d}" for i in range(n)], "label": labels, "prediction": scores}
    )
    return build_run_contract(spec, run_id="cand", predictions=predictions)


def test_unsuitable_accuracy_is_reported():
    """B11 (B05): accuracy при дисбалансе 1:50 — high; roc_auc — без находки."""
    spec = parse_task_spec(
        "task: {type: classification/binary, output: probabilities}\n"
        "metrics: {primary: accuracy, secondary: [roc_auc]}\n"
    )
    report = review(spec, _imbalanced(spec), groups=("metrics",))
    unsuitable = [f for f in report.findings if f.check == "metrics.unsuitable"]
    (finding,) = unsuitable
    assert finding.severity == "high"
    assert finding.facts["metric"] == "accuracy"
    assert finding.facts["role"] == "primary"
    assert finding.facts["baseline_value"] == 0.98
    assert finding.facts["positive_rate"] == 0.02
    assert finding.message.startswith("accuracy вырождена на этих данных: ")
    assert "0.980" in finding.message
    assert "B05" in finding.requirement


def test_secondary_unsuitable_metric_is_medium():
    """B11: дополнительная вырожденная метрика — medium."""
    spec = parse_task_spec(
        "task: {type: classification/binary, output: probabilities}\n"
        "metrics: {primary: pr_auc, secondary: [accuracy, brier]}\n"
    )
    report = review(spec, _imbalanced(spec), groups=("metrics",))
    by_metric = {
        f.facts["metric"]: f.severity for f in report.findings if f.check == "metrics.unsuitable"
    }
    assert by_metric == {"accuracy": "medium", "brier": "medium"}
