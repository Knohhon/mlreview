"""Метрики прогона и сравнение прогонов (FR-D01).

Спецификация: docs/specs/run-comparison.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlreview.analyzers.comparison import compare_runs, run_metrics
from mlreview.analyzers.evaluation import metric_names, metric_value
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

BINARY_PROBA = "task: {type: classification/binary, output: probabilities}"
REGRESSION = "task: {type: regression, output: values}"


def _spec(text: str):
    return parse_task_spec(text)


def _contract(spec, predictions, run_id="run"):
    return build_run_contract(spec, run_id=run_id, predictions=predictions)


def _binary_runs(n=400, seed=1, candidate_noise=0.5, baseline_noise=2.0):
    """Два прогона на одних примерах: кандидат точнее базового."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n)
    signal = np.where(label == 1, 1.0, -1.0)

    def proba(noise):
        logit = 2 * signal + rng.normal(0, noise, n)
        return 1 / (1 + np.exp(-logit))

    ids = [f"e{i:04d}" for i in range(n)]
    candidate = pd.DataFrame(
        {"example_id": ids, "label": label, "prediction": proba(candidate_noise)}
    )
    baseline = pd.DataFrame(
        {"example_id": ids, "label": label, "prediction": proba(baseline_noise)}
    )
    return candidate, baseline


def _by_metric(results, check):
    return {f.facts["metric"]: f for f in results.findings if f.check == check}


# --- B1: значения и наборы метрик ---


def test_binary_probability_metrics_match_manual():
    """B1: roc_auc, log_loss и brier совпадают с ручным расчётом."""
    task = _spec(BINARY_PROBA).task
    frame = pd.DataFrame(
        {"example_id": list("abcd"), "label": [0, 0, 1, 1], "prediction": [0.1, 0.6, 0.4, 0.8]}
    )
    # пары (pos, neg): (0.4>0.1), (0.4<0.6), (0.8>0.1), (0.8>0.6) → 3/4
    assert metric_value(task, frame, "roc_auc") == pytest.approx(0.75, abs=1e-9)
    expected_log_loss = -np.mean(np.log([0.9, 0.4, 0.4, 0.8]))
    assert metric_value(task, frame, "log_loss") == pytest.approx(expected_log_loss, abs=1e-9)
    expected_brier = np.mean([0.01, 0.36, 0.36, 0.04])
    assert metric_value(task, frame, "brier") == pytest.approx(expected_brier, abs=1e-9)


def test_roc_auc_ties_get_half_credit():
    """B1: равные значения у классов дают половину пары (средний ранг)."""
    task = _spec(BINARY_PROBA).task
    frame = pd.DataFrame({"example_id": list("ab"), "label": [0, 1], "prediction": [0.5, 0.5]})
    assert metric_value(task, frame, "roc_auc") == pytest.approx(0.5)


def test_binary_label_metrics_match_manual():
    """B1: f1, accuracy, precision, recall для бинарных меток."""
    task = _spec("task: {type: classification/binary, output: labels}").task
    # TP=2, FP=1, FN=1, TN=1
    frame = pd.DataFrame(
        {"example_id": list("abcde"), "label": [1, 1, 1, 0, 0], "prediction": [1, 1, 0, 1, 0]}
    )
    assert metric_value(task, frame, "accuracy") == pytest.approx(3 / 5)
    assert metric_value(task, frame, "precision") == pytest.approx(2 / 3)
    assert metric_value(task, frame, "recall") == pytest.approx(2 / 3)
    assert metric_value(task, frame, "f1") == pytest.approx(4 / 6)


def test_regression_metrics_match_manual():
    """B1: mae и rmse."""
    task = _spec(REGRESSION).task
    frame = pd.DataFrame(
        {"example_id": list("abcd"), "label": [1.0, 2.0, 3.0, 4.0], "prediction": [1, 3, 1, 5]}
    )
    assert metric_value(task, frame, "mae") == pytest.approx(1.0)
    assert metric_value(task, frame, "rmse") == pytest.approx(math.sqrt(1.5))


def test_multiclass_probability_metrics_match_manual():
    """B1: класс — колонка с наибольшей вероятностью; macro_f1, accuracy, log_loss."""
    task = _spec("task: {type: classification/multiclass, output: probabilities}").task
    frame = pd.DataFrame(
        {
            "example_id": list("abc"),
            "label": ["x", "y", "y"],
            "prediction:x": [0.7, 0.6, 0.2],
            "prediction:y": [0.3, 0.4, 0.8],
        }
    )
    # предсказано x, x, y: x — TP1 FP1 → F1 2/3; y — TP1 FN1 → F1 2/3
    assert metric_value(task, frame, "accuracy") == pytest.approx(2 / 3)
    assert metric_value(task, frame, "macro_f1") == pytest.approx(2 / 3)
    expected = -np.mean(np.log([0.7, 0.4, 0.8]))
    assert metric_value(task, frame, "log_loss") == pytest.approx(expected)


def test_multiclass_labels_compared_as_strings():
    """B1: метки multiclass сравниваются как строки (1 и "1" — один класс)."""
    task = _spec("task: {type: classification/multiclass, output: labels}").task
    frame = pd.DataFrame(
        {"example_id": list("abc"), "label": [1, 2, 3], "prediction": ["1", "2", "2"]}
    )
    assert metric_value(task, frame, "accuracy") == pytest.approx(2 / 3)
    # классы 1: F1 1; 2: TP1 FP1 → 2/3; 3: FN1 → 0
    assert metric_value(task, frame, "macro_f1") == pytest.approx((1 + 2 / 3 + 0) / 3)


@pytest.mark.parametrize(
    ("task_type", "output", "expected"),
    [
        ("classification/binary", "labels", ("f1", "accuracy", "precision", "recall")),
        ("classification/binary", "probabilities", ("roc_auc", "log_loss", "brier")),
        ("classification/binary", "scores", ("roc_auc",)),
        ("classification/multiclass", "labels", ("macro_f1", "accuracy")),
        ("classification/multiclass", "probabilities", ("macro_f1", "accuracy", "log_loss")),
        ("classification/multiclass", "scores", ("macro_f1", "accuracy")),
        ("regression", "values", ("mae", "rmse")),
    ],
)
def test_metric_sets_per_task(task_type, output, expected):
    """B1: набор и порядок метрик по паре task.type · task.output."""
    task = _spec(f"task: {{type: {task_type}, output: {output}}}").task
    assert metric_names(task) == expected


# --- B2: метрики одного прогона ---


def test_run_metrics_reports_value_with_interval():
    """B2: metrics.value по каждой метрике с 95% CI и n."""
    spec = _spec(BINARY_PROBA)
    candidate, _ = _binary_runs()
    results = run_metrics(spec, _contract(spec, candidate))

    found = _by_metric(results, "metrics.value")
    assert list(found) == ["roc_auc", "log_loss", "brier"]
    for name, finding in found.items():
        facts = finding.facts
        assert finding.severity == "info"
        assert facts["n"] == 400
        assert facts["ci_low"] <= facts["value"] <= facts["ci_high"]
        assert finding.message.startswith(f"{name} = {facts['value']:.3f}, 95% CI [")
        assert finding.message.endswith("n = 400")
    assert results.skipped == []


# --- B3–B5: дельты ---


def test_significant_improvement():
    """B3: кандидат значимо лучше — «значимо, лучше», info, факты дельты."""
    spec = _spec(BINARY_PROBA)
    candidate, baseline = _binary_runs()
    results = compare_runs(spec, _contract(spec, candidate), _contract(spec, baseline, "base"))

    found = _by_metric(results, "comparison.metric_delta")
    assert list(found) == ["roc_auc", "log_loss", "brier"]
    auc = found["roc_auc"]
    facts = auc.facts
    assert auc.severity == "info"
    assert facts["delta"] > 0
    assert facts["ci_low"] > 0
    assert facts["delta"] == pytest.approx(facts["candidate"] - facts["baseline"], abs=1e-6)
    assert facts["n"] == 400
    assert 0 <= facts["p_value"] < 0.05
    assert auc.message.startswith(f"roc_auc: {facts['baseline']:.3f} → {facts['candidate']:.3f}")
    assert auc.message.endswith("— значимо, лучше")
    # log_loss: меньше — лучше; у точного кандидата он ниже → тоже «лучше»
    assert found["log_loss"].facts["delta"] < 0
    assert found["log_loss"].message.endswith("— значимо, лучше")


def test_significant_decline_severity():
    """B4: значимо хуже — high для основной метрики, medium для остальных."""
    spec = _spec(BINARY_PROBA)
    good, bad = _binary_runs()
    results = compare_runs(spec, _contract(spec, bad), _contract(spec, good, "base"))

    found = _by_metric(results, "comparison.metric_delta")
    assert found["roc_auc"].severity == "high"
    assert found["roc_auc"].message.endswith("— значимо, хуже")
    assert found["log_loss"].severity == "medium"
    assert found["log_loss"].facts["delta"] > 0
    assert found["log_loss"].message.endswith("— значимо, хуже")


def test_identical_runs_within_noise():
    """B5: одинаковые предсказания — «в пределах шума», info."""
    spec = _spec(BINARY_PROBA)
    candidate, _ = _binary_runs()
    results = compare_runs(spec, _contract(spec, candidate), _contract(spec, candidate, "base"))

    for finding in _by_metric(results, "comparison.metric_delta").values():
        assert finding.severity == "info"
        assert finding.facts["delta"] == 0
        assert finding.message.endswith("— в пределах шума (CI включает 0)")


def test_partial_overlap_and_row_order():
    """B6: сравнение по общим примерам; порядок строк не влияет."""
    spec = _spec(BINARY_PROBA)
    candidate, baseline = _binary_runs()
    full = compare_runs(spec, _contract(spec, candidate), _contract(spec, baseline, "base"))

    shuffled = candidate.sample(frac=1, random_state=3)
    extra = pd.DataFrame({"example_id": ["only-c"], "label": [1], "prediction": [0.9]})
    partial_candidate = pd.concat([shuffled, extra], ignore_index=True)
    partial_baseline = baseline.iloc[:-1]  # без последнего примера
    partial = compare_runs(
        spec, _contract(spec, partial_candidate), _contract(spec, partial_baseline, "base")
    )

    assert _by_metric(partial, "comparison.metric_delta")["roc_auc"].facts["n"] == 399
    reordered = compare_runs(spec, _contract(spec, shuffled), _contract(spec, baseline, "base"))
    assert [f.facts for f in reordered.findings] == [f.facts for f in full.findings]


def test_repeated_analysis_is_identical():
    """B7: повторный анализ даёт идентичные находки (NFR-05)."""
    spec = _spec(BINARY_PROBA)
    candidate, baseline = _binary_runs()
    first = compare_runs(spec, _contract(spec, candidate), _contract(spec, baseline, "base"))
    second = compare_runs(spec, _contract(spec, candidate), _contract(spec, baseline, "base"))
    assert first == second


# --- E1–E4: деградация ---


def test_no_common_examples_skips_comparison():
    """E1: нет общих примеров — сравнение пропущено с причиной."""
    spec = _spec(BINARY_PROBA)
    candidate, baseline = _binary_runs()
    baseline = baseline.assign(example_id="b-" + baseline["example_id"])
    results = compare_runs(spec, _contract(spec, candidate), _contract(spec, baseline, "base"))

    assert results.findings == []
    assert [(s.check, s.reason) for s in results.skipped] == [
        (
            "comparison.metric_delta",
            "у прогонов нет общих примеров — парное сравнение невозможно",
        )
    ]


def test_different_labels_skip_comparison():
    """E2: метки общих примеров различаются — сравнение пропущено, в причине число и id."""
    spec = _spec(BINARY_PROBA)
    candidate, baseline = _binary_runs()
    baseline.loc[:2, "label"] = 1 - baseline.loc[:2, "label"]
    results = compare_runs(spec, _contract(spec, candidate), _contract(spec, baseline, "base"))

    assert results.findings == []
    (skipped,) = results.skipped
    assert skipped.check == "comparison.metric_delta"
    assert "метки различаются у 3 общих примеров" in skipped.reason
    assert "e0000" in skipped.reason


def test_undefined_metric_skipped_others_computed():
    """E3: roc_auc не определена при одном классе — пропуск, остальные метрики считаются."""
    spec = _spec(BINARY_PROBA)
    frame = pd.DataFrame(
        {"example_id": ["a", "b", "c"], "label": [1, 1, 1], "prediction": [0.9, 0.8, 0.7]}
    )
    results = run_metrics(spec, _contract(spec, frame))

    assert list(_by_metric(results, "metrics.value")) == ["log_loss", "brier"]
    (skipped,) = results.skipped
    assert skipped.check == "metrics.value"
    assert skipped.reason.startswith("roc_auc не определена")


def test_unstable_metric_skipped():
    """E4: метрика не определена на > 10% бутстрап-выборок — пропуск «оценка неустойчива»."""
    spec = _spec(BINARY_PROBA)
    # один положительный пример из 10: в ~35% выборок его нет → roc_auc не определена
    frame = pd.DataFrame(
        {
            "example_id": [f"e{i}" for i in range(10)],
            "label": [1] + [0] * 9,
            "prediction": np.linspace(0.9, 0.1, 10),
        }
    )
    results = run_metrics(spec, _contract(spec, frame))

    assert "roc_auc" not in _by_metric(results, "metrics.value")
    (skipped,) = results.skipped
    assert skipped.reason.startswith("roc_auc: оценка неустойчива")
