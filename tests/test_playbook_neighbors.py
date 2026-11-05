"""Плейбук neighbors: масштаб признаков и малое k (FR-D11, B14, B15).

Спецификация: docs/specs/playbook-neighbors.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from _factories import binary_spec, binary_tables
from mlreview.analyzers.review import review
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec
from mlreview.playbooks import builtin_playbooks

KNN = {"class": "sklearn.neighbors.KNeighborsClassifier"}
SCALE = "neighbors.unscaled_features"
SMALL_K = "neighbors.small_k"
REGRESSION = parse_task_spec("task: {type: regression, output: values}\n")


def _neighbors():
    return next(playbook for playbook in builtin_playbooks() if playbook.id == "neighbors")


def _contract(*, config=None, income_scale=1000.0, hard=False, with_examples=True, spec=None):
    predictions, examples = binary_tables()
    rng = np.random.default_rng(5)
    examples["income"] = rng.normal(0, 1, len(examples)) * income_scale
    if hard:
        predictions["prediction"] = (predictions["prediction"] > 0.5).astype(float)
    return build_run_contract(
        spec or binary_spec(),
        run_id="knn",
        predictions=predictions,
        examples=examples if with_examples else None,
        config=config,
        model=KNN,
    )


def _model(contract, spec=None):
    return review(spec or binary_spec(), contract, groups=("model",))


def _finding(report, check):
    found = [finding for finding in report.findings if finding.check == check]
    return found[0] if found else None


def _skip(report, check):
    return next((s.reason for s in report.skipped if s.check == check), None)


# --- Масштаб признаков (B14) ---------------------------------------------------------------------


def test_unscaled_features_with_explicit_no_scaler():
    """B1: scaler: none и признаки в разных масштабах — medium с фактами."""
    finding = _finding(_model(_contract(config={"scaler": "none", "k": 5})), SCALE)

    assert finding.severity == "medium"
    assert finding.requirement == "FR-D11, B14"
    facts = finding.facts
    assert (facts["largest_feature"], facts["smallest_feature"]) == ("income", "f0")
    assert facts["features"] == 2
    assert facts["scaler"] == "нет"
    assert facts["ratio"] == pytest.approx(facts["largest_std"] / facts["smallest_std"], rel=1e-5)
    assert 500 < facts["ratio"] < 2000
    assert finding.message == (
        f"признаки kNN в разных масштабах: стандартное отклонение на train различается"
        f" в {facts['ratio']:.1f} раз (income — {facts['largest_std']:.3g},"
        f" f0 — {facts['smallest_std']:.3g}); без масштабирования расстояния определяет income"
    )
    assert finding.suggestion == _neighbors().manifest.checks[SCALE].suggestion


def test_unscaled_features_without_preprocessing_info():
    """B2: о масштабировании ничего не известно — low с просьбой проверить."""
    finding = _finding(_model(_contract(config={"n_neighbors": "5"})), SCALE)

    assert finding.severity == "low"
    assert finding.facts["scaler"] == "неизвестно"
    assert finding.message.endswith(
        "; в config нет сведений о масштабировании — проверьте, есть ли скейлер в пайплайне"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"scaler": "standard"},
        {"standardscaler__copy": "True", "kneighborsclassifier__n_neighbors": "5"},
        {"steps": "[('standardscaler', StandardScaler()), ('knn', KNeighborsClassifier())]"},
    ],
)
def test_declared_scaler_means_no_finding(config):
    """B3: скейлер объявлен в config — находки нет."""
    assert _finding(_model(_contract(config=config)), SCALE) is None


def test_similar_scales_mean_no_finding():
    """B4: масштабы близки — находки нет."""
    assert _finding(_model(_contract(config={"scaler": "none"}, income_scale=3.0)), SCALE) is None


# --- Малое k (B15) -------------------------------------------------------------------------------


@pytest.mark.parametrize("config", [{"n_neighbors": "1"}, {"kneighborsclassifier__n_neighbors": 1}])
def test_small_k_with_degenerate_probabilities(config):
    """B5: k = 1 и вероятности 0/1 — medium с фактами о вероятностях."""
    finding = _finding(_model(_contract(config=config, hard=True)), SMALL_K)

    assert finding.severity == "medium"
    assert finding.requirement == "FR-D11, B15"
    assert finding.facts == {
        "n_neighbors": 1,
        "distinct_values": 2,
        "extreme_share": 1.0,
        "n_train": 200,
    }
    assert finding.message == (
        "kNN с k = 1: предсказание по 1 ближайшим соседям запоминает шум train; вероятности"
        " принимают 2 значений, ровно 0 или 1 — 1.0 — ROC AUC и калибровка по таким"
        " вероятностям почти бессмысленны"
    )
    assert finding.suggestion == _neighbors().manifest.checks[SMALL_K].suggestion


def test_small_k_with_smooth_probabilities():
    """B5: k = 2, вероятности не вырождены — без хвоста про ROC AUC."""
    finding = _finding(_model(_contract(config={"k": 2})), SMALL_K)

    assert finding.facts["n_neighbors"] == 2
    assert finding.facts["extreme_share"] < 0.9
    assert "ROC AUC" not in finding.message


def test_small_k_in_regression():
    """B6: регрессия с k = 1 — находка без частей о вероятностях."""
    predictions, examples = binary_tables()
    predictions["label"] = predictions["label"].astype(float)
    contract = build_run_contract(
        REGRESSION,
        run_id="knn-reg",
        predictions=predictions,
        examples=examples,
        config={"n_neighbors": 1},
        model={"class": "KNeighborsRegressor"},
    )

    finding = _finding(_model(contract, REGRESSION), SMALL_K)

    assert finding.facts == {"n_neighbors": 1, "n_train": 200}
    assert finding.message == (
        "kNN с k = 1: предсказание по 1 ближайшим соседям запоминает шум train"
    )


def test_enough_neighbors_mean_no_finding():
    """B7: k больше порога — находки нет."""
    assert _finding(_model(_contract(config={"n_neighbors": "15"})), SMALL_K) is None


def test_calibration_suggestion_is_overridden():
    """B8: подсказка calibration.ece для kNN — текст плейбука."""
    predictions, examples = binary_tables()
    flip = np.random.default_rng(3).random(len(predictions)) < 0.2
    predictions["prediction"] = np.where(flip, 1 - predictions["label"], predictions["label"])
    contract = build_run_contract(
        binary_spec(), run_id="knn", predictions=predictions, examples=examples, model=KNN
    )

    report = review(binary_spec(), contract, groups=("metrics",))

    ece = _finding(report, "calibration.ece")
    assert ece.suggestion == _neighbors().manifest.overrides["calibration.ece"]
    assert "weights" in ece.suggestion


def test_findings_are_reproducible():
    """B9: одинаковые входы — одинаковые находки."""
    first = _model(_contract(config={"scaler": "none", "k": 1}, hard=True)).to_dict()
    second = _model(_contract(config={"scaler": "none", "k": 1}, hard=True)).to_dict()
    assert first == second
    assert {f["check"] for f in first["findings"]} == {SCALE, SMALL_K}


# --- Деградация и ошибки -------------------------------------------------------------------------


def test_scale_check_skips_without_data():
    """E1: нет examples, нет train, мало числовых признаков — пропуск с причиной."""
    assert _skip(_model(_contract(with_examples=False)), SCALE) == "нет examples"

    predictions, examples = binary_tables()
    examples["split"] = examples["split"].replace({"train": "fit"})
    no_train = build_run_contract(
        binary_spec(), run_id="knn", predictions=predictions, examples=examples, model=KNN
    )
    assert _skip(_model(no_train), SCALE) == "в examples нет сплита train"

    predictions, examples = binary_tables()
    one_feature = build_run_contract(
        binary_spec(), run_id="knn", predictions=predictions, examples=examples, model=KNN
    )
    assert (
        _skip(_model(one_feature), SCALE) == "меньше двух числовых признаков с разбросом на train"
    )


def test_small_k_skips_without_param():
    """E2: нет n_neighbors — пропуск с алиасами."""
    assert _skip(_model(_contract(config={"scaler": "none"})), SMALL_K) == (
        "в config нет n_neighbors (алиасы: n_neighbors, k)"
    )


def test_small_k_skips_bad_value():
    """E3: n_neighbors не целое положительное — пропуск."""
    assert _skip(_model(_contract(config={"n_neighbors": "auto"})), SMALL_K) == (
        "n_neighbors=auto — ожидается целое число ≥ 1"
    )
