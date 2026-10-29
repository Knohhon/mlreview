"""Плейбук trees: глубина, ID-подобные признаки, экстраполяция (FR-D11, B16–B18).

Спецификация: docs/specs/playbook-trees.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _factories import binary_spec, binary_tables
from mlreview.analyzers.registry import Analyzer, register, unregister
from mlreview.analyzers.review import review
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec
from mlreview.playbooks import builtin_playbooks

TREE = {"class": "sklearn.tree.DecisionTreeClassifier"}
DEPTH = "tree.unbounded_depth"
ID_LIKE = "tree.id_like_importance"
EXTRAPOLATION = "tree.extrapolation"
REGRESSION = parse_task_spec("task: {type: regression, output: values}\n")
TEMPORAL = parse_task_spec(
    "task: {type: regression, output: values}\ndata: {split_strategy: temporal, time_column: ts}\n"
)


def _trees():
    return next(playbook for playbook in builtin_playbooks() if playbook.id == "trees")


def _binary(*, config=None, model=TREE, hard=False, importance=None, record_no=None):
    predictions, examples = binary_tables()
    if hard:
        predictions["prediction"] = (predictions["prediction"] > 0.5).astype(float)
    if record_no is not None:
        examples["record_no"] = record_no
    return build_run_contract(
        binary_spec(),
        run_id="tree",
        predictions=predictions,
        examples=examples,
        config=config,
        model=model,
        feature_importance=importance,
    )


def _regression(*, spec=REGRESSION, trend=0.0, model=None, config=None):
    """Train 200 + test 100; при trend > 0 метки test выше всех предсказаний."""
    rng = np.random.default_rng(7)
    n_train, n_test = 200, 100
    ids = [f"tr-{i:03d}" for i in range(n_train)] + [f"te-{i:03d}" for i in range(n_test)]
    examples = pd.DataFrame(
        {
            "example_id": ids,
            "split": ["train"] * n_train + ["test"] * n_test,
            "ts": pd.date_range("2024-01-01", periods=n_train + n_test, freq="h"),
            "x": rng.normal(0, 1, n_train + n_test),
        }
    )
    label = rng.normal(100, 10, n_test) + trend
    prediction = np.clip(label - trend + rng.normal(0, 3, n_test), 80, 120)
    predictions = pd.DataFrame(
        {"example_id": ids[n_train:], "label": label, "prediction": prediction}
    )
    return build_run_contract(
        spec,
        run_id="tree-reg",
        predictions=predictions,
        examples=examples,
        config=config,
        model=model or {"class": "DecisionTreeRegressor"},
    )


def _review(contract, spec=None, groups=("model",)):
    return review(spec or binary_spec(), contract, groups=groups)


def _finding(report, check):
    return next((finding for finding in report.findings if finding.check == check), None)


def _skip(report, check):
    return next((s.reason for s in report.skipped if s.check == check), None)


# --- Глубина дерева (B16) ------------------------------------------------------------------------


def test_unbounded_depth_with_pure_leaves():
    """B1: max_depth None, min_samples_leaf 1, вероятности 0/1 — medium."""
    config = {"max_depth": "None", "min_samples_leaf": "1"}
    finding = _finding(_review(_binary(config=config, hard=True)), DEPTH)

    assert finding.severity == "medium"
    assert finding.requirement == "FR-D11, B16"
    assert finding.facts == {"max_depth": None, "min_samples_leaf": 1, "extreme_share": 1.0}
    assert finding.message == (
        "дерево без ограничения глубины (max_depth не задан, min_samples_leaf = 1): листья"
        " запоминают train — у 1.0 предсказаний вероятность ровно 0 или 1"
    )
    assert finding.suggestion == _trees().manifest.checks[DEPTH].suggestion


def test_unbounded_depth_low_without_evidence():
    """B2: вероятности не вырождены или регрессия — low; min_samples_leaf не указан."""
    smooth = _finding(_review(_binary(config={"max_depth": None})), DEPTH)
    assert smooth.severity == "low"
    assert smooth.facts["min_samples_leaf"] is None
    assert smooth.facts["extreme_share"] < 0.9
    assert smooth.message == (
        "дерево без ограничения глубины (max_depth не задан, min_samples_leaf не указан):"
        " листья запоминают train"
    )

    regression = _regression(config={"max_depth": "None", "min_samples_leaf": 1})
    finding = _finding(_review(regression, REGRESSION), DEPTH)
    assert finding.severity == "low"
    assert finding.facts == {"max_depth": None, "min_samples_leaf": 1}


@pytest.mark.parametrize(
    "config",
    [
        {"max_depth": "4"},
        {"max_depth": "None", "min_samples_leaf": "20"},
        {"max_depth": "None", "max_leaf_nodes": "16"},
    ],
)
def test_bounded_tree_means_no_finding(config):
    """B3: глубина, лист или число листьев ограничены — находки нет."""
    assert _finding(_review(_binary(config=config, hard=True)), DEPTH) is None


def test_lightgbm_minus_one_is_unbounded():
    """B3: max_depth -1 — без ограничения (для явного семейства tree)."""
    model = {"class": "CustomTree", "family": "tree"}
    finding = _finding(_review(_binary(config={"max_depth": "-1"}, model=model)), DEPTH)
    assert finding.facts["max_depth"] is None


def test_forest_depth_is_not_checked():
    """B4: у случайного леса неограниченная глубина — норма: ни находки, ни пропуска."""
    report = _review(
        _binary(config={"max_depth": "None"}, model={"class": "RandomForestClassifier"})
    )
    assert _finding(report, DEPTH) is None
    assert _skip(report, DEPTH) is None


# --- ID-подобный признак (B17) -------------------------------------------------------------------


def _ids(n=400):
    return np.arange(n)


def _importance(feature="record_no", share=0.6, kind=None):
    frame = pd.DataFrame({"feature": [feature, "f0"], "importance": [share, 1 - share]})
    if kind is not None:
        frame["kind"] = kind
    return frame


def test_id_like_feature_is_reported():
    """B5: важный целый почти уникальный признак — high с доказательствами."""
    contract = _binary(importance=_importance(kind="impurity"), record_no=_ids())

    finding = _finding(_review(contract), ID_LIKE)

    assert finding.severity == "high"
    assert finding.requirement == "FR-D11, B17"
    assert finding.facts == {
        "feature": "record_no",
        "importance_share": 0.6,
        "kind": "impurity",
        "unique_share": 1.0,
        "n_train": 200,
    }
    assert finding.message == (
        "модель опирается на признак record_no, похожий на идентификатор: доля важности"
        " 0.6 (impurity), на train 1.0 значений уникальны и все целые — номер записи может"
        " кодировать порядок данных и метку"
    )
    assert finding.evidence == (
        *(f"tr-{i:03d}" for i in range(5)),
        *(f"tr-{i:03d}" for i in range(195, 200)),
    )
    assert finding.suggestion == _trees().manifest.checks[ID_LIKE].suggestion


@pytest.mark.parametrize(
    ("record_no", "importance"),
    [
        (np.arange(400) + 0.5, _importance()),
        (np.arange(400) // 10, _importance()),
        (_ids(), _importance(share=0.2)),
    ],
)
def test_ordinary_features_mean_no_finding(record_no, importance):
    """B6: дробный, неуникальный или неважный признак — находки нет."""
    contract = _binary(importance=importance, record_no=record_no)
    report = _review(contract)
    assert _finding(report, ID_LIKE) is None
    assert _skip(report, ID_LIKE) is None


def test_negative_permutation_importance_is_ignored():
    """B6: отрицательные важности permutation не входят в сумму."""
    importance = pd.DataFrame(
        {
            "feature": ["record_no", "f0", "size"],
            "importance": [0.2, 0.6, -0.5],
            "kind": ["permutation"] * 3,
        }
    )
    contract = _binary(importance=importance, record_no=_ids())
    finding = _finding(_review(contract), ID_LIKE)
    assert finding is None  # 0.2 / (0.2 + 0.6) = 0.25 < 0.3

    importance.loc[0, "importance"] = 0.3  # 0.3 / 0.9 ≈ 0.33
    assert _finding(_review(_binary(importance=importance, record_no=_ids())), ID_LIKE)


@pytest.mark.parametrize("model", ["RandomForestClassifier", "lightgbm.LGBMClassifier"])
def test_id_like_feature_in_ensembles(model):
    """B7: ансамбли деревьев — та же проверка."""
    contract = _binary(model={"class": model}, importance=_importance(), record_no=_ids())
    finding = _finding(_review(contract), ID_LIKE)
    assert finding.facts["kind"] == "unspecified"


# --- Экстраполяция (B18) -------------------------------------------------------------------------


def test_extrapolation_on_temporal_trend():
    """B8: тренд таргета и временной сплит — medium с доказательствами."""
    finding = _finding(_review(_regression(spec=TEMPORAL, trend=40.0), TEMPORAL), EXTRAPOLATION)

    assert finding.severity == "medium"
    assert finding.requirement == "FR-D11, B18"
    facts = finding.facts
    assert facts["outside_share"] >= 0.9
    assert facts["side_share"] == round(max(facts["above"], facts["below"]) / 100, 6)
    assert facts["above"] + facts["below"] == round(facts["outside_share"] * 100)
    assert facts["prediction_max"] == 120.0
    assert finding.message == (
        f"предсказания модели ограничены диапазоном [{facts['prediction_min']:.4g},"
        f" {facts['prediction_max']:.4g}]: {facts['outside_share']} меток вне его"
        f" (выше — {facts['above']}, ниже — {facts['below']}) — деревья не экстраполируют"
        " за пределы значений train; при временном сплите это признак тренда таргета"
    )
    assert len(finding.evidence) == 10


def test_extrapolation_without_temporal_split_is_low():
    """B8: без временного сплита — low, без хвоста про тренд."""
    model = {"class": "XGBRegressor"}
    finding = _finding(_review(_regression(trend=40.0, model=model), REGRESSION), EXTRAPOLATION)
    assert finding.severity == "low"
    assert "временном сплите" not in finding.message


def test_labels_inside_range_mean_no_finding():
    """B9: метки внутри диапазона или вне его с обеих сторон понемногу — находки нет."""
    assert _finding(_review(_regression(), REGRESSION), EXTRAPOLATION) is None
    contract = _regression()
    predictions = contract.predictions.assign(
        prediction=contract.predictions["label"].clip(
            contract.predictions["label"].quantile(0.15),
            contract.predictions["label"].quantile(0.85),
        )
    )
    both_sides = build_run_contract(
        REGRESSION,
        run_id="tree-reg",
        predictions=predictions,
        examples=contract.examples,
        model={"class": "DecisionTreeRegressor"},
    )
    assert _finding(_review(both_sides, REGRESSION), EXTRAPOLATION) is None


# --- Подсказки и воспроизводимость ---------------------------------------------------------------


@pytest.fixture
def general_checks():
    """Общие проверки с подсказками, которые переопределяет плейбук trees."""
    checks = ("calibration.ece", "curves.overfitting", "curves.underfitting", "curves.instability")

    def run(_ctx):
        results = CheckResults()
        for check in checks:
            results.findings.append(Finding(check, "low", "общая", "FR-D04", suggestion="lr"))
        return results

    analyzer = register(Analyzer("general", "data", "FR-D04", run))
    yield checks
    unregister(analyzer.name)


def test_overrides_for_tree_families(general_checks):
    """B10: подсказки калибровки и кривых — тексты плейбука trees."""
    overrides = _trees().manifest.overrides
    for model in ("DecisionTreeClassifier", "LGBMClassifier"):
        report = _review(_binary(model={"class": model}), groups=("data",))
        for check in general_checks:
            assert _finding(report, check).suggestion == overrides[check]
    assert "min_samples_leaf" in overrides["calibration.ece"]
    assert "ранн" in overrides["curves.overfitting"]


def test_findings_are_reproducible():
    """B11: одинаковые входы — одинаковые находки."""

    def run():
        contract = _binary(
            config={"max_depth": "None"}, hard=True, importance=_importance(), record_no=_ids()
        )
        return _review(contract).to_dict()

    first = run()
    assert first == run()
    assert {f["check"] for f in first["findings"]} == {DEPTH, ID_LIKE}


# --- Деградация и ошибки -------------------------------------------------------------------------


def test_depth_skips_without_param():
    """E1: нет max_depth — пропуск с алиасами."""
    assert _skip(_review(_binary(config={"criterion": "gini"})), DEPTH) == (
        "в config нет max_depth (алиасы: max_depth, depth)"
    )


def test_id_like_skips_without_inputs():
    """E2: нет важностей, examples или train — пропуск с причиной."""
    assert _skip(_review(_binary()), ID_LIKE) == (
        "нет feature_importance — передайте важности признаков"
        " (feature_importance.parquet или артефакт MLflow)"
    )
    predictions, examples = binary_tables()
    examples["split"] = examples["split"].replace({"train": "fit"})
    no_train = build_run_contract(
        binary_spec(),
        run_id="tree",
        predictions=predictions,
        examples=examples,
        model=TREE,
        feature_importance=_importance(feature="f0", share=1.0)[:1],
    )
    assert _skip(_review(no_train), ID_LIKE) == "в examples нет сплита train"


def test_extrapolation_only_for_regression():
    """E3: классификация — проверка не выполняется и пропуска не даёт."""
    report = _review(_binary(config={"max_depth": "4"}))
    assert _finding(report, EXTRAPOLATION) is None
    assert _skip(report, EXTRAPOLATION) is None
