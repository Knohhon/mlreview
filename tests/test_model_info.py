"""Сведения о модели в контракте (FR-C07): спецификация docs/specs/model-info.md."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from _factories import binary_spec, binary_tables
from mlreview.core.model_info import FAMILIES, ModelInfo
from mlreview.core.run_contract import ContractGap, RunContractError, build_run_contract


def _build(**overrides):
    predictions, examples = binary_tables(n=20)
    kwargs = {"run_id": "exp-1", "predictions": predictions, "examples": examples}
    kwargs.update(overrides)
    return build_run_contract(binary_spec(), **kwargs)


def _errors(**overrides) -> RunContractError:
    with pytest.raises(RunContractError) as exc:
        _build(**overrides)
    return exc.value


def _paths(error: RunContractError) -> list[str]:
    return [problem.path for problem in error.problems]


def _importance(**columns) -> pd.DataFrame:
    return pd.DataFrame({"feature": ["f0"], "importance": [1.0], **columns})


# --- Ожидаемое поведение -------------------------------------------------------------------------


def test_model_from_mapping():
    """B1: словарь class/family → ModelInfo с библиотекой из пути класса."""
    contract = _build(model={"class": "sklearn.tree.DecisionTreeClassifier", "family": "tree"})

    assert contract.model == ModelInfo(
        class_name="sklearn.tree.DecisionTreeClassifier",
        family="tree",
        library="sklearn",
        source="explicit",
    )
    assert contract.model.short_name == "DecisionTreeClassifier"


def test_model_short_class_or_family_only():
    """B2: короткое имя класса — без библиотеки; одно семейство — без класса."""
    short = _build(model={"class": "KNeighborsClassifier"}).model
    assert (short.class_name, short.library, short.family) == ("KNeighborsClassifier", None, None)
    assert short.short_name == "KNeighborsClassifier"

    family = _build(model={"family": "neighbors"}).model
    assert (family.class_name, family.family, family.short_name) == (None, "neighbors", None)


def test_model_info_is_kept_and_serialized():
    """B3: ModelInfo хранится как есть; to_dict — class, family, library, source."""
    info = ModelInfo(class_name="lightgbm.LGBMRegressor", library="lightgbm", source="tag")

    contract = _build(model=info)

    assert contract.model is info
    assert info.to_dict() == {
        "class": "lightgbm.LGBMRegressor",
        "family": None,
        "library": "lightgbm",
        "source": "tag",
    }
    assert "ensemble/boosting" in FAMILIES


def test_feature_importance_is_copied_and_not_hashed():
    """B4: важности — копия с индексом 0…n−1; на хеш данных не влияют."""
    importance = pd.DataFrame(
        {"feature": ["f0", "size"], "importance": [0.7, 0.3], "kind": ["impurity"] * 2},
        index=[5, 6],
    )

    contract = _build(feature_importance=importance)
    importance.loc[5, "importance"] = 0.0

    assert list(contract.feature_importance.index) == [0, 1]
    assert contract.feature_importance["importance"].tolist() == [0.7, 0.3]
    assert contract.data_hash == _build().data_hash


def test_missing_model_is_last_gap():
    """B5: без model — пробел model последним; без важностей пробела нет."""
    assert _build().gaps[-1] == ContractGap("model", "проверки по семейству модели (FR-D11)")
    assert _build(model={"family": "linear"}).gaps == [
        gap for gap in _build().gaps if gap.field != "model"
    ]
    assert _build().model is None
    assert _build().feature_importance is None


# --- Деградация и ошибки -------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["tree", 3, {}, {"library": "sklearn"}])
def test_bad_model(model):
    """E1: model не словарь/ModelInfo или без class и family — ошибка с путём model."""
    assert "model" in _paths(_errors(model=model))


def test_unknown_model_key():
    """E2: неизвестный ключ model — «неизвестное поле», гиперпараметры — в config."""
    error = _errors(model={"family": "tree", "params": {"max_depth": 3}})

    assert "model.params" in _paths(error)
    assert "неизвестное поле" in str(error)
    assert "config" in str(error)


@pytest.mark.parametrize(
    ("model", "path"),
    [
        ({"class": ""}, "model.class"),
        ({"class": 1}, "model.class"),
        ({"family": "tree", "library": " "}, "model.library"),
        ({"family": "forest"}, "model.family"),
    ],
)
def test_bad_model_values(model, path):
    """E3: пустые class/library и неизвестное семейство — ошибка с путём поля."""
    error = _errors(model=model)

    assert path in _paths(error)
    if path == "model.family":
        assert "ensemble/bagging" in str(error)


@pytest.mark.parametrize(
    ("importance", "path"),
    [
        ([{"feature": "f0", "importance": 1.0}], "feature_importance"),
        (pd.DataFrame({"feature": ["f0"]}), "feature_importance.importance"),
        (pd.DataFrame({"importance": [1.0]}), "feature_importance.feature"),
        (_importance(rank=[1]), "feature_importance.rank"),
    ],
)
def test_bad_importance_table(importance, path):
    """E4: не DataFrame, нет обязательной колонки или лишняя колонка — ошибка с путём."""
    assert path in _paths(_errors(feature_importance=importance))


@pytest.mark.parametrize(
    ("importance", "path"),
    [
        (pd.DataFrame({"feature": [""], "importance": [1.0]}), "feature_importance.feature"),
        (pd.DataFrame({"feature": [7], "importance": [1.0]}), "feature_importance.feature"),
        (pd.DataFrame({"feature": ["f0"], "importance": ["a"]}), "feature_importance.importance"),
        (_importance(importance=[math.inf]), "feature_importance.importance"),
        (_importance(importance=[math.nan]), "feature_importance.importance"),
        (_importance(importance=[-0.1]), "feature_importance.importance"),
        (
            pd.DataFrame({"feature": ["f0", "f0"], "importance": [0.5, 0.4]}),
            "feature_importance",
        ),
    ],
)
def test_bad_importance_values(importance, path):
    """E5: пустое имя, нечисло, бесконечность, NaN, отрицательное, повтор — ошибка."""
    assert path in _paths(_errors(feature_importance=importance))


def test_negative_permutation_importance_is_valid():
    """E5: отрицательная важность допустима при kind: permutation; пары feature/kind различны."""
    importance = pd.DataFrame(
        {
            "feature": ["f0", "f0"],
            "importance": [-0.02, 0.4],
            "kind": ["permutation", "impurity"],
        }
    )

    assert len(_build(feature_importance=importance).feature_importance) == 2


def test_importance_feature_must_be_in_examples():
    """E6: признака из важностей нет в examples — ошибка с примерами имён."""
    error = _errors(
        feature_importance=pd.DataFrame({"feature": ["f0", "age"], "importance": [1, 2]})
    )

    assert "feature_importance.feature" in _paths(error)
    assert "age" in str(error)
    predictions, _ = binary_tables(n=20)
    without_examples = _build(
        examples=None, predictions=predictions, feature_importance=_importance()
    )
    assert without_examples.feature_importance is not None
