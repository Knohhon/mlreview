"""Проекты бенчмарка: датасеты scikit-learn, линейные модели, kNN, деревья и версии с дефектами.

Датасеты поставляются вместе с scikit-learn — сеть не нужна (NFR-03). Время у примеров
синтетическое; все случайные шаги — с фиксированными сидами (NFR-05).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlreview.core.run_contract import RunContract, build_run_contract
from mlreview.core.task_spec import TaskSpec, parse_task_spec

START = pd.Timestamp("2024-01-01")
TEST_SHARE = 0.3
COPY_SHARE = 0.05
LEAK_NOISE = 0.01
IMBALANCE = 50
SEEDS = (11, 12, 13)
B08_PARAMS = (1.0, 1.05)  # базовый и кандидат: C логистической регрессии или alpha гребневой
KNN_K = 15
TREE_DEPTH = 4
TREE_LEAF = 5
RECORD_SHUFFLE = 0.1  # B17: доля перемешанных номеров записей train
TREND = 400.0  # B18: прирост метки от первого до последнего примера по времени


@dataclass(frozen=True)
class Project:
    """Базовый проект: датасет, задача, сегмент и значение сегмента для дефекта B12."""

    name: str
    task_type: str
    output: str
    loader: str
    segment: str
    segment_values: Callable[[pd.DataFrame], pd.Series]
    corrupted_value: str
    baseline_features: int  # базовый прогон — первые N признаков датасета
    corrupt_share: float  # B12: доля примеров сегмента с перемешанными предсказаниями
    l1_defects: tuple[str, ...] = ("B07", "B08", "B11")
    model: str = "linear"  # linear, knn или tree
    model_defects: tuple[str, ...] = ()  # дефекты плейбука семейства (B14–B18)

    @property
    def spec(self) -> TaskSpec:
        return self.spec_for("clean")

    def spec_for(self, version: str) -> TaskSpec:
        metrics = "metrics: {primary: accuracy}\n" if version == "B05" else ""
        return parse_task_spec(
            f"task: {{type: {self.task_type}, output: {self.output}, segments: [{self.segment}]}}\n"
            f"data: {{split_strategy: temporal, time_column: ts, version: {self.name}-v1}}\n"
            + metrics
        )


def _by_median(column: str, high: str, low: str) -> Callable[[pd.DataFrame], pd.Series]:
    def values(features: pd.DataFrame) -> pd.Series:
        series = features[column]
        return pd.Series(np.where(series > series.median(), high, low), index=features.index)

    return values


def _ink(features: pd.DataFrame) -> pd.Series:
    brightness = features.mean(axis=1)
    return pd.Series(
        np.where(brightness > brightness.median(), "heavy", "light"), index=features.index
    )


def _sex(features: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(features["sex"] > 0, "a", "b"), index=features.index)


PROJECTS: dict[str, Project] = {
    "breast_cancer": Project(
        name="breast_cancer",
        task_type="classification/binary",
        output="probabilities",
        loader="load_breast_cancer",
        segment="texture",
        segment_values=_by_median("mean texture", "rough", "smooth"),
        corrupted_value="smooth",
        baseline_features=1,
        corrupt_share=0.3,
        l1_defects=("B03", "B05", "B07", "B08", "B11"),
    ),
    "diabetes": Project(
        name="diabetes",
        task_type="regression",
        output="values",
        loader="load_diabetes",
        segment="sex_group",
        segment_values=_sex,
        corrupted_value="a",
        baseline_features=3,
        corrupt_share=0.3,
        l1_defects=("B03", "B07", "B08", "B11"),
    ),
    "digits": Project(
        name="digits",
        task_type="classification/multiclass",
        output="probabilities",
        loader="load_digits",
        segment="ink",
        segment_values=_ink,
        corrupted_value="light",
        baseline_features=16,
        corrupt_share=0.5,
    ),
}
PROJECTS.update(
    {
        "breast_cancer_knn": dataclasses.replace(
            PROJECTS["breast_cancer"],
            name="breast_cancer_knn",
            model="knn",
            model_defects=("B14", "B15"),
        ),
        "breast_cancer_tree": dataclasses.replace(
            PROJECTS["breast_cancer"],
            name="breast_cancer_tree",
            model="tree",
            model_defects=("B16", "B17"),
        ),
        "diabetes_tree": dataclasses.replace(
            PROJECTS["diabetes"], name="diabetes_tree", model="tree", model_defects=("B18",)
        ),
    }
)
L0_VERSIONS = ("clean", "B01", "B02", "B12")
L1_VERSIONS = ("B03", "B05", "B07", "B08", "B11")


def versions_for(name: str) -> tuple[str, ...]:
    """Версии проекта: линейного — clean, L0 и применимые L1; kNN и деревьев — clean и B14–B18."""
    project = PROJECTS[name]
    if project.model != "linear":
        return ("clean", *project.model_defects)
    defects = project.l1_defects
    return L0_VERSIONS + tuple(version for version in L1_VERSIONS if version in defects)


@dataclass(frozen=True)
class Runs:
    """Контракты кандидата и базового прогона одной версии проекта (B08 — с сидами)."""

    spec: TaskSpec
    candidate: RunContract
    baseline: RunContract
    candidate_seeds: tuple[RunContract, ...] = ()
    baseline_seeds: tuple[RunContract, ...] = ()


def _load(project: Project) -> tuple[pd.DataFrame, pd.Series]:
    import sklearn.datasets  # noqa: PLC0415 — scikit-learn нужен только бенчмарку

    data = getattr(sklearn.datasets, project.loader)(as_frame=True)
    return data.data.reset_index(drop=True), data.target.reset_index(drop=True)


def _frame(project: Project, version: str) -> pd.DataFrame:
    """Все примеры версии: id, сплит, сегмент, время, признаки и метка `target`."""
    features, target = _load(project)
    n = len(features)
    order = np.random.default_rng(0).permutation(n)
    frame = features.assign(
        example_id=[f"{project.name}-{i:04d}" for i in range(n)],
        **{project.segment: project.segment_values(features)},
        ts=START + pd.to_timedelta(order, unit="h"),
        target=target,
    )
    if version == "B02":
        test = np.zeros(n, dtype=bool)
        test[np.random.default_rng(1).permutation(n)[: round(TEST_SHARE * n)]] = True
    else:
        test = order >= n - round(TEST_SHARE * n)
    frame["split"] = np.where(test, "test", "train")
    if version == "B17":
        frame["record_no"] = _record_numbers(frame)
    if version == "B18":
        frame["target"] = frame["target"] + TREND * order / (n - 1)
    if version == "B01":
        test_rows = frame[frame["split"] == "test"]
        count = math.ceil(COPY_SHARE * len(test_rows))
        chosen = np.random.default_rng(2).choice(len(test_rows), count, replace=False)
        copies = test_rows.iloc[np.sort(chosen)].copy()
        copies["example_id"] = [f"{project.name}-copy-{i:03d}" for i in range(count)]
        copies["split"] = "train"
        frame = pd.concat([frame, copies], ignore_index=True)
    return frame


def _record_numbers(frame: pd.DataFrame) -> np.ndarray:
    """B17: номер записи — train по метке с частью перемешанных номеров, test — после train."""
    rng = np.random.default_rng(5)
    numbers = np.zeros(len(frame), dtype=np.int64)
    train = np.flatnonzero(frame["split"].to_numpy() == "train")
    ordered = train[np.argsort(frame["target"].to_numpy()[train], kind="mergesort")]
    shuffled = np.sort(
        rng.choice(len(ordered), round(RECORD_SHUFFLE * len(ordered)), replace=False)
    )
    ordered[shuffled] = ordered[rng.permutation(shuffled)]
    numbers[ordered] = np.arange(len(ordered))
    test = np.flatnonzero(frame["split"].to_numpy() != "train")
    numbers[rng.permutation(test)] = len(ordered) + np.arange(len(test))
    return numbers


def _hyperparameters(project: Project, version: str) -> dict:
    """Гиперпараметры модели версии: у kNN и деревьев дефекты B14–B16 — в них."""
    if project.model == "knn":
        return {
            "n_neighbors": 1 if version == "B15" else KNN_K,
            "scaler": "none" if version == "B14" else "standard",
            "weights": "uniform",
        }
    if project.model == "tree":
        unbounded = version == "B16"
        return {
            "max_depth": None if unbounded else TREE_DEPTH,
            "min_samples_leaf": 1 if unbounded else TREE_LEAF,
            "criterion": "squared_error" if project.task_type == "regression" else "gini",
        }
    return {"C": 1.0, "scaler": "standard", "class_weight": "none"}


def model_class(project: Project, param: float | None = None) -> str:
    """Класс модели sklearn для контракта (FR-C07)."""
    regression = project.task_type == "regression"
    if project.model == "knn":
        return (
            "sklearn.neighbors.KNeighborsRegressor"
            if regression
            else ("sklearn.neighbors.KNeighborsClassifier")
        )
    if project.model == "tree":
        kind = "Regressor" if regression else "Classifier"
        return f"sklearn.tree.DecisionTree{kind}"
    if regression:
        return (
            "sklearn.linear_model.LinearRegression"
            if param is None
            else ("sklearn.linear_model.Ridge")
        )
    return "sklearn.linear_model.LogisticRegression"


def _estimator(project: Project, version: str, param: float | None):
    from sklearn.linear_model import (  # noqa: PLC0415
        LinearRegression,
        LogisticRegression,
        Ridge,
    )
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: PLC0415

    regression = project.task_type == "regression"
    params = _hyperparameters(project, version)
    if project.model == "knn":
        knn = (KNeighborsRegressor if regression else KNeighborsClassifier)(
            n_neighbors=params["n_neighbors"], weights=params["weights"]
        )
        return knn if params["scaler"] == "none" else make_pipeline(StandardScaler(), knn)
    if project.model == "tree":
        return (DecisionTreeRegressor if regression else DecisionTreeClassifier)(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            criterion=params["criterion"],
            random_state=0,
        )
    if regression:
        estimator = LinearRegression() if param is None else Ridge(alpha=param)
        return make_pipeline(StandardScaler(), estimator)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=param or 1.0))


def _fit_predict(  # noqa: PLR0913 — проект, версия, данные, признаки, параметр B08
    project: Project,
    version: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    *,
    param: float | None = None,
):
    """Модель проекта; `param` — C логистической или alpha гребневой регрессии (B08).

    Возвращает предсказания, классы (или `None`) и важности признаков деревьев (или `None`).
    """
    from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415

    model = _estimator(project, version, param)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(train[columns], train["target"])
    importance = None
    if project.model == "tree":
        importance = pd.DataFrame(
            {"feature": columns, "importance": model.feature_importances_, "kind": "impurity"}
        )
    if project.task_type == "regression":
        return model.predict(test[columns]), None, importance
    return model.predict_proba(test[columns]), list(model.classes_), importance


def _predictions(project: Project, test: pd.DataFrame, raw, classes) -> pd.DataFrame:
    frame = pd.DataFrame({"example_id": test["example_id"].to_numpy(), "label": test["target"]})
    frame = frame.reset_index(drop=True)
    if project.task_type == "classification/binary":
        frame["prediction"] = raw[:, 1]
    elif project.task_type == "regression":
        frame["prediction"] = raw
    else:
        for i, cls in enumerate(classes):
            frame[f"prediction:{cls}"] = raw[:, i]
    return frame


def _corrupt(
    project: Project, candidate: pd.DataFrame, baseline: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """B12: на одном значении сегмента кандидат хуже базового прогона.

    Вне сегмента остаются предсказания кандидата; на сегменте — предсказания базового прогона,
    у доли `corrupt_share` примеров перемешанные между собой (перестановка с сидом).
    """
    candidate = candidate.copy()
    rows = np.flatnonzero(test[project.segment].to_numpy() == project.corrupted_value)
    rng = np.random.default_rng(3)
    chosen = np.sort(rng.choice(rows, round(project.corrupt_share * len(rows)), replace=False))
    donors = rng.permutation(chosen)
    columns = [c for c in candidate.columns if c.startswith("prediction")]
    candidate.loc[rows, columns] = baseline.loc[rows, columns].to_numpy()
    candidate.loc[chosen, columns] = baseline.loc[donors, columns].to_numpy()
    return candidate


@functools.cache
def _trained(name: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray, dict]:
    """Примеры, test, предсказания и важности обеих моделей варианта.

    У линейных проектов вариант — сплит (clean и B12 делят один сплит и модели), у kNN и
    деревьев — версия: дефекты B14–B18 меняют модель или данные.
    """
    project = PROJECTS[name]
    frame = _frame(project, variant)
    features = list(_load(project)[0].columns)
    extra = ["record_no"] if "record_no" in frame.columns else []
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    tables, importances = {}, {}
    baseline_columns = features[: project.baseline_features]
    for role, columns in (("candidate", features + extra), ("baseline", baseline_columns)):
        raw, classes, importances[role] = _fit_predict(project, variant, train, test, columns)
        tables[role] = _predictions(project, test, raw, classes)
    examples = frame[["example_id", "split", project.segment, "ts", *features, *extra]]
    return examples, test, tables, frame["target"].to_numpy(), importances


@functools.cache
def _seeded(name: str, param: float, seed: int) -> pd.DataFrame:
    """B08: модель на всех признаках, обученная на бутстрап-выборке train с сидом."""
    project = PROJECTS[name]
    frame = _frame(project, "clean")
    features = list(_load(project)[0].columns)
    train = frame[frame["split"] == "train"]
    rows = np.random.default_rng(seed).integers(0, len(train), len(train))
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    raw, classes, _ = _fit_predict(project, "clean", train.iloc[rows], test, features, param=param)
    return _predictions(project, test, raw, classes)


def clear_cache() -> None:
    """Сбрасывает кэш обученных моделей (для проверки воспроизводимости)."""
    _trained.cache_clear()
    _seeded.cache_clear()


def _configs(project: Project, version: str) -> tuple[dict, dict]:
    baseline = {"features": project.baseline_features, **_hyperparameters(project, version)}
    candidate = {**baseline, "features": "all"}
    if version == "B11":
        candidate.update(C=0.5, scaler="robust", class_weight="balanced")
    return candidate, baseline


def _imbalanced(predictions: pd.DataFrame) -> pd.DataFrame:
    """B05: все примеры мажоритарного класса и ⌈их / 50⌉ первых по id другого."""
    counts = predictions["label"].value_counts()
    majority = counts.idxmax()
    keep_minor = math.ceil(counts[majority] / IMBALANCE)
    minor = predictions[predictions["label"] != majority].sort_values("example_id").head(keep_minor)
    kept = set(predictions.loc[predictions["label"] == majority, "example_id"]) | set(
        minor["example_id"]
    )
    return predictions[predictions["example_id"].isin(kept)].reset_index(drop=True)


def _seed_runs(project: Project, spec: TaskSpec, examples: pd.DataFrame) -> Runs:
    """B08: по 3 сида на сторону; конфиги отличаются только параметром модели."""
    sides = {}
    for role, param in (("baseline", B08_PARAMS[0]), ("candidate", B08_PARAMS[1])):
        sides[role] = [
            build_run_contract(
                spec,
                run_id=f"{project.name}/B08/{role}/{seed}",
                predictions=_seeded(project.name, param, seed),
                examples=examples,
                config={"features": "all", "C": param, "seed": seed},
                data_version=f"{project.name}-v1",
                model={"class": model_class(project, param)},
            )
            for seed in SEEDS
        ]
    candidate, baseline = sides["candidate"], sides["baseline"]
    return Runs(spec, candidate[0], baseline[0], tuple(candidate[1:]), tuple(baseline[1:]))


def build_runs(project: Project, version: str) -> Runs:
    """Кандидат (все признаки) и базовый прогон (первые признаки) версии проекта."""
    spec = project.spec_for(version)
    if project.model == "linear":
        variant = version if version in ("B01", "B02") else "clean"
    else:
        variant = version
    examples, test, tables, targets, importances = _trained(project.name, variant)
    if version == "B08":
        return _seed_runs(project, spec, examples)
    tables = dict(tables)
    candidate_examples = baseline_examples = examples
    baseline_version = f"{project.name}-v1"
    if version == "B12":
        tables["candidate"] = _corrupt(project, tables["candidate"], tables["baseline"], test)
    if version == "B03":
        noise = np.random.default_rng(4).normal(0, LEAK_NOISE * targets.std(), len(targets))
        candidate_examples = baseline_examples = examples.assign(leak=targets + noise)
    if version == "B05":
        tables = {role: _imbalanced(table) for role, table in tables.items()}
    if version == "B07":
        train_rows = np.flatnonzero(examples["split"].to_numpy() == "train")
        baseline_examples = examples.drop(index=examples.index[train_rows[::10]])
        baseline_version = f"{project.name}-v0"
    configs = dict(zip(("candidate", "baseline"), _configs(project, version), strict=True))
    examples_of = {"candidate": candidate_examples, "baseline": baseline_examples}
    versions = {"candidate": f"{project.name}-v1", "baseline": baseline_version}
    contracts = {
        role: build_run_contract(
            spec,
            run_id=f"{project.name}/{version}/{role}",
            predictions=predictions,
            examples=examples_of[role],
            config=configs[role],
            data_version=versions[role],
            model={"class": model_class(project)},
            feature_importance=importances[role],
        )
        for role, predictions in tables.items()
    }
    return Runs(spec, contracts["candidate"], contracts["baseline"])
