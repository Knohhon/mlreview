"""Проекты бенчмарка: датасеты scikit-learn, линейные модели и версии с дефектами.

Датасеты поставляются вместе с scikit-learn — сеть не нужна (NFR-03). Время у примеров
синтетическое; все случайные шаги — с фиксированными сидами (NFR-05).
"""

from __future__ import annotations

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

    @property
    def spec(self) -> TaskSpec:
        return parse_task_spec(
            f"task: {{type: {self.task_type}, output: {self.output}, segments: [{self.segment}]}}\n"
            f"data: {{split_strategy: temporal, time_column: ts, version: {self.name}-v1}}\n"
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
VERSIONS = ("clean", "B01", "B02", "B12")


@dataclass(frozen=True)
class Runs:
    """Контракты кандидата и базового прогона одной версии проекта."""

    spec: TaskSpec
    candidate: RunContract
    baseline: RunContract


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
    if version == "B01":
        test_rows = frame[frame["split"] == "test"]
        count = math.ceil(COPY_SHARE * len(test_rows))
        chosen = np.random.default_rng(2).choice(len(test_rows), count, replace=False)
        copies = test_rows.iloc[np.sort(chosen)].copy()
        copies["example_id"] = [f"{project.name}-copy-{i:03d}" for i in range(count)]
        copies["split"] = "train"
        frame = pd.concat([frame, copies], ignore_index=True)
    return frame


def _fit_predict(project: Project, train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415
    from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import make_pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    if project.task_type == "regression":
        model = make_pipeline(StandardScaler(), LinearRegression())
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(train[columns], train["target"])
    if project.task_type == "regression":
        return model.predict(test[columns]), None
    return model.predict_proba(test[columns]), list(model.classes_)


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
def _trained(name: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Примеры, test и предсказания обеих моделей; clean и B12 делят один сплит и модели."""
    project = PROJECTS[name]
    frame = _frame(project, split)
    features = list(_load(project)[0].columns)
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    tables = {}
    baseline_columns = features[: project.baseline_features]
    for role, columns in (("candidate", features), ("baseline", baseline_columns)):
        raw, classes = _fit_predict(project, train, test, columns)
        tables[role] = _predictions(project, test, raw, classes)
    examples = frame[["example_id", "split", project.segment, "ts", *features]]
    return examples, test, tables


def clear_cache() -> None:
    """Сбрасывает кэш обученных моделей (для проверки воспроизводимости)."""
    _trained.cache_clear()


def build_runs(project: Project, version: str) -> Runs:
    """Кандидат (все признаки) и базовый прогон (первые признаки) версии проекта."""
    spec = project.spec
    split = version if version in ("B01", "B02") else "clean"
    examples, test, tables = _trained(project.name, split)
    tables = dict(tables)
    if version == "B12":
        tables["candidate"] = _corrupt(project, tables["candidate"], tables["baseline"], test)
    contracts = {
        role: build_run_contract(
            spec,
            run_id=f"{project.name}/{version}/{role}",
            predictions=predictions,
            examples=examples,
            data_version=f"{project.name}-v1",
        )
        for role, predictions in tables.items()
    }
    return Runs(spec, contracts["candidate"], contracts["baseline"])
