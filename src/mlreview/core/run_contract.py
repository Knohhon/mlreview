"""Контракт прогона: предсказания, метки, id и метаданные примеров, конфиг, кривые (FR-C02).

Контракт — всё, что ядро знает о прогоне: модель и фреймворк остаются за адаптером (NFR-01).
Формат предсказаний определяется спецификацией задачи (FR-C01). Поведение зафиксировано
в `docs/specs/run-contract.md`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlreview.core.task_spec import TaskSection, TaskSpec

CLASS_PREFIX = "prediction:"
EVIDENCE_LIMIT = 5
PROBABILITY_SUM_TOLERANCE = 1e-3

_EXAMPLES_PURPOSE = "срезы по сегментам (FR-D02), проверки данных между сплитами (FR-D03)"
_TIME_PURPOSE = "проверка временной утечки (FR-D03, B02)"
_SPLIT_PURPOSE = "нужно не меньше двух сплитов: проверка дублей между сплитами (FR-D03, B01)"
_CURVES_PURPOSE = "анализ кривых обучения (FR-D04)"
_CONFIG_PURPOSE = "логические проверки процесса (FR-D06)"
_VERSION_PURPOSE = "сверка версий данных между прогонами (FR-C04, FR-D06)"


@dataclass(frozen=True)
class ContractProblem:
    """Одна проблема контракта: путь к полю (`predictions.label`) и сообщение."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class RunContractError(ValueError):
    """Контракт прогона невалиден; содержит все найденные проблемы."""

    def __init__(self, run_id: str, problems: list[ContractProblem]) -> None:
        self.run_id = run_id
        self.problems = problems
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [f"прогон {self.run_id}: контракт прогона невалиден:"]
        lines += [f"  - {problem}" for problem in self.problems]
        return "\n".join(lines)


@dataclass(frozen=True)
class ContractGap:
    """Пробел контракта (X1): поля нет, и из-за этого не будет выполнено `purpose`."""

    field: str
    purpose: str


@dataclass(frozen=True, eq=False)
class RunContract:
    """Контракт прогона (FR-C02); собирается и проверяется через `build_run_contract`."""

    run_id: str
    predictions: pd.DataFrame
    examples: pd.DataFrame | None
    config: dict[str, Any]
    curves: pd.DataFrame | None
    data_version: str | None
    gaps: list[ContractGap]


def _evidence(labels: pd.Series | pd.Index, noun: str = "id") -> str:
    sample = ", ".join(str(label) for label in list(labels)[:EVIDENCE_LIMIT])
    return f"строк: {len(labels)}, например {noun}: {sample}"


def _row_evidence(frame: pd.DataFrame, mask: pd.Series) -> str:
    """Доказательства по строкам: id примеров, а без колонки id — номера строк."""
    if "example_id" in frame.columns:
        return _evidence(frame.loc[mask, "example_id"])
    return _evidence(frame.index[mask], noun="строки")


def _missing_columns(table: str, frame: pd.DataFrame, required: list[str]) -> list[ContractProblem]:
    return [
        ContractProblem(f"{table}.{column}", "обязательная колонка")
        for column in required
        if column not in frame.columns
    ]


def _id_problems(table: str, frame: pd.DataFrame) -> list[ContractProblem]:
    """Пропуски и повторы `example_id` (E4)."""
    if "example_id" not in frame.columns:
        return []
    ids = frame["example_id"]
    path = f"{table}.example_id"
    problems = []
    missing = ids.isna()
    if missing.any():
        problems.append(
            ContractProblem(path, f"пропуски id; {_evidence(frame.index[missing], 'строки')}")
        )
    repeated = ids[ids.notna() & ids.duplicated(keep=False)].drop_duplicates()
    if len(repeated):
        sample = ", ".join(str(value) for value in repeated.iloc[:EVIDENCE_LIMIT])
        message = f"id должны быть уникальны; повторяющихся id: {len(repeated)}, например: {sample}"
        problems.append(ContractProblem(path, message))
    return problems


def _is_finite(values: pd.Series) -> pd.Series:
    return pd.Series(np.isfinite(values), index=values.index)


def _is_binary(values: pd.Series) -> pd.Series:
    return values.isin([0.0, 1.0])


def _is_probability(values: pd.Series) -> pd.Series:
    return values.between(0.0, 1.0)


_NUMERIC_RULES: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    "finite": ("ожидается конечное число", _is_finite),
    "binary": ("ожидается 0 или 1 (положительный класс — 1)", _is_binary),
    "probability": ("ожидается вероятность в [0, 1]", _is_probability),
}


def _numeric_problems(frame: pd.DataFrame, column: str, rule: str) -> list[ContractProblem]:
    """Значения колонки предсказаний по правилу формата; пропуски проверяются отдельно (E5)."""
    expected, check = _NUMERIC_RULES[rule]
    series = frame[column]
    path = f"predictions.{column}"
    if not pd.api.types.is_numeric_dtype(series):
        return [ContractProblem(path, f"{expected}, а тип колонки {series.dtype}")]
    bad = series.notna() & ~check(series.astype(float))
    if bad.any():
        return [ContractProblem(path, f"{expected}; {_row_evidence(frame, bad)}")]
    return []


# Правила формата (docs/specs/run-contract.md): (тип задачи, выход) → (метка, предсказание).
_FORMATS: dict[tuple[str, str], tuple[str | None, str | None]] = {
    ("regression", "values"): ("finite", "finite"),
    ("classification/binary", "labels"): ("binary", "binary"),
    ("classification/binary", "probabilities"): ("binary", "probability"),
    ("classification/binary", "scores"): ("binary", "finite"),
    ("classification/multiclass", "labels"): (None, None),
    ("classification/multiclass", "probabilities"): (None, "probability"),
    ("classification/multiclass", "scores"): (None, "finite"),
}


def _class_problems(
    frame: pd.DataFrame, class_columns: list[str], rule: str
) -> list[ContractProblem]:
    """Multiclass с колонками `prediction:<класс>`: число классов, значения, метки, сумма."""
    if len(class_columns) < 2:  # noqa: PLR2004 — классификация различает хотя бы два класса
        message = (
            f"нужно не меньше двух колонок {CLASS_PREFIX}<класс>, найдено: {len(class_columns)}"
        )
        return [ContractProblem("predictions", message)]
    problems = []
    for column in class_columns:
        problems += _numeric_problems(frame, column, rule)
    if "label" in frame.columns:
        classes = [column.removeprefix(CLASS_PREFIX) for column in class_columns]
        labels = frame["label"]
        unknown = labels.notna() & ~labels.astype(str).isin(classes)
        if unknown.any():
            values = ", ".join(str(value) for value in labels[unknown].unique()[:EVIDENCE_LIMIT])
            message = (
                f"метки без колонки {CLASS_PREFIX}<класс>: {values}; "
                f"{_row_evidence(frame, unknown)}"
            )
            problems.append(ContractProblem("predictions.label", message))
    if rule == "probability" and not problems:
        scores = frame[class_columns]
        complete = scores.notna().all(axis=1)
        bad = complete & ((scores.sum(axis=1) - 1.0).abs() > PROBABILITY_SUM_TOLERANCE)
        if bad.any():
            message = (
                f"сумма вероятностей по строке должна быть 1 ± {PROBABILITY_SUM_TOLERANCE}; "
                f"{_row_evidence(frame, bad)}"
            )
            problems.append(ContractProblem("predictions", message))
    return problems


def _prediction_problems(task: TaskSection, predictions: Any) -> list[ContractProblem]:
    """Таблица предсказаний против формата из спецификации задачи (E2–E7)."""
    if not isinstance(predictions, pd.DataFrame):
        return [ContractProblem("predictions", "ожидается таблица pandas.DataFrame")]
    if predictions.empty:
        return [ContractProblem("predictions", "нет ни одного предсказания")]
    if task.type == "classification/multilabel":
        message = "формат предсказаний classification/multilabel на L0 не поддерживается"
        return [ContractProblem("predictions", message)]

    label_rule, prediction_rule = _FORMATS[(task.type, task.output)]
    per_class = task.type == "classification/multiclass" and task.output != "labels"
    class_columns = [
        column
        for column in predictions.columns
        if per_class and isinstance(column, str) and column.startswith(CLASS_PREFIX)
    ]
    required = ["example_id", "label"] if per_class else ["example_id", "label", "prediction"]
    problems = _missing_columns("predictions", predictions, required)
    problems += [
        ContractProblem(
            f"predictions.{column}",
            "неизвестная колонка; метаданные примеров передаются в examples",
        )
        for column in predictions.columns
        if column not in required and column not in class_columns
    ]
    problems += _id_problems("predictions", predictions)

    value_columns = [
        column for column in ["label", "prediction"] if column in required
    ] + class_columns
    for column in value_columns:
        if column not in predictions.columns:
            continue
        missing = predictions[column].isna()
        if missing.any():
            message = f"пропуски; {_row_evidence(predictions, missing)}"
            problems.append(ContractProblem(f"predictions.{column}", message))
    for column, rule in (("label", label_rule), ("prediction", prediction_rule)):
        if rule is not None and column in required and column in predictions.columns:
            problems += _numeric_problems(predictions, column, rule)
    if per_class and prediction_rule is not None:
        problems += _class_problems(predictions, class_columns, prediction_rule)
    return problems


def _is_nonblank_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _examples_problems(examples: Any, predictions: Any) -> list[ContractProblem]:
    """Метаданные примеров: колонки, id, сплиты и покрытие предсказаний (E4, E8)."""
    if not isinstance(examples, pd.DataFrame):
        return [ContractProblem("examples", "ожидается таблица pandas.DataFrame")]
    problems = _missing_columns("examples", examples, ["example_id", "split"])
    problems += _id_problems("examples", examples)
    if "split" in examples.columns:
        bad = ~examples["split"].map(_is_nonblank_text).astype(bool)
        if bad.any():
            message = f"ожидается непустое имя сплита; {_row_evidence(examples, bad)}"
            problems.append(ContractProblem("examples.split", message))
    if (
        isinstance(predictions, pd.DataFrame)
        and "example_id" in predictions.columns
        and "example_id" in examples.columns
    ):
        ids = predictions["example_id"]
        absent = ids.notna() & ~ids.isin(examples["example_id"])
        if absent.any():
            message = f"id предсказаний нет в examples; {_evidence(ids[absent])}"
            problems.append(ContractProblem("predictions.example_id", message))
    return problems


def _curves_problems(curves: Any) -> list[ContractProblem]:
    """Кривые обучения в длинном формате name, step, value (E9)."""
    if not isinstance(curves, pd.DataFrame):
        return [ContractProblem("curves", "ожидается таблица pandas.DataFrame")]
    problems = _missing_columns("curves", curves, ["name", "step", "value"])
    if "name" in curves.columns:
        bad = ~curves["name"].map(_is_nonblank_text).astype(bool)
        if bad.any():
            message = f"ожидается непустое имя кривой; {_evidence(curves.index[bad], 'строки')}"
            problems.append(ContractProblem("curves.name", message))
    if "step" in curves.columns:
        expected = "ожидается целое число ≥ 0"
        step = curves["step"]
        if not pd.api.types.is_numeric_dtype(step) or pd.api.types.is_bool_dtype(step):
            problems.append(
                ContractProblem("curves.step", f"{expected}, а тип колонки {step.dtype}")
            )
        else:
            values = step.astype(float)
            bad = ~(np.isfinite(values) & (values >= 0) & (values == np.floor(values)))
            if bad.any():
                message = f"{expected}; {_evidence(curves.index[bad], 'строки')}"
                problems.append(ContractProblem("curves.step", message))
    if "value" in curves.columns and not pd.api.types.is_numeric_dtype(curves["value"]):
        message = f"ожидается число, а тип колонки {curves['value'].dtype}"
        problems.append(ContractProblem("curves.value", message))
    if {"name", "step"} <= set(curves.columns):
        repeated = curves.loc[curves.duplicated(["name", "step"], keep=False), ["name", "step"]]
        if len(repeated):
            pairs = [f"{name}@{step}" for name, step in repeated.drop_duplicates().to_numpy()]
            message = (
                f"пара (name, step) должна быть уникальна; повторяются: "
                f"{', '.join(pairs[:EVIDENCE_LIMIT])}"
            )
            problems.append(ContractProblem("curves", message))
    return problems


def _gaps(
    spec: TaskSpec,
    examples: pd.DataFrame | None,
    curves: pd.DataFrame | None,
    config: dict[str, Any],
    data_version: str | None,
) -> list[ContractGap]:
    """Пробелы контракта в фиксированном порядке (G1–G6)."""
    gaps = []
    if examples is None:
        gaps.append(ContractGap("examples", _EXAMPLES_PURPOSE))
    else:
        gaps += [
            ContractGap(f"examples.{segment}", f"срез по сегменту {segment} (FR-D02)")
            for segment in spec.task.segments
            if segment not in examples.columns
        ]
        time_column = spec.data.time_column if spec.data else None
        if time_column and time_column not in examples.columns:
            gaps.append(ContractGap(f"examples.{time_column}", _TIME_PURPOSE))
        if examples["split"].nunique() < 2:  # noqa: PLR2004 — дубли ищутся между двумя сплитами
            gaps.append(ContractGap("examples.split", _SPLIT_PURPOSE))
    if curves is None:
        gaps.append(ContractGap("curves", _CURVES_PURPOSE))
    if not config:
        gaps.append(ContractGap("config", _CONFIG_PURPOSE))
    if data_version is None:
        gaps.append(ContractGap("data_version", _VERSION_PURPOSE))
    return gaps


def _copy_table(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    return None if frame is None else frame.reset_index(drop=True).copy(deep=True)


def build_run_contract(  # noqa: PLR0913 — поля контракта перечислены в FR-C02
    spec: TaskSpec,
    *,
    run_id: str,
    predictions: pd.DataFrame,
    examples: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    curves: pd.DataFrame | None = None,
    data_version: str | None = None,
) -> RunContract:
    """Собирает контракт прогона и проверяет его против спецификации задачи.

    Поднимает `RunContractError` со всеми найденными проблемами. Контракт хранит копии
    таблиц и конфига: изменения исходных объектов на него не влияют (NFR-05).
    """
    problems = []
    run_id_valid = isinstance(run_id, str) and bool(run_id.strip())
    if not run_id_valid:
        problems.append(ContractProblem("run_id", "ожидается непустая строка"))
    problems += _prediction_problems(spec.task, predictions)
    if examples is not None:
        problems += _examples_problems(examples, predictions)
    if curves is not None:
        problems += _curves_problems(curves)
    if config is not None and not (
        isinstance(config, Mapping) and all(isinstance(key, str) for key in config)
    ):
        problems.append(ContractProblem("config", "ожидается словарь со строковыми ключами"))
    if isinstance(data_version, int | float) and not isinstance(data_version, bool):
        # `data_version=3` — метка версии, как `version: 3` в спецификации задачи.
        data_version = str(data_version)
    if data_version is not None and not isinstance(data_version, str):
        problems.append(ContractProblem("data_version", "ожидается строка"))
    if problems:
        raise RunContractError(run_id if run_id_valid else "<без run_id>", problems)

    config = copy.deepcopy(dict(config or {}))
    examples = _copy_table(examples)
    curves = _copy_table(curves)
    return RunContract(
        run_id=run_id,
        predictions=_copy_table(predictions),
        examples=examples,
        config=config,
        curves=curves,
        data_version=data_version,
        gaps=_gaps(spec, examples, curves, config, data_version),
    )
