"""Сведения о модели в контракте: класс, библиотека, семейство, важности признаков (FR-C07).

Ядро не загружает модель — только её описание строками (NFR-01). Семейство по имени класса
определяют плейбуки; здесь — формат и проверки полей `model` и `feature_importance`.
Поведение зафиксировано в `docs/specs/model-info.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FAMILIES: tuple[str, ...] = (
    "linear",
    "neighbors",
    "tree",
    "ensemble/bagging",
    "ensemble/boosting",
    "other",
)
SOURCES: tuple[str, ...] = ("explicit", "tag", "autolog", "pipeline")
MODEL_KEYS: tuple[str, ...] = ("class", "family", "library")
IMPORTANCE_COLUMNS: tuple[str, ...] = ("feature", "importance", "kind")
SIGNED_KIND = "permutation"
EVIDENCE_LIMIT = 5


@dataclass(frozen=True)
class ModelInfo:
    """Описание модели прогона; `source` — откуда оно взято (`SOURCES`)."""

    class_name: str | None = None
    family: str | None = None
    library: str | None = None
    source: str = "explicit"

    @property
    def short_name(self) -> str | None:
        """Класс без модуля: `sklearn.tree.X` → `X`."""
        return None if self.class_name is None else self.class_name.rsplit(".", 1)[-1]

    def to_dict(self) -> dict[str, str | None]:
        return {
            "class": self.class_name,
            "family": self.family,
            "library": self.library,
            "source": self.source,
        }


def library_of(class_name: str | None) -> str | None:
    """Библиотека — первая часть пути класса; у короткого имени её нет."""
    if class_name is None or "." not in class_name:
        return None
    return class_name.split(".", 1)[0]


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def model_problems(model: Any) -> list[tuple[str, str]]:
    """Проблемы поля `model` как пары (путь, сообщение) (E1–E3)."""
    if isinstance(model, ModelInfo):
        values = {"class": model.class_name, "family": model.family, "library": model.library}
    elif isinstance(model, Mapping):
        values = dict(model)
    else:
        return [("model", "ожидается словарь с ключами class, family, library или ModelInfo")]
    problems = [
        (f"model.{key}", "неизвестное поле; гиперпараметры передаются в config")
        for key in values
        if key not in MODEL_KEYS
    ]
    if values.get("class") is None and values.get("family") is None:
        problems.append(("model", "нужен class или family"))
    for key in ("class", "library"):
        if values.get(key) is not None and not _is_text(values[key]):
            problems.append((f"model.{key}", "ожидается непустая строка"))
    family = values.get("family")
    if family is not None and family not in FAMILIES:
        allowed = ", ".join(FAMILIES)
        problems.append(("model.family", f"неизвестное семейство; допустимо: {allowed}"))
    return problems


def to_model_info(model: ModelInfo | Mapping[str, Any]) -> ModelInfo:
    """Проверенный словарь `model` → `ModelInfo`; `ModelInfo` возвращается как есть."""
    if isinstance(model, ModelInfo):
        return model
    class_name = model.get("class")
    library = model.get("library")
    return ModelInfo(
        class_name=class_name,
        family=model.get("family"),
        library=library if library is not None else library_of(class_name),
    )


def _rows(frame: pd.DataFrame, mask: pd.Series) -> str:
    sample = ", ".join(str(label) for label in list(frame.index[mask])[:EVIDENCE_LIMIT])
    return f"строк: {int(mask.sum())}, например строки: {sample}"


def importance_problems(importance: Any, examples: pd.DataFrame | None) -> list[tuple[str, str]]:
    """Проблемы таблицы `feature_importance` как пары (путь, сообщение) (E4–E6)."""
    if not isinstance(importance, pd.DataFrame):
        return [("feature_importance", "ожидается таблица pandas.DataFrame")]
    problems = [
        (f"feature_importance.{column}", "обязательная колонка")
        for column in ("feature", "importance")
        if column not in importance.columns
    ]
    problems += [
        (
            f"feature_importance.{column}",
            "неизвестная колонка; допустимо: feature, importance, kind",
        )
        for column in importance.columns
        if column not in IMPORTANCE_COLUMNS
    ]
    if "feature" in importance.columns:
        bad = ~importance["feature"].map(_is_text).astype(bool)
        if bad.any():
            message = f"ожидается непустое имя признака; {_rows(importance, bad)}"
            problems.append(("feature_importance.feature", message))
        elif examples is not None:
            absent = sorted(set(importance["feature"]) - set(map(str, examples.columns)))
            if absent:
                message = (
                    f"признаков нет среди колонок examples: {len(absent)}, например: "
                    f"{', '.join(absent[:EVIDENCE_LIMIT])}"
                )
                problems.append(("feature_importance.feature", message))
    if "kind" in importance.columns:
        bad = ~importance["kind"].map(_is_text).astype(bool)
        if bad.any():
            message = f"ожидается непустой способ подсчёта; {_rows(importance, bad)}"
            problems.append(("feature_importance.kind", message))
    if "importance" in importance.columns:
        values = importance["importance"]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            message = f"ожидается число, а тип колонки {values.dtype}"
            problems.append(("feature_importance.importance", message))
        else:
            numbers = values.astype(float)
            signed = (
                importance["kind"].eq(SIGNED_KIND)
                if "kind" in importance.columns
                else pd.Series(False, index=importance.index)
            )
            bad = ~np.isfinite(numbers) | ((numbers < 0) & ~signed)
            if bad.any():
                message = (
                    "ожидается конечное число ≥ 0 (отрицательное — только при kind: "
                    f"{SIGNED_KIND}); {_rows(importance, bad)}"
                )
                problems.append(("feature_importance.importance", message))
    keys = [column for column in ("feature", "kind") if column in importance.columns]
    if "feature" in keys and importance.duplicated(keys).any():
        repeated = importance.loc[importance.duplicated(keys, keep=False), "feature"]
        sample = ", ".join(map(str, repeated.drop_duplicates().iloc[:EVIDENCE_LIMIT]))
        problems.append(
            (
                "feature_importance",
                f"пара (feature, kind) должна быть уникальна; повторяются: {sample}",
            )
        )
    return problems
