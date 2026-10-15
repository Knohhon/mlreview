"""Метрики задачи для анализаторов: раздел `metrics:` спецификации или набор L0 (FR-M02, FR-M05).

`spec_metrics(spec)` — метрики в порядке «основная, дополнительные»; каждая — `Metric` над
`Outputs` (как в `mlreview.analyzers.evaluation`) и `compute` над массивами контракта.
Поведение зафиксировано в `docs/specs/metrics.md`.
"""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlreview.analyzers.evaluation import METRICS, Metric, Outputs, metric_names
from mlreview.core.task_spec import CustomMetric, TaskSection, TaskSpec
from mlreview.metrics import kernels
from mlreview.metrics.catalog import CATALOG, MetricInfo, param_value, parse_name

CUSTOM_UNDEFINED = "пользовательская метрика вернула не конечное число"


class MetricsError(ValueError):
    """Метрика не загружена: файл пользовательской метрики, функция, имя."""


@dataclass(frozen=True)
class SpecMetric:
    """Метрика задачи: имя из спецификации, роль, функция и описание."""

    name: str
    role: str  # primary | secondary
    metric: Metric
    bounded: str | None
    info: MetricInfo | None = None

    @property
    def higher_is_better(self) -> bool:
        return self.metric.higher_is_better

    def compute(self, labels: Any, predictions: Any) -> float:
        """Значение на массивах меток и предсказаний (multiclass — коды классов и матрица)."""
        labels = np.asarray(labels)
        values = np.asarray(predictions)
        return self.metric(Outputs(np.arange(len(labels)).astype(str), labels, values))


def classes(task: TaskSection, values: np.ndarray, threshold: float) -> np.ndarray:
    """Предсказанные классы: порог для бинарных вероятностей, argmax для матрицы."""
    if values.ndim == 2:  # noqa: PLR2004 — матрица оценок по классам
        return np.argmax(values, axis=1)
    if task.type == "classification/binary" and task.output == "probabilities":
        return (values >= threshold).astype(int)
    return values.astype(int)


def _catalog_metric(name: str, task: TaskSection, threshold: float) -> Metric:
    base, raw = parse_name(name)
    info = CATALOG[base]
    param = param_value(info, raw)
    kernel = getattr(kernels, base)

    def fn(outputs: Outputs) -> float:
        if len(outputs) == 0:
            return float("nan")
        values = outputs.values
        if info.kind == "classes":
            values = classes(task, values, threshold)
        if param is None:
            return float(kernel(outputs.labels, values))
        return float(kernel(outputs.labels, values, param))

    return Metric(name, fn, info.higher_is_better, "не определена на этих примерах")


_LOADED: dict[str, Callable[..., Any]] = {}


def load_custom(custom: CustomMetric) -> Callable[..., Any]:
    """Функция пользовательской метрики из файла (E2)."""
    key = f"{custom.path}::{custom.function}"
    if key in _LOADED:
        return _LOADED[key]
    path = Path(custom.path)
    if not path.is_file():
        raise MetricsError(f"метрика {custom.name}: файл не найден: {path}")
    module_name = "mlreview_custom_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    loader_spec = importlib.util.spec_from_file_location(module_name, path)
    if loader_spec is None or loader_spec.loader is None:
        raise MetricsError(f"метрика {custom.name}: {path} не загружается как модуль Python")
    module = importlib.util.module_from_spec(loader_spec)
    try:
        loader_spec.loader.exec_module(module)
    except Exception as exc:  # код пользователя может бросить что угодно
        raise MetricsError(f"метрика {custom.name}: ошибка импорта {path}: {exc}") from exc
    function = getattr(module, custom.function, None)
    if not callable(function):
        raise MetricsError(f"метрика {custom.name}: в {path} нет функции {custom.function}")
    _LOADED[key] = function
    return function


def _custom_metric(custom: CustomMetric) -> Metric:
    function = load_custom(custom)

    def fn(outputs: Outputs) -> float:
        try:
            value = float(function(outputs.labels, outputs.values))
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    return Metric(custom.name, fn, custom.higher_is_better, CUSTOM_UNDEFINED)


def metric_by_name(spec: TaskSpec, name: str, role: str = "secondary") -> SpecMetric:
    """Метрика задачи по имени: пользовательская, из каталога или из набора L0."""
    section = spec.metrics
    if section is not None:
        for custom in section.custom:
            if custom.name == name:
                return SpecMetric(name, role, _custom_metric(custom), custom.bounded)
        threshold = section.threshold
    else:
        threshold = 0.5
    base, _ = parse_name(name)
    info = CATALOG.get(base)
    if section is None and name in METRICS:
        return SpecMetric(name, role, METRICS[name], info.bounded if info else None, info)
    if info is None:
        raise MetricsError(f"неизвестная метрика {name!r}")
    return SpecMetric(name, role, _catalog_metric(name, spec.task, threshold), info.bounded, info)


def spec_metrics(spec: TaskSpec) -> list[SpecMetric]:
    """Метрики задачи: из раздела `metrics:` или набор L0; основная — первой."""
    names = spec.metrics.names if spec.metrics is not None else list(metric_names(spec.task))
    return [
        metric_by_name(spec, name, "primary" if i == 0 else "secondary")
        for i, name in enumerate(names)
    ]


def primary(spec: TaskSpec) -> Metric:
    """Основная метрика задачи: по ней работают срезы и сиды."""
    return spec_metrics(spec)[0].metric
