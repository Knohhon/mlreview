"""Спецификация задачи: формат YAML, схема и валидация (FR-C01).

Спецификация описывает задачу, а не модель: тип задачи и выход, цену ошибок, важные сегменты,
стратегию сплита и версию данных; раздел `metrics:` — метрики задачи и ответы диалога о них
(FR-M01–M05). Поведение зафиксировано в `docs/specs/task-spec.md` и `docs/specs/metrics.md`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from mlreview.metrics.catalog import CATALOG, check_name

TaskType = Literal[
    "classification/binary",
    "classification/multiclass",
    "classification/multilabel",
    "regression",
]
TASK_TYPES: tuple[str, ...] = get_args(TaskType)
SplitStrategy = Literal["random", "stratified", "temporal"]
CostLevel = Literal["low", "medium", "high"]

CLASSIFICATION_OUTPUTS = ("labels", "probabilities", "scores")
REGRESSION_OUTPUTS = ("values",)
CLASSIFICATION_COSTS = ("false_positive", "false_negative")
REGRESSION_COSTS = ("underestimation", "overestimation")


@dataclass(frozen=True)
class SpecProblem:
    """Одна проблема спецификации: путь к полю (`task.output`) и сообщение."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}" if self.path else self.message


class TaskSpecError(ValueError):
    """Спецификация не прочитана или невалидна; содержит все найденные проблемы."""

    def __init__(self, source: str, problems: list[SpecProblem]) -> None:
        self.source = source
        self.problems = problems
        super().__init__(str(self))

    def __str__(self) -> str:
        if len(self.problems) == 1 and not self.problems[0].path:
            return f"{self.source}: {self.problems[0]}"
        lines = [f"{self.source}: спецификация задачи невалидна:"]
        lines += [f"  - {problem}" for problem in self.problems]
        return "\n".join(lines)


class _SectionError(ValueError):
    """Проблемы, найденные проверками раздела; пути — относительно раздела."""

    def __init__(self, problems: list[SpecProblem]) -> None:
        self.problems = problems
        super().__init__("; ".join(str(problem) for problem in problems))


def _allowed(values: Iterable[str]) -> str:
    return ", ".join(values)


def _task_problems(section: Mapping[str, Any]) -> list[SpecProblem]:
    """Согласованность раздела task: выход и цена ошибок по типу задачи, имена сегментов.

    Работает и на проверенных значениях, и на сыром YAML: поля неверного типа пропускает —
    о них сообщает схема.
    """
    problems = []
    task_type = section.get("type")
    if task_type in TASK_TYPES:
        is_classification = task_type.startswith("classification/")
        outputs = CLASSIFICATION_OUTPUTS if is_classification else REGRESSION_OUTPUTS
        costs = CLASSIFICATION_COSTS if is_classification else REGRESSION_COSTS
        output = section.get("output")
        if isinstance(output, str) and output not in outputs:
            message = f"выход {output!r} недопустим для {task_type}; допустимо: {_allowed(outputs)}"
            problems.append(SpecProblem("output", message))
        error_costs = section.get("error_costs")
        if isinstance(error_costs, Mapping):
            problems += [
                SpecProblem(
                    f"error_costs.{key}",
                    f"цена ошибки {key!r} не подходит к {task_type}; допустимо: {_allowed(costs)}",
                )
                for key in error_costs
                if key not in costs
            ]
    segments = section.get("segments")
    if isinstance(segments, list):
        names = [name for name in segments if isinstance(name, str)]
        if any(not name.strip() for name in names):
            problems.append(SpecProblem("segments", "пустое имя сегмента"))
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            message = f"повторяются сегменты: {_allowed(duplicates)}"
            problems.append(SpecProblem("segments", message))
    return problems


def _data_problems(section: Mapping[str, Any]) -> list[SpecProblem]:
    """Согласованность раздела data: для временного сплита нужна колонка времени (B02)."""
    if section.get("split_strategy") == "temporal" and not section.get("time_column"):
        message = (
            "обязательное поле при split_strategy: temporal — без колонки времени "
            "нельзя проверить временную утечку"
        )
        return [SpecProblem("time_column", message)]
    return []


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskSection(_Section):
    """Раздел `task`: что предсказывается и как используются предсказания."""

    type: TaskType
    output: str
    decision: str | None = None
    error_costs: dict[str, CostLevel] = {}
    segments: list[str] = []

    @model_validator(mode="after")
    def _check_consistency(self) -> TaskSection:
        problems = _task_problems(self.model_dump())
        if problems:
            raise _SectionError(problems)
        return self


class DataSection(_Section):
    """Раздел `data`: сплит, версия данных и известные проблемы."""

    split_strategy: SplitStrategy | None = None
    time_column: str | None = None
    version: str | None = None
    known_issues: list[str] = []

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: Any) -> Any:
        # `version: 3` в YAML — число; версия данных при этом остаётся меткой.
        return str(value) if isinstance(value, int | float) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> DataSection:
        problems = _data_problems(self.model_dump())
        if problems:
            raise _SectionError(problems)
        return self


Decision = Literal["threshold", "ranking", "probability"]


class MetricsContext(_Section):
    """Ответы диалога о контексте решения (FR-M01)."""

    decision: Decision | None = None
    budget_k: int | None = Field(default=None, ge=1)
    cost_ratio: float | None = Field(default=None, gt=0)
    calibration: bool | None = None
    large_errors: bool | None = None
    critical_segments: list[str] = []


class CustomMetric(_Section):
    """Пользовательская метрика: функция `f(labels, predictions) -> float` из файла (FR-M05)."""

    name: str
    path: str
    function: str
    higher_is_better: bool = True
    bounded: Literal["high", "low"] | None = None


class MetricsSection(_Section):
    """Раздел `metrics`: основная и дополнительные метрики задачи (FR-M01–M05)."""

    primary: str
    secondary: list[str] = []
    threshold: float = 0.5
    context: MetricsContext | None = None
    custom: list[CustomMetric] = []

    @field_validator("threshold")
    @classmethod
    def _threshold_inside(cls, value: float) -> float:
        if not 0 < value < 1:
            raise ValueError("порог — число в (0, 1)")
        return value

    @property
    def names(self) -> list[str]:
        """Метрики задачи: основная первой."""
        return [self.primary, *self.secondary]


def _metrics_problems(task: TaskSection, metrics: MetricsSection) -> list[SpecProblem]:
    """Метрики подходят к `task.type · task.output`, имена не повторяются (E1)."""
    problems = []
    custom = frozenset(metric.name for metric in metrics.custom)
    for index, metric in enumerate(metrics.custom):
        if metric.name.partition("@")[0] in CATALOG:
            message = f"повтор: имя {metric.name!r} совпадает с метрикой каталога"
            problems.append(SpecProblem(f"metrics.custom.{index}.name", message))
    seen: set[str] = set()
    paths = ["metrics.primary"] + [f"metrics.secondary.{i}" for i in range(len(metrics.secondary))]
    for path, name in zip(paths, metrics.names, strict=True):
        if name in seen:
            problems.append(SpecProblem(path, f"повтор метрики {name!r}"))
            continue
        seen.add(name)
        problem = check_name(name, task.type, task.output, custom)
        if problem:
            problems.append(SpecProblem(path, problem))
    return problems


class TaskSpec(_Section):
    """Спецификация задачи (FR-C01)."""

    task: TaskSection
    data: DataSection | None = None
    metrics: MetricsSection | None = None

    @model_validator(mode="after")
    def _check_metrics(self) -> TaskSpec:
        if self.metrics is not None:
            problems = _metrics_problems(self.task, self.metrics)
            if problems:
                raise _SectionError(problems)
        return self


_SIMPLE_MESSAGES = {
    "missing": "обязательное поле",
    "extra_forbidden": "неизвестное поле",
    "string_type": "ожидается строка",
    "list_type": "ожидается список",
    "dict_type": "ожидается словарь",
    "model_type": "ожидается словарь",
    "model_attributes_type": "ожидается словарь",
    "int_type": "ожидается целое число",
    "int_parsing": "ожидается целое число",
    "int_from_float": "ожидается целое число",
    "float_type": "ожидается число",
    "float_parsing": "ожидается число",
    "bool_type": "ожидается true или false",
    "bool_parsing": "ожидается true или false",
}
_BOUNDS = {
    "greater_than": ("gt", ">"),
    "greater_than_equal": ("ge", "≥"),
    "less_than": ("lt", "<"),
    "less_than_equal": ("le", "≤"),
}


def _translate(error: dict[str, Any]) -> list[SpecProblem]:
    """Переводит ошибку pydantic в проблемы спецификации с путями и русскими сообщениями."""
    path = ".".join(str(part) for part in error["loc"])
    kind = error["type"]
    cause = error.get("ctx", {}).get("error")
    if isinstance(cause, _SectionError):
        return [
            SpecProblem(f"{path}.{problem.path}" if path else problem.path, problem.message)
            for problem in cause.problems
        ]
    if kind in _BOUNDS:
        key, sign = _BOUNDS[kind]
        return [SpecProblem(path, f"ожидается число {sign} {error['ctx'][key]}")]
    if kind == "value_error" and cause is not None:
        return [SpecProblem(path, str(cause))]
    if kind == "literal_error":
        allowed = re.findall(r"'([^']*)'", error["ctx"]["expected"])
        message = f"недопустимое значение {error['input']!r}; допустимо: {', '.join(allowed)}"
        return [SpecProblem(path, message)]
    return [SpecProblem(path, _SIMPLE_MESSAGES.get(kind, error["msg"]))]


def _resolve_custom_paths(data: dict[str, Any], base_dir: Path) -> None:
    """Пути пользовательских метрик — относительно файла спецификации (FR-M05)."""
    metrics = data.get("metrics")
    custom = metrics.get("custom") if isinstance(metrics, dict) else None
    if not isinstance(custom, list):
        return
    for item in custom:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            path = Path(item["path"])
            if not path.is_absolute():
                item["path"] = str((base_dir / path).resolve())


def parse_task_spec(text: str, source: str = "<строка>", base_dir: Path | None = None) -> TaskSpec:
    """Разбирает спецификацию задачи из строки YAML; `source` попадает в сообщения об ошибках.

    `base_dir` — каталог файла спецификации: от него считаются пути пользовательских метрик.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f", строка {mark.line + 1}" if mark is not None else ""
        reason = getattr(exc, "problem", None) or exc
        problem = SpecProblem("", f"некорректный YAML{where}: {reason}")
        raise TaskSpecError(source, [problem]) from exc
    if not isinstance(data, dict):
        found = "пустой документ" if data is None else f"{type(data).__name__}"
        message = f"верхний уровень должен быть словарём с разделом task, а не {found}"
        raise TaskSpecError(source, [SpecProblem("", message)])
    if base_dir is not None:
        _resolve_custom_paths(data, base_dir)
    try:
        return TaskSpec.model_validate(data)
    except ValidationError as exc:
        problems = [problem for error in exc.errors() for problem in _translate(error)]
        # Раздел с ошибками полей не доходит до проверок согласованности — запускаем их
        # на сыром YAML, чтобы пользователь увидел все проблемы сразу (E8).
        for name, check in (("task", _task_problems), ("data", _data_problems)):
            section = data.get(name)
            if isinstance(section, Mapping):
                problems += [
                    SpecProblem(f"{name}.{problem.path}", problem.message)
                    for problem in check(section)
                ]
        unique = dict.fromkeys(problems)
        raise TaskSpecError(source, sorted(unique, key=lambda problem: problem.path)) from None


def load_task_spec(path: str | Path) -> TaskSpec:
    """Читает и проверяет спецификацию задачи из YAML-файла."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TaskSpecError(str(path), [SpecProblem("", "файл не найден")]) from None
    except (OSError, UnicodeDecodeError) as exc:
        problem = SpecProblem("", f"не удалось прочитать файл: {exc}")
        raise TaskSpecError(str(path), [problem]) from None
    return parse_task_spec(text, source=str(path), base_dir=path.parent)
