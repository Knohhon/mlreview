"""Единый интерфейс анализаторов: контекст ревью → находки и пропуски (FR-E01).

Внутренний реестр: встроенные анализаторы регистрируются в `mlreview.analyzers.builtin`,
`review()` запускает анализаторы выбранных групп. Поведение зафиксировано
в `docs/specs/analyzers.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mlreview.core.findings import CheckResults
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec

GROUPS: tuple[str, ...] = ("data", "segments", "metrics", "model", "curves", "process")


@dataclass(frozen=True)
class ReviewContext:
    """Всё, что анализатор знает о ревью: спецификация, прогоны, сиды, история."""

    spec: TaskSpec
    contract: RunContract
    baseline: RunContract | None = None
    candidate_seeds: tuple[RunContract, ...] = ()
    baseline_seeds: tuple[RunContract, ...] = ()
    prior_runs: int | None = None

    @property
    def candidates(self) -> tuple[RunContract, ...]:
        """Основной прогон кандидата и его сиды."""
        return (self.contract, *self.candidate_seeds)

    @property
    def baselines(self) -> tuple[RunContract, ...]:
        """Основной базовый прогон и его сиды; без базового — пусто."""
        return () if self.baseline is None else (self.baseline, *self.baseline_seeds)


@dataclass(frozen=True)
class Analyzer:
    """Анализатор: имя, группа ревью, опора и функция проверки."""

    name: str
    group: str
    requirement: str
    run: Callable[[ReviewContext], CheckResults]


_REGISTRY: dict[str, Analyzer] = {}


def register(analyzer: Analyzer) -> Analyzer:
    """Регистрирует анализатор; имя уникально, группа — из `GROUPS`."""
    if analyzer.group not in GROUPS:
        raise ValueError(f"неизвестная группа {analyzer.group!r}; допустимо: {', '.join(GROUPS)}")
    if analyzer.name in _REGISTRY:
        raise ValueError(f"анализатор {analyzer.name!r} уже зарегистрирован")
    _REGISTRY[analyzer.name] = analyzer
    return analyzer


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def registered(groups: Sequence[str] = GROUPS) -> list[Analyzer]:
    """Анализаторы выбранных групп: по порядку групп, внутри — по порядку регистрации."""
    unknown = [group for group in groups if group not in GROUPS]
    if unknown:
        raise ValueError(
            f"неизвестные группы проверок: {', '.join(unknown)}; допустимо: {', '.join(GROUPS)}"
        )
    return [
        analyzer
        for group in GROUPS
        if group in groups
        for analyzer in _REGISTRY.values()
        if analyzer.group == group
    ]


def run_analyzers(ctx: ReviewContext, groups: Sequence[str] = GROUPS) -> CheckResults:
    """Запускает анализаторы; исключение анализатора становится пропуском (B2, X1)."""
    results = CheckResults()
    for analyzer in registered(groups):
        try:
            results.extend(analyzer.run(ctx))
        except Exception as exc:  # анализатор не должен ронять ревью целиком
            results.skip(analyzer.name, f"ошибка анализатора: {type(exc).__name__}: {exc}")
    return results
