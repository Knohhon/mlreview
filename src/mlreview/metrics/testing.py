"""Юнит-тесты метрик: тривиальные бейзлайны, вырожденность, бутстрап-шум, MDE (FR-M03, FR-M05).

Тесты считаются на метках и предсказаниях прогона; бутстрап — 1000 выборок, сид 0 (NFR-05).
Поведение зафиксировано в `docs/specs/metrics.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mlreview.analyzers.evaluation import Outputs, estimate_value, rounded
from mlreview.core.task_spec import TaskSection
from mlreview.metrics.library import SpecMetric

HIGH_LIMIT = 0.9
LOW_LIMIT = 0.1
Z = 1.959964
MDE_FACTOR = 2.8 * math.sqrt(2)
MAJORITY = "константа мажоритарного класса"
PRIOR = "константа доли класса 1"


def _majority(labels: np.ndarray) -> int:
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def trivial_outputs(
    task: TaskSection, outputs: Outputs, *, data_dependent: bool = False
) -> list[tuple[str, Outputs]]:
    """Константные предсказания той же формы, что у модели.

    `data_dependent` — только бейзлайны, зависящие от данных (мажоритарный класс, доля
    класса 1, среднее): для анализатора пригодности в ревью.
    """
    n = len(outputs)
    ids, labels = outputs.ids, outputs.labels

    def make(values: np.ndarray) -> Outputs:
        return Outputs(ids, labels, values, outputs.n_classes)

    if task.type == "regression":
        mean, median = float(np.mean(labels)), float(np.median(labels))
        return [("среднее", make(np.full(n, mean))), ("медиана", make(np.full(n, median)))]
    majority = _majority(labels)
    if task.type == "classification/multiclass":
        if outputs.values.ndim == 1:
            return [(MAJORITY, make(np.full(n, majority)))]
        size = outputs.values.shape[1]
        onehot = np.zeros((n, size))
        onehot[:, majority] = 1.0
        prior = np.bincount(labels, minlength=size) / n
        return [
            (MAJORITY, make(onehot)),
            ("константа частот классов", make(np.tile(prior, (n, 1)))),
        ]
    if task.output == "scores":
        return [("константа", make(np.zeros(n)))]
    other = 1 - majority
    result = [(MAJORITY, make(np.full(n, float(majority))))]
    if task.output == "probabilities":
        result.append((PRIOR, make(np.full(n, float(np.mean(labels == 1))))))
    if not data_dependent:
        result.append((f"всегда класс {other}", make(np.full(n, float(other)))))
    return result


def trivial_scores(
    item: SpecMetric, task: TaskSection, outputs: Outputs, *, data_dependent: bool = False
) -> list[tuple[str, float]]:
    """Значения метрики у тривиальных бейзлайнов (неопределённые — пропускаются)."""
    scores = []
    for name, trivial in trivial_outputs(task, outputs, data_dependent=data_dependent):
        value = item.metric(trivial)
        if np.isfinite(value):
            scores.append((name, float(value)))
    return scores


def best_trivial(item: SpecMetric, scores: list[tuple[str, float]]) -> tuple[str, float] | None:
    if not scores:
        return None
    if item.higher_is_better:
        return max(scores, key=lambda pair: pair[1])
    return min(scores, key=lambda pair: pair[1])


def is_degenerate(item: SpecMetric, value: float) -> bool:
    """Тривиальное значение у границы ограниченной метрики (≥ 0,9 или ≤ 0,1)."""
    if item.bounded == "high":
        return value >= HIGH_LIMIT
    if item.bounded == "low":
        return value <= LOW_LIMIT
    return False


@dataclass(frozen=True)
class TestResult:
    """Результат одного теста: `passed` — `None` для информационных."""

    name: str
    passed: bool | None
    message: str

    @property
    def mark(self) -> str:
        return {True: "✓", False: "✗", None: "i"}[self.passed]


@dataclass
class MetricTestReport:
    """Тесты одной метрики на прогоне."""

    metric: SpecMetric
    value: float | None
    results: list[TestResult] = field(default_factory=list)
    trivial: list[tuple[str, float]] = field(default_factory=list)

    @property
    def failed(self) -> list[TestResult]:
        return [result for result in self.results if result.passed is False]


def _degenerate_result(item: SpecMetric, best: tuple[str, float] | None) -> TestResult:
    name = "вырожденное решение"
    if item.bounded is None:
        return TestResult(name, None, "не применимо — метрика не ограничена")
    if best is None:
        return TestResult(name, None, "тривиальные бейзлайны не определены")
    limit = HIGH_LIMIT if item.bounded == "high" else LOW_LIMIT
    sign = "≥" if item.bounded == "high" else "≤"
    if is_degenerate(item, best[1]):
        return TestResult(
            name,
            False,
            f"{best[0]} получает {best[1]:.3f} ({sign} {limit}) — метрика не различает модели",
        )
    return TestResult(name, True, f"лучший тривиальный бейзлайн — {best[0]}: {best[1]:.3f}")


def run_metric_tests(item: SpecMetric, task: TaskSection, outputs: Outputs) -> MetricTestReport:
    """Тесты метрики на выходах прогона (docs/specs/metrics.md, таблица тестов)."""
    trivial = trivial_scores(item, task, outputs)
    report = MetricTestReport(item, None, trivial=trivial)
    estimate = estimate_value(item.metric, outputs)
    best = best_trivial(item, trivial)
    report.results.append(_degenerate_result(item, best))
    if isinstance(estimate, str):
        report.results.append(TestResult("значение", False, f"метрика не определена: {estimate}"))
        return report
    report.value = estimate.value
    half = (estimate.high - estimate.low) / 2
    se = (estimate.high - estimate.low) / (2 * Z)
    mde = MDE_FACTOR * se
    report.results.append(
        TestResult(
            "бутстрап-шум",
            None,
            f"95% CI [{estimate.low:.3f}; {estimate.high:.3f}] (±{half:.3f}) на n = {estimate.n}",
        )
    )
    report.results.append(
        TestResult(
            "минимально различимый эффект",
            None,
            f"MDE ≈ {mde:.3f} — разницу меньше между двумя моделями не различить",
        )
    )
    if best is not None:
        gap = estimate.value - best[1] if item.higher_is_better else best[1] - estimate.value
        verdict = gap > mde
        relation = "больше" if verdict else "не больше"
        report.results.append(
            TestResult(
                "отличает модель",
                verdict,
                f"{estimate.value:.3f} против {best[1]:.3f} у «{best[0]}»: разница"
                f" {gap:+.3f} {relation} MDE {mde:.3f}",
            )
        )
    return report


def positive_rate(task: TaskSection, outputs: Outputs) -> float | None:
    if task.type != "classification/binary":
        return None
    return rounded(float(np.mean(outputs.labels == 1)))
