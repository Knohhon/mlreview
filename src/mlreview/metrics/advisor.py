"""Рекомендации метрик по ответам диалога о контексте решения (FR-M01, FR-M02).

Правила — таблица в `docs/specs/metrics.md`: решение (порог, ранжирование, вероятность),
цена ошибок, бюджет K, калибровка, дисбаланс классов.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mlreview.core.task_spec import MetricsContext, TaskSpec
from mlreview.metrics.catalog import CATALOG, parse_name

IMBALANCE = 0.1
DEFAULT_K = 10
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Recommendation:
    """Рекомендованная метрика: роль, обоснование и ловушки."""

    name: str
    role: str
    reason: str
    pitfalls: tuple[str, ...]

    @property
    def title(self) -> str:
        return CATALOG[parse_name(self.name)[0]].title


@dataclass
class MetricPlan:
    """Набор метрик для задачи и пометки о допущениях."""

    items: list[Recommendation]
    context: MetricsContext
    notes: list[str] = field(default_factory=list)
    limited_support: bool = False

    @property
    def primary(self) -> str | None:
        return self.items[0].name if self.items else None

    @property
    def secondary(self) -> list[str]:
        return [item.name for item in self.items[1:]]

    def section(self) -> dict[str, Any]:
        """Раздел `metrics:` спецификации для YAML."""
        context = self.context.model_dump(exclude_none=True)
        if not context.get("critical_segments"):
            context.pop("critical_segments", None)
        section: dict[str, Any] = {"primary": self.primary, "secondary": self.secondary}
        if context:
            section["context"] = context
        return section


def _format(value: float) -> str:
    return f"{round(value, 3):g}"


class _Builder:
    def __init__(self, spec: TaskSpec, context: MetricsContext, rate: float | None) -> None:
        self.task = spec.task
        self.context = context
        self.rate = rate
        self.names: list[str] = []
        self.reasons: dict[str, str] = {}
        self.notes: list[str] = []

    @property
    def imbalanced(self) -> bool:
        return self.rate is not None and self.rate < IMBALANCE

    def add(self, name: str, why: str) -> None:
        if name not in self.names:
            self.names.append(name)
            self.reasons[name] = why

    def ranking_metric(self) -> str:
        return "pr_auc" if self.imbalanced else "roc_auc"

    def ranking_why(self) -> str:
        if self.imbalanced:
            return f"класс 1 редкий (доля {self.rate:.1%}): ранжирование без порога"
        return "качество ранжирования без выбора порога"

    def binary(self) -> None:
        output = self.task.output
        decision = self.context.decision or "threshold"
        if decision == "ranking" and output == "labels":
            self.notes.append("ранжирование при output: labels невозможно — метрики по классам")
            decision = "threshold"
        if decision == "probability" and output != "probabilities":
            self.notes.append(f"output: {output} — не вероятности; метрики вероятностей недоступны")
            decision = "threshold"
        if decision == "threshold" and output == "scores":
            self.notes.append("у output: scores нет порога — метрики ранжирования")
            first = self.ranking_metric()
            self.add(first, self.ranking_why())
            self.add("roc_auc" if first == "pr_auc" else "pr_auc", "вторая метрика ранжирования")
            return
        if decision == "ranking":
            k = self.context.budget_k or DEFAULT_K
            if self.context.budget_k is None:
                self.notes.append(f"бюджет проверки K не задан — взят {k}")
            self.add(f"precision_at_k@{k}", f"проверяют {k} первых объектов очереди")
            self.add(f"recall_at_k@{k}", f"какую часть класса 1 покрывают {k} проверок")
            self.add(self.ranking_metric(), self.ranking_why())
        elif decision == "threshold":
            ratio = self.context.cost_ratio or 1.0
            beta = math.sqrt(ratio)
            if abs(beta - 1) < 1e-9:  # noqa: PLR2004
                self.add("f1", "пропуск и ложная тревога стоят одинаково")
            else:
                self.add(
                    f"fbeta@{_format(beta)}",
                    f"пропуск дороже ложной тревоги в {_format(ratio)} раз: β = √r",
                )
            self.add("precision", "доля верных срабатываний при выбранном пороге")
            self.add("recall", "доля найденных примеров класса 1 при выбранном пороге")
            if output != "labels":
                self.add(self.ranking_metric(), self.ranking_why())
        else:
            self.add("log_loss", "вероятность используют напрямую")
            self.add("ece", "вероятность должна соответствовать частоте")
            if not self.imbalanced:
                self.add("brier", "точность вероятностей")
            self.add("roc_auc", "разделимость классов независимо от калибровки")

    def multiclass(self) -> None:
        output = self.task.output
        decision = self.context.decision or "threshold"
        if decision == "ranking" and output == "labels":
            self.notes.append("ранжирование при output: labels невозможно — метрики по классам")
            decision = "threshold"
        if decision == "probability" and output != "probabilities":
            self.notes.append(f"output: {output} — не вероятности; метрики вероятностей недоступны")
            decision = "threshold"
        if decision == "ranking":
            k = self.context.budget_k or DEFAULT_TOP_K
            self.add(f"top_k_accuracy@{k}", f"пользователь выбирает из {k} подсказок")
            self.add("macro_f1", "все классы важны одинаково")
        elif decision == "threshold":
            self.add("macro_f1", "все классы важны одинаково, включая редкие")
            self.add("balanced_accuracy", "средняя полнота по классам")
            self.add("accuracy", "доля верных ответов для отчёта")
        else:
            self.add("log_loss", "вероятности используют напрямую")
            self.add("ece", "уверенность должна соответствовать частоте верных ответов")
            self.add("macro_f1", "качество выбранного класса")

    def regression(self) -> None:
        ratio = self.context.cost_ratio or 1.0
        if abs(ratio - 1) > 1e-9:  # noqa: PLR2004
            tau = ratio / (1 + ratio)
            self.add(
                f"pinball@{_format(tau)}",
                f"недооценка дороже переоценки в {_format(ratio)} раз: τ = r / (1 + r)",
            )
            self.add("mae", "типичный размер ошибки")
        elif self.context.large_errors:
            self.add("rmse", "крупные ошибки непропорционально дороже")
            self.add("mae", "типичный размер ошибки")
        else:
            self.add("mae", "ошибки стоят пропорционально размеру")
            self.add("rmse", "чувствительность к крупным ошибкам")


def recommend(
    spec: TaskSpec, context: MetricsContext, positive_rate: float | None = None
) -> MetricPlan:
    """Метрики для задачи по ответам диалога (`docs/specs/metrics.md`, правила)."""
    builder = _Builder(spec, context, positive_rate)
    task_type = spec.task.type
    if task_type == "classification/binary":
        builder.binary()
    elif task_type == "classification/multiclass":
        builder.multiclass()
    elif task_type == "regression":
        builder.regression()
    else:
        note = f"ограниченная поддержка: в библиотеке нет метрик для {task_type}"
        return MetricPlan([], context, [note], limited_support=True)
    if context.calibration and spec.task.output == "probabilities":
        builder.add("ece", "калибровка важна по ответу диалога")
    items = []
    for index, name in enumerate(builder.names):
        info = CATALOG[parse_name(name)[0]]
        # у метрик с параметром причина уже содержит формулу параметра из каталога
        reason = (
            builder.reasons[name] if info.param else f"{builder.reasons[name]}; {info.rationale}"
        )
        items.append(
            Recommendation(name, "primary" if index == 0 else "secondary", reason, info.pitfalls)
        )
    return MetricPlan(items, context, builder.notes)
