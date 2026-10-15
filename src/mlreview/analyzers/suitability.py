"""Пригодность метрик задачи на данных прогона: вырожденные метрики (FR-M03, B05).

Тест «вырожденное решение» с бейзлайнами, зависящими от данных: константа мажоритарного
класса и доля класса 1. Поведение зафиксировано в `docs/specs/metrics.md`.
"""

from __future__ import annotations

from mlreview.analyzers.evaluation import prepare, rounded
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec
from mlreview.metrics.catalog import CATALOG
from mlreview.metrics.library import metric_by_name, spec_metrics
from mlreview.metrics.testing import best_trivial, is_degenerate, positive_rate, trivial_scores

CHECK = "metrics.unsuitable"
ALTERNATIVES = ("pr_auc", "balanced_accuracy", "roc_auc", "log_loss", "macro_f1", "mae")


def _alternatives(spec: TaskSpec, outputs, used: set[str], kind: str | None) -> list[str]:
    """До двух метрик каталога, которые на этих данных не вырождены; того же вида — первыми."""
    found = []
    for name in sorted(ALTERNATIVES, key=lambda candidate: CATALOG[candidate].kind != kind):
        info = CATALOG[name]
        if name in used or (spec.task.type, spec.task.output) not in info.tasks:
            continue
        item = metric_by_name(spec, name)
        best = best_trivial(item, trivial_scores(item, spec.task, outputs, data_dependent=True))
        if best is None or not is_degenerate(item, best[1]):
            found.append(name)
        if len(found) == 2:  # noqa: PLR2004 — двух альтернатив достаточно
            break
    return found


def suitability_findings(spec: TaskSpec, contract: RunContract) -> CheckResults:
    """Метрики задачи, вырожденные на метках кандидата (B05)."""
    results = CheckResults()
    outputs = prepare(spec.task, contract.predictions)
    items = spec_metrics(spec)
    used = {item.name for item in items}
    rate = positive_rate(spec.task, outputs)
    for item in items:
        best = best_trivial(item, trivial_scores(item, spec.task, outputs, data_dependent=True))
        if best is None or not is_degenerate(item, best[1]):
            continue
        name, value = best
        facts = {
            "metric": item.name,
            "role": item.role,
            "baseline": name,
            "baseline_value": rounded(value),
        }
        if rate is not None:
            facts["positive_rate"] = rate
        message = (
            f"{item.name} вырождена на этих данных: {name} получает {facts['baseline_value']:.3f}"
            " — метрика не различает модели"
        )
        if rate is not None:
            message += f" (доля класса 1 — {rate:.1%})"
        kind = item.info.kind if item.info else None
        alternatives = _alternatives(spec, outputs, used, kind)
        hint = f"например, {', '.join(alternatives)}; " if alternatives else ""
        suggestion = (
            f"заменить {item.name} метрикой, которая на этих данных отличает модель от константы"
            f" ({hint}подобрать метрики под задачу: mlreview metrics init)"
        )
        severity = "high" if item.role == "primary" else "medium"
        results.findings.append(
            Finding(CHECK, severity, message, "FR-M03, B05", (), facts, suggestion)
        )
    return results
