"""Группа ревью `model`: проверки применимых плейбуков и их подсказки (FR-D11, X1).

Поведение зафиксировано в `docs/specs/playbooks.md`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from mlreview.analyzers.registry import ReviewContext
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import RunContract
from mlreview.playbooks import loader
from mlreview.playbooks.context import PlaybookContext
from mlreview.playbooks.loader import Playbook
from mlreview.playbooks.resolve import model_params, resolve_family

MODEL_CHECK = "model"
NO_MODEL = (
    "в контракте нет сведений о модели — проверки по семейству не выполнены;"
    " укажите model в run.yaml или теги mlreview.model.*"
)
UNKNOWN_FAMILY = (
    "семейство модели не определено для класса {name}; укажите model.family (run.yaml)"
    " или тег mlreview.model.family"
)


def applicable(contract: RunContract, task_type: str) -> list[Playbook]:
    """Плейбуки, применимые к семейству модели прогона и типу задачи."""
    family = resolve_family(contract.model)
    return [
        playbook for playbook in loader.builtin_playbooks() if playbook.applies(family, task_type)
    ]


def model_findings(ctx: ReviewContext) -> CheckResults:
    """Проверки плейбуков для модели кандидата; без модели — пропуск с причиной (B9, E1, E2)."""
    results = CheckResults()
    model = ctx.contract.model
    family = resolve_family(model)
    if model is None:
        results.skip(MODEL_CHECK, NO_MODEL)
        return results
    if family is None:
        results.skip(MODEL_CHECK, UNKNOWN_FAMILY.format(name=model.class_name))
        return results
    for playbook in applicable(ctx.contract, ctx.spec.task.type):
        if playbook.run is None:
            continue
        params = model_params(ctx.contract.config, playbook)
        try:
            results.extend(playbook.run(PlaybookContext(ctx, playbook, family, params)))
        except Exception as exc:  # плейбук не должен ронять ревью целиком
            results.skip(
                f"playbook.{playbook.id}", f"ошибка анализатора: {type(exc).__name__}: {exc}"
            )
    return results


def apply_overrides(
    findings: Iterable[Finding], contract: RunContract, task_type: str
) -> list[Finding]:
    """Подсказки общих проверок из `overrides` применимых плейбуков (B11)."""
    overrides: dict[str, str] = {}
    for playbook in applicable(contract, task_type):
        for check, text in playbook.manifest.overrides.items():
            overrides.setdefault(check, text)
    return [
        dataclasses.replace(finding, suggestion=overrides[finding.check])
        if finding.check in overrides
        else finding
        for finding in findings
    ]


def model_summary(contract: RunContract) -> dict[str, str | None] | None:
    """Класс и определённое семейство модели для отчёта и контекста LLM (B12)."""
    if contract.model is None:
        return None
    return {"class": contract.model.class_name, "family": resolve_family(contract.model)}
