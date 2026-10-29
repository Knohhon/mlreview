"""Инструменты MCP: операции CLI в виде функций со словарями JSON на выходе (FR-R02).

Модуль не зависит от пакета `mcp` — сервер (`server.py`) только регистрирует функции.
Поведение зафиксировано в `docs/specs/mcp-server.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from mlreview import api
from mlreview.analyzers.evaluation import prepare
from mlreview.analyzers.review import GROUPS
from mlreview.cli.common import CliError, open_store
from mlreview.llm.service import ask_outcome, hypotheses_outcome
from mlreview.metrics.advisor import context_from_answers, default_answers, recommend
from mlreview.metrics.library import spec_metrics
from mlreview.metrics.testing import positive_rate, run_metric_tests


def review(  # noqa: PLR0913 — те же параметры, что у mlreview review/diagnose
    *,
    run: str,
    spec: str,
    baseline: str | None = None,
    seeds: list[str] | None = None,
    baseline_seeds: list[str] | None = None,
    groups: list[str] | None = None,
    min_severity: str = "info",
    limit: int = 0,
    store: bool = True,
    tracking_uri: str | None = None,
    hypotheses: bool = False,
    focus: str | None = None,
) -> dict[str, Any]:
    """Ревью прогона (каталог или id MLflow) или одной группы проверок, как `--format json`."""
    task = api.load_spec(spec)
    result = api.review_runs(
        task,
        run,
        baseline,
        seeds=seeds or [],
        baseline_seeds=baseline_seeds or [],
        groups=groups or GROUPS,
        tracking_uri=tracking_uri,
        store=store,
    )
    payload = result.to_dict(min_severity=min_severity, limit=limit)
    if hypotheses:
        view = result.view(min_severity=min_severity, limit=limit)
        items = [
            (finding, entry.id if entry is not None else f"L{number}")
            for number, (finding, entry) in enumerate(view.items, start=1)
        ]
        outcome = hypotheses_outcome(task, result.report, items, result.contract, focus=focus)
        payload["llm"] = outcome.to_dict()
    return payload


def contract_check(run: str, spec: str, tracking_uri: str | None = None) -> dict[str, Any]:
    """Контракт прогона: размеры, сплиты, кривые, версия и хеш данных, модель, пробелы."""
    contract = api.load_run(api.load_spec(spec), run, tracking_uri=tracking_uri)
    examples = contract.examples
    return {
        "run_id": contract.run_id,
        "predictions": len(contract.predictions),
        "examples": None if examples is None else len(examples),
        "splits": [] if examples is None else sorted(examples["split"].astype(str).unique()),
        "curves": [] if contract.curves is None else sorted(contract.curves["name"].unique()),
        "params": len(contract.config),
        "data_version": contract.data_version,
        "data_hash": contract.data_hash,
        "gaps": [{"field": gap.field, "purpose": gap.purpose} for gap in contract.gaps],
        "model": None if contract.model is None else contract.model.to_dict(),
        "feature_importance": (
            None if contract.feature_importance is None else len(contract.feature_importance)
        ),
    }


def metrics_recommend(
    spec: str,
    answers: dict[str, Any] | None = None,
    run: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Рекомендации метрик по ответам о контексте решения (без записи в спецификацию)."""
    task = api.load_spec(spec)
    merged = {**default_answers(task), **(answers or {})}
    try:
        context = context_from_answers(task, merged)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise CliError(f"недопустимые ответы: {error['loc']}: {error['msg']}") from None
    rate = None
    if run:
        contract = api.load_run(task, run, tracking_uri=tracking_uri)
        rate = positive_rate(task.task, prepare(task.task, contract.predictions))
    plan = recommend(task, context, rate)
    return {
        "primary": plan.primary,
        "secondary": plan.secondary,
        "items": [
            {"name": i.name, "role": i.role, "reason": i.reason, "pitfalls": list(i.pitfalls)}
            for i in plan.items
        ],
        "notes": plan.notes,
        "limited_support": plan.limited_support,
        "section": None if plan.limited_support else plan.section(),
    }


def metrics_test(run: str, spec: str, tracking_uri: str | None = None) -> dict[str, Any]:
    """Юнит-тесты метрик задачи на прогоне: бейзлайны, вырожденность, шум, MDE."""
    task = api.load_spec(spec)
    contract = api.load_run(task, run, tracking_uri=tracking_uri)
    outputs = prepare(task.task, contract.predictions)
    metrics = []
    for item in spec_metrics(task):
        report = run_metric_tests(item, task.task, outputs)
        metrics.append(
            {
                "name": item.name,
                "role": item.role,
                "value": report.value,
                "tests": [
                    {"name": r.name, "passed": r.passed, "message": r.message}
                    for r in report.results
                ],
                "trivial": [{"name": name, "value": value} for name, value in report.trivial],
            }
        )
    return {"run_id": contract.run_id, "n": len(outputs), "metrics": metrics}


def findings_list(
    status: str | None = None, min_severity: str = "low", check: str | None = None
) -> dict[str, Any]:
    """Находки хранилища (по умолчанию без info)."""
    items = open_store().findings(status=status, min_severity=min_severity, check=check)
    return {"findings": [item.to_dict() for item in items]}


def finding_show(finding_id: str) -> dict[str, Any]:
    """Находка: факты, доказательства, история и вхождения."""
    item = open_store().finding(finding_id)
    return {
        **item.to_dict(),
        "history": [vars(event) for event in item.history],
        "occurrences_detail": [
            {
                "run_id": o.run_id,
                "baseline_id": o.baseline_id,
                "severity": o.severity,
                "data_version": o.data_version,
                "reviewed_at": o.reviewed_at,
                "hidden": o.hidden,
            }
            for o in item.occurrences
        ],
    }


def finding_set_status(finding_id: str, status: str, reason: str | None = None) -> dict[str, Any]:
    """Статус находки: accepted, fixed, new, dismissed (critical — только с причиной)."""
    store = open_store()
    store.set_status(finding_id, status, reason=reason)
    return store.finding(finding_id).to_dict()


def journal_add(  # noqa: PLR0913 — поля записи журнала
    *,
    kind: str,
    text: str,
    runs: list[str] | None = None,
    findings: list[str] | None = None,
    hypothesis: str | None = None,
    outcome: str | None = None,
    data_version: str | None = None,
) -> dict[str, Any]:
    """Гипотеза, решение или заметка в журнал."""
    entry = open_store().add_entry(
        kind,
        text,
        runs=runs or [],
        findings=findings or [],
        hypothesis=hypothesis,
        outcome=outcome,
        data_version=data_version,
    )
    return entry.to_dict()


def journal_search(query: str, spec: str | None = None) -> dict[str, Any]:
    """Поиск по журналу и конфигам прогонов; пометка, если данные сменились."""
    current = None
    if spec:
        task = api.load_spec(spec)
        current = task.data.version if task.data else None
    result = open_store().search(query, current_version=current)
    return {
        "hits": [
            {
                "entry": hit.entry.to_dict(),
                "linked": [linked.to_dict() for linked in hit.linked],
                "stale": hit.stale,
            }
            for hit in result.hits
        ],
        "config_matches": [
            {"run_id": run, "key": key, "value": value} for run, key, value in result.config_matches
        ],
    }


def ask(
    question: str,
    spec: str | None = None,
    run: str | None = None,
    baseline: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Вопрос по находкам и журналу; без LLM — статус disabled и поиск по журналу."""
    result = ask_outcome(
        question,
        spec=api.load_spec(spec) if spec else None,
        run=run,
        baseline=baseline,
        tracking_uri=tracking_uri,
    )
    payload = result.outcome.to_dict()
    payload["journal_search"] = result.search_lines
    return payload


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., dict[str, Any]]


TOOLS: tuple[Tool, ...] = (
    Tool("review", "Ревью прогона MLReview: находки по важности, что не проверено", review),
    Tool("contract_check", "Проверка контракта прогона и его пробелов", contract_check),
    Tool("metrics_recommend", "Рекомендации метрик по контексту решения", metrics_recommend),
    Tool("metrics_test", "Юнит-тесты метрик задачи на прогоне", metrics_test),
    Tool("findings_list", "Находки хранилища со статусами", findings_list),
    Tool("finding_show", "Находка: факты, история, вхождения", finding_show),
    Tool(
        "finding_set_status",
        "Принять, исправить, вернуть или отклонить находку",
        finding_set_status,
    ),
    Tool("journal_add", "Запись в журнал гипотез и решений", journal_add),
    Tool("journal_search", "Поиск по журналу гипотез и решений", journal_search),
    Tool("ask", "Вопрос по находкам и журналу (нужен LLM)", ask),
)
