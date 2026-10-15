"""Сценарии LLM: гипотезы к ревью и ответ на вопрос — для CLI, MCP и API (FR-D07, FR-L01–L03).

Контекст собирается без сырых данных (NFR-04); примеры — только с согласием (X6); без LLM
возвращается явный статус, а не ошибка (S16, X2). Поведение — `docs/specs/llm.md`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlreview.adapters.base import load_run
from mlreview.analyzers.review import review
from mlreview.core.findings import Finding, Report
from mlreview.core.run_contract import RunContract
from mlreview.core.store import Store
from mlreview.core.task_spec import TaskSpec
from mlreview.core.workspace import find_workspace
from mlreview.llm.assistant import (
    ANSWER_SCHEMA,
    ANSWER_TASK,
    DISABLED,
    HYPOTHESES_SCHEMA,
    HYPOTHESES_TASK,
    Assistant,
    LLMOutcome,
)
from mlreview.llm.config import load_llm_config
from mlreview.llm.context import describe_rows, example_rows, finding_dict, run_dict, task_dict
from mlreview.llm.grounding import Grounding, validate_answer, validate_hypotheses
from mlreview.reporting import render_search

MAX_STORED = 30
NO_CONSENT = "примеры не отправлены: нет согласия (--yes)"

Consent = Callable[[list[dict[str, Any]]], tuple[bool, str | None]]


class LLMContextError(ValueError):
    """Нечего передать в LLM или не хватает входов (E4)."""


def consent_flag(yes: bool) -> Consent:
    """Согласие только флагом: без него примеры не уходят (X6)."""

    def decide(rows: list[dict[str, Any]]) -> tuple[bool, str | None]:
        if yes:
            return True, f"в LLM отправлены примеры: {describe_rows(rows)}"
        return False, NO_CONSENT

    return decide


def make_assistant() -> Assistant:
    """Помощник по настройкам рабочего пространства; без него LLM выключен."""
    workspace = find_workspace()
    settings = {} if workspace is None else workspace.config()
    config = load_llm_config(settings)
    return Assistant(config, None if workspace is None else workspace.llm_log_path)


def _examples(
    contract: RunContract | None, ids: Sequence[str], consent: Consent
) -> tuple[list[dict[str, Any]], list[str]]:
    if not ids:
        return [], []
    if contract is None:
        return [], ["примеры не отправлены: не задан прогон (--run)"]
    rows, missing = example_rows(contract, ids)
    notes = [f"примеров нет в прогоне: {', '.join(missing)}"] if missing else []
    if not rows:
        return [], notes
    send, note = consent(rows)
    if note:
        notes.append(note)
    return (rows if send else []), notes


def _row_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, default=str) for row in rows]


def hypotheses_outcome(  # noqa: PLR0913 — отчёт, находки с id, контракт и параметры LLM
    spec: TaskSpec,
    report: Report,
    items: Sequence[tuple[Finding, str]],
    contract: RunContract,
    *,
    focus: str | None = None,
    example_ids: Sequence[str] = (),
    consent: Consent | None = None,
    dry_run: bool = False,
) -> LLMOutcome:
    """Гипотезы со ссылками на находки и различающий эксперимент (FR-D07, S03)."""
    helper = make_assistant()
    if not helper.config.enabled:
        return LLMOutcome("disabled", reason=DISABLED)
    findings = [finding_dict(finding, finding_id) for finding, finding_id in items]
    rows, notes = _examples(contract, example_ids, consent or consent_flag(False))
    if not findings:
        return LLMOutcome(
            "unavailable", label=helper.config.label, reason="нет находок для гипотез", notes=notes
        )
    context: dict[str, Any] = {
        "task": task_dict(spec),
        "run": run_dict(report),
        "findings": findings,
    }
    if focus:
        context["focus"] = focus
    if rows:
        context["examples"] = rows
    texts = [focus or "", json.dumps(context["run"], default=str), *_row_texts(rows)]
    grounding = Grounding.build(findings, texts)
    ids = {finding["id"] for finding in findings}
    outcome = helper.run(
        "hypotheses",
        HYPOTHESES_TASK,
        context,
        HYPOTHESES_SCHEMA,
        lambda data: validate_hypotheses(data, ids, grounding),
        dry_run=dry_run,
    )
    outcome.notes = notes + outcome.notes
    return outcome


@dataclass
class AskResult:
    """Ответ на вопрос и детерминированная выдача поиска по журналу (без LLM)."""

    outcome: LLMOutcome
    search_lines: list[str] = field(default_factory=list)


def _journal_context(store: Store, question: str, current: str | None) -> tuple[list, list]:
    result = store.search(question, current_version=current)
    wanted: dict[str, str | None] = {}
    for hit in result.hits:
        wanted[hit.entry.id] = hit.stale
        for linked in hit.linked:
            wanted.setdefault(linked.id, None)
    entries = []
    for entry in store.entries():
        if entry.id in wanted:
            item = entry.to_dict()
            item["stale"] = wanted[entry.id]
            entries.append(item)
    configs = [{"run_id": r, "key": k, "value": v} for r, k, v in result.config_matches]
    return entries, configs


def _stored_findings(store: Store) -> list[dict[str, Any]]:
    stored = [item for item in store.findings() if item.status != "dismissed"][-MAX_STORED:]
    result = []
    for item in stored:
        last = item.last
        finding = Finding(
            item.check,
            last.severity,
            last.message,
            item.requirement,
            (),
            last.facts,
            last.suggestion,
        )
        result.append(
            {**finding_dict(finding, item.id), "status": item.status, "last_run": last.run_id}
        )
    return result


def _run_review(
    spec: TaskSpec | None, run: str | None, baseline: str | None, tracking_uri: str | None
) -> tuple[list, dict | None, RunContract | None]:
    if not run:
        return [], None, None
    if spec is None:
        raise LLMContextError("для прогона нужна спецификация задачи (--spec)")
    contract = load_run(spec, run, tracking_uri=tracking_uri)
    base = load_run(spec, baseline, tracking_uri=tracking_uri) if baseline else None
    report = review(spec, contract, base)
    findings = [finding_dict(f, f"L{i}") for i, f in enumerate(report.findings, start=1)]
    return findings, run_dict(report), contract


def ask_outcome(  # noqa: PLR0913 — вопрос, прогоны и параметры LLM
    question: str,
    *,
    spec: TaskSpec | None = None,
    run: str | None = None,
    baseline: str | None = None,
    tracking_uri: str | None = None,
    example_ids: Sequence[str] = (),
    consent: Consent | None = None,
    dry_run: bool = False,
) -> AskResult:
    """Ответ по находкам хранилища, журналу и ревью прогона (S05); без LLM — поиск."""
    workspace = find_workspace()
    helper = make_assistant()
    store = None if workspace is None else Store(workspace.store_path)
    current = spec.data.version if spec and spec.data else None
    if not helper.config.enabled:
        lines = []
        if store is not None:
            lines = render_search(store.search(question, current_version=current))
        return AskResult(LLMOutcome("disabled", reason=DISABLED), lines)
    journal, configs = ([], []) if store is None else _journal_context(store, question, current)
    findings = [] if store is None else _stored_findings(store)
    run_findings, run_info, contract = _run_review(spec, run, baseline, tracking_uri)
    findings += run_findings
    if not findings and not journal and not configs:
        raise LLMContextError("нет фактов для ответа: в хранилище нет находок и записей журнала")
    rows, notes = _examples(contract, example_ids, consent or consent_flag(False))
    context: dict[str, Any] = {"question": question, "findings": findings, "journal": journal}
    if configs:
        context["run_configs"] = configs
    if spec is not None:
        context["task"] = task_dict(spec)
    if run_info is not None:
        context["run"] = run_info
    if rows:
        context["examples"] = rows
    texts = [question, *(entry["text"] for entry in journal), json.dumps(configs, default=str)]
    texts += [json.dumps(run_info, default=str), *_row_texts(rows)]
    grounding = Grounding.build(findings, texts)
    ids = {f["id"] for f in findings} | {entry["id"] for entry in journal}
    ids |= {run_id for entry in journal for run_id in entry["runs"]}
    ids |= {c["run_id"] for c in configs}
    if run_info is not None:
        ids |= {run_info["run_id"], run_info["baseline_id"]} - {None}
    outcome = helper.run(
        "ask",
        ANSWER_TASK,
        context,
        ANSWER_SCHEMA,
        lambda data: validate_answer(data, ids, grounding),
        dry_run=dry_run,
    )
    outcome.notes = notes + outcome.notes
    return AskResult(outcome)


def _refs(refs: Any) -> str:
    return f" [{', '.join(str(ref) for ref in refs)}]" if isinstance(refs, list) and refs else ""


def _status_lines(outcome: LLMOutcome, kind: str) -> list[str] | None:
    """Строки для статусов без ответа модели; `None` — ответ есть."""
    if outcome.status == "disabled":
        what = "гипотезы требуют LLM" if kind == "hypotheses" else "требует LLM"
        return [f"LLM: {what} — {DISABLED}"]
    if outcome.status == "dry_run":
        return [
            "запрос к LLM (не отправлен):",
            json.dumps(outcome.payload, ensure_ascii=False, indent=2),
        ]
    if outcome.status == "unavailable":
        cost = [outcome.cost_line()] if outcome.input_tokens else []
        return [f"LLM: недоступен — {outcome.reason}", *cost]
    if outcome.status == "blocked":
        return [
            f"LLM: ответ заблокирован сверкой с фактами (FR-L02): {'; '.join(outcome.problems)}"
            " — показаны только детерминированные находки",
            outcome.cost_line(),
        ]
    return None


def outcome_lines(outcome: LLMOutcome, kind: str = "hypotheses") -> list[str]:
    """Строки LLM для текстового вывода: гипотезы, ответ или статус."""
    lines = list(outcome.notes)
    special = _status_lines(outcome, kind)
    if special is not None:
        return lines + special
    data = outcome.data or {}
    if kind == "answer":
        return [
            *lines,
            str(data.get("answer", "")),
            f"ссылки: {', '.join(str(ref) for ref in data.get('references', []))}",
            outcome.cost_line(),
        ]
    lines.append(f"гипотезы (LLM: {outcome.label}):")
    for hypothesis in data.get("hypotheses", []):
        refs = _refs(hypothesis.get("findings"))
        lines.append(f"  {hypothesis.get('id')}. {hypothesis.get('statement')}{refs}")
        lines.append(f"      альтернатива: {hypothesis.get('alternative')}")
    experiment = data.get("experiment") or {}
    refs = _refs(experiment.get("findings"))
    lines.append(f"  различающий эксперимент: {experiment.get('description')}{refs}")
    lines.append(outcome.cost_line())
    return lines
