"""LLM в командах: гипотезы к ревью и ответ на вопрос `ask` (FR-D07, FR-L01–L03, FR-R01).

Без LLM команды не падают и явно говорят, что не сделано (S16, X2). Примеры уходят в LLM
только с согласием (X6). Поведение зафиксировано в `docs/specs/llm.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mlreview.adapters.base import load_run
from mlreview.analyzers.review import review
from mlreview.cli.common import RUN_HELP, CliError, add_tracking_argument
from mlreview.cli.journal import render_search
from mlreview.core.findings import Finding, Report
from mlreview.core.run_contract import RunContract
from mlreview.core.store import Store
from mlreview.core.task_spec import TaskSpec, load_task_spec
from mlreview.core.workspace import Workspace, find_workspace
from mlreview.llm.assistant import (
    ANSWER_SCHEMA,
    ANSWER_TASK,
    DISABLED,
    HYPOTHESES_SCHEMA,
    HYPOTHESES_TASK,
    Assistant,
    LLMOutcome,
)
from mlreview.llm.config import LLMConfigError, load_llm_config
from mlreview.llm.context import describe_rows, example_rows, finding_dict, run_dict, task_dict
from mlreview.llm.grounding import Grounding, validate_answer, validate_hypotheses

ASK_HINT = (
    "детерминированно без LLM: mlreview journal search <запрос>,"
    " mlreview review <run_id> --spec <путь>"
)
MAX_STORED = 30


def add_llm_arguments(parser: argparse.ArgumentParser, *, hypotheses: bool) -> None:
    """Флаги LLM: гипотезы, фокус, явно выбранные примеры, согласие, пробный запуск."""
    if hypotheses:
        parser.add_argument(
            "--hypotheses",
            action="store_true",
            help="гипотезы и различающий эксперимент от LLM поверх находок (FR-D07)",
        )
        parser.add_argument("--focus", help="о чём гипотезы, например «сегмент language=de»")
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="id примера, который можно отправить в LLM (можно несколько; нужно согласие)",
    )
    parser.add_argument("--yes", action="store_true", help="согласие отправить примеры в LLM")
    parser.add_argument(
        "--dry-run", action="store_true", help="показать запрос к LLM и не отправлять его"
    )


def assistant() -> Assistant:
    workspace = find_workspace()
    settings = {} if workspace is None else workspace.config()
    try:
        config = load_llm_config(settings)
    except LLMConfigError as exc:
        raise CliError(f"настройки LLM: {exc}") from exc
    return Assistant(config, None if workspace is None else workspace.llm_log_path)


def _consent(rows: list[dict[str, Any]], yes: bool) -> tuple[bool, str | None]:
    """Согласие на отправку примеров (X6): --yes, вопрос в терминале или отказ."""
    if not rows:
        return False, None
    if yes:
        return True, f"в LLM отправлены примеры: {describe_rows(rows)}"
    if not sys.stdin.isatty():
        return False, "примеры не отправлены: нет согласия (--yes)"
    print(f"в LLM будут отправлены примеры: {describe_rows(rows)}")
    print("Отправить? (да/нет): ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    if answer in {"да", "д", "yes", "y"}:
        return True, f"в LLM отправлены примеры: {describe_rows(rows)}"
    return False, "примеры не отправлены: нет согласия"


def _examples(
    contract: RunContract | None, ids: Sequence[str], yes: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    if not ids:
        return [], notes
    if contract is None:
        return [], ["примеры не отправлены: не задан прогон (--run)"]
    rows, missing = example_rows(contract, ids)
    if missing:
        notes.append(f"примеров нет в прогоне: {', '.join(missing)}")
    send, note = _consent(rows, yes)
    if note:
        notes.append(note)
    return (rows if send else []), notes


def _row_texts(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, default=str) for row in rows]


def hypotheses_for(
    args: argparse.Namespace,
    spec: TaskSpec,
    report: Report,
    items: Sequence[tuple[Finding, str]],
    contract: RunContract,
) -> LLMOutcome:
    """Гипотезы со ссылками на находки и различающий эксперимент (FR-D07, S03)."""
    helper = assistant()
    if not helper.config.enabled:
        return LLMOutcome("disabled", reason=DISABLED)
    findings = [finding_dict(finding, finding_id) for finding, finding_id in items]
    rows, notes = _examples(contract, args.example, args.yes)
    context: dict[str, Any] = {
        "task": task_dict(spec),
        "run": run_dict(report),
        "findings": findings,
    }
    if args.focus:
        context["focus"] = args.focus
    if rows:
        context["examples"] = rows
    if not findings:
        outcome = LLMOutcome(
            "unavailable", label=helper.config.label, reason="нет находок для гипотез"
        )
        outcome.notes = notes
        return outcome
    texts = [args.focus or "", json.dumps(context["run"], default=str), *_row_texts(rows)]
    grounding = Grounding.build(findings, texts)
    ids = {finding["id"] for finding in findings}
    outcome = helper.run(
        "hypotheses",
        HYPOTHESES_TASK,
        context,
        HYPOTHESES_SCHEMA,
        lambda data: validate_hypotheses(data, ids, grounding),
        dry_run=args.dry_run,
    )
    outcome.notes = notes + outcome.notes
    return outcome


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
    """Строки LLM в конце текстового отчёта."""
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


def add_ask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", nargs="+", help="вопрос по истории проекта и находкам")
    parser.add_argument("--spec", type=Path, help="спецификация задачи (текущая версия данных)")
    parser.add_argument("--run", help="прогон, по ревью которого спрашивают: " + RUN_HELP)
    parser.add_argument("--baseline", help="базовый прогон для ревью: " + RUN_HELP)
    add_tracking_argument(parser)
    add_llm_arguments(parser, hypotheses=False)


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


def _ask_review(
    args: argparse.Namespace, spec: TaskSpec | None
) -> tuple[list, dict | None, RunContract | None]:
    if not args.run:
        return [], None, None
    if spec is None:
        raise CliError("для --run нужна спецификация задачи (--spec)")
    contract = load_run(spec, args.run, tracking_uri=args.tracking_uri)
    baseline = (
        load_run(spec, args.baseline, tracking_uri=args.tracking_uri) if args.baseline else None
    )
    report = review(spec, contract, baseline)
    findings = [finding_dict(f, f"L{i}") for i, f in enumerate(report.findings, start=1)]
    return findings, run_dict(report), contract


def run_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question)
    workspace: Workspace | None = find_workspace()
    helper = assistant()
    store = None if workspace is None else Store(workspace.store_path)
    spec = load_task_spec(args.spec) if args.spec else None
    current = spec.data.version if spec and spec.data else None
    if not helper.config.enabled:
        if store is not None:
            print("без LLM — поиск по журналу:")
            print("\n".join(render_search(store.search(question, current_version=current))))
        print(
            f"mlreview ask: требует LLM — {DISABLED} (.mlreview/config.yaml); {ASK_HINT}",
            file=sys.stderr,
        )
        return 1
    journal, configs = ([], []) if store is None else _journal_context(store, question, current)
    findings = [] if store is None else _stored_findings(store)
    run_findings, run, contract = _ask_review(args, spec)
    findings += run_findings
    if not findings and not journal and not configs:
        raise CliError("нет фактов для ответа: в хранилище нет находок и записей журнала")
    rows, notes = _examples(contract, args.example, args.yes)
    context: dict[str, Any] = {"question": question, "findings": findings, "journal": journal}
    if configs:
        context["run_configs"] = configs
    if spec is not None:
        context["task"] = task_dict(spec)
    if run is not None:
        context["run"] = run
    if rows:
        context["examples"] = rows
    texts = [question, *(entry["text"] for entry in journal), json.dumps(configs, default=str)]
    texts += [json.dumps(run, default=str), *_row_texts(rows)]
    grounding = Grounding.build(findings, texts)
    ids = {f["id"] for f in findings} | {entry["id"] for entry in journal}
    ids |= {run_id for entry in journal for run_id in entry["runs"]} | {
        c["run_id"] for c in configs
    }
    if run is not None:
        ids |= {run["run_id"], run["baseline_id"]} - {None}
    outcome = helper.run(
        "ask",
        ANSWER_TASK,
        context,
        ANSWER_SCHEMA,
        lambda data: validate_answer(data, ids, grounding),
        dry_run=args.dry_run,
    )
    outcome.notes = notes + outcome.notes
    lines = outcome_lines(outcome, "answer")
    if outcome.status in ("ok", "dry_run"):
        print("\n".join(lines))
        return 0
    print("\n".join(lines), file=sys.stderr)
    return 1
