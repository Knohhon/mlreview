"""Команда journal: гипотезы, решения, заметки и поиск по ним (FR-C06, S05)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlreview.cli.common import open_store
from mlreview.core.store import KINDS, OUTCOMES, JournalEntry, SearchResult
from mlreview.core.task_spec import load_task_spec

KIND_NAMES = {"hypothesis": "гипотеза", "decision": "решение", "note": "заметка"}


def add_journal_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="journal_action", metavar="<действие>", required=True)
    add_help = "добавить гипотезу, решение или заметку"
    add = actions.add_parser("add", help=add_help, description=add_help)
    add.add_argument("kind", choices=tuple(KINDS), help="вид записи")
    add.add_argument("text", help="текст записи")
    add.add_argument(
        "--run", action="append", default=[], help="связанный прогон (можно несколько)"
    )
    add.add_argument(
        "--finding", action="append", default=[], help="связанная находка (можно несколько)"
    )
    add.add_argument("--hypothesis", help="гипотеза, к которой относится решение")
    add.add_argument("--outcome", choices=OUTCOMES, help="исход решения")
    add.add_argument("--data-version", help="версия данных, если прогоны не проходили ревью")
    list_help = "записи журнала"
    listing = actions.add_parser("list", help=list_help, description=list_help)
    listing.add_argument("--kind", choices=tuple(KINDS), help="только этот вид")
    show_help = "запись журнала"
    show = actions.add_parser("show", help=show_help, description=show_help)
    show.add_argument("entry_id", help="id записи, например H-0001")
    search_help = "поиск по журналу и конфигам прогонов"
    search = actions.add_parser("search", help=search_help, description=search_help)
    search.add_argument("query", help="запрос, например «focal loss»")
    search.add_argument("--spec", type=Path, help="спецификация: текущая версия данных")


def _summary(entry: JournalEntry) -> str:
    state = entry.status or entry.outcome
    state_text = f" [{state}]" if state else ""
    return f"{entry.id} {KIND_NAMES[entry.kind]}{state_text} — {entry.text}"


def _details(entry: JournalEntry) -> list[str]:
    lines = [_summary(entry), f"  время: {entry.created_at}"]
    if entry.status:
        lines.append(f"  статус: {entry.status}")
    if entry.outcome:
        lines.append(f"  исход: {entry.outcome}")
    if entry.hypothesis:
        lines.append(f"  гипотеза: {entry.hypothesis}")
    if entry.runs:
        lines.append(f"  прогоны: {', '.join(entry.runs)}")
    if entry.findings:
        lines.append(f"  находки: {', '.join(entry.findings)}")
    if entry.data_versions:
        lines.append(f"  версии данных: {', '.join(entry.data_versions)}")
    return lines


def render_search(result: SearchResult) -> list[str]:
    """Выдача поиска (B3–B7)."""
    if result.hits:
        lines = [f"найдено: {len(result.hits)}"]
        for hit in result.hits:
            lines.append(f"  {_summary(hit.entry)}")
            if hit.entry.runs:
                lines.append(f"     прогоны: {', '.join(hit.entry.runs)}")
            if hit.entry.data_versions:
                lines.append(f"     версии данных: {', '.join(hit.entry.data_versions)}")
            lines += [f"     связано: {_summary(linked)}" for linked in hit.linked]
            if hit.stale:
                lines.append(f"     {hit.stale}")
        return lines
    if result.config_matches:
        found = ", ".join(f"{run} ({key}={value})" for run, key, value in result.config_matches)
        return [
            f"в журнале записей нет; найдено в конфигах прогонов: {found};"
            " причина решения не зафиксирована"
        ]
    return ["ничего не найдено"]


def run_journal(args: argparse.Namespace) -> int:
    store = open_store()
    action = args.journal_action
    if action == "add":
        entry = store.add_entry(
            args.kind,
            args.text,
            runs=args.run,
            findings=args.finding,
            hypothesis=args.hypothesis,
            outcome=args.outcome,
            data_version=args.data_version,
        )
        print(entry.id)
    elif action == "list":
        entries = store.entries(args.kind)
        print("\n".join(_summary(entry) for entry in entries) if entries else "записей нет")
    elif action == "show":
        print("\n".join(_details(store.entry(args.entry_id))))
    else:
        current = None
        if args.spec is not None:
            spec = load_task_spec(args.spec)
            current = spec.data.version if spec.data else None
        print("\n".join(render_search(store.search(args.query, current_version=current))))
    return 0
