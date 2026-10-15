"""Команды init, findings и dismiss: рабочее пространство и статусы находок (FR-C05, FR-R05)."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlreview.cli.common import add_severity_argument, open_store
from mlreview.core.store import STATUS_NAMES, STATUSES, Store, StoredFinding
from mlreview.core.workspace import init_workspace


def add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "directory", nargs="?", type=Path, help="каталог проекта (по умолчанию текущий)"
    )


def run_init(args: argparse.Namespace) -> int:
    workspace, created = init_workspace(args.directory)
    Store(workspace.store_path).initialize()
    if created:
        print(f"рабочее пространство создано: {workspace.root}")
        print(f"  настройки: {workspace.config_path} (LLM выключен)")
    else:
        print(f"уже инициализировано: {workspace.root}")
    return 0


def add_findings_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="findings_action", metavar="<действие>", required=True)
    list_help = "находки хранилища"
    listing = actions.add_parser("list", help=list_help, description=list_help)
    listing.add_argument("--status", choices=STATUSES, help="только с этим статусом")
    add_severity_argument(listing, "low")
    listing.add_argument("--check", help="только эта проверка, например data.split_duplicates")
    show_help = "находка: факты, доказательства, история и вхождения"
    show = actions.add_parser("show", help=show_help, description=show_help)
    show.add_argument("finding_id", help="id находки, например F-0001")
    for action, help_text in (
        ("accept", "принять находку"),
        ("fix", "отметить находку исправленной"),
        ("reopen", "вернуть находку в новые"),
        ("dismiss", "отклонить находку («не важно»)"),
    ):
        sub = actions.add_parser(action, help=help_text, description=help_text)
        _add_status_arguments(sub)


def _add_status_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("finding_id", help="id находки, например F-0001")
    parser.add_argument("--reason", help="причина; для отклонения critical-находки обязательна")


def add_dismiss_arguments(parser: argparse.ArgumentParser) -> None:
    _add_status_arguments(parser)


ACTION_STATUS = {"accept": "accepted", "fix": "fixed", "reopen": "new", "dismiss": "dismissed"}


def _with_reason(status: str, reason: str | None) -> str:
    name = STATUS_NAMES[status]
    return f"{name} ({reason})" if reason else name


def _set_status(finding_id: str, action: str, reason: str | None) -> int:
    store = open_store()
    status = ACTION_STATUS[action]
    store.set_status(finding_id, status, reason=reason)
    stored = store.finding(finding_id)
    print(f"{stored.id}: {_with_reason(stored.status, stored.reason)}")
    return 0


def run_dismiss(args: argparse.Namespace) -> int:
    return _set_status(args.finding_id, "dismiss", args.reason)


def _list_line(item: StoredFinding) -> str:
    return (
        f"{item.id}  {STATUS_NAMES[item.status]:<10}  {item.severity:<8}  {item.check}"
        f"  вхождений: {len(item.occurrences)}  прогон: {item.last.run_id}  {item.message}"
    )


def _show(item: StoredFinding) -> list[str]:
    last = item.last
    lines = [
        f"{item.id} {item.check} ({item.requirement})",
        f"  статус: {_with_reason(item.status, item.reason)}",
        f"  важность: {last.severity}",
        f"  сообщение: {last.message}",
    ]
    if last.facts:
        facts = ", ".join(f"{key}={value}" for key, value in last.facts.items())
        lines.append(f"  факты: {facts}")
    if last.evidence:
        lines.append(f"  примеры: {', '.join(last.evidence)}")
    if last.suggestion:
        lines.append(f"  предложение: {last.suggestion}")
    lines.append(f"  вхождения: {len(item.occurrences)}")
    for occurrence in item.occurrences:
        baseline = f", базовый: {occurrence.baseline_id}" if occurrence.baseline_id else ""
        hidden = " [скрыта]" if occurrence.hidden else ""
        lines.append(
            f"    - {occurrence.reviewed_at} {occurrence.run_id}{baseline}:"
            f" {occurrence.severity}, версия данных {occurrence.data_version or 'нет'}{hidden}"
        )
    lines.append("  история:")
    for event in item.history:
        note = f" — {event.note}" if event.note else ""
        lines.append(f"    - {event.at}: {_with_reason(event.status, event.reason)}{note}")
    return lines


def run_findings(args: argparse.Namespace) -> int:
    action = args.findings_action
    if action in ACTION_STATUS:
        return _set_status(args.finding_id, action, args.reason)
    store = open_store()
    if action == "show":
        print("\n".join(_show(store.finding(args.finding_id))))
        return 0
    items = store.findings(status=args.status, min_severity=args.min_severity, check=args.check)
    print("\n".join(_list_line(item) for item in items) if items else "находок нет")
    return 0
