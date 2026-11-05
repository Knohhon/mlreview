"""Команда playbooks: какие плейбуки есть и к чему они применимы (FR-E02, FR-D11)."""

from __future__ import annotations

import argparse

from mlreview.core.task_spec import TASK_TYPES
from mlreview.playbooks import builtin_playbooks
from mlreview.playbooks.loader import Playbook


def add_playbooks_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="playbooks_action", metavar="<действие>", required=True)
    list_help = "встроенные плейбуки: семейства моделей, типы задач, проверки"
    actions.add_parser("list", help=list_help, description=list_help)


def playbook_line(playbook: Playbook) -> str:
    """Строка плейбука: id, название, семейства, типы задач, проверки (B14)."""
    applies_to = playbook.manifest.applies_to
    task_types = applies_to.task_types
    tasks = "все" if set(task_types) == set(TASK_TYPES) else ", ".join(task_types)
    checks = ", ".join(playbook.manifest.checks) or "без описаний"
    if playbook.run is None:
        checks = "только распознавание"
    return (
        f"{playbook.id} — {playbook.manifest.title}; семейства: {', '.join(applies_to.families)};"
        f" типы задач: {tasks}; проверки: {checks}"
    )


def run_playbooks(args: argparse.Namespace) -> int:  # noqa: ARG001 — одно действие list
    print("\n".join(playbook_line(playbook) for playbook in builtin_playbooks()))
    return 0
