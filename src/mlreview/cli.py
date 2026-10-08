"""Командная строка: review, metrics, diagnose, ask, spec (FR-R01)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from mlreview import __version__
from mlreview.core.task_spec import TaskSpecError, load_task_spec

COMMANDS: dict[str, str] = {
    "review": "ревью прогона или сравнение прогонов",
    "metrics": "проектирование и проверка метрик",
    "diagnose": "диагностика данных, сегментов и кривых обучения",
    "ask": "вопрос по истории проекта и находкам",
    "spec": "работа со спецификацией задачи",
}


def _add_spec_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="spec_action", metavar="<действие>", required=True)
    validate_help = "проверить файл спецификации задачи (FR-C01)"
    validate = actions.add_parser("validate", help=validate_help, description=validate_help)
    validate.add_argument("path", type=Path, help="путь к YAML-файлу спецификации")


def _run_spec(args: argparse.Namespace) -> int:
    try:
        spec = load_task_spec(args.path)
    except TaskSpecError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"спецификация валидна: {args.path} "
        f"(тип задачи: {spec.task.type}, выход: {spec.task.output})"
    )
    return 0


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {"spec": _run_spec}
ACTIONS: dict[str, Callable[[argparse.ArgumentParser], None]] = {"spec": _add_spec_actions}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlreview",
        description="Ассистент-критик для ML-экспериментов.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<команда>")
    for name, help_text in COMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        if name in ACTIONS:
            ACTIONS[name](subparser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handler = HANDLERS.get(args.command)
    if handler is None:
        print(f"mlreview {args.command}: команда пока не реализована", file=sys.stderr)
        return 1
    return handler(args)
