"""Командная строка: review, metrics, diagnose, ask и служебные команды (FR-R01).

Команды хранилища и журнала — FR-C05, FR-C06, FR-R05. Ошибки команд печатаются в stderr
с кодом 1; ошибки использования — код 2 от argparse.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from mlreview import __version__
from mlreview.cli import journal, review, store
from mlreview.cli.common import ERRORS
from mlreview.cli.review import render_report

__all__ = ["COMMANDS", "build_parser", "main", "render_report"]

COMMANDS: dict[str, str] = {
    "review": "ревью прогона или сравнение прогонов",
    "metrics": "проектирование и проверка метрик",
    "diagnose": "диагностика данных, сегментов и кривых обучения",
    "ask": "вопрос по истории проекта и находкам",
    "spec": "работа со спецификацией задачи",
    "contract": "проверка контракта прогона",
    "init": "создать рабочее пространство .mlreview в проекте",
    "findings": "находки хранилища и их статусы",
    "dismiss": "отметить находку как неважную",
    "journal": "журнал гипотез и решений",
}


def _run_ask(_args: argparse.Namespace) -> int:
    print(
        "mlreview ask: требует LLM — LLM-слой не подключён (появится на L1, FR-L01); "
        "детерминированные находки без LLM: mlreview review <run_id> --spec <путь>",
        file=sys.stderr,
    )
    return 1


def _add_ask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", nargs="*", help="вопрос по истории проекта и находкам")


def _run_metrics(_args: argparse.Namespace) -> int:
    print(
        "mlreview metrics: команда пока не реализована: "
        "проектирование метрик (FR-M01–M06) — уровень L1",
        file=sys.stderr,
    )
    return 1


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "review": review.run_review,
    "metrics": _run_metrics,
    "diagnose": review.run_diagnose,
    "ask": _run_ask,
    "spec": review.run_spec,
    "contract": review.run_contract,
    "init": store.run_init,
    "findings": store.run_findings,
    "dismiss": store.run_dismiss,
    "journal": journal.run_journal,
}
ACTIONS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "review": review.add_review_arguments,
    "diagnose": review.add_diagnose_actions,
    "ask": _add_ask_arguments,
    "spec": review.add_spec_actions,
    "contract": review.add_contract_actions,
    "init": store.add_init_arguments,
    "findings": store.add_findings_actions,
    "dismiss": store.add_dismiss_arguments,
    "journal": journal.add_journal_actions,
}


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
    try:
        return handler(args)
    except ERRORS as exc:
        print(exc, file=sys.stderr)
        return 1
