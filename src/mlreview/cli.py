"""Командная строка: review, metrics, diagnose, ask (FR-R01)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from mlreview import __version__

COMMANDS: dict[str, str] = {
    "review": "ревью прогона или сравнение прогонов",
    "metrics": "проектирование и проверка метрик",
    "diagnose": "диагностика данных, сегментов и кривых обучения",
    "ask": "вопрос по истории проекта и находкам",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlreview",
        description="Ассистент-критик для ML-экспериментов.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<команда>")
    for name, help_text in COMMANDS.items():
        subparsers.add_parser(name, help=help_text, description=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(f"mlreview {args.command}: команда пока не реализована", file=sys.stderr)
    return 1
