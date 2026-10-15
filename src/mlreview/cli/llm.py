"""LLM в командах: флаги, согласие в терминале, команда ask (FR-D07, FR-L01–L03, FR-R01).

Логика — `mlreview.llm.service`; здесь только аргументы и вывод. Поведение зафиксировано
в `docs/specs/llm.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mlreview.cli.common import RUN_HELP, add_tracking_argument
from mlreview.core.task_spec import load_task_spec
from mlreview.llm.assistant import DISABLED
from mlreview.llm.context import describe_rows
from mlreview.llm.service import Consent, ask_outcome, consent_flag, outcome_lines

ASK_HINT = (
    "детерминированно без LLM: mlreview journal search <запрос>,"
    " mlreview review <run_id> --spec <путь>"
)


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


def terminal_consent(yes: bool) -> Consent:
    """Согласие на отправку примеров (X6): --yes, вопрос в терминале или отказ."""
    if yes or not sys.stdin.isatty():
        return consent_flag(yes)

    def ask(rows: list[dict[str, Any]]) -> tuple[bool, str | None]:
        print(f"в LLM будут отправлены примеры: {describe_rows(rows)}")
        print("Отправить? (да/нет): ", end="", flush=True)
        if sys.stdin.readline().strip().lower() in {"да", "д", "yes", "y"}:
            return True, f"в LLM отправлены примеры: {describe_rows(rows)}"
        return False, "примеры не отправлены: нет согласия"

    return ask


def add_ask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", nargs="+", help="вопрос по истории проекта и находкам")
    parser.add_argument("--spec", type=Path, help="спецификация задачи (текущая версия данных)")
    parser.add_argument("--run", help="прогон, по ревью которого спрашивают: " + RUN_HELP)
    parser.add_argument("--baseline", help="базовый прогон для ревью: " + RUN_HELP)
    add_tracking_argument(parser)
    add_llm_arguments(parser, hypotheses=False)


def run_ask(args: argparse.Namespace) -> int:
    result = ask_outcome(
        " ".join(args.question),
        spec=load_task_spec(args.spec) if args.spec else None,
        run=args.run,
        baseline=args.baseline,
        tracking_uri=args.tracking_uri,
        example_ids=args.example,
        consent=terminal_consent(args.yes),
        dry_run=args.dry_run,
    )
    outcome = result.outcome
    if outcome.status == "disabled":
        if result.search_lines:
            print("без LLM — поиск по журналу:")
            print("\n".join(result.search_lines))
        print(
            f"mlreview ask: требует LLM — {DISABLED} (.mlreview/config.yaml); {ASK_HINT}",
            file=sys.stderr,
        )
        return 1
    lines = outcome_lines(outcome, "answer")
    if outcome.status in ("ok", "dry_run"):
        print("\n".join(lines))
        return 0
    print("\n".join(lines), file=sys.stderr)
    return 1
