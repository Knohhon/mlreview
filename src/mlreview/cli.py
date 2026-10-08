"""Командная строка: review, metrics, diagnose, ask, spec, contract (FR-R01)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from mlreview import __version__
from mlreview.adapters.mlflow_adapter import MlflowAdapterError, load_mlflow_contract
from mlreview.core.run_contract import RunContract, RunContractError
from mlreview.core.task_spec import TaskSpecError, load_task_spec

COMMANDS: dict[str, str] = {
    "review": "ревью прогона или сравнение прогонов",
    "metrics": "проектирование и проверка метрик",
    "diagnose": "диагностика данных, сегментов и кривых обучения",
    "ask": "вопрос по истории проекта и находкам",
    "spec": "работа со спецификацией задачи",
    "contract": "проверка контракта прогона из трекера",
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


def _add_contract_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="contract_action", metavar="<действие>", required=True)
    check_help = "собрать контракт прогона из MLflow и проверить его (FR-C02, FR-C03)"
    check = actions.add_parser("check", help=check_help, description=check_help)
    check.add_argument("run_id", help="id прогона MLflow")
    check.add_argument(
        "--spec", type=Path, required=True, help="путь к YAML-файлу спецификации задачи"
    )
    check.add_argument(
        "--tracking-uri", help="адрес трекера MLflow; по умолчанию — MLFLOW_TRACKING_URI"
    )


def _contract_summary(contract: RunContract) -> list[str]:
    examples = contract.examples
    if examples is None:
        examples_line = "примеров: нет"
    else:
        splits = ", ".join(sorted(examples["split"].unique()))
        examples_line = f"примеров: {len(examples)}, сплиты: {splits}"
    curves = 0 if contract.curves is None else contract.curves["name"].nunique()
    lines = [
        f"контракт прогона валиден: {contract.run_id}",
        f"  предсказаний: {len(contract.predictions)}",
        f"  {examples_line}",
        f"  кривых: {curves}",
        f"  параметров: {len(contract.config)}",
        f"  версия данных: {contract.data_version or 'нет'}",
    ]
    lines += [f"  нет {gap.field} — недоступно: {gap.purpose}" for gap in contract.gaps]
    return lines


def _run_contract(args: argparse.Namespace) -> int:
    try:
        spec = load_task_spec(args.spec)
        contract = load_mlflow_contract(spec, args.run_id, tracking_uri=args.tracking_uri)
    except (TaskSpecError, RunContractError, MlflowAdapterError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print("\n".join(_contract_summary(contract)))
    return 0


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "spec": _run_spec,
    "contract": _run_contract,
}
ACTIONS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "spec": _add_spec_actions,
    "contract": _add_contract_actions,
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
    return handler(args)
