"""Командная строка: review, metrics, diagnose, ask, spec, contract (FR-R01)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from mlreview import __version__
from mlreview.adapters.mlflow_adapter import MlflowAdapterError, load_mlflow_contract
from mlreview.analyzers.review import review
from mlreview.core.findings import Report
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


def _add_run_arguments(parser: argparse.ArgumentParser, *, baseline: bool) -> None:
    """Общие аргументы команд анализа прогона (FR-R01)."""
    parser.add_argument("run_id", help="id прогона MLflow")
    parser.add_argument(
        "--spec", type=Path, required=True, help="путь к YAML-файлу спецификации задачи"
    )
    if baseline:
        parser.add_argument("--baseline", help="id базового прогона MLflow для сравнения")
    parser.add_argument(
        "--tracking-uri", help="адрес трекера MLflow; по умолчанию — MLFLOW_TRACKING_URI"
    )
    parser.add_argument("--format", choices=("text", "json"), default="text", help="формат отчёта")


def _add_review_arguments(parser: argparse.ArgumentParser) -> None:
    _add_run_arguments(parser, baseline=True)


def _add_diagnose_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="diagnose_action", metavar="<группа>", required=True)
    data_help = "проверки данных: дубли между сплитами, временная утечка, сильные признаки"
    data = actions.add_parser("data", help=data_help, description=data_help)
    _add_run_arguments(data, baseline=False)
    segments_help = "срезы: сегменты хуже остальных и регрессии сегментов"
    segments = actions.add_parser("segments", help=segments_help, description=segments_help)
    _add_run_arguments(segments, baseline=True)


def _add_ask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("question", nargs="*", help="вопрос по истории проекта и находкам")


def render_report(report: Report, title: str) -> str:
    """Текстовый отчёт: заголовок, находки по важности, что не проверено, строка LLM."""
    lines = [f"{title} {report.run_id}"]
    if report.baseline_id is not None:
        lines.append(
            f"  базовый прогон: {report.baseline_id}, общих примеров: {report.compared_examples}"
        )
        version = f"{report.baseline_data_version or 'нет'} → {report.data_version or 'нет'}"
    else:
        version = report.data_version or "нет"
    lines.append(f"  версия данных: {version}")
    lines.append(f"находки: {len(report.findings)}" if report.findings else "находок нет")
    for number, finding in enumerate(report.findings, start=1):
        lines.append(
            f"  {number}. [{finding.severity}] {finding.check} — {finding.message}"
            f" ({finding.requirement})"
        )
        if finding.evidence:
            lines.append(f"     примеры: {', '.join(finding.evidence)}")
        if finding.suggestion:
            lines.append(f"     предложение: {finding.suggestion}")
    if report.skipped:
        lines.append(f"не проверено: {len(report.skipped)}")
        lines += [f"  - {skipped.check}: {skipped.reason}" for skipped in report.skipped]
    lines.append("LLM: не используется — отчёт детерминированный")
    return "\n".join(lines)


def _analyze(args: argparse.Namespace, groups: tuple[str, ...], title: str) -> int:
    try:
        spec = load_task_spec(args.spec)
        contract = load_mlflow_contract(spec, args.run_id, tracking_uri=args.tracking_uri)
        baseline = None
        if getattr(args, "baseline", None):
            baseline = load_mlflow_contract(spec, args.baseline, tracking_uri=args.tracking_uri)
    except (TaskSpecError, RunContractError, MlflowAdapterError) as exc:
        print(exc, file=sys.stderr)
        return 1
    report = review(spec, contract, baseline, groups=groups)
    if args.format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_report(report, title))
    return 0


def _run_review(args: argparse.Namespace) -> int:
    return _analyze(args, ("data", "segments", "metrics"), "ревью прогона")


DIAGNOSE_TITLES = {
    "data": "диагностика данных прогона",
    "segments": "диагностика сегментов прогона",
}


def _run_diagnose(args: argparse.Namespace) -> int:
    group = args.diagnose_action
    return _analyze(args, (group,), DIAGNOSE_TITLES[group])


def _run_ask(_args: argparse.Namespace) -> int:
    print(
        "mlreview ask: требует LLM — LLM-слой не подключён (появится на L1, FR-L01); "
        "детерминированные находки без LLM: mlreview review <run_id> --spec <путь>",
        file=sys.stderr,
    )
    return 1


def _run_metrics(_args: argparse.Namespace) -> int:
    print(
        "mlreview metrics: команда пока не реализована: "
        "проектирование метрик (FR-M01–M06) — уровень L1",
        file=sys.stderr,
    )
    return 1


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "review": _run_review,
    "metrics": _run_metrics,
    "diagnose": _run_diagnose,
    "ask": _run_ask,
    "spec": _run_spec,
    "contract": _run_contract,
}
ACTIONS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "review": _add_review_arguments,
    "diagnose": _add_diagnose_actions,
    "ask": _add_ask_arguments,
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
