"""Команды spec, contract, review, diagnose и отчёт ревью (FR-R01, FR-C05, FR-R05)."""

from __future__ import annotations

import argparse
import json

from mlreview.adapters.base import load_run
from mlreview.analyzers.review import GROUPS
from mlreview.api import review_runs
from mlreview.cli.common import (
    RUN_HELP,
    add_severity_argument,
    add_spec_argument,
    add_tracking_argument,
)
from mlreview.cli.llm import add_llm_arguments, terminal_consent
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import load_task_spec
from mlreview.llm.service import hypotheses_outcome, outcome_lines
from mlreview.reporting import render_report, report_payload


def add_spec_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="spec_action", metavar="<действие>", required=True)
    validate_help = "проверить файл спецификации задачи (FR-C01)"
    validate = actions.add_parser("validate", help=validate_help, description=validate_help)
    validate.add_argument("path", help="путь к YAML-файлу спецификации")


def run_spec(args: argparse.Namespace) -> int:
    spec = load_task_spec(args.path)
    print(
        f"спецификация валидна: {args.path} "
        f"(тип задачи: {spec.task.type}, выход: {spec.task.output})"
    )
    return 0


def add_contract_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="contract_action", metavar="<действие>", required=True)
    check_help = "собрать контракт прогона и проверить его (FR-C02, FR-C03)"
    check = actions.add_parser("check", help=check_help, description=check_help)
    check.add_argument("run_id", help=RUN_HELP)
    add_spec_argument(check)
    add_tracking_argument(check)


def _model_line(contract: RunContract) -> str:
    """Модель прогона для сводки контракта (FR-C07)."""
    model = contract.model
    if model is None:
        return "модель: не указана"
    family = model.family or "не указано"
    name = model.class_name or "класс не указан"
    return f"модель: {name} (семейство: {family}, источник: {model.source})"


def _contract_summary(contract: RunContract) -> list[str]:
    examples = contract.examples
    if examples is None:
        examples_line = "примеров: нет"
    else:
        splits = ", ".join(sorted(examples["split"].astype(str).unique()))
        examples_line = f"примеров: {len(examples)}, сплиты: {splits}"
    curves = 0 if contract.curves is None else contract.curves["name"].nunique()
    lines = [
        f"контракт прогона валиден: {contract.run_id}",
        f"  предсказаний: {len(contract.predictions)}",
        f"  {examples_line}",
        f"  кривых: {curves}",
        f"  параметров: {len(contract.config)}",
        f"  версия данных: {contract.data_version or 'нет'}",
        f"  хеш данных: {contract.data_hash}",
        f"  {_model_line(contract)}",
    ]
    if contract.feature_importance is not None:
        lines.append(f"  важностей признаков: {len(contract.feature_importance)}")
    lines += [f"  нет {gap.field} — недоступно: {gap.purpose}" for gap in contract.gaps]
    return lines


def run_contract(args: argparse.Namespace) -> int:
    spec = load_task_spec(args.spec)
    contract = load_run(spec, args.run_id, tracking_uri=args.tracking_uri)
    print("\n".join(_contract_summary(contract)))
    return 0


def add_run_arguments(
    parser: argparse.ArgumentParser, *, baseline: bool, seeds: bool = False
) -> None:
    """Общие аргументы команд анализа прогона (FR-R01, FR-R05, FR-D05)."""
    if seeds:
        parser.add_argument(
            "run_id", nargs="+", help=RUN_HELP + "; следующие — сиды той же конфигурации"
        )
    else:
        parser.add_argument("run_id", help=RUN_HELP)
    add_spec_argument(parser)
    if baseline:
        parser.add_argument(
            "--baseline",
            nargs="+" if seeds else None,
            help="базовый прогон для сравнения: "
            + RUN_HELP
            + ("; следующие — его сиды" if seeds else ""),
        )
    add_tracking_argument(parser)
    parser.add_argument("--format", choices=("text", "json"), default="text", help="формат отчёта")
    add_severity_argument(parser, "info")
    parser.add_argument(
        "--limit", type=int, default=0, help="показать не больше N находок (0 — все)"
    )
    parser.add_argument("--no-store", action="store_true", help="не записывать находки в хранилище")
    add_llm_arguments(parser, hypotheses=True)


def add_review_arguments(parser: argparse.ArgumentParser) -> None:
    add_run_arguments(parser, baseline=True, seeds=True)


DIAGNOSE_HELP = {
    "data": "проверки данных: дубли между сплитами, временная утечка, сильные признаки",
    "segments": "срезы: сегменты хуже остальных и регрессии сегментов",
    "curves": "кривые обучения: переобучение, недообучение, нестабильность",
    "process": "процесс: разные данные, несколько изменений, многократная валидация",
}


def add_diagnose_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="diagnose_action", metavar="<группа>", required=True)
    for group, help_text in DIAGNOSE_HELP.items():
        action = actions.add_parser(group, help=help_text, description=help_text)
        add_run_arguments(action, baseline=group != "data")


def _analyze(args: argparse.Namespace, groups: tuple[str, ...], title: str) -> int:
    spec = load_task_spec(args.spec)
    runs = [args.run_id] if isinstance(args.run_id, str) else list(args.run_id)
    baseline = getattr(args, "baseline", None)
    baselines = [] if baseline is None else [baseline] if isinstance(baseline, str) else baseline
    result = review_runs(
        spec,
        runs[0],
        baselines[0] if baselines else None,
        seeds=runs[1:],
        baseline_seeds=baselines[1:],
        groups=groups,
        tracking_uri=args.tracking_uri,
        store=not args.no_store,
        title=title,
    )
    view = result.view(min_severity=args.min_severity, limit=args.limit)
    outcome = None
    if args.hypotheses:
        items = [
            (finding, entry.id if entry is not None else f"L{number}")
            for number, (finding, entry) in enumerate(view.items, start=1)
        ]
        outcome = hypotheses_outcome(
            spec,
            result.report,
            items,
            result.contract,
            focus=args.focus,
            example_ids=args.example,
            consent=terminal_consent(args.yes),
            dry_run=args.dry_run,
        )
    if args.format == "json":
        payload = report_payload(result.report, view)
        payload["llm"] = None if outcome is None else outcome.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif outcome is None:
        print(render_report(result.report, title, view))
    else:
        llm_line = "\n".join(outcome_lines(outcome))
        print(render_report(result.report, title, view, llm_line=llm_line))
    return 0


def run_review(args: argparse.Namespace) -> int:
    return _analyze(args, GROUPS, "ревью прогона")


DIAGNOSE_TITLES = {
    "data": "диагностика данных прогона",
    "segments": "диагностика сегментов прогона",
    "curves": "диагностика кривых обучения прогона",
    "process": "диагностика процесса прогона",
}


def run_diagnose(args: argparse.Namespace) -> int:
    group = args.diagnose_action
    return _analyze(args, (group,), DIAGNOSE_TITLES[group])
