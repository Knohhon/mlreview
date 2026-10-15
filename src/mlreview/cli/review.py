"""Команды spec, contract, review, diagnose и отчёт ревью (FR-R01, FR-C05, FR-R05)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mlreview.adapters.base import load_run, run_source
from mlreview.analyzers.review import GROUPS, review
from mlreview.cli.common import (
    RUN_HELP,
    add_severity_argument,
    add_spec_argument,
    add_tracking_argument,
)
from mlreview.core.findings import Finding, Report, severity_rank
from mlreview.core.run_contract import RunContract
from mlreview.core.store import STATUS_NAMES, Entry, ReviewRecord, RunRecord, Store
from mlreview.core.task_spec import load_task_spec
from mlreview.core.workspace import INIT_HINT, find_workspace

NOT_CONNECTED = f"не подключено — {INIT_HINT.split(' — ')[-1]}"
NOT_USED = "не используется (--no-store)"
DETERMINISTIC = "LLM: не используется — отчёт детерминированный"


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
    ]
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


@dataclass
class ReportView:
    """Что показать из отчёта: находки после фильтров и сведения о хранилище."""

    items: list[tuple[Finding, Entry | None]]
    hidden_dismissed: list[str] = field(default_factory=list)
    hidden_by_filter: int = 0
    store: str | None = None


def build_view(
    report: Report,
    record: ReviewRecord | None = None,
    *,
    min_severity: str = "info",
    limit: int = 0,
    store: str | None = None,
) -> ReportView:
    """Находки для показа: по важности, не ниже порога, не больше лимита (B9)."""
    if record is not None:
        items: list[tuple[Finding, Entry | None]] = [(e.finding, e) for e in record.entries]
        hidden = [entry.id for entry in record.hidden]
    else:
        items = [(finding, None) for finding in report.findings]
        hidden = []
    shown = [
        item for item in items if severity_rank(item[0].severity) <= severity_rank(min_severity)
    ]
    if limit > 0:
        shown = shown[:limit]
    return ReportView(shown, hidden, len(items) - len(shown), store)


def _status_text(entry: Entry) -> str:
    if entry.status == "new" and entry.occurrence > 1:
        return f"повторяется: {entry.occurrence}-е вхождение"
    return STATUS_NAMES.get(entry.status, entry.status)


def _version(version: str | None, data_hash: str | None) -> str:
    text = version or "нет"
    return f"{text} ({data_hash})" if data_hash else text


def render_report(
    report: Report,
    title: str,
    view: ReportView | None = None,
    *,
    llm_line: str = DETERMINISTIC,
) -> str:
    """Текстовый отчёт: заголовок, находки по важности, что скрыто и не проверено, строка LLM."""
    view = view or build_view(report)
    lines = [f"{title} {report.run_id}"]
    if report.baseline_id is not None:
        lines.append(
            f"  базовый прогон: {report.baseline_id}, общих примеров: {report.compared_examples}"
        )
        before = _version(report.baseline_data_version, report.baseline_data_hash)
        version = f"{before} → {_version(report.data_version, report.data_hash)}"
    else:
        version = _version(report.data_version, report.data_hash)
    lines.append(f"  версия данных: {version}")
    if view.store is not None:
        lines.append(f"  хранилище находок: {view.store}")
    lines.append(f"находки: {len(view.items)}" if view.items else "находок нет")
    for number, (finding, entry) in enumerate(view.items, start=1):
        label = f"{entry.id} " if entry is not None else ""
        lines.append(
            f"  {number}. [{finding.severity}] {label}{finding.check} — {finding.message}"
            f" ({finding.requirement})"
        )
        if entry is not None:
            lines.append(f"     статус: {_status_text(entry)}")
            if entry.note:
                lines.append(f"     пометка: {entry.note}")
        if finding.evidence:
            lines.append(f"     примеры: {', '.join(finding.evidence)}")
        if finding.suggestion:
            lines.append(f"     предложение: {finding.suggestion}")
    if view.hidden_dismissed:
        lines.append(f"скрыто отклонённых: {len(view.hidden_dismissed)}")
    if view.hidden_by_filter:
        lines.append(f"скрыто: {view.hidden_by_filter} (ниже порога или сверх лимита)")
    if report.skipped:
        lines.append(f"не проверено: {len(report.skipped)}")
        lines += [f"  - {skipped.check}: {skipped.reason}" for skipped in report.skipped]
    lines.append(llm_line)
    return "\n".join(lines)


def report_payload(report: Report, view: ReportView) -> dict[str, Any]:
    """Отчёт для `--format json`: находки после фильтров с полями хранилища (B8)."""
    payload = report.to_dict()
    payload["findings"] = [
        finding.to_dict() if entry is None else entry.to_dict() for finding, entry in view.items
    ]
    payload["hidden_dismissed"] = view.hidden_dismissed
    payload["hidden_by_filter"] = view.hidden_by_filter
    payload["store"] = view.store if view.store not in (NOT_CONNECTED, NOT_USED) else None
    return payload


def _store() -> Store | None:
    workspace = find_workspace()
    return None if workspace is None else Store(workspace.store_path)


def record_report(
    report: Report,
    contracts: Sequence[tuple[RunContract, str]],
    *,
    store: Store | None,
    no_store: bool,
) -> tuple[ReviewRecord | None, str]:
    """Записывает отчёт в хранилище рабочего пространства; строка — состояние хранилища."""
    if no_store:
        return None, NOT_USED
    if store is None:
        return None, NOT_CONNECTED
    runs = [RunRecord.from_contract(contract, source) for contract, source in contracts]
    return store.record_review(report, runs=runs), str(store.path)


def _refs(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _analyze(args: argparse.Namespace, groups: tuple[str, ...], title: str) -> int:
    spec = load_task_spec(args.spec)
    loaded = {}
    for role in ("run_id", "baseline"):
        loaded[role] = [
            (
                load_run(spec, ref, tracking_uri=args.tracking_uri),
                run_source(ref, args.tracking_uri),
            )
            for ref in _refs(getattr(args, role, None))
        ]
    candidates, baselines = loaded["run_id"], loaded["baseline"]
    contract = candidates[0][0]
    baseline = baselines[0][0] if baselines else None
    store = None if args.no_store else _store()
    prior = None
    if store is not None:
        prior = store.count_runs(contract.data_hash, exclude=[contract.run_id])
    report = review(
        spec,
        contract,
        baseline,
        groups=groups,
        candidate_seeds=[run for run, _ in candidates[1:]],
        baseline_seeds=[run for run, _ in baselines[1:]],
        prior_runs=prior,
    )
    record, state = record_report(
        report, [*candidates, *baselines], store=store, no_store=args.no_store
    )
    view = build_view(report, record, min_severity=args.min_severity, limit=args.limit, store=state)
    if args.format == "json":
        print(json.dumps(report_payload(report, view), ensure_ascii=False, indent=2))
    else:
        print(render_report(report, title, view))
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
