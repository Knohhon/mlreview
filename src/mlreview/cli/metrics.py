"""Команда metrics: диалог и рекомендации, юнит-тесты, экспорт, каталог (FR-M01–M05)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mlreview.adapters.base import load_run
from mlreview.analyzers.evaluation import prepare
from mlreview.cli.common import (
    RUN_HELP,
    CliError,
    add_spec_argument,
    add_tracking_argument,
    open_store,
)
from mlreview.core.task_spec import (
    CLASSIFICATION_OUTPUTS,
    REGRESSION_OUTPUTS,
    TASK_TYPES,
    MetricsContext,
    TaskSpec,
    load_task_spec,
    parse_task_spec,
)
from mlreview.metrics.advisor import MetricPlan, context_from_answers, default_answers, recommend
from mlreview.metrics.catalog import CATALOG
from mlreview.metrics.export import export_metrics
from mlreview.metrics.library import spec_metrics
from mlreview.metrics.testing import positive_rate, run_metric_tests

ROLE_NAMES = {"primary": "основная", "secondary": "дополнительная"}
YES = {"да", "д", "yes", "y", "true", "1"}
NO = {"нет", "н", "no", "n", "false", "0"}
ATTEMPTS = 3


def add_metrics_actions(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="metrics_action", metavar="<действие>", required=True)
    init_help = "диалог о контексте решения и рекомендации метрик (FR-M01, FR-M02)"
    init = actions.add_parser("init", help=init_help, description=init_help)
    add_spec_argument(init)
    init.add_argument("--answers", type=Path, help="ответы диалога в YAML (без вопросов)")
    init.add_argument("--run", help="прогон для доли класса 1: " + RUN_HELP)
    add_tracking_argument(init)
    init.add_argument("--write", action="store_true", help="записать раздел metrics: в файл")
    test_help = "юнит-тесты метрик задачи на прогоне (FR-M03, FR-M05)"
    test = actions.add_parser("test", help=test_help, description=test_help)
    test.add_argument("run_id", help=RUN_HELP)
    add_spec_argument(test)
    add_tracking_argument(test)
    test.add_argument("--keep", help="оставить метрику, не прошедшую тесты (решение в журнал)")
    test.add_argument("--reason", help="причина для --keep")
    export_help = "экспорт метрик задачи в исполняемый модуль metrics.py (FR-M04)"
    export = actions.add_parser("export", help=export_help, description=export_help)
    add_spec_argument(export)
    export.add_argument("-o", "--output", type=Path, default=Path("metrics.py"), help="файл")
    list_help = "каталог метрик: обоснование и ловушки"
    listing = actions.add_parser("list", help=list_help, description=list_help)
    listing.add_argument("--task", choices=TASK_TYPES, help="тип задачи")
    listing.add_argument(
        "--output", choices=sorted({*CLASSIFICATION_OUTPUTS, *REGRESSION_OUTPUTS}), help="выход"
    )


def _ask(prompt: str, parse, default: Any) -> Any:
    """Вопрос в терминале; пустой ответ — значение по умолчанию."""
    for _ in range(ATTEMPTS):
        print(prompt, end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            print()
            return default
        answer = line.strip()
        if not answer:
            return default
        try:
            return parse(answer)
        except ValueError as exc:
            print(f"  не понял ответ: {exc}")
    return default


def _yes_no(answer: str) -> bool:
    if answer.lower() in YES:
        return True
    if answer.lower() in NO:
        return False
    raise ValueError("ожидается «да» или «нет»")


def _positive(kind):
    def parse(answer: str):
        value = kind(answer.replace(",", "."))
        if value <= 0:
            raise ValueError("ожидается положительное число")
        return value

    return parse


def _decision(answer: str) -> str:
    choices = {"1": "threshold", "2": "ranking", "3": "probability"}
    if answer not in choices:
        raise ValueError("ожидается 1, 2 или 3")
    return choices[answer]


def _segments(answer: str) -> list[str]:
    return [part.strip() for part in answer.split(",") if part.strip()]


def _dialog(spec: TaskSpec, defaults: dict[str, Any]) -> dict[str, Any]:
    """Вопросы о контексте решения (FR-M01); ответы по умолчанию — из спецификации."""
    answers = dict(defaults)
    task = spec.task
    segments_default = ", ".join(defaults["critical_segments"]) or "нет"
    if task.type == "regression":
        answers["cost_ratio"] = _ask(
            "Во сколько раз недооценка дороже переоценки? (1 — одинаково)"
            f" [по умолчанию {defaults['cost_ratio']:g}]: ",
            _positive(float),
            defaults["cost_ratio"],
        )
        answers["large_errors"] = _ask(
            "Крупные ошибки непропорционально дороже? (да/нет) [по умолчанию нет]: ",
            _yes_no,
            defaults["large_errors"],
        )
    else:
        names = {"threshold": "1", "ranking": "2", "probability": "3"}
        answers["decision"] = _ask(
            "Как используется предсказание? 1 — решение по порогу (да/нет), 2 — ранжирование"
            " (проверяют K первых), 3 — вероятность используют напрямую"
            f" [по умолчанию {names[defaults['decision']]}]: ",
            _decision,
            defaults["decision"],
        )
        if answers["decision"] == "ranking":
            answers["budget_k"] = _ask(
                f"Сколько объектов проверяют за раз (K)? [по умолчанию {defaults['budget_k']}]: ",
                _positive(int),
                defaults["budget_k"],
            )
        if task.type == "classification/binary":
            answers["cost_ratio"] = _ask(
                "Во сколько раз пропуск дороже ложной тревоги? (1 — одинаково)"
                f" [по умолчанию {defaults['cost_ratio']:g}]: ",
                _positive(float),
                defaults["cost_ratio"],
            )
        if task.output == "probabilities":
            answers["calibration"] = _ask(
                "Важна ли калибровка вероятностей? (да/нет) [по умолчанию нет]: ",
                _yes_no,
                defaults["calibration"],
            )
    answers["critical_segments"] = _ask(
        f"Сегменты, где деградация недопустима (через запятую) [по умолчанию {segments_default}]: ",
        _segments,
        defaults["critical_segments"],
    )
    return answers


def _read_answers(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CliError(f"не удалось прочитать ответы {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CliError(f"{path}: ответы — словарь")
    try:
        MetricsContext.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise CliError(f"{path}: недопустимые ответы: {problems}") from None
    return data


def _render_plan(spec: TaskSpec, plan: MetricPlan) -> list[str]:
    lines = [f"рекомендации метрик ({spec.task.type} · {spec.task.output}):"]
    for item in plan.items:
        lines.append(f"  {ROLE_NAMES[item.role]}: {item.name} — {item.title}")
        lines.append(f"    почему: {item.reason}")
        lines.append(f"    ловушки: {'; '.join(item.pitfalls)}")
    lines += [f"пометка: {note}" for note in plan.notes]
    return lines


def _replace_section(text: str, section_yaml: str) -> str:
    """Заменяет верхнеуровневый раздел `metrics:` или дописывает его в конец."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.match(r"metrics\s*:", line)), None)
    if start is None:
        prefix = text if text.endswith("\n") or not text else text + "\n"
        return prefix + section_yaml
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
        end += 1
    return "".join(lines[:start]) + section_yaml + "".join(lines[end:])


def _run_init(args: argparse.Namespace) -> int:
    spec = load_task_spec(args.spec)
    defaults = default_answers(spec)
    if args.answers is not None:
        answers = {**defaults, **_read_answers(args.answers)}
    elif sys.stdin.isatty():
        answers = _dialog(spec, defaults)
    else:
        print("ответы по умолчанию из спецификации (stdin — не терминал; ответы — --answers)")
        answers = defaults
    rate = None
    if args.run:
        contract = load_run(spec, args.run, tracking_uri=args.tracking_uri)
        rate = positive_rate(spec.task, prepare(spec.task, contract.predictions))
    plan = recommend(spec, context_from_answers(spec, answers), rate)
    print("\n".join(_render_plan(spec, plan)))
    if plan.limited_support:
        return 0
    section = yaml.safe_dump(
        {"metrics": plan.section()}, allow_unicode=True, sort_keys=False, default_flow_style=None
    )
    print("раздел для спецификации:")
    print(section, end="")
    if args.write:
        text = _replace_section(args.spec.read_text(encoding="utf-8"), section)
        parse_task_spec(text, source=str(args.spec), base_dir=args.spec.parent)
        args.spec.write_text(text, encoding="utf-8")
        print(f"записано в {args.spec}")
    return 0


def _run_test(args: argparse.Namespace) -> int:
    spec = load_task_spec(args.spec)
    contract = load_run(spec, args.run_id, tracking_uri=args.tracking_uri)
    outputs = prepare(spec.task, contract.predictions)
    reports = [run_metric_tests(item, spec.task, outputs) for item in spec_metrics(spec)]
    lines = [
        f"тесты метрик на прогоне {contract.run_id}"
        f" ({spec.task.type} · {spec.task.output}, n = {len(outputs)})"
    ]
    for report in reports:
        value = "не определена" if report.value is None else f"{report.value:.3f}"
        lines.append(f"{report.metric.name} ({ROLE_NAMES[report.metric.role]}) = {value}")
        lines += [f"  {r.mark} {r.name}: {r.message}" for r in report.results]
        if report.trivial:
            shown = ", ".join(f"{name} = {score:.3f}" for name, score in report.trivial)
            lines.append(f"  тривиальные бейзлайны: {shown}")
    failed = [report.metric.name for report in reports if report.failed]
    lines.append(f"не прошли тесты: {', '.join(failed)}" if failed else "все метрики прошли тесты")
    print("\n".join(lines))
    if args.keep:
        return _keep(args, contract.run_id, failed, {r.metric.name for r in reports})
    return 0


def _keep(args: argparse.Namespace, run_id: str, failed: list[str], names: set[str]) -> int:
    if args.keep not in names:
        raise CliError(f"метрики {args.keep} нет в спецификации")
    if args.keep not in failed:
        raise CliError(f"метрика {args.keep} прошла тесты — фиксировать нечего")
    if not args.reason or not args.reason.strip():
        raise CliError("для --keep нужна причина (--reason)")
    store = open_store()
    text = f"оставить метрику {args.keep}, несмотря на непройденные тесты: {args.reason.strip()}"
    entry = store.add_entry("decision", text, runs=[run_id], outcome="accept")
    print(f"решение записано в журнал: {entry.id}")
    return 0


def _run_export(args: argparse.Namespace) -> int:
    spec = load_task_spec(args.spec)
    args.output.write_text(export_metrics(spec), encoding="utf-8")
    print(f"метрики записаны в {args.output}: {', '.join(m.name for m in spec_metrics(spec))}")
    return 0


def _run_list(args: argparse.Namespace) -> int:
    lines = []
    for info in CATALOG.values():
        if args.task and not any(task == args.task for task, _ in info.tasks):
            continue
        if args.output and not any(
            output == args.output and (args.task is None or task == args.task)
            for task, output in info.tasks
        ):
            continue
        name = f"{info.name}@{info.param.name}" if info.param else info.name
        direction = "больше лучше" if info.higher_is_better else "меньше лучше"
        lines.append(f"{name} — {info.title} ({direction})")
        lines.append(f"  когда: {info.rationale}")
        lines.append(f"  ловушки: {'; '.join(info.pitfalls)}")
    print("\n".join(lines) if lines else "метрик нет")
    return 0


HANDLERS = {"init": _run_init, "test": _run_test, "export": _run_export, "list": _run_list}


def run_metrics(args: argparse.Namespace) -> int:
    return HANDLERS[args.metrics_action](args)
