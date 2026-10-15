"""Логические проверки процесса: разные данные, несколько изменений, валидация (FR-D06).

Ловит дефекты бенчмарка B07 (сравнение на разных версиях данных) и B11 (несколько
изменений в одном эксперименте). Поведение зафиксировано в `docs/specs/process-checks.md`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mlreview.analyzers.registry import ReviewContext
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import RunContract

SEED_KEYS = frozenset({"seed", "random_state", "random_seed", "rng_seed"})
MIN_CHANGES = 2
REUSE_THRESHOLD = 20
MISSING = "нет"
VERSION_CHECK = "process.data_version_mismatch"
CHANGES_CHECK = "process.multiple_changes"
SEED_CHECK = "process.seed_config_mismatch"
REUSE_CHECK = "process.validation_reuse"


def flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Плоский конфиг: вложенные словари — ключи через точку, значения — строки."""
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def is_seed_key(key: str) -> bool:
    last = key.rsplit(".", 1)[-1].lower()
    return last in SEED_KEYS or last.endswith("_seed")


def config_changes(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[tuple[str, str, str]]:
    """Изменённые ключи (кроме ключей сида): ключ, значение базового, значение кандидата."""
    before, after = flatten(baseline), flatten(candidate)
    keys = sorted((set(before) | set(after)) - {k for k in (*before, *after) if is_seed_key(k)})
    return [
        (key, before.get(key, MISSING), after.get(key, MISSING))
        for key in keys
        if before.get(key) != after.get(key)
    ]


def _data_mismatch(candidate: RunContract, baseline: RunContract) -> Finding | None:
    if candidate.data_hash == baseline.data_hash:
        return None
    before, after = baseline.data_version, candidate.data_version
    if before != after:
        message = (
            f"сравнение на разных версиях данных: {before or MISSING} → {after or MISSING}"
            " — разница метрик может объясняться данными, а не моделью"
        )
    else:
        message = (
            f"сравнение на разных данных: хеш {baseline.data_hash} → {candidate.data_hash}"
            f" при версии {after or 'не задана'} — метка версии не отражает изменение"
        )
    facts = {
        "baseline_version": before,
        "candidate_version": after,
        "baseline_hash": baseline.data_hash,
        "candidate_hash": candidate.data_hash,
    }
    suggestion = (
        "сравнивать на одной версии данных: переобучить базовую конфигурацию на данных"
        " кандидата; различающий эксперимент — если разница сохранится, её причина в модели,"
        " а не в данных"
    )
    return Finding(VERSION_CHECK, "high", message, "FR-D06, B07", (), facts, suggestion)


def _multiple_changes(candidate: RunContract, baseline: RunContract) -> Finding | None:
    changes = config_changes(baseline.config, candidate.config)
    if len(changes) < MIN_CHANGES:
        return None
    shown = ", ".join(f"{key} ({before} → {after})" for key, before, after in changes)
    message = (
        f"в эксперименте изменено сразу {len(changes)} параметра: {shown} — эффект нельзя"
        " приписать одному изменению"
    )
    if len(changes) >= 5:  # noqa: PLR2004 — «5 параметров»
        message = message.replace("параметра", "параметров", 1)
    keys = ", ".join(key for key, _, _ in changes)
    suggestion = (
        f"абляция: менять параметры по одному от базового прогона ({keys}); различающий"
        f" эксперимент — базовая конфигурация с одним изменением {changes[0][0]}"
    )
    facts = {"changed": len(changes), "keys": keys}
    return Finding(CHANGES_CHECK, "medium", message, "FR-D06, B11", (), facts, suggestion)


def _seed_mismatch(side: str, runs: Sequence[RunContract]) -> Finding | None:
    if len(runs) < 2:  # noqa: PLR2004 — сравнивать сиды можно начиная с двух
        return None
    keys = sorted(
        {key for run in runs[1:] for key, _, _ in config_changes(runs[0].config, run.config)}
    )
    if not keys:
        return None
    message = f"прогоны по сидам {side} различаются не только сидом: {', '.join(keys)}"
    suggestion = "сиды одной конфигурации должны отличаться только сидом; перезапустить прогоны"
    facts = {"side": side, "keys": ", ".join(keys)}
    return Finding(SEED_CHECK, "medium", message, "FR-D05, FR-D06", (), facts, suggestion)


def _validation_reuse(contract: RunContract, prior_runs: int) -> Finding | None:
    runs = prior_runs + 1
    if runs < REUSE_THRESHOLD:
        return None
    message = (
        f"на одних данных ({contract.data_hash}) прошли ревью {runs} прогонов — решения"
        " принимаются по одной валидации, риск переобучения на ней; проверьте итоговую модель"
        " на отложенной выборке"
    )
    suggestion = (
        "отложить нетронутую выборку для итоговой оценки и не подбирать по ней гиперпараметры;"
        " различающий эксперимент — сравнить лучшую и медианную модель на отложенной выборке"
    )
    facts = {"runs": runs, "data_hash": contract.data_hash, "threshold": REUSE_THRESHOLD}
    return Finding(REUSE_CHECK, "medium", message, "FR-D06", (), facts, suggestion)


def process_findings(ctx: ReviewContext) -> CheckResults:
    """Логические проверки процесса для контекста ревью."""
    results = CheckResults()
    candidate, baseline = ctx.contract, ctx.baseline
    if baseline is None:
        results.skip(VERSION_CHECK, "нет базового прогона")
        results.skip(CHANGES_CHECK, "нет базового прогона")
    else:
        finding = _data_mismatch(candidate, baseline)
        if finding is not None:
            results.findings.append(finding)
        if not candidate.config or not baseline.config:
            side = "кандидата" if not candidate.config else "базового прогона"
            results.skip(CHANGES_CHECK, f"нет config у {side}")
        else:
            finding = _multiple_changes(candidate, baseline)
            if finding is not None:
                results.findings.append(finding)
    for side, runs in (("кандидата", ctx.candidates), ("базового", ctx.baselines)):
        finding = _seed_mismatch(side, runs)
        if finding is not None:
            results.findings.append(finding)
    if ctx.prior_runs is None:
        results.skip(
            REUSE_CHECK, "нет хранилища находок — история прогонов неизвестна (mlreview init)"
        )
    else:
        finding = _validation_reuse(candidate, ctx.prior_runs)
        if finding is not None:
            results.findings.append(finding)
    return results
