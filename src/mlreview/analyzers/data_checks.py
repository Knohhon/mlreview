"""Проверки данных: дубли между сплитами, временная утечка, подозрительно сильные признаки.

Закрывает FR-D03 и дефекты бенчмарка B01 (дубли train/test) и B02 (временная утечка).
Признаки — колонки `examples`, кроме id, сплита, сегментов и времени. Поведение
зафиксировано в `docs/specs/data-checks.md`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlreview.analyzers.evaluation import roc_auc, rounded
from mlreview.core.findings import CheckResults, Finding, evidence_ids
from mlreview.core.run_contract import RunContract
from mlreview.core.task_spec import TaskSpec

DUPLICATES_CHECK = "data.split_duplicates"
TEMPORAL_CHECK = "data.temporal_leak"
STRONG_CHECK = "data.strong_feature"
CRITICAL_SHARE = 0.01
STRONG_THRESHOLD = 0.99
MIN_VALUES = 30

SPLIT_RANKS = {
    "train": 0,
    "validation": 1,
    "valid": 1,
    "val": 1,
    "dev": 1,
    "test": 2,
    "holdout": 2,
}
EXPECTED_SPLITS = "train, validation (valid, val, dev), test (holdout)"
NO_FEATURES = "в examples нет признаков (колонок, кроме example_id, split, сегментов и времени)"


def _ordered_splits(splits: pd.Series) -> list[str]:
    names = {str(name) for name in splits.unique()}
    return sorted(names, key=lambda name: (SPLIT_RANKS.get(name, len(SPLIT_RANKS)), name))


def _features(spec: TaskSpec, examples: pd.DataFrame) -> list[str]:
    service = {"example_id", "split", *spec.task.segments}
    if spec.data and spec.data.time_column:
        service.add(spec.data.time_column)
    return [column for column in examples.columns if column not in service]


def _later_pairs(order: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(order) for b in order[i + 1 :]]


def _sorted_ids(ids: pd.Series) -> list[str]:
    return sorted(ids.astype(str))


def _duplicates(examples: pd.DataFrame, features: list[str]) -> CheckResults:
    """Строки более позднего сплита, совпадающие по всем признакам со строками раннего (B01)."""
    results = CheckResults()
    keys = examples.groupby(features, dropna=False, sort=False).ngroup()
    split = examples["split"].astype(str)
    for earlier, later in _later_pairs(_ordered_splits(examples["split"])):
        in_later = split == later
        copied = in_later & keys.isin(set(keys[split == earlier]))
        count, total = int(copied.sum()), int(in_later.sum())
        if not count:
            continue
        share = count / total
        message = (
            f"{count} из {total} примеров {later} ({share:.1%}) полностью совпадают с примерами"
            f" {earlier} по {len(features)} признакам — метрика на {later}, вероятно, завышена"
        )
        facts = {
            "earlier_split": earlier,
            "later_split": later,
            "duplicates": count,
            "total": total,
            "share": rounded(share),
            "features": len(features),
        }
        suggestion = (
            f"убрать из {earlier} примеры, совпадающие с {later}, или делить данные по группам"
            " одинаковых примеров, и переоценить модель"
        )
        results.findings.append(
            Finding(
                DUPLICATES_CHECK,
                "critical" if share >= CRITICAL_SHARE else "medium",
                message,
                "FR-D03, B01",
                evidence_ids(_sorted_ids(examples.loc[copied, "example_id"])),
                facts,
                suggestion,
            )
        )
    return results


def _parse_time(values: pd.Series) -> pd.Series | None:
    """Время как число или дата; `None`, если непустые значения не читаются."""
    if pd.api.types.is_bool_dtype(values):
        return None
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pandas предупреждает, если формат даты не угадан
        parsed = pd.to_datetime(values, errors="coerce")
    if (parsed.isna() & values.notna()).any():
        return None
    return parsed


def _format_time(value: object) -> str:
    if isinstance(value, int | float | np.number):
        return f"{float(value):g}"
    return str(value)


def _temporal(spec: TaskSpec, examples: pd.DataFrame) -> CheckResults:
    """Примеры позднего сплита раньше последнего примера раннего при temporal (B02)."""
    results = CheckResults()
    strategy = spec.data.split_strategy if spec.data else None
    if strategy != "temporal":
        shown = strategy or "не задан"
        results.skip(TEMPORAL_CHECK, f"split_strategy в спецификации — {shown}, а не temporal")
        return results
    column = spec.data.time_column if spec.data else None
    if column not in examples.columns:
        results.skip(TEMPORAL_CHECK, f"нет examples.{column}")
        return results
    times = _parse_time(examples[column])
    if times is None:
        results.skip(TEMPORAL_CHECK, f"колонка времени {column} не читается как число или дата")
        return results

    order = _ordered_splits(examples["split"])
    unknown = [name for name in order if name not in SPLIT_RANKS]
    if unknown:
        results.skip(
            TEMPORAL_CHECK,
            f"порядок сплитов неизвестен: {', '.join(unknown)}; ожидаются {EXPECTED_SPLITS}",
        )
    known = [name for name in order if name in SPLIT_RANKS]
    split = examples["split"].astype(str)
    for earlier, later in _later_pairs(known):
        if SPLIT_RANKS[earlier] == SPLIT_RANKS[later]:
            continue
        boundary = times[split == earlier].max()
        in_later = (split == later) & times.notna()
        leaked = in_later & (times < boundary)
        count, total = int(leaked.sum()), int(in_later.sum())
        if not count:
            continue
        share = count / total
        boundary_text = _format_time(boundary)
        message = (
            f"{count} из {total} примеров {later} ({share:.1%}) раньше последнего примера"
            f" {earlier} ({boundary_text}) при split_strategy: temporal — модель обучена"
            f" на данных из будущего относительно {later}"
        )
        facts = {
            "earlier_split": earlier,
            "later_split": later,
            "leaked": count,
            "total": total,
            "share": rounded(share),
            "boundary": boundary_text,
        }
        rows = pd.DataFrame(
            {"time": times[leaked], "id": examples.loc[leaked, "example_id"].astype(str)}
        ).sort_values(["time", "id"], kind="mergesort")
        suggestion = (
            f"разделить по времени: {earlier} — до даты отсечки, {later} — после;"
            " переобучить и переоценить модель"
        )
        results.findings.append(
            Finding(
                TEMPORAL_CHECK,
                "critical" if share >= CRITICAL_SHARE else "high",
                message,
                "FR-D03, B02",
                evidence_ids(rows["id"]),
                facts,
                suggestion,
            )
        )
    return results


def _strength(task_type: str, labels: np.ndarray, values: np.ndarray) -> tuple[str, float]:
    if task_type == "regression":
        rho = spearmanr(values, labels).statistic
        return "spearman", abs(float(rho))
    auc = roc_auc(labels, values)
    return "roc_auc", max(auc, 1 - auc)


def _strong_features(
    spec: TaskSpec, examples: pd.DataFrame, predictions: pd.DataFrame, features: list[str]
) -> CheckResults:
    """Признаки, почти идеально предсказывающие метку оцениваемых примеров."""
    results = CheckResults()
    task_type = spec.task.type
    if task_type not in ("classification/binary", "regression"):
        results.skip(STRONG_CHECK, "на L0 — только для бинарной классификации и регрессии")
        return results
    labels = predictions[["example_id", "label"]].astype({"example_id": str})
    rows = examples.astype({"example_id": str}).merge(labels, on="example_id", how="inner")
    if task_type != "regression" and rows["label"].nunique() < 2:  # noqa: PLR2004
        results.skip(STRONG_CHECK, "в оцениваемых примерах один класс")
        return results

    unchecked = []
    for feature in sorted(features, key=str):
        column = rows[feature]
        if not pd.api.types.is_numeric_dtype(column):
            unchecked.append(f"{feature} (нечисловой)")
            continue
        present = column.notna()
        values = column[present].astype(float).to_numpy()
        target = rows.loc[present, "label"].astype(float).to_numpy()
        if len(values) < MIN_VALUES:
            unchecked.append(f"{feature} (меньше {MIN_VALUES} непустых значений)")
            continue
        if np.unique(values).size == 1:
            unchecked.append(f"{feature} (константа)")
            continue
        if task_type != "regression" and np.unique(target).size < 2:  # noqa: PLR2004
            unchecked.append(f"{feature} (один класс среди непустых значений)")
            continue
        statistic, strength = _strength(task_type, target, values)
        if not np.isfinite(strength) or strength < STRONG_THRESHOLD:
            continue
        value = rounded(strength)
        shown = f"AUC {value:.3f}" if statistic == "roc_auc" else f"|ρ Спирмена| {value:.3f}"
        message = (
            f"признак {feature} почти идеально предсказывает метку на оцениваемых примерах:"
            f" {shown} (n = {len(values)}) — проверьте утечку таргета"
        )
        facts = {"feature": str(feature), "statistic": statistic, "value": value, "n": len(values)}
        suggestion = (
            "проверить, как вычисляется признак: не использует ли он метку или информацию,"
            " доступную только после события; если связь легитимна — записать это в"
            " data.known_issues спецификации"
        )
        results.findings.append(
            Finding(STRONG_CHECK, "high", message, "FR-D03", (), facts, suggestion)
        )
    if unchecked:
        results.skip(STRONG_CHECK, "не проверены признаки: " + ", ".join(unchecked))
    return results


def data_findings(spec: TaskSpec, contract: RunContract) -> CheckResults:
    """Проверки данных прогона: дубли, временная утечка, сильные признаки."""
    results = CheckResults()
    examples = contract.examples
    if examples is None:
        for check in (DUPLICATES_CHECK, TEMPORAL_CHECK, STRONG_CHECK):
            results.skip(check, "нет examples")
        return results
    features = _features(spec, examples)
    single_split = examples["split"].nunique() < 2  # noqa: PLR2004

    if single_split:
        results.skip(DUPLICATES_CHECK, "в examples один сплит")
    elif not features:
        results.skip(DUPLICATES_CHECK, NO_FEATURES)
    else:
        results.extend(_duplicates(examples, features))

    if single_split:
        results.skip(TEMPORAL_CHECK, "в examples один сплит")
    else:
        results.extend(_temporal(spec, examples))

    if not features:
        results.skip(STRONG_CHECK, NO_FEATURES)
    else:
        results.extend(_strong_features(spec, examples, contract.predictions, features))
    return results
