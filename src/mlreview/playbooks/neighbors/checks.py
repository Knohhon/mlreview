"""Проверки плейбука neighbors: масштаб признаков и малое k (FR-D11, B14, B15).

Поведение зафиксировано в `docs/specs/playbook-neighbors.md`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mlreview.analyzers.data_checks import feature_columns
from mlreview.analyzers.evaluation import rounded
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.run_contract import CLASS_PREFIX, RunContract
from mlreview.playbooks.context import PlaybookContext
from mlreview.playbooks.resolve import flatten

SCALE_CHECK = "neighbors.unscaled_features"
SMALL_K_CHECK = "neighbors.small_k"
NO_SCALER_VALUES = frozenset({"none", "null", "false", "no", ""})
SCALER_KEY = re.compile(r"scaler|normalizer|transformer", re.IGNORECASE)
SCALER_VALUE = re.compile(r"Scaler|Normalizer|QuantileTransformer|PowerTransformer")
TRAIN = "train"


def _scaler(config: dict) -> str | None:
    """Скейлер в config: `"есть"`, `"нет"` (объявлен как отсутствующий) или `None`."""
    flat = flatten(config)
    declared_absent = False
    for key, value in flat.items():
        last = re.split(r"[.]|__", key)[-1].lower()
        if last == "scaler":
            if str(value).strip().lower() in NO_SCALER_VALUES:
                declared_absent = True
                continue
            return "есть"
        step = key.split("__", 1)[0].rsplit(".", 1)[-1]
        if SCALER_KEY.search(step) or SCALER_VALUE.search(str(value)):
            return "есть"
    return "нет" if declared_absent else None


def _scale_finding(ctx: PlaybookContext, results: CheckResults) -> None:
    contract, spec = ctx.review.contract, ctx.review.spec
    examples = contract.examples
    if examples is None:
        results.skip(SCALE_CHECK, "нет examples")
        return
    train = examples[examples["split"].astype(str) == TRAIN]
    if train.empty:
        results.skip(SCALE_CHECK, "в examples нет сплита train")
        return
    stds = {}
    for column in feature_columns(spec, examples):
        values = train[column]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            std = float(np.nanstd(values.astype(float).to_numpy()))
            if np.isfinite(std) and std > 0:
                stds[str(column)] = std
    if len(stds) < 2:  # noqa: PLR2004 — отношение масштабов нужно хотя бы для двух признаков
        results.skip(SCALE_CHECK, "меньше двух числовых признаков с разбросом на train")
        return
    largest = max(stds, key=lambda name: (stds[name], name))
    smallest = min(stds, key=lambda name: (stds[name], name))
    ratio = rounded(stds[largest] / stds[smallest])
    scaler = _scaler(contract.config)
    if ratio < ctx.thresholds["scale_ratio"] or scaler == "есть":
        return
    facts = {
        "ratio": ratio,
        "largest_feature": largest,
        "largest_std": rounded(stds[largest]),
        "smallest_feature": smallest,
        "smallest_std": rounded(stds[smallest]),
        "features": len(stds),
        "scaler": scaler or "неизвестно",
    }
    message = (
        "признаки kNN в разных масштабах: стандартное отклонение на train различается"
        f" в {facts['ratio']:.1f} раз ({largest} — {facts['largest_std']:.3g},"
        f" {smallest} — {facts['smallest_std']:.3g}); без масштабирования расстояния"
        f" определяет {largest}"
    )
    severity = "medium" if scaler == "нет" else "low"
    if scaler is None:
        message += (
            "; в config нет сведений о масштабировании — проверьте, есть ли скейлер в пайплайне"
        )
    suggestion = ctx.playbook.manifest.checks[SCALE_CHECK].suggestion
    results.findings.append(
        Finding(SCALE_CHECK, severity, message, "FR-D11, B14", (), facts, suggestion)
    )


def _probabilities(contract: RunContract) -> np.ndarray:
    """Вероятности для оценки грубости: `prediction` бинарной или максимум строки multiclass."""
    predictions = contract.predictions
    if "prediction" in predictions.columns:
        return predictions["prediction"].astype(float).to_numpy()
    columns = [column for column in predictions.columns if column.startswith(CLASS_PREFIX)]
    return predictions[columns].astype(float).to_numpy().max(axis=1)


def _small_k_finding(ctx: PlaybookContext, results: CheckResults) -> None:
    if "n_neighbors" not in ctx.params:
        aliases = ", ".join(ctx.playbook.manifest.params["n_neighbors"].aliases)
        results.skip(SMALL_K_CHECK, f"в config нет n_neighbors (алиасы: {aliases})")
        return
    k = ctx.params["n_neighbors"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        results.skip(SMALL_K_CHECK, f"n_neighbors={k} — ожидается целое число ≥ 1")
        return
    if k > ctx.thresholds["small_k"]:
        return
    contract, task = ctx.review.contract, ctx.review.spec.task
    examples = contract.examples
    n_train = None if examples is None else int((examples["split"].astype(str) == TRAIN).sum())
    message = f"kNN с k = {k}: предсказание по {k} ближайшим соседям запоминает шум train"
    facts: dict = {"n_neighbors": k}
    if task.type != "regression" and task.output == "probabilities":
        values = _probabilities(contract)
        distinct = int(np.unique(values).size)
        extreme = rounded(np.mean((values == 0) | (values == 1)))
        facts.update({"distinct_values": distinct, "extreme_share": extreme})
        message += f"; вероятности принимают {distinct} значений, ровно 0 или 1 — {extreme}"
        if extreme >= ctx.thresholds["extreme_share"]:
            message += " — ROC AUC и калибровка по таким вероятностям почти бессмысленны"
    facts["n_train"] = n_train
    suggestion = ctx.playbook.manifest.checks[SMALL_K_CHECK].suggestion
    results.findings.append(
        Finding(SMALL_K_CHECK, "medium", message, "FR-D11, B15", (), facts, suggestion)
    )


def run(ctx: PlaybookContext) -> CheckResults:
    """Проверки kNN для прогона кандидата."""
    results = CheckResults()
    _scale_finding(ctx, results)
    _small_k_finding(ctx, results)
    return results
