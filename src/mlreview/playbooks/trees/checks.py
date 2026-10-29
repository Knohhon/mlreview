"""Проверки плейбука trees: глубина, ID-подобные признаки, экстраполяция (FR-D11, B16–B18).

Поведение зафиксировано в `docs/specs/playbook-trees.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlreview.analyzers.evaluation import rounded
from mlreview.core.findings import CheckResults, Finding, evidence_ids
from mlreview.playbooks.common import extreme_share, probabilities, train_rows
from mlreview.playbooks.context import PlaybookContext

DEPTH_CHECK = "tree.unbounded_depth"
ID_LIKE_CHECK = "tree.id_like_importance"
EXTRAPOLATION_CHECK = "tree.extrapolation"
SINGLE_TREE = "tree"
NO_KIND = "unspecified"
EDGE = 5
NO_IMPORTANCE = (
    "нет feature_importance — передайте важности признаков"
    " (feature_importance.parquet или артефакт MLflow)"
)


def _limits_leaves(value: object) -> bool:
    """`min_samples_leaf` ограничивает лист: целое > 1 или доля train в (0, 1)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return value > 1 or 0 < value < 1


def _depth_finding(ctx: PlaybookContext, results: CheckResults) -> None:
    if ctx.family != SINGLE_TREE:
        return  # у ансамблей неограниченная глубина отдельных деревьев — норма
    if "max_depth" not in ctx.params:
        aliases = ", ".join(ctx.playbook.manifest.params["max_depth"].aliases)
        results.skip(DEPTH_CHECK, f"в config нет max_depth (алиасы: {aliases})")
        return
    leaf = ctx.params.get("min_samples_leaf")
    if (
        ctx.params["max_depth"] is not None
        or _limits_leaves(leaf)
        or ctx.params.get("max_leaf_nodes") is not None
    ):
        return
    leaf_text = "min_samples_leaf не указан" if leaf is None else f"min_samples_leaf = {leaf}"
    message = (
        f"дерево без ограничения глубины (max_depth не задан, {leaf_text}): листья запоминают train"
    )
    facts: dict = {"max_depth": None, "min_samples_leaf": leaf}
    severity = "low"
    task = ctx.review.spec.task
    if task.type != "regression" and task.output == "probabilities":
        share = extreme_share(probabilities(ctx.review.contract))
        facts["extreme_share"] = share
        if share >= ctx.thresholds["extreme_share"]:
            severity = "medium"
            message += f" — у {share} предсказаний вероятность ровно 0 или 1"
    suggestion = ctx.playbook.manifest.checks[DEPTH_CHECK].suggestion
    results.findings.append(
        Finding(DEPTH_CHECK, severity, message, "FR-D11, B16", (), facts, suggestion)
    )


def _importance_shares(importance: pd.DataFrame) -> dict[str, tuple[float, str]]:
    """Наибольшая доля важности признака среди способов подсчёта; отрицательные — не в сумме."""
    frame = importance.assign(
        kind=importance["kind"].astype(str) if "kind" in importance.columns else NO_KIND,
        importance=importance["importance"].astype(float).clip(lower=0),
    )
    shares: dict[str, tuple[float, str]] = {}
    for kind, rows in frame.groupby("kind", sort=True):
        total = rows["importance"].sum()
        if total <= 0:
            continue
        for feature, value in zip(rows["feature"], rows["importance"], strict=True):
            share = rounded(value / total)
            if share > shares.get(str(feature), (-1.0, ""))[0]:
                shares[str(feature)] = (share, str(kind))
    return shares


def _id_like(values: pd.Series, thresholds: dict) -> float | None:
    """Доля уникальных значений, если признак целый и почти уникальный, иначе `None`."""
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return None
    numbers = values.astype(float).to_numpy()
    numbers = numbers[np.isfinite(numbers)]
    if len(numbers) < thresholds["min_train"] or not np.all(numbers == np.floor(numbers)):
        return None
    unique = rounded(np.unique(numbers).size / len(numbers))
    return unique if unique >= thresholds["id_unique_share"] else None


def _id_like_findings(ctx: PlaybookContext, results: CheckResults) -> None:
    contract = ctx.review.contract
    if contract.feature_importance is None:
        results.skip(ID_LIKE_CHECK, NO_IMPORTANCE)
        return
    if contract.examples is None:
        results.skip(ID_LIKE_CHECK, "нет examples")
        return
    train = train_rows(contract.examples)
    if train.empty:
        results.skip(ID_LIKE_CHECK, "в examples нет сплита train")
        return
    shares = _importance_shares(contract.feature_importance)
    ranked = sorted(shares.items(), key=lambda item: (-item[1][0], item[0]))
    suggestion = ctx.playbook.manifest.checks[ID_LIKE_CHECK].suggestion
    for feature, (share, kind) in ranked:
        if share < ctx.thresholds["importance_share"]:
            break
        unique = _id_like(train[feature], ctx.thresholds)
        if unique is None:
            continue
        ordered = train.sort_values([feature, "example_id"], kind="mergesort")["example_id"]
        evidence = evidence_ids([*ordered.iloc[:EDGE], *ordered.iloc[-EDGE:]])
        message = (
            f"модель опирается на признак {feature}, похожий на идентификатор: доля важности"
            f" {share} ({kind}), на train {unique} значений уникальны и все целые — номер"
            " записи может кодировать порядок данных и метку"
        )
        facts = {
            "feature": feature,
            "importance_share": share,
            "kind": kind,
            "unique_share": unique,
            "n_train": len(train),
        }
        results.findings.append(
            Finding(ID_LIKE_CHECK, "high", message, "FR-D11, B17", evidence, facts, suggestion)
        )


def _extrapolation_finding(ctx: PlaybookContext, results: CheckResults) -> None:
    spec = ctx.review.spec
    if spec.task.type != "regression":
        return
    predictions = ctx.review.contract.predictions
    values = predictions["prediction"].astype(float).to_numpy()
    labels = predictions["label"].astype(float).to_numpy()
    low, high = float(values.min()), float(values.max())
    excess = np.maximum(labels - high, low - labels)
    outside = excess > 0
    share = rounded(outside.mean())
    if share < ctx.thresholds["outside_share"]:
        return
    above, below = int((labels > high).sum()), int((labels < low).sum())
    facts = {
        "outside_share": share,
        "above": above,
        "below": below,
        "prediction_min": rounded(low),
        "prediction_max": rounded(high),
    }
    message = (
        f"предсказания модели ограничены диапазоном [{facts['prediction_min']:.4g},"
        f" {facts['prediction_max']:.4g}]: {share} меток вне его (выше — {above}, ниже —"
        f" {below}) — деревья не экстраполируют за пределы значений train"
    )
    temporal = spec.data is not None and spec.data.split_strategy == "temporal"
    if temporal:
        message += "; при временном сплите это признак тренда таргета"
    order = np.argsort(-excess, kind="mergesort")
    evidence = evidence_ids(predictions["example_id"].to_numpy()[order][outside[order]])
    suggestion = ctx.playbook.manifest.checks[EXTRAPOLATION_CHECK].suggestion
    severity = "medium" if temporal else "low"
    results.findings.append(
        Finding(EXTRAPOLATION_CHECK, severity, message, "FR-D11, B18", evidence, facts, suggestion)
    )


def run(ctx: PlaybookContext) -> CheckResults:
    """Проверки деревьев и ансамблей деревьев для прогона кандидата."""
    results = CheckResults()
    _depth_finding(ctx, results)
    _id_like_findings(ctx, results)
    _extrapolation_finding(ctx, results)
    return results
