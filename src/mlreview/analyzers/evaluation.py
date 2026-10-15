"""Метрики прогона и бутстрап для сравнения прогонов и срезов (FR-D01, FR-D02, NFR-05).

Набор метрик задаётся парой `task.type · task.output` (FR-C01); интервалы — перцентильный
бутстрап с фиксированным сидом. Поведение зафиксировано в `docs/specs/run-comparison.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from mlreview.core.findings import evidence_ids
from mlreview.core.run_contract import CLASS_PREFIX, RunContract
from mlreview.core.task_spec import TaskSection

RESAMPLES = 1000
SEED = 0
CONFIDENCE = 0.95
MAX_UNDEFINED_SHARE = 0.1
EPSILON = 1e-15
NO_COMMON_EXAMPLES = "у прогонов нет общих примеров — парное сравнение невозможно"


@dataclass(frozen=True)
class Outputs:
    """Метки и предсказания прогона в виде массивов, упорядоченные по `example_id`.

    В multiclass метки и предсказанные классы — коды 0…`n_classes`−1; при `probabilities` и
    `scores` значения — матрица с колонкой на класс.
    """

    ids: np.ndarray
    labels: np.ndarray
    values: np.ndarray
    n_classes: int | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, index: np.ndarray) -> Outputs:
        return Outputs(self.ids[index], self.labels[index], self.values[index], self.n_classes)

    def resample(self, index: np.ndarray) -> Outputs:
        """Бутстрап-выборка без id: метрикам id не нужны, а копировать строки дорого."""
        return Outputs(_NO_IDS, self.labels[index], self.values[index], self.n_classes)


_NO_IDS = np.empty(0, dtype=str)
MetricFn = Callable[[Outputs], float]
Sampler = Callable[[Outputs], Callable[[np.ndarray], float]]


@dataclass(frozen=True)
class Metric:
    """Метрика: функция, направление и причина, по которой она бывает не определена."""

    name: str
    fn: MetricFn
    higher_is_better: bool
    undefined: str
    sampler: Sampler | None = None

    def __call__(self, outputs: Outputs) -> float:
        return self.fn(outputs)

    def on_resamples(self, outputs: Outputs) -> Callable[[np.ndarray], float]:
        """Функция индексов бутстрап-выборки → значение метрики на ней.

        Быстрый путь `sampler` (например, для ROC AUC) даёт то же значение, что метрика
        на `outputs.resample(index)`, без пересортировки на каждой выборке (NFR-06).
        """
        if self.sampler is not None:
            return self.sampler(outputs)
        return lambda index: self.fn(outputs.resample(index))


def auc_sampler(outputs: Outputs) -> Callable[[np.ndarray], float]:
    """ROC AUC на бутстрап-выборках через веса повторов: O(n) на выборку (NFR-06).

    Выборка с повторами — это веса примеров; AUC = Σ по положительным весам (число
    отрицательных ниже + ½ равных) / (P·N) — то же, что Манн — Уитни со средними рангами.
    """
    scores = np.asarray(outputs.values, dtype=float)
    positive = outputs.labels == 1
    _, groups = np.unique(scores, return_inverse=True)
    size = int(groups.max()) + 1 if len(groups) else 0
    n = len(scores)

    def value(index: np.ndarray) -> float:
        counts = np.bincount(index, minlength=n).astype(float)
        pos = counts * positive
        neg = counts - pos
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        neg_group = np.bincount(groups, weights=neg, minlength=size)
        pos_group = np.bincount(groups, weights=pos, minlength=size)
        below = np.cumsum(neg_group) - neg_group
        return float(np.sum(pos_group * (below + 0.5 * neg_group)) / (n_pos * n_neg))

    return value


def _predicted_class(outputs: Outputs) -> np.ndarray:
    if outputs.values.ndim == 1:
        return outputs.values
    return np.argmax(outputs.values, axis=1)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _confusion(outputs: Outputs) -> tuple[int, int, int]:
    positive_label = outputs.labels == 1
    positive_prediction = outputs.values == 1
    tp = int(np.sum(positive_label & positive_prediction))
    fp = int(np.sum(~positive_label & positive_prediction))
    fn = int(np.sum(positive_label & ~positive_prediction))
    return tp, fp, fn


def _accuracy(outputs: Outputs) -> float:
    return float(np.mean(_predicted_class(outputs) == outputs.labels))


def _precision(outputs: Outputs) -> float:
    tp, fp, _ = _confusion(outputs)
    return _ratio(tp, tp + fp)


def _recall(outputs: Outputs) -> float:
    tp, _, fn = _confusion(outputs)
    return _ratio(tp, tp + fn)


def _f1(outputs: Outputs) -> float:
    tp, fp, fn = _confusion(outputs)
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _macro_f1(outputs: Outputs) -> float:
    predicted = _predicted_class(outputs)
    labels = outputs.labels
    size = outputs.n_classes
    tp = np.bincount(labels[predicted == labels], minlength=size)
    total = np.bincount(labels, minlength=size) + np.bincount(predicted, minlength=size)
    present = total > 0  # классы, встречающиеся в метках или предсказаниях
    return float(np.mean(2 * tp[present] / total[present]))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC по рангам (Манн — Уитни); равные значения получают средний ранг."""
    positive = labels == 1
    n_pos = int(np.sum(positive))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _roc_auc(outputs: Outputs) -> float:
    return roc_auc(outputs.labels, outputs.values)


def _true_class_probability(outputs: Outputs) -> np.ndarray:
    if outputs.values.ndim == 1:
        return np.where(outputs.labels == 1, outputs.values, 1 - outputs.values)
    return outputs.values[np.arange(len(outputs)), outputs.labels]


def _log_loss(outputs: Outputs) -> float:
    probability = np.clip(_true_class_probability(outputs), EPSILON, 1 - EPSILON)
    return float(-np.mean(np.log(probability)))


def _brier(outputs: Outputs) -> float:
    return float(np.mean((outputs.values - outputs.labels) ** 2))


def _mae(outputs: Outputs) -> float:
    return float(np.mean(np.abs(outputs.labels - outputs.values)))


def _rmse(outputs: Outputs) -> float:
    return float(np.sqrt(np.mean((outputs.labels - outputs.values) ** 2)))


METRICS: dict[str, Metric] = {
    metric.name: metric
    for metric in (
        Metric("accuracy", _accuracy, True, "нет примеров"),
        Metric("precision", _precision, True, "нет предсказаний класса 1"),
        Metric("recall", _recall, True, "нет примеров класса 1"),
        Metric("f1", _f1, True, "нет ни примеров, ни предсказаний класса 1"),
        Metric("macro_f1", _macro_f1, True, "нет примеров"),
        Metric("roc_auc", _roc_auc, True, "в примерах один класс", auc_sampler),
        Metric("log_loss", _log_loss, False, "нет примеров"),
        Metric("brier", _brier, False, "нет примеров"),
        Metric("mae", _mae, False, "нет примеров"),
        Metric("rmse", _rmse, False, "нет примеров"),
    )
}

_METRIC_SETS: dict[tuple[str, str], tuple[str, ...]] = {
    ("classification/binary", "labels"): ("f1", "accuracy", "precision", "recall"),
    ("classification/binary", "probabilities"): ("roc_auc", "log_loss", "brier"),
    ("classification/binary", "scores"): ("roc_auc",),
    ("classification/multiclass", "labels"): ("macro_f1", "accuracy"),
    ("classification/multiclass", "probabilities"): ("macro_f1", "accuracy", "log_loss"),
    ("classification/multiclass", "scores"): ("macro_f1", "accuracy"),
    ("regression", "values"): ("mae", "rmse"),
}


def metric_names(task: TaskSection) -> tuple[str, ...]:
    """Метрики задачи; первая — основная (по ней работают срезы)."""
    return _METRIC_SETS[(task.type, task.output)]


def primary_metric(task: TaskSection) -> Metric:
    return METRICS[metric_names(task)[0]]


def prepare(task: TaskSection, predictions: pd.DataFrame) -> Outputs:
    """Таблица предсказаний контракта → массивы, упорядоченные по `example_id` как строке."""
    frame = predictions.assign(example_id=predictions["example_id"].astype(str))
    frame = frame.sort_values("example_id", kind="mergesort").reset_index(drop=True)
    ids = frame["example_id"].to_numpy(dtype=str)
    if task.type == "classification/multiclass":
        labels = frame["label"].astype(str)
        if task.output == "labels":
            predicted = frame["prediction"].astype(str)
            classes = sorted(set(labels) | set(predicted))
            code = {name: i for i, name in enumerate(classes)}
            return Outputs(
                ids,
                labels.map(code).to_numpy(dtype=int),
                predicted.map(code).to_numpy(dtype=int),
                len(classes),
            )
        columns = [c for c in frame.columns if isinstance(c, str) and c.startswith(CLASS_PREFIX)]
        code = {column.removeprefix(CLASS_PREFIX): i for i, column in enumerate(columns)}
        return Outputs(
            ids,
            labels.map(code).to_numpy(dtype=int),
            frame[columns].to_numpy(dtype=float),
            len(columns),
        )
    labels = frame["label"].to_numpy(dtype=float)
    return Outputs(ids, labels, frame["prediction"].to_numpy(dtype=float))


def metric_value(task: TaskSection, predictions: pd.DataFrame, name: str) -> float:
    """Значение метрики `name` на всей таблице предсказаний."""
    return METRICS[name](prepare(task, predictions))


def example_errors(task: TaskSection, outputs: Outputs) -> np.ndarray | None:
    """Ошибка каждого примера для доказательств; для бинарных `scores` не определена."""
    if task.type == "regression" or (
        task.type == "classification/binary" and task.output == "probabilities"
    ):
        return np.abs(outputs.labels - outputs.values)
    if task.type == "classification/multiclass" and task.output == "probabilities":
        return 1 - _true_class_probability(outputs)
    if task.output == "labels" or task.type == "classification/multiclass":
        return (_predicted_class(outputs) != outputs.labels).astype(float)
    return None


def top_ids(ids: np.ndarray, scores: np.ndarray | None) -> tuple[str, ...]:
    """До 10 id с наибольшим значением `scores`; при равенстве — по id."""
    if scores is None:
        return ()
    order = np.lexsort((np.asarray(ids, dtype=str), -scores))
    return evidence_ids(ids[order])


@dataclass(frozen=True)
class Estimate:
    """Точечное значение (или дельта) и бутстрап-интервал."""

    value: float
    low: float
    high: float
    p_value: float
    n: int


def _bootstrap(statistic: Callable[[np.random.Generator], float]) -> np.ndarray | int:
    """Бутстрап-распределение статистики или число выборок, где она не определена."""
    rng = np.random.default_rng(SEED)
    samples = np.array([statistic(rng) for _ in range(RESAMPLES)], dtype=float)
    valid = samples[np.isfinite(samples)]
    undefined = RESAMPLES - len(valid)
    if undefined > MAX_UNDEFINED_SHARE * RESAMPLES:
        return undefined
    return valid


def _estimate(value: float, samples: np.ndarray, n: int) -> Estimate:
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    p_value = min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))
    return Estimate(value, float(low), float(high), float(p_value), n)


def unstable_reason(name: str, undefined: int) -> str:
    return (
        f"{name}: оценка неустойчива — метрика не определена на {undefined} из {RESAMPLES}"
        " бутстрап-выборок"
    )


def undefined_reason(metric: Metric) -> str:
    return f"{metric.name} не определена: {metric.undefined}"


def estimate_value(metric: Metric, outputs: Outputs) -> Estimate | str:
    """Значение метрики с интервалом; строка — причина, по которой оценки нет."""
    value = metric(outputs)
    if not np.isfinite(value):
        return undefined_reason(metric)
    n = len(outputs)
    on_sample = metric.on_resamples(outputs)
    samples = _bootstrap(lambda rng: on_sample(rng.integers(0, n, n)))
    if isinstance(samples, int):
        return unstable_reason(metric.name, samples)
    return _estimate(value, samples, n)


def paired_delta(metric: Metric, candidate: Outputs, baseline: Outputs) -> Estimate | str:
    """Дельта метрики кандидата и базового прогона на одних примерах (выровненных)."""
    delta = metric(candidate) - metric(baseline)
    if not np.isfinite(delta):
        return undefined_reason(metric)
    n = len(candidate)
    on_candidate, on_baseline = metric.on_resamples(candidate), metric.on_resamples(baseline)

    def statistic(rng: np.random.Generator) -> float:
        index = rng.integers(0, n, n)
        return on_candidate(index) - on_baseline(index)

    samples = _bootstrap(statistic)
    if isinstance(samples, int):
        return unstable_reason(metric.name, samples)
    return _estimate(delta, samples, n)


def independent_delta(metric: Metric, group: Outputs, rest: Outputs) -> Estimate | str:
    """Дельта метрики группы и остальных примеров; выборки — независимо в каждой части."""
    delta = metric(group) - metric(rest)
    if not np.isfinite(delta):
        return undefined_reason(metric)
    n_group, n_rest = len(group), len(rest)
    on_group, on_rest = metric.on_resamples(group), metric.on_resamples(rest)

    def statistic(rng: np.random.Generator) -> float:
        group_value = on_group(rng.integers(0, n_group, n_group))
        return group_value - on_rest(rng.integers(0, n_rest, n_rest))

    samples = _bootstrap(statistic)
    if isinstance(samples, int):
        return unstable_reason(metric.name, samples)
    return _estimate(delta, samples, n_group)


def holm(p_values: Sequence[float]) -> list[float]:
    """p-значения с поправкой Холма на множественные проверки."""
    order = np.argsort(p_values, kind="mergesort")
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def is_worse(metric: Metric, delta: float) -> bool:
    return delta < 0 if metric.higher_is_better else delta > 0


@dataclass(frozen=True)
class Pair:
    """Кандидат и базовый прогон, выровненные по общим примерам."""

    candidate: Outputs
    baseline: Outputs


def pair_runs(task: TaskSection, candidate: RunContract, baseline: RunContract) -> Pair | str:
    """Выравнивает прогоны по общим `example_id`; строка — причина, почему сравнить нельзя."""
    cand = prepare(task, candidate.predictions)
    base = prepare(task, baseline.predictions)
    common = np.intersect1d(cand.ids, base.ids)
    if len(common) == 0:
        return NO_COMMON_EXAMPLES
    cand = cand.take(np.searchsorted(cand.ids, common))
    base = base.take(np.searchsorted(base.ids, common))
    labels = [
        contract.predictions.set_index(contract.predictions["example_id"].astype(str))[
            "label"
        ].reindex(common)
        for contract in (candidate, baseline)
    ]
    if task.type == "classification/multiclass":
        differ = (labels[0].astype(str) != labels[1].astype(str)).to_numpy()
    else:
        first, second = (series.to_numpy(dtype=float) for series in labels)
        differ = ~np.isclose(first, second, rtol=1e-9, atol=1e-12)
    if differ.any():
        sample = ", ".join(evidence_ids(common[differ])[:5])
        return (
            f"метки различаются у {int(differ.sum())} общих примеров (например: {sample})"
            " — прогоны оценены на разных данных"
        )
    return Pair(cand, base)


def rounded(value: float) -> float:
    """Число для фактов находки: 6 знаков, чтобы отчёт был стабилен (NFR-05)."""
    return round(float(value), 6)
