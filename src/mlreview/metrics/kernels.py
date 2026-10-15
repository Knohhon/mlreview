"""Реализации метрик на numpy (FR-M02, FR-M04).

Функции самодостаточны (только numpy), поэтому `mlreview metrics export` копирует их
исходники в `metrics.py` без изменений. Метки бинарной задачи — 0/1, multiclass — коды
классов 0…C−1; `classes` — предсказанные классы, `scores` — оценки или вероятности.
"""

import numpy as np

EPSILON = 1e-15
ECE_BINS = 10


def _average_ranks(values):
    """Ранги 1…n; равные значения получают средний ранг."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2)[inverse]


def roc_auc(labels, scores):
    positive = labels == 1
    n_pos = int(np.sum(positive))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=float))
    return float((np.sum(ranks[positive]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(labels, scores):
    """Average precision: Σ (Rₙ − Rₙ₋₁)·Pₙ по порогам (равные оценки — один порог)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = (labels[order] == 1).astype(float)
    sorted_scores = scores[order]
    last = np.concatenate((np.flatnonzero(np.diff(sorted_scores)), [len(scores) - 1]))
    tp = np.cumsum(hits)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def log_loss(labels, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        true = np.where(labels == 1, probabilities, 1 - probabilities)
    else:
        true = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true, EPSILON, 1 - EPSILON))))


def brier(labels, probabilities):
    return float(np.mean((np.asarray(probabilities, dtype=float) - labels) ** 2))


def ece(labels, probabilities):
    """Ожидаемая ошибка калибровки, 10 корзин равной ширины."""
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0:
        return float("nan")
    if probabilities.ndim == 1:
        confidence, correct = probabilities, (labels == 1).astype(float)
    else:
        confidence = probabilities.max(axis=1)
        correct = (np.argmax(probabilities, axis=1) == labels).astype(float)
    index = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    conf = np.bincount(index, weights=confidence, minlength=ECE_BINS)
    hits = np.bincount(index, weights=correct, minlength=ECE_BINS)
    return float(np.sum(np.abs(hits - conf)) / len(labels))


def accuracy(labels, classes):
    return float(np.mean(classes == labels))


def balanced_accuracy(labels, classes):
    recalls = [np.mean(classes[labels == value] == value) for value in np.unique(labels)]
    return float(np.mean(recalls))


def _confusion(labels, classes):
    positive, predicted = labels == 1, classes == 1
    tp = float(np.sum(positive & predicted))
    return tp, float(np.sum(~positive & predicted)), float(np.sum(positive & ~predicted))


def precision(labels, classes):
    tp, fp, _ = _confusion(labels, classes)
    return tp / (tp + fp) if tp + fp else float("nan")


def recall(labels, classes):
    tp, _, fn = _confusion(labels, classes)
    return tp / (tp + fn) if tp + fn else float("nan")


def f1(labels, classes):
    tp, fp, fn = _confusion(labels, classes)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def fbeta(labels, classes, beta):
    tp, fp, fn = _confusion(labels, classes)
    weight = beta * beta
    denominator = (1 + weight) * tp + weight * fn + fp
    return (1 + weight) * tp / denominator if denominator else float("nan")


def macro_f1(labels, classes):
    """Среднее F1 по классам, встречающимся в метках или предсказаниях."""
    present = np.union1d(labels, classes)
    scores = []
    for value in present:
        tp = np.sum((labels == value) & (classes == value))
        total = np.sum(labels == value) + np.sum(classes == value)
        scores.append(2 * tp / total)
    return float(np.mean(scores))


def precision_at_k(labels, scores, k):
    k = min(int(k), len(labels))
    top = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:k]
    return float(np.mean(labels[top] == 1))


def recall_at_k(labels, scores, k):
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return float("nan")
    k = min(int(k), len(labels))
    top = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:k]
    return float(np.sum(labels[top] == 1) / n_pos)


def top_k_accuracy(labels, scores, k):
    scores = np.asarray(scores, dtype=float)
    k = min(int(k), scores.shape[1])
    top = np.argsort(-scores, axis=1, kind="mergesort")[:, :k]
    return float(np.mean(np.any(top == np.asarray(labels)[:, None], axis=1)))


def mae(labels, values):
    return float(np.mean(np.abs(labels - values)))


def rmse(labels, values):
    return float(np.sqrt(np.mean((labels - values) ** 2)))


def median_ae(labels, values):
    return float(np.median(np.abs(labels - values)))


def mape(labels, values):
    if np.any(labels == 0):
        return float("nan")
    return float(np.mean(np.abs(labels - values) / np.abs(labels)))


def pinball(labels, values, tau):
    diff = labels - values
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def r2(labels, values):
    total = np.sum((labels - np.mean(labels)) ** 2)
    if total == 0:
        return float("nan")
    return float(1 - np.sum((labels - values) ** 2) / total)
