"""Общие помощники проверок плейбуков: train, вероятности, доля вырожденных (FR-D11)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlreview.analyzers.evaluation import rounded
from mlreview.core.run_contract import CLASS_PREFIX, RunContract

TRAIN = "train"


def train_rows(examples: pd.DataFrame) -> pd.DataFrame:
    """Строки examples сплита train."""
    return examples[examples["split"].astype(str) == TRAIN]


def probabilities(contract: RunContract) -> np.ndarray:
    """Вероятности: `prediction` бинарной задачи или максимум строки multiclass."""
    predictions = contract.predictions
    if "prediction" in predictions.columns:
        return predictions["prediction"].astype(float).to_numpy()
    columns = [column for column in predictions.columns if column.startswith(CLASS_PREFIX)]
    return predictions[columns].astype(float).to_numpy().max(axis=1)


def extreme_share(values: np.ndarray) -> float:
    """Доля вероятностей ровно 0 или 1 (округлена для фактов)."""
    return rounded(np.mean((values == 0) | (values == 1)))
