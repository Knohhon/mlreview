"""Синтетические спецификации и контракты для тестов L1 (детерминированные, NFR-05)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from mlreview.core.run_contract import RunContract, build_run_contract
from mlreview.core.task_spec import TaskSpec, parse_task_spec

BINARY_SPEC_TEXT = (
    "task: {type: classification/binary, output: probabilities, segments: [size]}\n"
    "data: {split_strategy: stratified, version: v1}\n"
)


def binary_spec(text: str = BINARY_SPEC_TEXT) -> TaskSpec:
    return parse_task_spec(text)


def binary_tables(
    *, n: int = 200, seed: int = 1, duplicates: int = 0, shift: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train `n` + test `n`; первые `duplicates` строк train копируют признак строк test.

    `shift` — сдвиг признака f0 во всех строках (меняет данные, а не модель).
    """
    rng = np.random.default_rng(0)
    examples = pd.DataFrame(
        {
            "example_id": [f"tr-{i:03d}" for i in range(n)] + [f"te-{i:03d}" for i in range(n)],
            "split": ["train"] * n + ["test"] * n,
            "size": ["small", "large"] * n,
            "f0": rng.normal(0, 1, 2 * n) + shift,
        }
    )
    if duplicates:
        examples.loc[: duplicates - 1, "f0"] = examples.loc[n : n + duplicates - 1, "f0"].to_numpy()
    label = rng.integers(0, 2, n)
    logit = np.where(label == 1, 1.5, -1.5) + np.random.default_rng(seed).normal(0, 1, n)
    predictions = pd.DataFrame(
        {
            "example_id": examples["example_id"].iloc[n:].to_numpy(),
            "label": label,
            "prediction": 1 / (1 + np.exp(-logit)),
        }
    )
    return predictions, examples


def binary_contract(  # noqa: PLR0913 — параметры синтетического прогона
    run_id: str = "run-1",
    *,
    spec: TaskSpec | None = None,
    seed: int = 1,
    duplicates: int = 0,
    shift: float = 0.0,
    data_version: str | None = "v1",
    config: Mapping[str, Any] | None = None,
    curves: pd.DataFrame | None = None,
    with_examples: bool = True,
) -> RunContract:
    predictions, examples = binary_tables(seed=seed, duplicates=duplicates, shift=shift)
    return build_run_contract(
        spec or binary_spec(),
        run_id=run_id,
        predictions=predictions,
        examples=examples if with_examples else None,
        config=config,
        curves=curves,
        data_version=data_version,
    )
