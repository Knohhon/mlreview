"""Производительность ревью без LLM на синтетическом прогоне (NFR-06: 100 тыс. примеров ≤ 2 мин).

Спецификация — раздел «Производительность» в `docs/specs/benchmark.md`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlreview.analyzers.review import review
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

DEFAULT_N = 100_000
LIMIT_SECONDS = 120
FEATURES = 20
SPEC = (
    "task: {type: classification/binary, output: probabilities, segments: [region, device]}\n"
    "data: {split_strategy: temporal, time_column: ts, version: perf-v1}\n"
)


@dataclass(frozen=True)
class PerfResult:
    n: int
    contract_seconds: float
    review_seconds: float
    findings: int

    @property
    def total_seconds(self) -> float:
        return self.contract_seconds + self.review_seconds

    @property
    def passed(self) -> bool:
        return self.total_seconds <= LIMIT_SECONDS

    def describe(self) -> str:
        verdict = "уложилось" if self.passed else "НЕ уложилось"
        return (
            f"NFR-06: {self.n} примеров test (+{self.n} train, {FEATURES} признаков, 2 сегмента):"
            f" контракты {self.contract_seconds:.1f} с, ревью {self.review_seconds:.1f} с,"
            f" всего {self.total_seconds:.1f} с — {verdict} в {LIMIT_SECONDS} с;"
            f" находок {self.findings}"
        )


def _predictions(ids: np.ndarray, labels: np.ndarray, seed: int) -> pd.DataFrame:
    noise = np.random.default_rng(seed).normal(0, 1, len(labels))
    logit = np.where(labels == 1, 1.0, -1.0) + noise
    return pd.DataFrame(
        {"example_id": ids, "label": labels, "prediction": 1 / (1 + np.exp(-logit))}
    )


def measure(n: int = DEFAULT_N) -> PerfResult:
    """Сборка двух контрактов и полное ревью кандидата против базового прогона."""
    rng = np.random.default_rng(0)
    spec = parse_task_spec(SPEC)
    ids = np.array([f"e{i:07d}" for i in range(2 * n)])
    examples = pd.DataFrame(
        {
            "example_id": ids,
            "split": ["train"] * n + ["test"] * n,
            "region": rng.choice(["a", "b", "c", "d"], 2 * n),
            "device": rng.choice(["mobile", "desktop"], 2 * n),
            "ts": np.arange(2 * n),
            **{f"f{j}": rng.normal(size=2 * n) for j in range(FEATURES)},
        }
    )
    labels = rng.integers(0, 2, n)
    start = time.perf_counter()
    contracts = [
        build_run_contract(
            spec,
            run_id=role,
            predictions=_predictions(ids[n:], labels, seed),
            examples=examples,
            config={"lr": seed},
            data_version="perf-v1",
        )
        for role, seed in (("candidate", 1), ("baseline", 2))
    ]
    built = time.perf_counter()
    report = review(spec, contracts[0], contracts[1])
    done = time.perf_counter()
    return PerfResult(n, built - start, done - built, len(report.findings))
