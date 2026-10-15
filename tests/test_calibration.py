"""Калибровка и расхождение двух моделей (FR-D08).

Спецификация: docs/specs/calibration.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _factories import binary_spec, binary_tables
from mlreview.analyzers.calibration import calibration_findings, ece
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

N = 2000


def _calibrated(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, N)
    labels = (rng.uniform(0, 1, N) < p).astype(int)
    return labels, p


def _contract(run_id, labels, p, spec=None, examples=None):
    predictions = pd.DataFrame(
        {"example_id": [f"e{i:04d}" for i in range(len(labels))], "label": labels, "prediction": p}
    )
    return build_run_contract(
        spec or binary_spec(), run_id=run_id, predictions=predictions, examples=examples
    )


def _by_check(results):
    return {finding.check: finding for finding in results.findings}


def test_ece_value():
    """ECE: все вероятности 0,9 при доле положительных 0,5 — 0,4."""
    labels = np.array([0, 1] * 50)
    assert ece(labels, np.full(100, 0.9)) == pytest.approx(0.4)


def test_overconfident_probabilities():
    """B1: ECE ≥ 0,1 — medium, доказательства из худшей корзины."""
    labels = np.array([0, 1] * 100)
    contract = _contract("cand", labels, np.full(200, 0.9))
    finding = _by_check(calibration_findings(binary_spec(), contract))["calibration.ece"]
    assert finding.severity == "medium"
    assert finding.message == (
        "калибровка: ECE 0.400 (10 корзин) — вероятности систематически расходятся с частотами"
    )
    assert finding.facts == {
        "ece": 0.4,
        "bins": 10,
        "n": 200,
        "worst_bin": "0.9–1.0",
        "worst_gap": 0.4,
    }
    assert finding.evidence == tuple(f"e{i:04d}" for i in range(10))
    assert finding.requirement == "FR-D08"


def test_calibrated_probabilities_have_no_finding():
    """B2: откалиброванные вероятности — без находки."""
    labels, p = _calibrated()
    assert calibration_findings(binary_spec(), _contract("cand", labels, p)).findings == []


def test_calibration_got_worse():
    """B3: кандидат переуверен относительно базового."""
    labels, p = _calibrated()
    sharp = p**4 / (p**4 + (1 - p) ** 4)
    results = calibration_findings(
        binary_spec(), _contract("cand", labels, sharp), _contract("base", labels, p)
    )
    finding = _by_check(results)["calibration.ece_delta"]
    assert finding.severity == "medium"
    assert finding.message.startswith("калибровка ухудшилась: ECE ")
    assert finding.facts["ci_low"] > 0
    assert finding.facts["n"] == N
    assert "comparison.disagreement" not in _by_check(results)  # порядок вероятностей тот же


def test_disagreement_examples():
    """B4, B5: расхождение предсказанных классов и концентрация в сегменте."""
    predictions, examples = binary_tables()
    labels = predictions["label"].to_numpy()
    p = predictions["prediction"].to_numpy()
    flipped = p.copy()
    flip = np.arange(len(p)) % 2 == 0  # te-000, te-002, … — сегмент size=small
    flip[40:] = False
    flipped[flip] = 1 - p[flip]
    base = build_run_contract(
        binary_spec(), run_id="base", predictions=predictions, examples=examples
    )
    cand = build_run_contract(
        binary_spec(),
        run_id="cand",
        predictions=predictions.assign(prediction=flipped),
        examples=examples,
    )
    finding = _by_check(calibration_findings(binary_spec(), cand, base))["comparison.disagreement"]
    assert finding.severity == "info"
    differ = int(np.sum((flipped > 0.5) != (p > 0.5)))
    cand_right = int(np.sum(((flipped > 0.5) == labels) & ((flipped > 0.5) != (p > 0.5))))
    assert finding.facts["disagree"] == differ
    assert finding.facts["candidate_right"] == cand_right
    assert finding.facts["baseline_right"] == differ - cand_right
    assert finding.message.startswith(f"модели расходятся на {differ} из 200 примеров")
    assert finding.message.endswith("; 100% из них — size=small (против 50% среди всех)")
    assert finding.facts["segment"] == "size"
    assert finding.facts["segment_value"] == "small"
    assert len(finding.evidence) == 10


def test_same_predictions_have_no_disagreement():
    """B6: модели не расходятся."""
    labels, p = _calibrated()
    results = calibration_findings(
        binary_spec(), _contract("cand", labels, p), _contract("base", labels, p)
    )
    assert "comparison.disagreement" not in _by_check(results)


def test_skips_for_unsupported_outputs():
    """E1, E2: не probabilities; регрессия."""
    spec = parse_task_spec("task: {type: classification/binary, output: labels}\n")
    labels = np.array([0, 1] * 20)
    contract = _contract("cand", labels, labels, spec=spec)
    reasons = {s.check: s.reason for s in calibration_findings(spec, contract).skipped}
    assert reasons["calibration.ece"] == "калибровка — только для output: probabilities"

    regression = parse_task_spec("task: {type: regression, output: values}\n")
    values = np.linspace(0, 1, 40)
    cand = _contract("cand", values, values, spec=regression)
    base = _contract("base", values, values + 0.1, spec=regression)
    reasons = {s.check: s.reason for s in calibration_findings(regression, cand, base).skipped}
    assert reasons["comparison.disagreement"] == (
        "расхождение классов не определено для regression · values"
    )


def test_no_common_examples_is_skipped():
    """E3: нет общих примеров."""
    labels, p = _calibrated()
    cand = _contract("cand", labels, p)
    other = pd.DataFrame({"example_id": ["x1", "x2"], "label": [0, 1], "prediction": [0.2, 0.7]})
    base = build_run_contract(binary_spec(), run_id="base", predictions=other)
    reasons = {s.check: s.reason for s in calibration_findings(binary_spec(), cand, base).skipped}
    assert "нет общих примеров" in reasons["calibration.ece_delta"]
    assert "нет общих примеров" in reasons["comparison.disagreement"]
