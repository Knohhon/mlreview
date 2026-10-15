"""Находки и ревью прогона (FR-R01, FR-D01–D03).

Спецификация: docs/specs/review.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlreview.analyzers.review import review
from mlreview.core.findings import Finding, sort_findings
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

SPEC = parse_task_spec(
    "task: {type: classification/binary, output: probabilities, segments: [size]}\n"
    "data: {split_strategy: stratified, version: v1}\n"
)


def _run(run_id="cand", noise_large=4.0, seed=1, with_examples=True, data_version="v2"):
    """Train 300 + test 300; 6 строк test скопированы в train; сегмент large хуже."""
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 300
    examples = pd.DataFrame(
        {
            "example_id": [f"tr-{i:03d}" for i in range(n_train)]
            + [f"te-{i:03d}" for i in range(n_test)],
            "split": ["train"] * n_train + ["test"] * n_test,
            "size": ["small", "large"] * ((n_train + n_test) // 2),
            "f0": rng.normal(0, 1, n_train + n_test),
            "f1": rng.normal(0, 1, n_train + n_test),
        }
    )
    examples.loc[:5, ["f0", "f1"]] = examples.loc[n_train : n_train + 5, ["f0", "f1"]].to_numpy()
    test = examples.iloc[n_train:].reset_index(drop=True)
    label = rng.integers(0, 2, n_test)
    noise = np.where(test["size"] == "large", noise_large, 0.5)
    logit = (
        np.where(label == 1, 2.0, -2.0) + np.random.default_rng(seed).normal(0, 1, n_test) * noise
    )
    predictions = pd.DataFrame(
        {"example_id": test["example_id"], "label": label, "prediction": 1 / (1 + np.exp(-logit))}
    )
    return build_run_contract(
        SPEC,
        run_id=run_id,
        predictions=predictions,
        examples=examples if with_examples else None,
        data_version=data_version,
    )


def _finding(check, severity, message="m"):
    return Finding(check=check, severity=severity, message=message, requirement="FR-D01")


def test_findings_sorted_by_severity_check_message():
    """B1: сортировка по severity, затем check, затем сообщению."""
    findings = [
        _finding("b", "info"),
        _finding("b", "critical", "z"),
        _finding("a", "medium"),
        _finding("b", "critical", "a"),
        _finding("a", "high"),
        _finding("c", "low"),
    ]
    ordered = [(f.check, f.severity, f.message) for f in sort_findings(findings)]
    assert ordered == [
        ("b", "critical", "a"),
        ("b", "critical", "z"),
        ("a", "high", "m"),
        ("a", "medium", "m"),
        ("c", "low", "m"),
        ("b", "info", "m"),
    ]


def test_review_without_baseline_runs_all_groups():
    """B2: находки групп data, segments, metrics; базового прогона нет."""
    report = review(SPEC, _run())

    checks = [f.check for f in report.findings]
    assert checks[0] == "data.split_duplicates"
    assert report.findings[0].severity == "critical"
    assert "slices.segment_worse" in checks
    assert "metrics.value" in checks
    assert report.findings == sort_findings(report.findings)
    assert report.run_id == "cand"
    assert report.baseline_id is None
    assert report.compared_examples is None
    assert report.data_version == "v2"


def test_review_single_group():
    """B3: groups=("data",) — только находки и пропуски группы data."""
    report = review(SPEC, _run(), groups=("data",))
    assert {f.check for f in report.findings} == {"data.split_duplicates"}
    assert all(s.check.startswith("data.") for s in report.skipped)


def test_review_with_baseline():
    """B4: с базовым прогоном — поля базового и дельты метрик."""
    baseline = _run("base", noise_large=1.0, seed=2, data_version="v1")
    report = review(SPEC, _run(), baseline)

    assert report.baseline_id == "base"
    assert report.baseline_data_version == "v1"
    assert report.compared_examples == 300
    checks = {f.check for f in report.findings}
    assert "comparison.metric_delta" in checks
    assert "metrics.value" not in checks


def test_review_is_reproducible_and_json_serializable():
    """B5: два запуска дают равные to_dict(); словарь сериализуется в JSON."""
    baseline = _run("base", noise_large=1.0, seed=2)
    first = review(SPEC, _run(), baseline).to_dict()
    second = review(SPEC, _run(), baseline).to_dict()
    assert first == second
    data = json.loads(json.dumps(first, ensure_ascii=False))
    assert data["run_id"] == "cand"
    assert data["findings"][0]["check"] == "data.split_duplicates"
    assert set(data["findings"][0]) == {
        "check",
        "severity",
        "message",
        "requirement",
        "evidence",
        "facts",
        "suggestion",
    }
    # temporal_leak пропущена: split_strategy — stratified
    assert data["skipped"][0] == {
        "check": "data.temporal_leak",
        "reason": "split_strategy в спецификации — stratified, а не temporal",
    }
    assert data["llm"] is None


def test_review_without_examples_degrades_explicitly():
    """E1: без examples — проверки data и segments в skipped, метрики на месте."""
    report = review(SPEC, _run(with_examples=False))

    skipped = [s.check for s in report.skipped]
    assert skipped == [
        "data.split_duplicates",
        "data.temporal_leak",
        "data.strong_feature",
        "slices.segment_worse",
    ]
    assert {f.check for f in report.findings} == {"metrics.value"}
