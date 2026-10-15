"""Срезы по сегментам (FR-D02, B12).

Спецификация: docs/specs/slices.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlreview.analyzers.slices import segment_findings
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

BINARY_SPEC = "task: {type: classification/binary, output: probabilities, segments: [size]}"
REGRESSION_SPEC = "task: {type: regression, output: values, segments: [size]}"


def _labels(sizes: dict[str, int], seed: int = 0) -> pd.DataFrame:
    """Примеры с сегментом `size` и бинарной меткой; id по порядку."""
    rng = np.random.default_rng(seed)
    values = [value for value, count in sizes.items() for _ in range(count)]
    n = len(values)
    return pd.DataFrame(
        {
            "example_id": [f"e{i:04d}" for i in range(n)],
            "size": values,
            "label": rng.integers(0, 2, n),
        }
    )


def _proba(frame: pd.DataFrame, noise: dict[str, float], seed: int) -> pd.Series:
    """Вероятности: чем больше шум значения сегмента, тем хуже разделение."""
    rng = np.random.default_rng(seed)
    signal = np.where(frame["label"] == 1, 2.0, -2.0)
    scale = frame["size"].map(noise).to_numpy()
    logit = signal + rng.normal(0, 1, len(frame)) * scale
    return pd.Series(1 / (1 + np.exp(-logit)), index=frame.index)


def _contract(spec, frame, prediction, run_id="run", examples=None):
    predictions = pd.DataFrame(
        {"example_id": frame["example_id"], "label": frame["label"], "prediction": prediction}
    )
    if examples is None:
        examples = frame[["example_id", "size"]].assign(split="test")
    return build_run_contract(spec, run_id=run_id, predictions=predictions, examples=examples)


def _checks(results, check):
    return [f for f in results.findings if f.check == check]


# --- B1, B2, B5: сегмент хуже остальных ---


def test_segment_worse_than_rest():
    """B1: сегмент заметно хуже — medium, факты, доказательства с наибольшей ошибкой."""
    spec = parse_task_spec(BINARY_SPEC)
    frame = _labels({"small": 200, "large": 200})
    prediction = _proba(frame, {"small": 0.5, "large": 4.0}, seed=1)
    results = segment_findings(spec, _contract(spec, frame, prediction))

    (finding,) = _checks(results, "slices.segment_worse")
    facts = finding.facts
    assert finding.severity == "medium"
    assert (facts["segment"], facts["value"], facts["metric"]) == ("size", "large", "roc_auc")
    assert facts["segment_metric"] < facts["rest_metric"]
    assert facts["ci_high"] < 0
    assert facts["n"] == 200
    assert finding.message.startswith(
        f"сегмент size=large хуже остальных: roc_auc {facts['segment_metric']:.3f} "
        f"против {facts['rest_metric']:.3f}"
    )
    assert finding.message.endswith("n = 200)")

    error = (frame["label"] - prediction).abs()
    in_segment = frame["size"] == "large"
    assert len(finding.evidence) == 10
    evidence_mask = frame["example_id"].isin(finding.evidence)
    assert in_segment[evidence_mask].all()
    assert error[evidence_mask].min() >= error[in_segment & ~evidence_mask].max()


def test_equal_segments_have_no_findings():
    """B2: сегменты одного качества — находок нет."""
    spec = parse_task_spec(BINARY_SPEC)
    frame = _labels({"small": 200, "large": 200})
    prediction = _proba(frame, {"small": 1.0, "large": 1.0}, seed=1)
    results = segment_findings(spec, _contract(spec, frame, prediction))

    assert results.findings == []
    assert results.skipped == []


def test_lower_is_better_metric_segment_worse():
    """B5: для регрессии (mae) хуже — рост ошибки."""
    spec = parse_task_spec(REGRESSION_SPEC)
    rng = np.random.default_rng(0)
    frame = _labels({"small": 150, "large": 150}).assign(label=rng.normal(10, 2, 300))
    noise = np.where(frame["size"] == "large", 3.0, 0.5)
    prediction = frame["label"] + rng.normal(0, 1, 300) * noise
    results = segment_findings(spec, _contract(spec, frame, prediction))

    (finding,) = _checks(results, "slices.segment_worse")
    assert finding.facts["value"] == "large"
    assert finding.facts["metric"] == "mae"
    assert finding.facts["segment_metric"] > finding.facts["rest_metric"]


# --- B3, B4: регрессия сегмента между прогонами ---


def _b12_runs():
    frame = _labels({"a": 200, "b": 200, "small": 100})
    baseline = _proba(frame, {"a": 3.0, "b": 3.0, "small": 3.0}, seed=1)
    candidate = _proba(frame, {"a": 0.3, "b": 0.3, "small": 6.0}, seed=2)
    return frame, baseline, candidate


def test_segment_regression_while_overall_improves():
    """B3: кандидат лучше в целом, но хуже на size=small — high с фактами и доказательствами."""
    spec = parse_task_spec(BINARY_SPEC)
    frame, baseline, candidate = _b12_runs()
    results = segment_findings(
        spec,
        _contract(spec, frame, candidate),
        _contract(spec, frame, baseline, run_id="base"),
    )

    (finding,) = _checks(results, "slices.segment_regression")
    facts = finding.facts
    assert finding.severity == "high"
    assert (facts["segment"], facts["value"], facts["metric"]) == ("size", "small", "roc_auc")
    assert facts["delta"] < 0 < facts["overall_delta"]
    assert facts["ci_high"] < 0
    assert facts["n"] == 100
    assert finding.message.startswith(
        f"регрессия сегмента size=small: roc_auc {facts['baseline']:.3f} → {facts['candidate']:.3f}"
    )
    assert finding.message.endswith(f"в целом по прогону Δ {facts['overall_delta']:+.3f}")

    growth = (frame["label"] - candidate).abs() - (frame["label"] - baseline).abs()
    in_segment = frame["size"] == "small"
    evidence_mask = frame["example_id"].isin(finding.evidence)
    assert len(finding.evidence) == 10
    assert in_segment[evidence_mask].all()
    assert growth[evidence_mask].min() >= growth[in_segment & ~evidence_mask].max()


def test_identical_runs_have_no_segment_regression():
    """B4: одинаковые предсказания — регрессий сегментов нет."""
    spec = parse_task_spec(BINARY_SPEC)
    frame, baseline, _ = _b12_runs()
    results = segment_findings(
        spec, _contract(spec, frame, baseline), _contract(spec, frame, baseline, run_id="base")
    )
    assert _checks(results, "slices.segment_regression") == []


def test_repeated_analysis_is_identical():
    """B6: одинаковые входы — идентичные находки (NFR-05)."""
    spec = parse_task_spec(BINARY_SPEC)
    frame, baseline, candidate = _b12_runs()
    runs = (_contract(spec, frame, candidate), _contract(spec, frame, baseline, run_id="base"))
    assert segment_findings(spec, *runs) == segment_findings(spec, *runs)


# --- E1–E6: деградация ---


def test_no_segments_in_spec():
    """E1: нет task.segments — обе проверки пропущены."""
    spec = parse_task_spec("task: {type: classification/binary, output: probabilities}")
    frame, baseline, candidate = _b12_runs()
    results = segment_findings(
        spec, _contract(spec, frame, candidate), _contract(spec, frame, baseline, "base")
    )
    assert results.findings == []
    assert [(s.check, s.reason) for s in results.skipped] == [
        ("slices.segment_worse", "в спецификации нет task.segments"),
        ("slices.segment_regression", "в спецификации нет task.segments"),
    ]


def test_no_examples():
    """E2: нет examples — проверки пропущены."""
    spec = parse_task_spec(BINARY_SPEC)
    frame = _labels({"small": 50})
    predictions = frame[["example_id", "label"]].assign(prediction=0.5)
    contract = build_run_contract(spec, run_id="run", predictions=predictions)
    results = segment_findings(spec, contract)

    assert [(s.check, s.reason) for s in results.skipped] == [
        ("slices.segment_worse", "нет examples — значения сегментов неизвестны"),
    ]


def test_missing_segment_column_others_checked():
    """E3: колонки сегмента нет — пропуск, другие сегменты проверяются."""
    spec = parse_task_spec(
        "task: {type: classification/binary, output: probabilities, segments: [region, size]}"
    )
    frame = _labels({"small": 200, "large": 200})
    prediction = _proba(frame, {"small": 0.5, "large": 4.0}, seed=1)
    results = segment_findings(spec, _contract(spec, frame, prediction))

    assert [f.facts["value"] for f in _checks(results, "slices.segment_worse")] == ["large"]
    assert [(s.check, s.reason) for s in results.skipped] == [
        ("slices.segment_worse", "нет examples.region"),
    ]


def test_small_segment_values_listed():
    """E4: значения с менее чем 30 примерами не проверяются и перечисляются."""
    spec = parse_task_spec(BINARY_SPEC)
    frame = _labels({"small": 200, "large": 200, "tiny": 10, "rare": 3})
    prediction = _proba(frame, {"small": 1.0, "large": 1.0, "tiny": 1.0, "rare": 1.0}, seed=1)
    results = segment_findings(spec, _contract(spec, frame, prediction))

    assert [(s.check, s.reason) for s in results.skipped] == [
        (
            "slices.segment_worse",
            "size: значения с менее чем 30 примерами не проверены: rare (3), tiny (10)",
        ),
    ]


def test_metric_undefined_in_segment():
    """E5: в сегменте один класс — roc_auc не определена, пропуск с сегментом и значением."""
    spec = parse_task_spec(BINARY_SPEC)
    frame = _labels({"small": 200, "large": 200, "mono": 40})
    frame.loc[frame["size"] == "mono", "label"] = 1
    prediction = _proba(frame, {"small": 1.0, "large": 1.0, "mono": 1.0}, seed=1)
    results = segment_findings(spec, _contract(spec, frame, prediction))

    (skipped,) = results.skipped
    assert skipped.check == "slices.segment_worse"
    assert skipped.reason.startswith("size=mono: roc_auc не определена")


def test_regression_check_needs_baseline_and_common_examples():
    """E6: без базового прогона регрессия не запускается; без общих примеров — пропуск."""
    spec = parse_task_spec(BINARY_SPEC)
    frame, baseline, candidate = _b12_runs()
    alone = segment_findings(spec, _contract(spec, frame, candidate))
    assert all(s.check != "slices.segment_regression" for s in alone.skipped)
    assert _checks(alone, "slices.segment_regression") == []

    other = frame.assign(example_id="b-" + frame["example_id"])
    results = segment_findings(
        spec, _contract(spec, frame, candidate), _contract(spec, other, baseline, "base")
    )
    assert (
        "slices.segment_regression",
        "у прогонов нет общих примеров — парное сравнение невозможно",
    ) in [(s.check, s.reason) for s in results.skipped]
