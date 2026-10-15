"""Проверки данных: дубли между сплитами, временная утечка, сильные признаки (FR-D03).

Спецификация: docs/specs/data-checks.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlreview.analyzers.data_checks import data_findings
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

TEMPORAL_SPEC = """
task: {type: classification/binary, output: probabilities, segments: [size]}
data: {split_strategy: temporal, time_column: ts}
"""
START = pd.Timestamp("2024-01-01")


def _data(n_train=400, n_test=171, seed=0):
    """Train и test с пятью признаками, сегментом и временем; test позже train."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    examples = pd.DataFrame(
        {
            "example_id": [f"tr-{i:03d}" for i in range(n_train)]
            + [f"te-{i:03d}" for i in range(n_test)],
            "split": ["train"] * n_train + ["test"] * n_test,
            "size": rng.choice(["small", "large"], n),
            "ts": START + pd.to_timedelta(np.arange(n), unit="h"),
        }
    )
    for k in range(5):
        examples[f"f{k}"] = rng.normal(0, 1, n).round(6)
    test = examples[examples["split"] == "test"]
    predictions = pd.DataFrame(
        {"example_id": test["example_id"], "label": rng.integers(0, 2, n_test), "prediction": 0.5}
    )
    return examples, predictions


def _run(examples, predictions, spec_text=TEMPORAL_SPEC):
    spec = parse_task_spec(spec_text)
    contract = build_run_contract(spec, run_id="run", predictions=predictions, examples=examples)
    return data_findings(spec, contract)


def _copy_into_train(examples, test_ids, prefix="copy"):
    rows = examples[examples["example_id"].isin(test_ids)].copy()
    rows["example_id"] = [f"{prefix}-{i}" for i in range(len(rows))]
    rows["split"] = "train"
    rows["ts"] = START - pd.Timedelta(hours=1)  # раньше всего — временную проверку не задевает
    return pd.concat([examples, rows], ignore_index=True)


def _checks(results, check):
    return [f for f in results.findings if f.check == check]


# --- B1–B3: дубли между сплитами ---


def test_duplicates_between_train_and_test_critical():
    """B1: 9 из 171 строк test совпадают с train по признакам — critical с фактами."""
    examples, predictions = _data()
    copied = [f"te-{i:03d}" for i in range(0, 90, 10)]
    results = _run(_copy_into_train(examples, copied), predictions)

    (finding,) = _checks(results, "data.split_duplicates")
    assert finding.severity == "critical"
    assert finding.message == (
        "9 из 171 примеров test (5.3%) полностью совпадают с примерами train по 5 признакам"
        " — метрика на test, вероятно, завышена"
    )
    assert finding.evidence == tuple(copied)
    assert finding.facts == {
        "earlier_split": "train",
        "later_split": "test",
        "duplicates": 9,
        "total": 171,
        "share": pytest.approx(9 / 171, abs=1e-6),
        "features": 5,
    }
    assert finding.requirement == "FR-D03, B01"
    assert finding.suggestion


def test_few_duplicates_medium_and_all_split_pairs():
    """B2: < 1% — medium; проверяются все пары сплитов; NaN в одной позиции — совпадение."""
    examples, predictions = _data()
    examples.loc[examples["example_id"] == "te-000", "f0"] = np.nan
    examples = _copy_into_train(examples, ["te-000"])
    # validation: копия строки te-001
    row = examples[examples["example_id"] == "te-001"].assign(example_id="va-0", split="val")
    examples = pd.concat([examples, row], ignore_index=True)
    results = _run(examples, predictions)

    found = {
        (f.facts["earlier_split"], f.facts["later_split"]): f
        for f in _checks(results, "data.split_duplicates")
    }
    assert set(found) == {("train", "test"), ("val", "test")}
    assert found[("train", "test")].severity == "medium"
    assert found[("train", "test")].evidence == ("te-000",)
    assert found[("val", "test")].evidence == ("te-001",)


def test_segment_and_time_are_not_features():
    """B3: строки с одинаковыми признаками — дубли, даже если сегмент и время различаются."""
    examples, predictions = _data()
    examples = _copy_into_train(examples, ["te-005"])
    copy = examples["example_id"] == "copy-0"
    examples.loc[copy, "size"] = "other"
    results = _run(examples, predictions)

    (finding,) = _checks(results, "data.split_duplicates")
    assert finding.evidence == ("te-005",)
    assert finding.facts["features"] == 5


# --- B4, B5: временная утечка ---


def _shuffle_times(examples, leaked_ids):
    """Переносит время примеров test раньше последнего примера train."""
    examples = examples.copy()
    mask = examples["example_id"].isin(leaked_ids)
    examples.loc[mask, "ts"] = START + pd.to_timedelta(np.arange(mask.sum())[::-1] * 3, unit="h")
    return examples


def test_temporal_leak_critical():
    """B4: 40 из 171 примеров test раньше последнего train — critical."""
    examples, predictions = _data()
    leaked = [f"te-{i:03d}" for i in range(40)]
    results = _run(_shuffle_times(examples, leaked), predictions)

    (finding,) = _checks(results, "data.temporal_leak")
    boundary = str(START + pd.Timedelta(hours=399))
    assert finding.severity == "critical"
    assert finding.message == (
        f"40 из 171 примеров test (23.4%) раньше последнего примера train ({boundary})"
        " при split_strategy: temporal — модель обучена на данных из будущего относительно test"
    )
    # по возрастанию времени: te-039 получил самое раннее время
    assert finding.evidence[:3] == ("te-039", "te-038", "te-037")
    assert len(finding.evidence) == 10
    assert finding.facts == {
        "earlier_split": "train",
        "later_split": "test",
        "leaked": 40,
        "total": 171,
        "share": pytest.approx(40 / 171, abs=1e-6),
        "boundary": boundary,
    }
    assert finding.requirement == "FR-D03, B02"
    assert finding.suggestion


def test_small_temporal_leak_high_and_boundary_tie_is_not_leak():
    """B5: < 1% — high; время, равное границе, — не утечка."""
    examples, predictions = _data()
    examples.loc[examples["example_id"] == "te-000", "ts"] = START  # раньше границы
    examples.loc[examples["example_id"] == "te-001", "ts"] = START + pd.Timedelta(hours=399)
    results = _run(examples, predictions)

    (finding,) = _checks(results, "data.temporal_leak")
    assert finding.severity == "high"
    assert finding.evidence == ("te-000",)


def test_correct_temporal_split_and_string_dates():
    """B5: правильный временной сплит — находок нет; время строкой читается как дата."""
    examples, predictions = _data()
    as_text = examples.assign(ts=examples["ts"].dt.strftime("%Y-%m-%d %H:%M"))
    assert _checks(_run(examples, predictions), "data.temporal_leak") == []
    results = _run(as_text, predictions)
    assert results.findings == []
    assert all(s.check != "data.temporal_leak" for s in results.skipped)

    leaked = _shuffle_times(examples, ["te-000", "te-001"])
    leaked_text = leaked.assign(ts=leaked["ts"].dt.strftime("%Y-%m-%d %H:%M"))
    assert len(_checks(_run(leaked_text, predictions), "data.temporal_leak")) == 1


def test_numeric_time_column():
    """B4: колонка времени — число."""
    examples, predictions = _data()
    numeric = examples.assign(ts=np.arange(len(examples), dtype=float))
    numeric.loc[numeric["example_id"] == "te-003", "ts"] = 5.0
    (finding,) = _checks(_run(numeric, predictions), "data.temporal_leak")
    assert finding.evidence == ("te-003",)
    assert finding.facts["boundary"] == "399"


# --- B6, B7: подозрительно сильные признаки ---


def test_strong_feature_binary():
    """B6: признак почти идеально разделяет классы — high с AUC."""
    examples, predictions = _data()
    label = predictions.set_index("example_id")["label"]
    test = examples["split"] == "test"
    examples.loc[test, "f3"] = examples.loc[test, "example_id"].map(label) * 10 + examples.loc[
        test, "f3"
    ].clip(-1, 1)
    results = _run(examples, predictions)

    (finding,) = _checks(results, "data.strong_feature")
    assert finding.severity == "high"
    assert finding.facts == {"feature": "f3", "statistic": "roc_auc", "value": 1.0, "n": 171}
    assert finding.message == (
        "признак f3 почти идеально предсказывает метку на оцениваемых примерах:"
        " AUC 1.000 (n = 171) — проверьте утечку таргета"
    )


def test_strong_feature_inverse_direction():
    """B6: обратная связь (1 − AUC ≥ 0,99) тоже сильный признак."""
    examples, predictions = _data()
    label = predictions.set_index("example_id")["label"]
    test = examples["split"] == "test"
    examples.loc[test, "f1"] = -examples.loc[test, "example_id"].map(label).astype(float)
    (finding,) = _checks(_run(examples, predictions), "data.strong_feature")
    assert finding.facts["feature"] == "f1"
    assert finding.facts["value"] == pytest.approx(1.0)


def test_strong_feature_regression():
    """B6: регрессия — |Спирмен| ≥ 0,99."""
    examples, predictions = _data()
    rng = np.random.default_rng(5)
    predictions = predictions.assign(label=rng.normal(0, 1, len(predictions)), prediction=0.0)
    label = predictions.set_index("example_id")["label"]
    test = examples["split"] == "test"
    examples.loc[test, "f2"] = examples.loc[test, "example_id"].map(label) * 3 + 1
    spec = TEMPORAL_SPEC.replace(
        "classification/binary, output: probabilities", "regression, output: values"
    )
    (finding,) = _checks(_run(examples, predictions, spec), "data.strong_feature")
    assert finding.facts == {"feature": "f2", "statistic": "spearman", "value": 1.0, "n": 171}


def test_ordinary_features_no_finding():
    """B7: обычный признак (AUC около 0,95) — находок нет."""
    examples, predictions = _data()
    label = predictions.set_index("example_id")["label"]
    test = examples["split"] == "test"
    examples.loc[test, "f4"] = (
        examples.loc[test, "example_id"].map(label) * 2.3 + examples.loc[test, "f4"]
    )
    results = _run(examples, predictions)
    assert _checks(results, "data.strong_feature") == []
    assert results.findings == []
    assert results.skipped == []


# --- E1–E7: деградация ---


def _reasons(results):
    return [(s.check, s.reason) for s in results.skipped]


def test_no_examples():
    """E1: нет examples — все три проверки пропущены."""
    _, predictions = _data()
    spec = parse_task_spec(TEMPORAL_SPEC)
    results = data_findings(spec, build_run_contract(spec, run_id="r", predictions=predictions))
    assert _reasons(results) == [
        ("data.split_duplicates", "нет examples"),
        ("data.temporal_leak", "нет examples"),
        ("data.strong_feature", "нет examples"),
    ]


def test_single_split():
    """E2: один сплит — дубли и временная утечка пропущены."""
    examples, predictions = _data()
    examples = examples[examples["split"] == "test"]
    assert _reasons(_run(examples, predictions)) == [
        ("data.split_duplicates", "в examples один сплит"),
        ("data.temporal_leak", "в examples один сплит"),
    ]


def test_no_features():
    """E3: в examples нет признаков — дубли и сильные признаки пропущены."""
    examples, predictions = _data()
    examples = examples[["example_id", "split", "size", "ts"]]
    reason = "в examples нет признаков (колонок, кроме example_id, split, сегментов и времени)"
    assert _reasons(_run(examples, predictions)) == [
        ("data.split_duplicates", reason),
        ("data.strong_feature", reason),
    ]


@pytest.mark.parametrize(
    ("data_section", "shown"),
    [("data: {split_strategy: random}", "random"), ("", "не задан")],
)
def test_not_temporal_strategy(data_section, shown):
    """E4: split_strategy не temporal — временная проверка пропущена с причиной."""
    examples, predictions = _data()
    spec = "task: {type: classification/binary, output: probabilities, segments: [size]}\n"
    results = _run(examples.drop(columns="ts"), predictions, spec + data_section)
    assert _reasons(results) == [
        ("data.temporal_leak", f"split_strategy в спецификации — {shown}, а не temporal"),
    ]


def test_time_column_missing_or_unreadable():
    """E5: нет колонки времени или она не читается — пропуск с именем колонки."""
    examples, predictions = _data()
    assert _reasons(_run(examples.drop(columns="ts"), predictions)) == [
        ("data.temporal_leak", "нет examples.ts"),
    ]
    garbage = examples.assign(ts="когда-то")
    assert _reasons(_run(garbage, predictions)) == [
        ("data.temporal_leak", "колонка времени ts не читается как число или дата"),
    ]


def test_unknown_split_names():
    """E6: неизвестные сплиты перечисляются; известные проверяются."""
    examples, predictions = _data()
    extra = examples[examples["split"] == "train"].head(5).copy()
    extra["example_id"] = [f"x-{i}" for i in range(5)]
    extra["split"] = "extra"
    extra[["f0", "f1"]] = 99.0
    examples = pd.concat([examples, extra], ignore_index=True)
    leaked = _shuffle_times(examples, ["te-000"])
    results = _run(leaked, predictions)

    assert len(_checks(results, "data.temporal_leak")) == 1
    assert _reasons(results) == [
        (
            "data.temporal_leak",
            "порядок сплитов неизвестен: extra; ожидаются train, validation (valid, val, dev),"
            " test (holdout)",
        ),
    ]


def test_strong_feature_skips():
    """E7: multiclass — пропуск; нечисловые и непроверяемые признаки перечисляются."""
    examples, predictions = _data()
    multiclass = predictions.drop(columns="prediction").assign(
        **{"prediction:0": 0.5, "prediction:1": 0.5}
    )
    spec = TEMPORAL_SPEC.replace("binary", "multiclass")
    assert ("data.strong_feature", "на L0 — только для бинарной классификации и регрессии") in (
        _reasons(_run(examples, multiclass, spec))
    )

    examples = examples.assign(color="red", const=1.0)
    examples.loc[examples["split"] == "test", "f0"] = np.nan
    examples.loc[examples["example_id"] == "te-000", "f0"] = 1.0
    assert _reasons(_run(examples, predictions)) == [
        (
            "data.strong_feature",
            "не проверены признаки: color (нечисловой), const (константа),"
            " f0 (меньше 30 непустых значений)",
        ),
    ]
