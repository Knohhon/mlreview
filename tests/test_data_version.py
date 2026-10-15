"""Версия и хеш данных прогона (FR-C04).

Спецификация: docs/specs/data-version.md.
"""

from __future__ import annotations

import re

from _factories import binary_contract, binary_spec, binary_tables
from mlreview.analyzers.review import review
from mlreview.cli import render_report
from mlreview.core.run_contract import build_run_contract


def _contract(predictions, examples=None, **kwargs):
    return build_run_contract(
        binary_spec(), run_id="r", predictions=predictions, examples=examples, **kwargs
    )


def test_hash_format():
    """B1: sha256: и 16 hex."""
    assert re.fullmatch(r"sha256:[0-9a-f]{16}", binary_contract().data_hash)


def test_hash_ignores_row_and_column_order():
    """B2: порядок строк и колонок не влияет."""
    predictions, examples = binary_tables()
    first = _contract(predictions, examples).data_hash
    shuffled_p = predictions.sample(frac=1, random_state=3)[["prediction", "label", "example_id"]]
    shuffled_e = examples.sample(frac=1, random_state=4)[["f0", "size", "split", "example_id"]]
    assert _contract(shuffled_p, shuffled_e).data_hash == first


def test_hash_changes_with_value_or_label():
    """B3: изменение признака или метки меняет хеш."""
    predictions, examples = binary_tables()
    base = _contract(predictions, examples).data_hash
    changed = examples.copy()
    changed.loc[0, "f0"] += 1e-6
    assert _contract(predictions, changed).data_hash != base
    flipped = predictions.copy()
    flipped.loc[0, "label"] = 1 - flipped.loc[0, "label"]
    assert _contract(flipped, examples).data_hash != base


def test_hash_ignores_integer_vs_float():
    """B4: 1 и 1.0 дают один хеш."""
    predictions, examples = binary_tables()
    ints = examples.assign(f1=[1, 2] * (len(examples) // 2))
    floats = examples.assign(f1=[1.0, 2.0] * (len(examples) // 2))
    assert _contract(predictions, ints).data_hash == _contract(predictions, floats).data_hash


def test_hash_without_examples_uses_predictions():
    """B5: без examples — по id и меткам predictions; прогнозы модели не влияют."""
    predictions, _ = binary_tables()
    other_model = predictions.assign(prediction=1 - predictions["prediction"])
    assert _contract(predictions).data_hash == _contract(other_model).data_hash


def test_report_shows_versions_and_hashes():
    """B6: отчёт несёт хеши; текстовая строка версии с хешем."""
    candidate = binary_contract("cand", seed=1, data_version="v2", shift=0.5)
    baseline = binary_contract("base", seed=2, data_version="v1")
    report = review(binary_spec(), candidate, baseline, groups=("metrics",))
    assert report.data_hash == candidate.data_hash
    assert report.baseline_data_hash == baseline.data_hash
    assert report.to_dict()["data_hash"] == candidate.data_hash
    text = render_report(report, "ревью прогона")
    expected = f"  версия данных: v1 ({baseline.data_hash}) → v2 ({candidate.data_hash})"
    assert expected in text.splitlines()

    single = review(binary_spec(), binary_contract(data_version=None), groups=("metrics",))
    assert f"  версия данных: нет ({single.data_hash})" in render_report(single, "x").splitlines()


def test_missing_version_keeps_gap():
    """E1: без data_version пробел остаётся, хеш есть."""
    contract = binary_contract(data_version=None)
    assert "data_version" in [gap.field for gap in contract.gaps]
    assert contract.data_hash.startswith("sha256:")
