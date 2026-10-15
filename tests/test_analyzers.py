"""Интерфейс анализаторов, группы ревью и сиды в CLI (FR-E01, FR-R01, FR-D05).

Спецификация: docs/specs/analyzers.md.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from _factories import BINARY_SPEC_TEXT, binary_contract, binary_spec, binary_tables
from mlreview.adapters.files_adapter import save_run_files
from mlreview.analyzers.registry import GROUPS, Analyzer, register, unregister
from mlreview.analyzers.review import review
from mlreview.cli import main
from mlreview.core.findings import CheckResults, Finding


@pytest.fixture
def custom():
    added = []

    def add(analyzer):
        added.append(register(analyzer))
        return analyzer

    yield add
    for analyzer in added:
        unregister(analyzer.name)


def test_groups_order():
    """Порядок групп."""
    assert GROUPS == ("data", "segments", "metrics", "curves", "process")


def test_registered_analyzer_runs_after_builtin(custom):
    """B1: свой анализатор группы data — после встроенных."""

    def run(ctx):
        results = CheckResults()
        message = f"{ctx.contract.run_id}: своя проверка"
        results.findings.append(Finding("custom.check", "low", message, "FR-E01"))
        results.skip("custom.other", "нечего проверять")
        return results

    custom(Analyzer(name="custom", group="data", requirement="FR-E01", run=run))
    report = review(binary_spec(), binary_contract(duplicates=10), groups=("data",))
    assert "custom.check" in [f.check for f in report.findings]
    assert report.skipped[-1].check == "custom.other"


def test_failing_analyzer_is_reported(custom):
    """B2: исключение анализатора — пропуск, остальные работают."""

    def boom(_ctx):
        raise RuntimeError("boom")

    custom(Analyzer(name="broken", group="data", requirement="FR-E01", run=boom))
    report = review(binary_spec(), binary_contract(duplicates=10), groups=("data",))
    assert "data.split_duplicates" in [f.check for f in report.findings]
    assert ("broken", "ошибка анализатора: RuntimeError: boom") in [
        (s.check, s.reason) for s in report.skipped
    ]


def test_unknown_group():
    """B3: неизвестная группа."""
    with pytest.raises(ValueError, match="допустимо: data, segments, metrics, curves, process"):
        review(binary_spec(), binary_contract(), groups=("nope",))


def test_seeds_reach_seed_analyzer():
    """B4: сиды обеих сторон — в анализатор сидов; отчёт знает их id."""
    report = review(
        binary_spec(),
        binary_contract("c1", seed=1),
        binary_contract("b1", seed=4),
        groups=("metrics",),
        candidate_seeds=(binary_contract("c2", seed=2), binary_contract("c3", seed=3)),
        baseline_seeds=(binary_contract("b2", seed=5), binary_contract("b3", seed=6)),
    )
    seeds = [f for f in report.findings if f.check.startswith("seeds.")]
    assert seeds
    assert seeds[0].facts["candidate_seeds"] == 3
    assert report.candidate_seeds == ("c2", "c3")
    assert report.baseline_seeds == ("b2", "b3")


def test_cli_review_with_seeds(tmp_path, capsys):
    """B5: review A1 A2 A3 --baseline B1 B2 B3."""
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    for name, seed in (("a1", 1), ("a2", 2), ("a3", 3), ("b1", 4), ("b2", 5), ("b3", 6)):
        predictions, examples = binary_tables(seed=seed)
        save_run_files(tmp_path / name, predictions, examples=examples, data_version="v1")
    argv = ["review", "a1", "a2", "a3", "--spec", "task.yaml", "--baseline", "b1", "b2", "b3"]
    assert main([*argv, "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == "a1"
    assert data["baseline_id"] == "b1"
    assert data["candidate_seeds"] == ["a2", "a3"]
    assert data["baseline_seeds"] == ["b2", "b3"]
    assert any(f["check"].startswith("seeds.") for f in data["findings"])


@pytest.mark.parametrize(("group", "prefix"), [("curves", "curves"), ("process", "process.")])
def test_diagnose_curves_and_process(tmp_path, capsys, group, prefix):
    """B6: diagnose curves / process — только своя группа."""
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    predictions, examples = binary_tables()
    curves = pd.DataFrame(
        {"name": ["train_loss"] * 5, "step": range(5), "value": [1, 0.8, float("nan"), 0, 0]}
    )
    save_run_files(tmp_path / "run", predictions, examples=examples, curves=curves)
    argv = ["diagnose", group, "run", "--spec", "task.yaml", "--format", "json"]
    assert main(argv) == 0
    data = json.loads(capsys.readouterr().out)
    assert all(f["check"].startswith(prefix) for f in data["findings"])
    assert all(s["check"].startswith(prefix) for s in data["skipped"])
    assert data["findings"] or data["skipped"]


def test_missing_inputs_are_skipped_per_analyzer():
    """E1: нет кривых, конфига, базового, хранилища — пропуски, остальное работает."""
    report = review(binary_spec(), binary_contract())
    skipped = {s.check for s in report.skipped}
    assert {"curves", "process.multiple_changes", "process.validation_reuse"} <= skipped
    assert "metrics.value" in {f.check for f in report.findings}
