"""Приёмочный бенчмарк L0–L1: дефекты B01–B12; метрики Q1, Q2; NFR-06.

Спецификация: docs/specs/benchmark.md.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from benchmark.perf import measure
from benchmark.projects import clear_cache, versions_for
from benchmark.run import PROJECTS, main, run_benchmark
from mlreview.analyzers.evaluation import METRICS, Outputs, auc_sampler


@pytest.fixture(scope="module")
def results():
    pytest.importorskip("sklearn")
    return run_benchmark()


def test_every_version_evaluated(results):
    """B1: все проекты и версии; Q1 и Q2 считаются по версиям."""
    seen = [(v.project, v.version) for v in results.versions]
    assert seen == [(project, version) for project in PROJECTS for version in versions_for(project)]
    assert len([v for v in results.versions if v.version != "clean"]) == 21

    defects = [v for v in results.versions if v.version != "clean"]
    clean = [v for v in results.versions if v.version == "clean"]
    assert all(v.found is None for v in clean)
    assert results.recall == sum(v.found for v in defects) / len(defects)
    assert results.false_critical == sum(v.critical for v in clean) / len(clean)


def test_l1_gate_and_target(results):
    """B2: гейт L1 (Q1 ≥ 70%, Q2 ≤ 1) пройден; на текущем наборе — все дефекты, 0 critical."""
    assert results.passed
    missed = [(v.project, v.version) for v in results.versions if v.found is False]
    assert missed == []
    assert results.recall == 1.0
    assert results.false_critical == 0


def test_b12_overall_metric_not_worse(results):
    """B3: в версии B12 кандидат в целом не хуже базового прогона."""
    for version in results.versions:
        if version.version == "B12":
            assert version.overall_improvement >= 0, version.project


def test_benchmark_is_reproducible():
    """B4: повторный запуск даёт идентичные результаты (NFR-05)."""
    pytest.importorskip("sklearn")
    clear_cache()
    first = run_benchmark(projects=("breast_cancer",))
    clear_cache()  # модели обучаются заново, а не берутся из кэша
    second = run_benchmark(projects=("breast_cancer",))
    assert [v.report.to_dict() for v in first.versions] == [
        v.report.to_dict() for v in second.versions
    ]


def test_main_prints_table(capsys):
    """B5: python -m benchmark — таблица версий, строка Q1/Q2, код 0 при пройденном гейте."""
    pytest.importorskip("sklearn")
    assert main(["--project", "breast_cancer"]) == 0
    out = capsys.readouterr().out
    assert "breast_cancer" in out
    assert "B01" in out
    assert "Q1" in out
    assert "Q2" in out


def test_main_without_sklearn(monkeypatch, capsys):
    """E1: без scikit-learn — подсказка установки и код 1."""
    monkeypatch.setitem(sys.modules, "sklearn", None)
    assert main([]) == 1
    assert "pip install -r requirements.txt" in capsys.readouterr().err


def test_fast_bootstrap_auc_matches_resampled_auc():
    """B6: быстрый бутстрап ROC AUC равен ROC AUC на выборке с повторами."""
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.integers(0, 2, n)
    scores = np.round(rng.uniform(0, 1, n), 2)  # много равных оценок
    outputs = Outputs(np.arange(n).astype(str), labels, scores)
    fast = auc_sampler(outputs)
    for _ in range(20):
        index = rng.integers(0, n, n)
        assert fast(index) == pytest.approx(METRICS["roc_auc"](outputs.take(index)), abs=1e-12)


def test_perf_measure_reports_times():
    """B6: замер производительности — время контрактов и ревью, число находок."""
    result = measure(2000)
    assert result.n == 2000
    assert result.review_seconds > 0
    assert result.findings > 0
    assert result.passed
