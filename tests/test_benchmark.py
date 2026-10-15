"""Приёмочный бенчмарк L0: дефекты B01, B02, B12; метрики Q1, Q2.

Спецификация: docs/specs/benchmark.md.
"""

from __future__ import annotations

import sys

import pytest

from benchmark.projects import clear_cache
from benchmark.run import PROJECTS, VERSIONS, main, run_benchmark


@pytest.fixture(scope="module")
def results():
    pytest.importorskip("sklearn")
    return run_benchmark()


def test_every_version_evaluated(results):
    """B1: все проекты и версии; Q1 и Q2 считаются по версиям."""
    seen = [(v.project, v.version) for v in results.versions]
    assert seen == [(project, version) for project in PROJECTS for version in VERSIONS]

    defects = [v for v in results.versions if v.version != "clean"]
    clean = [v for v in results.versions if v.version == "clean"]
    assert all(v.found is None for v in clean)
    assert results.recall == sum(v.found for v in defects) / len(defects)
    assert results.false_critical == sum(v.critical for v in clean) / len(clean)


def test_l0_gate_and_target(results):
    """B2: гейт L0 (Q1 ≥ 50%, Q2 ≤ 3) пройден; на текущем наборе — все дефекты, 0 critical."""
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
