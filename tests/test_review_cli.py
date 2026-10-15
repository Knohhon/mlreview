"""Команды review, diagnose, ask, metrics (FR-R01).

Спецификация: docs/specs/review.md (пункты C*, E2).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from mlreview.cli import main

SPEC_TEXT = (
    "task: {type: classification/binary, output: probabilities, segments: [size]}\n"
    "data: {split_strategy: stratified}\n"
)


def _tables(seed: int, duplicates: bool):
    """Train 200 + test 200; при `duplicates` 5 строк test скопированы в train."""
    rng = np.random.default_rng(0)
    n = 200
    examples = pd.DataFrame(
        {
            "example_id": [f"tr-{i:03d}" for i in range(n)] + [f"te-{i:03d}" for i in range(n)],
            "split": ["train"] * n + ["test"] * n,
            "size": ["small", "large"] * n,
            "f0": rng.normal(0, 1, 2 * n),
        }
    )
    if duplicates:
        examples.loc[:4, "f0"] = examples.loc[n : n + 4, "f0"].to_numpy()
    label = rng.integers(0, 2, n)
    logit = np.where(label == 1, 2.0, -2.0) + np.random.default_rng(seed).normal(0, 1, n)
    predictions = pd.DataFrame(
        {
            "example_id": examples["example_id"].iloc[n:].to_numpy(),
            "label": label,
            "prediction": 1 / (1 + np.exp(-logit)),
        }
    )
    return predictions, examples


@pytest.fixture
def runs(tmp_path, mlflow_store):
    """Кандидат с дублями и базовый прогон в локальном MLflow; файл спецификации."""
    ids = {}
    for name, seed, duplicates, version in (("cand", 1, True, "v2"), ("base", 2, False, "v1")):
        predictions, examples = _tables(seed, duplicates)
        ids[name] = mlflow_store.run(
            artifacts={"predictions.parquet": predictions, "examples.parquet": examples},
            tags={"mlreview.data_version": version},
        )
    spec = tmp_path / "task.yaml"
    spec.write_text(SPEC_TEXT, encoding="utf-8")
    return ids, str(spec), mlflow_store.uri


def test_review_text_report(runs, capsys):
    """C1, C3, E2: текстовый отчёт со всеми группами, базовым прогоном и строкой LLM."""
    ids, spec, uri = runs
    argv = ["review", ids["cand"], "--spec", spec, "--baseline", ids["base"], "--tracking-uri", uri]
    assert main(argv) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"ревью прогона {ids['cand']}"
    assert lines[1] == f"  базовый прогон: {ids['base']}, общих примеров: 200"
    assert lines[2] == "  версия данных: v1 → v2"
    assert lines[3].startswith("находки: ")
    assert lines[4].startswith("  1. [critical] data.split_duplicates — 5 из 200 примеров test")
    assert lines[4].endswith("(FR-D03, B01)")
    assert lines[5] == "     примеры: te-000, te-001, te-002, te-003, te-004"
    assert lines[6].startswith("     предложение: ")
    assert any("comparison.metric_delta — roc_auc: " in line for line in lines)
    assert "не проверено: 1" in lines
    assert (
        "  - data.temporal_leak: split_strategy в спецификации — stratified, а не temporal" in lines
    )
    assert lines[-1] == "LLM: не используется — отчёт детерминированный"


def test_review_json_report(runs, capsys):
    """C4: --format json — Report.to_dict() с полем llm: null, кириллица без экранирования."""
    ids, spec, uri = runs
    argv = ["review", ids["cand"], "--spec", spec, "--tracking-uri", uri, "--format", "json"]
    assert main(argv) == 0

    out = capsys.readouterr().out
    assert "примеров" in out
    data = json.loads(out)
    assert data["run_id"] == ids["cand"]
    assert data["baseline_id"] is None
    assert data["llm"] is None
    assert data["findings"][0]["check"] == "data.split_duplicates"
    assert {f["check"] for f in data["findings"]} >= {"metrics.value"}


def test_diagnose_data(runs, capsys):
    """C2: diagnose data — только группа data."""
    ids, spec, uri = runs
    argv = ["diagnose", "data", ids["cand"], "--spec", spec, "--tracking-uri", uri]
    assert main(argv) == 0

    out = capsys.readouterr().out
    assert out.startswith(f"диагностика данных прогона {ids['cand']}\n")
    assert "data.split_duplicates" in out
    assert "metrics.value" not in out
    assert "slices." not in out


def test_diagnose_segments_with_baseline(runs, capsys):
    """C2: diagnose segments — только группа segments, базовый прогон учитывается."""
    ids, spec, uri = runs
    argv = ["diagnose", "segments", ids["cand"], "--spec", spec, "--tracking-uri", uri]
    assert main([*argv, "--baseline", ids["base"], "--format", "json"]) == 0

    data = json.loads(capsys.readouterr().out)
    assert data["baseline_id"] == ids["base"]
    assert all(f["check"].startswith("slices.") for f in data["findings"])
    assert all(s["check"].startswith("slices.") for s in data["skipped"])


def test_review_no_findings_line(tmp_path, mlflow_store, capsys):
    """C3: без находок — строка «находок нет»."""
    predictions, examples = _tables(1, duplicates=False)
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": predictions, "examples.parquet": examples}
    )
    spec = tmp_path / "task.yaml"
    spec.write_text(SPEC_TEXT, encoding="utf-8")
    argv = ["diagnose", "data", run_id, "--spec", str(spec), "--tracking-uri", mlflow_store.uri]
    assert main(argv) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "  версия данных: нет"
    assert lines[2] == "находок нет"


def test_review_errors(runs, tmp_path, capsys):
    """C5: невалидная спецификация, неизвестный базовый прогон, без mlflow — код 1."""
    ids, spec, uri = runs
    bad = tmp_path / "bad.yaml"
    bad.write_text("task: {output: probabilities}\n", encoding="utf-8")
    assert main(["review", ids["cand"], "--spec", str(bad), "--tracking-uri", uri]) == 1
    assert "task.type" in capsys.readouterr().err

    argv = ["review", ids["cand"], "--spec", spec, "--baseline", "0" * 32, "--tracking-uri", uri]
    assert main(argv) == 1
    assert "0" * 32 in capsys.readouterr().err


def test_review_without_mlflow(tmp_path, monkeypatch, capsys):
    """C5: без пакета mlflow — код 1 и подсказка установки."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    spec = tmp_path / "task.yaml"
    spec.write_text(SPEC_TEXT, encoding="utf-8")
    assert main(["review", "run-1", "--spec", str(spec)]) == 1
    assert 'pip install "mlreview[mlflow]"' in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["review"],
        ["review", "run-1"],
        ["diagnose"],
        ["diagnose", "run-1", "--spec", "s.yaml"],
        ["diagnose", "data", "run-1"],
        ["review", "run-1", "--spec", "s.yaml", "--format", "xml"],
    ],
)
def test_review_usage_errors(argv):
    """C6: нет --spec, id прогона или группы diagnose — ошибка использования, код 2."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_ask_requires_llm(capsys):
    """C7: ask без LLM — явный отказ, код 1."""
    assert main(["ask", "почему", "хуже?"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("mlreview ask: требует LLM — LLM-слой не подключён")
    assert "mlreview review <run_id> --spec <путь>" in err


def test_metrics_not_implemented(capsys):
    """C8: metrics — «пока не реализована», уровень L1."""
    assert main(["metrics"]) == 1
    err = capsys.readouterr().err
    assert "команда пока не реализована" in err
    assert "L1" in err
