"""Адаптер локальных файлов (FR-C03, L1).

Спецификация: docs/specs/files-adapter.md.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from _factories import binary_spec, binary_tables
from mlreview.adapters.base import AdapterError, load_run
from mlreview.adapters.files_adapter import (
    FilesAdapterError,
    load_files_contract,
    save_run_files,
)
from mlreview.cli import main
from mlreview.core.run_contract import RunContractError

CURVES = pd.DataFrame(
    {"name": ["train_loss"] * 3, "step": [0, 1, 2], "value": [1.0, 0.7, 0.5]},
)


def _write_parquet_run(path, *, config=None, meta=None):
    predictions, examples = binary_tables()
    path.mkdir(parents=True)
    predictions.to_parquet(path / "predictions.parquet", index=False)
    examples.to_parquet(path / "examples.parquet", index=False)
    CURVES.to_parquet(path / "curves.parquet", index=False)
    (path / "config.yaml").write_text(yaml.safe_dump(config or {"lr": 0.01}), encoding="utf-8")
    meta = {"run_id": "exp-1", "data_version": "v1"} if meta is None else meta
    (path / "run.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    return predictions, examples


def test_loads_full_parquet_run(tmp_path):
    """B1: все файлы в parquet и run.yaml → полный контракт без пробелов."""
    predictions, examples = _write_parquet_run(tmp_path / "exp-1")
    contract = load_files_contract(binary_spec(), tmp_path / "exp-1")
    assert contract.run_id == "exp-1"
    assert contract.data_version == "v1"
    assert contract.config == {"lr": 0.01}
    pd.testing.assert_frame_equal(contract.predictions, predictions)
    pd.testing.assert_frame_equal(contract.examples, examples)
    pd.testing.assert_frame_equal(contract.curves, CURVES)
    assert contract.gaps == []


def test_loads_csv_and_json_run(tmp_path):
    """B2: CSV, config.json и run.json дают тот же контракт; id остаются строками."""
    predictions, examples = binary_tables()
    predictions = predictions.assign(example_id=["007", *predictions["example_id"].iloc[1:]])
    examples.loc[examples["example_id"] == "te-000", "example_id"] = "007"
    run = tmp_path / "csv-run"
    run.mkdir()
    predictions.to_csv(run / "predictions.csv", index=False)
    examples.to_csv(run / "examples.csv", index=False)
    CURVES.to_csv(run / "curves.csv", index=False)
    (run / "config.json").write_text(json.dumps({"lr": 0.01}), encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"run_id": "exp-2", "data_version": 3}), "utf-8")
    contract = load_files_contract(binary_spec(), run)
    assert contract.run_id == "exp-2"
    assert contract.data_version == "3"
    assert contract.predictions["example_id"].iloc[0] == "007"
    assert "007" in set(contract.examples["example_id"])
    assert contract.config == {"lr": 0.01}


def test_only_predictions_uses_directory_name(tmp_path):
    """B3: только predictions.csv → run_id — имя каталога, пробелы перечислены."""
    predictions, _ = binary_tables()
    run = tmp_path / "bare"
    run.mkdir()
    predictions.to_csv(run / "predictions.csv", index=False)
    contract = load_files_contract(binary_spec(), run)
    assert contract.run_id == "bare"
    assert [gap.field for gap in contract.gaps] == ["examples", "curves", "config", "data_version"]


def test_save_run_files_roundtrip(tmp_path):
    """B4: save_run_files пишет каталог, который читается в равный контракт."""
    predictions, examples = binary_tables()
    run = tmp_path / "runs" / "exp-3"
    save_run_files(
        run,
        predictions,
        examples=examples,
        curves=CURVES,
        config={"lr": 0.1, "loss": "focal"},
        data_version="ds-v3",
        run_id="exp-3",
    )
    assert sorted(path.name for path in run.iterdir()) == [
        "config.yaml",
        "curves.parquet",
        "examples.parquet",
        "predictions.parquet",
        "run.yaml",
    ]
    contract = load_files_contract(binary_spec(), run)
    assert contract.run_id == "exp-3"
    assert contract.data_version == "ds-v3"
    assert contract.config == {"lr": 0.1, "loss": "focal"}
    pd.testing.assert_frame_equal(contract.predictions, predictions)
    pd.testing.assert_frame_equal(contract.examples, examples)


def test_load_run_dispatches_directory_to_files_adapter(tmp_path, monkeypatch):
    """B5: каталог → адаптер файлов; иначе — адаптер MLflow."""
    _write_parquet_run(tmp_path / "exp-1")
    assert load_run(binary_spec(), str(tmp_path / "exp-1")).run_id == "exp-1"

    calls = []

    def fake_mlflow(_spec, run_id, *, tracking_uri=None):
        calls.append((run_id, tracking_uri))
        return "mlflow-contract"

    monkeypatch.setattr("mlreview.adapters.base.load_mlflow_contract", fake_mlflow)
    assert load_run(binary_spec(), "abc123", tracking_uri="sqlite:///x.db") == "mlflow-contract"
    assert calls == [("abc123", "sqlite:///x.db")]


def test_contract_check_command_reads_directory(tmp_path, capsys):
    """B6: contract check по каталогу — код 0 и сводка."""
    _write_parquet_run(tmp_path / "exp-1")
    spec = tmp_path / "task.yaml"
    spec.write_text(
        "task: {type: classification/binary, output: probabilities, segments: [size]}\n",
        encoding="utf-8",
    )
    assert main(["contract", "check", str(tmp_path / "exp-1"), "--spec", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "контракт прогона валиден: exp-1" in out
    assert "версия данных: v1" in out


def test_missing_directory(tmp_path):
    """E1: каталога нет."""
    with pytest.raises(FilesAdapterError, match="каталог прогона не найден"):
        load_files_contract(binary_spec(), tmp_path / "nope")


def test_missing_predictions_lists_both_names(tmp_path):
    """E2: нет предсказаний → оба имени файла в сообщении."""
    (tmp_path / "run").mkdir()
    with pytest.raises(FilesAdapterError) as error:
        load_files_contract(binary_spec(), tmp_path / "run")
    assert "predictions.parquet" in str(error.value)
    assert "predictions.csv" in str(error.value)
    assert isinstance(error.value, AdapterError)


@pytest.mark.parametrize(
    ("first", "second"),
    [("predictions.parquet", "predictions.csv"), ("config.yaml", "config.json")],
)
def test_ambiguous_files(tmp_path, first, second):
    """E3: два формата одного файла — ошибка неоднозначности."""
    predictions, _ = binary_tables()
    run = tmp_path / "run"
    run.mkdir()
    predictions.to_parquet(run / "predictions.parquet", index=False)
    for name in (first, second):
        path = run / name
        if name.endswith(".csv"):
            predictions.to_csv(path, index=False)
        elif name.endswith((".yaml", ".json")):
            path.write_text("{}", encoding="utf-8")
    with pytest.raises(FilesAdapterError) as error:
        load_files_contract(binary_spec(), run)
    assert first in str(error.value)
    assert second in str(error.value)


@pytest.mark.parametrize(
    ("name", "content"),
    [("predictions.parquet", b"not parquet"), ("config.yaml", b"lr: [1, 2")],
)
def test_unreadable_file(tmp_path, name, content):
    """E4: файл не читается → путь файла и причина."""
    predictions, _ = binary_tables()
    run = tmp_path / "run"
    run.mkdir()
    if name != "predictions.parquet":
        predictions.to_parquet(run / "predictions.parquet", index=False)
    (run / name).write_bytes(content)
    with pytest.raises(FilesAdapterError, match=name):
        load_files_contract(binary_spec(), run)


@pytest.mark.parametrize("name", ["config.yaml", "run.yaml"])
def test_config_must_be_mapping(tmp_path, name):
    """E5: config и run — словари."""
    predictions, _ = binary_tables()
    run = tmp_path / "run"
    run.mkdir()
    predictions.to_parquet(run / "predictions.parquet", index=False)
    (run / name).write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(FilesAdapterError, match="ожидается словарь"):
        load_files_contract(binary_spec(), run)


def test_invalid_contract_is_reported_by_core(tmp_path):
    """E6: таблицы не по формату → RunContractError ядра."""
    predictions, _ = binary_tables()
    run = tmp_path / "run"
    run.mkdir()
    predictions.drop(columns="label").to_parquet(run / "predictions.parquet", index=False)
    with pytest.raises(RunContractError, match=r"predictions\.label"):
        load_files_contract(binary_spec(), run)
