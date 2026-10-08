"""Тесты адаптера MLflow (docs/specs/mlflow-adapter.md).

Прогоны создаются в локальном sqlite-трекере (фикстура `mlflow_store`); без пакета mlflow
эти тесты пропускаются, кроме E1.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from mlreview.adapters.mlflow_adapter import (
    MlflowAdapterError,
    load_mlflow_contract,
    log_contract_artifacts,
)
from mlreview.core.run_contract import RunContractError
from mlreview.core.task_spec import TaskSpec, parse_task_spec

SPEC = parse_task_spec("task: {type: classification/binary, output: probabilities}\n")
INSTALL_HINT = 'pip install "mlreview[mlflow]"'


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["007", "a2", "a3"],
            "label": [0, 1, 1],
            "prediction": [0.1, 0.8, 0.4],
        }
    )


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["t1", "t2", "007", "a2", "a3"],
            "split": ["train", "train", "test", "test", "test"],
            "language": ["en", "de", "en", "de", "en"],
        }
    )


def _gap_fields(contract) -> list[str]:
    return [gap.field for gap in contract.gaps]


def _load(store, run_id: str, spec: TaskSpec = SPEC):
    return load_mlflow_contract(spec, run_id, tracking_uri=store.uri)


# --- Ожидаемое поведение -------------------------------------------------------------------------


def test_full_run_becomes_contract(mlflow_store):
    """B1: параметры, история метрик, тег и артефакты parquet попадают в контракт."""
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions(), "examples.parquet": _examples()},
        params={"C": "1.0", "penalty": "l2"},
        metrics={"val_loss": [0.8, 0.6, 0.65], "train_loss": [0.7, 0.5, 0.4], "auc": [0.9]},
        tags={"mlreview.data_version": "ds-v3"},
    )

    contract = _load(mlflow_store, run_id)

    assert contract.run_id == run_id
    assert contract.config == {"C": "1.0", "penalty": "l2"}
    assert contract.data_version == "ds-v3"
    pd.testing.assert_frame_equal(contract.predictions, _predictions())
    assert contract.examples is not None
    pd.testing.assert_frame_equal(contract.examples, _examples())
    expected_curves = pd.DataFrame(
        {
            "name": ["train_loss"] * 3 + ["val_loss"] * 3,
            "step": [0, 1, 2, 0, 1, 2],
            "value": [0.7, 0.5, 0.4, 0.8, 0.6, 0.65],
        }
    )
    assert contract.curves is not None
    pd.testing.assert_frame_equal(contract.curves, expected_curves)
    assert contract.gaps == []


def test_csv_artifacts_keep_ids_as_text(mlflow_store):
    """B2: таблицы в CSV дают тот же контракт; id `007` остаётся строкой."""
    run_id = mlflow_store.run(
        artifacts={"predictions.csv": _predictions(), "examples.csv": _examples()}
    )

    contract = _load(mlflow_store, run_id)

    pd.testing.assert_frame_equal(contract.predictions, _predictions())
    assert contract.examples is not None
    pd.testing.assert_frame_equal(contract.examples, _examples())
    assert contract.predictions["example_id"].tolist() == ["007", "a2", "a3"]


def test_single_point_metric_is_not_a_curve(mlflow_store):
    """B3: метрика с одной точкой — не кривая."""
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions()},
        metrics={"auc": [0.9], "train_loss": [0.7, 0.5, 0.4]},
    )

    contract = _load(mlflow_store, run_id)

    assert contract.curves is not None
    assert contract.curves["name"].unique().tolist() == ["train_loss"]


def test_no_curves_is_gap(mlflow_store):
    """B3: без метрик из двух и более точек кривых нет — пробел curves."""
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions()}, metrics={"auc": [0.9]}
    )

    contract = _load(mlflow_store, run_id)

    assert contract.curves is None
    assert "curves" in _gap_fields(contract)


def test_predictions_only_run(mlflow_store):
    """B4: только предсказания — контракт валиден, пробелы перечислены."""
    run_id = mlflow_store.run(artifacts={"predictions.parquet": _predictions()})

    contract = _load(mlflow_store, run_id)

    assert _gap_fields(contract) == ["examples", "curves", "config", "data_version"]


def test_default_tracking_uri_from_environment(mlflow_store, monkeypatch):
    """B5: без tracking_uri используется MLFLOW_TRACKING_URI."""
    run_id = mlflow_store.run(artifacts={"predictions.parquet": _predictions()})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_store.uri)

    contract = load_mlflow_contract(SPEC, run_id)

    assert contract.run_id == run_id


def test_logged_artifacts_round_trip(mlflow_store, monkeypatch):
    """B6: log_contract_artifacts пишет таблицы без индекса и тег версии; чтение их возвращает."""
    import mlflow  # noqa: PLC0415 — mlflow есть, раз фикстура mlflow_store не пропустила тест

    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_store.uri)
    predictions = _predictions().set_index(pd.Index([10, 11, 12]))
    with mlflow.start_run(experiment_id=mlflow_store.experiment_id) as run:
        log_contract_artifacts(predictions, examples=_examples(), data_version="ds-v3")

    run_id = run.info.run_id
    paths = [item.path for item in mlflow_store.client.list_artifacts(run_id, "mlreview")]
    assert sorted(paths) == ["mlreview/examples.parquet", "mlreview/predictions.parquet"]
    contract = _load(mlflow_store, run_id)
    pd.testing.assert_frame_equal(contract.predictions, _predictions())
    assert contract.examples is not None
    pd.testing.assert_frame_equal(contract.examples, _examples())
    assert contract.data_version == "ds-v3"


# --- Деградация и ошибки -------------------------------------------------------------------------


def test_missing_mlflow_load(monkeypatch):
    """E1: без пакета mlflow чтение — ошибка с подсказкой установки, а не ImportError."""
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(MlflowAdapterError) as exc:
        load_mlflow_contract(SPEC, "run-1", tracking_uri="sqlite:///nowhere.db")

    assert INSTALL_HINT in str(exc.value)


def test_missing_mlflow_log(monkeypatch):
    """E1: без пакета mlflow запись — та же ошибка с подсказкой."""
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(MlflowAdapterError) as exc:
        log_contract_artifacts(_predictions())

    assert INSTALL_HINT in str(exc.value)


def test_unknown_run(mlflow_store):
    """E2: прогона нет — ошибка с id прогона и адресом трекера."""
    run_id = "0" * 32

    with pytest.raises(MlflowAdapterError) as exc:
        _load(mlflow_store, run_id)

    message = str(exc.value)
    assert run_id in message
    assert mlflow_store.uri in message


def test_missing_predictions_artifact(mlflow_store):
    """E3: нет артефакта предсказаний — ошибка с ожидаемыми путями."""
    run_id = mlflow_store.run(artifacts={"examples.parquet": _examples()})

    with pytest.raises(MlflowAdapterError) as exc:
        _load(mlflow_store, run_id)

    message = str(exc.value)
    assert "mlreview/predictions.parquet" in message
    assert "mlreview/predictions.csv" in message


def test_ambiguous_table_format(mlflow_store):
    """E4: таблица и в parquet, и в CSV — ошибка неоднозначности с обоими путями."""
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions(), "predictions.csv": _predictions()}
    )

    with pytest.raises(MlflowAdapterError) as exc:
        _load(mlflow_store, run_id)

    message = str(exc.value)
    assert "mlreview/predictions.parquet" in message
    assert "mlreview/predictions.csv" in message


@pytest.mark.parametrize(
    ("name", "content"),
    [("predictions.parquet", b"not a parquet file"), ("predictions.csv", b"")],
)
def test_unreadable_artifact(mlflow_store, name, content):
    """E5: повреждённый артефакт — ошибка с путём артефакта."""
    run_id = mlflow_store.run(artifacts={name: content})

    with pytest.raises(MlflowAdapterError) as exc:
        _load(mlflow_store, run_id)

    assert f"mlreview/{name}" in str(exc.value)


def test_invalid_contract_is_reported_by_core(mlflow_store):
    """E6: таблица не по формату — RunContractError из ядра с путём поля."""
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions().drop(columns="label")}
    )

    with pytest.raises(RunContractError) as exc:
        _load(mlflow_store, run_id)

    assert "predictions.label" in [problem.path for problem in exc.value.problems]


def test_repeated_metric_steps_are_reported(mlflow_store):
    """E6: метрика, залогированная несколько раз с одним step, — ошибка контракта в curves."""
    run_id = mlflow_store.run(artifacts={"predictions.parquet": _predictions()})
    for value in (0.7, 0.5):
        mlflow_store.client.log_metric(run_id, "train_loss", value, step=0)

    with pytest.raises(RunContractError) as exc:
        _load(mlflow_store, run_id)

    assert "curves" in [problem.path for problem in exc.value.problems]


def test_log_requires_active_run(mlflow_store, monkeypatch):
    """E7: без активного прогона запись — ошибка, новый прогон не создаётся."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_store.uri)

    with pytest.raises(MlflowAdapterError) as exc:
        log_contract_artifacts(_predictions())

    assert "нет активного прогона" in str(exc.value)
    runs = mlflow_store.client.search_runs([mlflow_store.experiment_id, "0"])
    assert runs == []
