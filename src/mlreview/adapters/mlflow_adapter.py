"""Адаптер MLflow: прогон MLflow → контракт прогона (FR-C03, FR-C02, FR-C07).

Таблицы контракта лежат в артефактах прогона `mlreview/<таблица>.parquet|csv`, конфиг — в
параметрах, кривые — в истории метрик, версия данных — в теге `mlreview.data_version`,
модель — в тегах `mlreview.model.*` или автологирования sklearn (`docs/specs/model-info.md`).
Пакет `mlflow` (extra `mlflow`) импортируется лениво, чтобы без него работали остальные
команды (NFR-01). Поведение зафиксировано в `docs/specs/mlflow-adapter.md`.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from mlreview.adapters.errors import AdapterError
from mlreview.core.model_info import ModelInfo, library_of
from mlreview.core.run_contract import RunContract, build_run_contract
from mlreview.core.task_spec import TaskSpec

ARTIFACT_DIR = "mlreview"
DATA_VERSION_TAG = "mlreview.data_version"
MODEL_TAG_PREFIX = "mlreview.model."
AUTOLOG_CLASS_TAG = "estimator_class"
# Шаг Pipeline в repr параметра `steps`: `('knn', KNeighborsClassifier(n_neighbors=3))`.
_PIPELINE_STEP = re.compile(r"\(\s*['\"][^'\"]+['\"]\s*,\s*([A-Za-z_][\w.]*)\(")
TABLE_FORMATS = ("parquet", "csv")
INSTALL_HINT = 'pip install "mlreview[mlflow]"'


class MlflowAdapterError(AdapterError):
    """Прогон MLflow не прочитан или не записан; сообщение объясняет причину."""


def _mlflow() -> Any:
    try:
        import mlflow  # noqa: PLC0415 — extra mlflow необязателен, импорт только по требованию
    except ImportError:
        message = f"адаптер MLflow требует пакет mlflow: {INSTALL_HINT}"
        raise MlflowAdapterError(message) from None
    return mlflow


def _table_path(run_id: str, table: str, artifacts: set[str]) -> str | None:
    """Путь артефакта таблицы; ни одного — `None`, оба формата сразу — ошибка (E4)."""
    found = [f"{ARTIFACT_DIR}/{table}.{fmt}" for fmt in TABLE_FORMATS]
    found = [path for path in found if path in artifacts]
    if len(found) > 1:
        message = (
            f"прогон {run_id}: таблица {table} залогирована в нескольких форматах "
            f"({', '.join(found)}); оставьте один"
        )
        raise MlflowAdapterError(message)
    return found[0] if found else None


def _read_table(client: Any, run_id: str, path: str) -> pd.DataFrame:
    """Скачивает и читает артефакт таблицы; `example_id` из CSV читается как строка (B2)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(run_id, path, tmp)
            if path.endswith(".csv"):
                return pd.read_csv(local, dtype={"example_id": str})
            return pd.read_parquet(local)
        except Exception as exc:  # читатели parquet и CSV бросают разные типы ошибок
            message = f"прогон {run_id}: не удалось прочитать артефакт {path}: {exc}"
            raise MlflowAdapterError(message) from exc


def _curves(client: Any, run_id: str, metric_names: list[str]) -> pd.DataFrame | None:
    """История метрик с двумя и более точками в формате name, step, value (B3)."""
    rows = []
    for name in sorted(metric_names):
        history = client.get_metric_history(run_id, name)
        if len(history) < 2:  # noqa: PLR2004 — одна точка — итоговое число, а не кривая
            continue
        points = sorted(history, key=lambda metric: (metric.step, metric.timestamp))
        rows += [(name, point.step, point.value) for point in points]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["name", "step", "value"])


def _autolog_model(tags: Mapping[str, str], params: Mapping[str, str]) -> ModelInfo | None:
    """Модель из автологирования sklearn; у Pipeline — последний шаг из `steps` (M2, M3)."""
    estimator = tags.get(AUTOLOG_CLASS_TAG)
    if not estimator:
        return None
    if estimator.rsplit(".", 1)[-1] == "Pipeline":
        steps = _PIPELINE_STEP.findall(params.get("steps", ""))
        if steps:
            return ModelInfo(class_name=steps[-1], library=library_of(steps[-1]), source="pipeline")
    return ModelInfo(class_name=estimator, library=library_of(estimator), source="autolog")


def _model(tags: Mapping[str, str], params: Mapping[str, str]) -> ModelInfo | dict | None:
    """Модель прогона: теги `mlreview.model.*` важнее автологирования (M1, M4)."""
    autolog = _autolog_model(tags, params)
    explicit = {
        key[len(MODEL_TAG_PREFIX) :]: value
        for key, value in tags.items()
        if key.startswith(MODEL_TAG_PREFIX)
    }
    if not explicit:
        return autolog
    if set(explicit) - {"class", "family", "library"}:
        return explicit  # неизвестный ключ покажет ядро с путём model.<ключ>
    class_name = explicit.get("class") or (autolog.class_name if autolog else None)
    return ModelInfo(
        class_name=class_name,
        family=explicit.get("family"),
        library=explicit.get("library") or library_of(class_name),
        source="tag",
    )


def load_mlflow_contract(
    spec: TaskSpec, run_id: str, *, tracking_uri: str | None = None
) -> RunContract:
    """Читает прогон MLflow и собирает его контракт (FR-C02) против спецификации задачи.

    Ошибки чтения — `MlflowAdapterError`; ошибки содержимого — `RunContractError` из ядра.
    Без `tracking_uri` используется адрес трекера по умолчанию MLflow (`MLFLOW_TRACKING_URI`).
    """
    mlflow = _mlflow()
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    uri = tracking_uri or mlflow.get_tracking_uri()
    try:
        run = client.get_run(run_id)
        artifacts = {item.path for item in client.list_artifacts(run_id, ARTIFACT_DIR)}
    except mlflow.exceptions.MlflowException as exc:
        raise MlflowAdapterError(f"прогон {run_id} не получен из {uri}: {exc.message}") from exc

    predictions_path = _table_path(run_id, "predictions", artifacts)
    if predictions_path is None:
        expected = " или ".join(f"{ARTIFACT_DIR}/predictions.{fmt}" for fmt in TABLE_FORMATS)
        raise MlflowAdapterError(f"прогон {run_id}: нет артефакта предсказаний {expected}")
    examples_path = _table_path(run_id, "examples", artifacts)
    importance_path = _table_path(run_id, "feature_importance", artifacts)
    params = dict(run.data.params)

    return build_run_contract(
        spec,
        run_id=run.info.run_id,
        predictions=_read_table(client, run_id, predictions_path),
        examples=_read_table(client, run_id, examples_path) if examples_path else None,
        config=params,
        curves=_curves(client, run_id, list(run.data.metrics)),
        data_version=run.data.tags.get(DATA_VERSION_TAG),
        model=_model(run.data.tags, params),
        feature_importance=(
            _read_table(client, run_id, importance_path) if importance_path else None
        ),
    )


def log_contract_artifacts(
    predictions: pd.DataFrame,
    *,
    examples: pd.DataFrame | None = None,
    data_version: str | None = None,
    model: ModelInfo | Mapping[str, Any] | None = None,
    feature_importance: pd.DataFrame | None = None,
) -> None:
    """Записывает таблицы контракта, версию данных и модель в активный прогон MLflow (B6, M6).

    Вызывается из скрипта обучения внутри `mlflow.start_run()`; таблицы пишутся в parquet
    без индекса. Без активного прогона — ошибка, новый прогон не создаётся (E7).
    """
    mlflow = _mlflow()
    if mlflow.active_run() is None:
        message = "нет активного прогона MLflow: вызывайте внутри mlflow.start_run()"
        raise MlflowAdapterError(message)
    with tempfile.TemporaryDirectory() as tmp:
        tables = (
            ("predictions", predictions),
            ("examples", examples),
            ("feature_importance", feature_importance),
        )
        for table, frame in tables:
            if frame is None:
                continue
            path = Path(tmp) / f"{table}.parquet"
            frame.to_parquet(path, index=False)
            mlflow.log_artifact(str(path), artifact_path=ARTIFACT_DIR)
    if data_version is not None:
        mlflow.set_tag(DATA_VERSION_TAG, str(data_version))
    if model is not None:
        values = model.to_dict() if isinstance(model, ModelInfo) else dict(model)
        values.pop("source", None)
        for key, value in values.items():
            if value is not None:
                mlflow.set_tag(f"{MODEL_TAG_PREFIX}{key}", str(value))


def finished_runs(experiment: str, *, tracking_uri: str | None = None) -> list[str]:
    """Id завершённых (`FINISHED`) прогонов эксперимента по времени окончания (FR-R04)."""
    mlflow = _mlflow()
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    uri = tracking_uri or mlflow.get_tracking_uri()
    try:
        found = client.get_experiment_by_name(experiment)
        if found is None:
            raise MlflowAdapterError(f"эксперимент {experiment} не найден в {uri}")
        runs = client.search_runs(
            [found.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.end_time ASC"],
        )
    except mlflow.exceptions.MlflowException as exc:
        raise MlflowAdapterError(
            f"прогоны эксперимента {experiment} не получены: {exc.message}"
        ) from exc
    return [run.info.run_id for run in runs]
