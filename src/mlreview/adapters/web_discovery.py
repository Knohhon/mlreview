"""Постраничное обнаружение MLflow metadata без загрузки predictions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from mlreview.adapters.errors import AdapterError
from mlreview.adapters.mlflow_adapter import _mlflow


@contextmanager
def tracking_errors():
    """Недоступность трекера не превращается в необработанную ошибку HTTP."""
    try:
        yield
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"MLflow недоступен: {exc}") from exc


def experiments(uri: str) -> list[dict]:
    """Доступные эксперименты явно выбранного трекера."""
    with tracking_errors():
        client = _mlflow().MlflowClient(tracking_uri=uri)
        result = []
        token = None
        while True:
            page = client.search_experiments(max_results=100, page_token=token)
            result.extend({"id": item.experiment_id, "name": item.name} for item in page)
            token = page.token
            if not token:
                return result


def discover(uri: str, experiment_ids: list[str]) -> Iterator[dict]:
    """Metadata прогона; импорт MLflow остаётся в адаптере."""
    with tracking_errors():
        client = _mlflow().MlflowClient(tracking_uri=uri)
        token = None
        while True:
            page = client.search_runs(experiment_ids, max_results=100, page_token=token)
            for run in page:
                yield {
                    "native_id": run.info.run_id,
                    "name": run.data.tags.get("mlflow.runName", run.info.run_id),
                    "experiment": run.info.experiment_id,
                    "status": run.info.status.lower(),
                    "parameters": dict(run.data.params),
                    "metrics": dict(run.data.metrics),
                    "started_at": run.info.start_time,
                    "data_version": run.data.tags.get("mlreview.data_version"),
                    "ref": run.info.run_id,
                }
            token = page.token
            if not token:
                return
