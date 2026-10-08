"""Общие фикстуры тестов."""

from __future__ import annotations

import shutil
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_NETWORK_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})


def _blocked(*_args, **_kwargs):
    raise RuntimeError("сеть в тестах запрещена: тесты должны проходить офлайн (NFR-03)")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Запрещает TCP/UDP-соединения; Unix-сокеты остаются доступны."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(self, address):
        if self.family in _NETWORK_FAMILIES:
            _blocked()
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in _NETWORK_FAMILIES:
            _blocked()
        return original_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(scope="session")
def _mlflow_db_template(tmp_path_factory):
    """Пустая sqlite-база MLflow: миграции схемы выполняются один раз за сессию."""
    mlflow = pytest.importorskip("mlflow")
    path = tmp_path_factory.mktemp("mlflow-template") / "mlflow.db"
    mlflow.MlflowClient(tracking_uri=f"sqlite:///{path}").search_experiments()
    return path


@dataclass
class MlflowStore:
    """Локальный трекер MLflow во временном каталоге для прогонов тестов."""

    uri: str
    root: Path
    client: Any
    experiment_id: str

    def run(
        self,
        *,
        artifacts: dict[str, pd.DataFrame | bytes] | None = None,
        params: dict[str, str] | None = None,
        metrics: dict[str, list[float]] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Создаёт завершённый прогон; артефакты кладутся в `mlreview/<имя>`."""
        run_id = self.client.create_run(self.experiment_id).info.run_id
        for key, value in (params or {}).items():
            self.client.log_param(run_id, key, value)
        for name, values in (metrics or {}).items():
            for step, value in enumerate(values):
                self.client.log_metric(run_id, name, value, step=step)
        for key, value in (tags or {}).items():
            self.client.set_tag(run_id, key, value)
        source = self.root / "to-log" / run_id
        source.mkdir(parents=True)
        for name, content in (artifacts or {}).items():
            path = source / name
            if isinstance(content, bytes):
                path.write_bytes(content)
            elif name.endswith(".csv"):
                content.to_csv(path, index=False)
            else:
                content.to_parquet(path, index=False)
            self.client.log_artifact(run_id, str(path), artifact_path="mlreview")
        self.client.set_terminated(run_id)
        return run_id


@pytest.fixture
def mlflow_store(tmp_path, _mlflow_db_template):
    """Трекер MLflow на копии пустой базы; артефакты — во временном каталоге теста."""
    mlflow = pytest.importorskip("mlflow")
    db = tmp_path / "mlflow.db"
    shutil.copy(_mlflow_db_template, db)
    uri = f"sqlite:///{db}"
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        "tests", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return MlflowStore(uri=uri, root=tmp_path, client=client, experiment_id=experiment_id)
