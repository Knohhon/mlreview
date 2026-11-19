"""Поведение локального Web UI: docs/specs/web-ui.md."""

from pathlib import Path

import pytest

from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main
from mlreview.core.store import Store
from mlreview.core.workspace import init_workspace
from tests._factories import BINARY_SPEC_TEXT, binary_tables

TestClient = pytest.importorskip("fastapi.testclient").TestClient
pytest.importorskip("httpx")


def client_for(tmp_path):
    from mlreview.web.app import create_app  # noqa: PLC0415 — проверка до реализации

    workspace, _ = init_workspace(tmp_path)
    Store(workspace.store_path).initialize()
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    return TestClient(create_app(workspace), base_url="http://127.0.0.1:8765")


def headers(client):
    token = client.get("/api/v1/workspace").json()["csrf_token"]
    return {"X-CSRF-Token": token, "Origin": "http://127.0.0.1:8765"}


def add_run(tmp_path, name="run-1"):

    predictions, examples = binary_tables(n=20)
    save_run_files(tmp_path / "runs" / name, predictions, examples=examples, run_id=name)


def test_shell_and_api(tmp_path):
    """B1, E7: прямые ссылки отдают shell, неизвестный API — JSON 404."""
    with client_for(tmp_path) as client:
        assert client.get("/runs/source:run").status_code == 200
        assert client.get("/api/v1/missing").status_code == 404
        assert client.get("/api/v1/health").json()["status"] == "ok"


def test_sources_runs_contract(tmp_path):
    """B2, B8: локальный источник индексируется и проверяется без LLM."""
    add_run(tmp_path)
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/v1/sources",
            json={"name": "Локальные", "kind": "files", "path": str(tmp_path / "runs")},
            headers=headers(client),
        )
        assert response.status_code == 200, response.text
        assert wait_job(client, response.json()["job"])["status"] == "completed"
        listing = client.get("/api/v1/runs").json()
        assert listing["total"] == 1
        run = listing["items"][0]
        assert ":" in run["id"]
        checked = client.post(f"/api/v1/runs/{run['id']}/contract-check", headers=headers(client))
        assert checked.status_code == 200, checked.text
        assert checked.json()["status"] == "partial"


def test_csrf_and_origin(tmp_path):
    """E3: мутации требуют CSRF, Origin/Host проверяются."""
    with client_for(tmp_path) as client:
        assert client.post("/api/v1/journal", json={}).status_code == 403
        assert client.get("/api/v1/workspace", headers={"Host": "evil.test"}).status_code == 403
        assert (
            client.get("/api/v1/workspace", headers={"Origin": "http://evil.test"}).status_code
            == 403
        )


def test_task_conflict(tmp_path):
    """B6, E4: редактор валидирует YAML, сохраняет backup и ловит конфликт."""
    with client_for(tmp_path) as client:
        task = client.get("/api/v1/task").json()
        payload = {"text": BINARY_SPEC_TEXT + "# Изменение\n", "etag": task["etag"]}
        assert client.put("/api/v1/task", json=payload, headers=headers(client)).status_code == 200
        assert client.put("/api/v1/task", json=payload, headers=headers(client)).status_code == 409
        assert Path(str(tmp_path / "task.yaml") + ".bak").exists()


def test_missing_workspace(tmp_path):
    """E1: явный project без workspace не создаётся."""

    assert main(["ui", "--project", str(tmp_path), "--no-browser"]) == 1
    assert not (tmp_path / ".mlreview").exists()


def test_store_collision(tmp_path):
    """E5: другой источник не перезаписывает существующий run ID."""
    from mlreview.core.store import RunRecord, StoreError  # noqa: PLC0415

    store = Store(tmp_path / "store.sqlite")
    store.record_runs([RunRecord("same", "files:/a", "v1", "hash")])

    with pytest.raises(StoreError, match="источник"):
        store.record_runs([RunRecord("same", "files:/b", "v1", "hash")])
    assert store.run("same").source == "files:/a"


def wait_job(client, job):
    import time  # noqa: PLC0415

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current = client.get("/api/v1/jobs/" + job["id"]).json()
        if current["status"] not in ("queued", "running"):
            return current
        time.sleep(0.01)
    raise AssertionError("Задание не завершилось")


def configured_run(client, tmp_path):
    result = client.post(
        "/api/v1/sources",
        json={"name": "Файлы", "kind": "files", "path": str(tmp_path / "runs")},
        headers=headers(client),
    )
    assert result.status_code == 200, result.text
    assert wait_job(client, result.json()["job"])["status"] == "completed"
    return client.get("/api/v1/runs").json()["items"][0]["id"]


def test_review_persistence_and_cli_parity(tmp_path):
    """B3, B4, B8: те же факты API, сохранённый отчёт, смена spec делает stale."""
    from mlreview.api import review_runs  # noqa: PLC0415
    from tests._factories import binary_spec  # noqa: PLC0415

    add_run(tmp_path)
    workspace, _ = init_workspace(tmp_path)
    expected = review_runs(
        binary_spec(), str(tmp_path / "runs" / "run-1"), workspace=workspace
    ).to_dict()
    with client_for(tmp_path) as client:
        run_id = configured_run(client, tmp_path)
        detail = client.get("/api/v1/runs/" + run_id).json()
        assert detail["metrics"]["roc_auc"]["value"] is not None
        request = client.post("/api/v1/reviews", json={"run_id": run_id}, headers=headers(client))
        job = wait_job(client, request.json())
        assert job["status"] == "completed", job
        result = client.get("/api/v1/runs/" + run_id + "/review").json()
        assert result["status"] == "completed"
        assert not result["stale"]
        assert result["report"]["skipped"] == expected["skipped"]
        assert [f["facts"] for f in result["report"]["findings"]] == [
            f["facts"] for f in expected["findings"]
        ]
        assert not {"predictions", "examples"} & detail.keys()
    with client_for(tmp_path) as client:
        result = client.get("/api/v1/runs/" + run_id + "/review").json()
        assert result["status"] == "completed"
        (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT + "# новая версия\n", encoding="utf-8")
        assert client.get("/api/v1/runs/" + run_id + "/review").json()["stale"]
        assert client.get("/api/v1/runs/" + run_id).status_code == 200


def test_journal_shared_store(tmp_path):
    """B5: UI сохраняет безопасный текст заметки в общем Store."""
    with client_for(tmp_path) as client:
        result = client.post(
            "/api/v1/journal",
            json={"kind": "note", "text": "<script>alert(1)</script>"},
            headers=headers(client),
        )
        assert result.status_code == 200
        assert (
            Store(tmp_path / ".mlreview" / "store.sqlite").entry(result.json()["id"]).text
            == "<script>alert(1)</script>"
        )
        assert len(client.get("/api/v1/journal").json()) == 1


def test_interrupted_job(tmp_path):
    """B4: после перезапуска состояние running не сохраняется."""
    from mlreview.web.state import State  # noqa: PLC0415

    workspace, _ = init_workspace(tmp_path)
    state = State(workspace)
    state.save_job("unfinished", "running", {"kind": "review"})
    restarted = State(workspace)
    assert restarted.job("unfinished")["status"] == "interrupted"


def test_custom_metric_not_imported(tmp_path):
    """E6: Web не исполняет Python, указанный в task.yaml."""
    add_run(tmp_path)
    with client_for(tmp_path) as client:
        run_id = configured_run(client, tmp_path)
        text = (
            BINARY_SPEC_TEXT + "metrics:\n  primary: unsafe\n  custom:\n"
            "    - {name: unsafe, path: evil.py, function: score}\n"
        )
        (tmp_path / "task.yaml").write_text(text, encoding="utf-8")
        (tmp_path / "evil.py").write_text("raise RuntimeError('EXECUTED')", encoding="utf-8")
        response = client.post("/api/v1/reviews", json={"run_id": run_id}, headers=headers(client))
        assert response.status_code == 422
        assert response.json()["code"] == "custom_metric_disabled"


def test_external_source_symlink(tmp_path):
    """E3: добавленный каталог не разрешает читать symlink за его пределами."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "escape").symlink_to(outside, target_is_directory=True)
    with client_for(tmp_path) as client:
        run_id = configured_run(client, tmp_path)
        response = client.get("/api/v1/runs/" + run_id)
        assert response.status_code == 403


def test_source_unavailable_preserves_catalog(tmp_path):
    """E2: недоступность одного источника не уничтожает предыдущий индекс."""
    add_run(tmp_path)
    with client_for(tmp_path) as client:
        run_id = configured_run(client, tmp_path)
        source = client.get("/api/v1/sources").json()[0]
        (tmp_path / "runs").rename(tmp_path / "moved")
        job = client.post(
            "/api/v1/sources/" + source["id"] + "/refresh", headers=headers(client)
        ).json()
        assert wait_job(client, job)["status"] == "failed"
        assert client.get("/api/v1/runs").json()["total"] == 1
        assert client.get("/api/v1/runs/" + run_id).status_code == 404


def test_index_pagination_10k(tmp_path):
    """B2: сервер возвращает страницу из 10k metadata без чтения predictions."""
    with client_for(tmp_path) as client:
        state = client.app.state.service.state
        with state.connect() as db:
            import json  # noqa: PLC0415

            db.executemany(
                "INSERT INTO catalog VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        f"s:{i}",
                        "s",
                        str(i),
                        f"run-{i:05}",
                        "local",
                        "unknown",
                        "unused",
                        json.dumps({"id": f"s:{i}", "name": f"run-{i:05}"}),
                    )
                    for i in range(10_000)
                ],
            )
        result = client.get("/api/v1/runs?limit=25&offset=100&sort=-name").json()
        assert result["total"] == 10_000
        assert len(result["items"]) == 25
        assert result["items"][0]["name"] == "run-09899"
        assert client.get("/api/v1/runs?q=run-00001").json()["total"] == 1


def test_critical_dismiss_requires_reason(tmp_path):
    """B5: правило critical dismissal сохраняется и в HTTP."""
    from mlreview.core.findings import Finding, Report  # noqa: PLC0415

    with client_for(tmp_path) as client:
        store = client.app.state.service.store
        record = store.record_review(
            Report("run", [Finding("check", "critical", "Проблема", "FR-C05")], [])
        )
        finding_id = record.entries[0].id
        route = "/api/v1/findings/" + finding_id + "/status"
        assert (
            client.post(route, json={"status": "dismissed"}, headers=headers(client)).status_code
            == 422
        )
        assert (
            client.post(
                route,
                json={"status": "dismissed", "reason": "Проверено вручную"},
                headers=headers(client),
            ).status_code
            == 200
        )
        assert store.finding(finding_id).reason == "Проверено вручную"


def test_mlflow_local_discovery(tmp_path, mlflow_store):
    """B2, E2: настоящий локальный MLflow доступен через слой адаптеров."""
    predictions, examples = binary_tables(n=20)
    native = mlflow_store.run(
        artifacts={"predictions.parquet": predictions, "examples.parquet": examples}
    )
    with client_for(tmp_path) as client:
        response = client.post(
            "/api/v1/sources",
            json={
                "name": "MLflow",
                "kind": "mlflow",
                "tracking_uri": mlflow_store.uri,
                "experiments": [mlflow_store.experiment_id],
            },
            headers=headers(client),
        )
        assert response.status_code == 200, response.text
        assert wait_job(client, response.json()["job"])["status"] == "completed"
        run = client.get("/api/v1/runs").json()["items"][0]
        assert run["native_id"] == native
        assert client.get("/api/v1/runs/" + run["id"]).json()["contract"]["status"] == "partial"


def test_compare_requires_compatible_groups(tmp_path):
    """B7: сравнение разных версий сохраняет предупреждение ядра, baseline явный."""
    from mlreview.adapters.files_adapter import save_run_files  # noqa: PLC0415

    predictions, examples = binary_tables(n=20)
    for i in range(2):
        predictions, examples = binary_tables(n=20, shift=i * 0.5)
        save_run_files(
            tmp_path / "runs" / f"run-{i}",
            predictions,
            examples=examples,
            run_id=f"run-{i}",
            data_version=f"v{i}",
        )
    with client_for(tmp_path) as client:
        configured_run(client, tmp_path)
        ids = [run["id"] for run in client.get("/api/v1/runs").json()["items"]]
        result = client.post(
            "/api/v1/reviews", json={"run_id": ids[1], "baseline": ids[0]}, headers=headers(client)
        )
        assert wait_job(client, result.json())["status"] == "completed"
        report = client.get("/api/v1/runs/" + ids[1] + "/review").json()["report"]
        assert report["baseline_data_version"] == "v0"
        assert report["data_version"] == "v1"
        assert any(f["check"] == "process.data_version_mismatch" for f in report["findings"])


def test_copy_cli_groups(capsys):
    """B3: эквивалент CLI воспроизводит выбранные группы анализаторов."""

    with pytest.raises(SystemExit) as exc:
        main(["review", "--help"])
    assert exc.value.code == 0
    assert "--groups" in capsys.readouterr().out


@pytest.mark.parametrize("native", ["nested/id", "nested/review", "percent%2Fid"])
def test_encoded_native_id(tmp_path, native):
    """B1, B2: URL-encoded native ID со слешем остаётся ID, а не путём."""
    from urllib.parse import quote  # noqa: PLC0415

    predictions, examples = binary_tables(n=20)
    save_run_files(tmp_path / "runs" / "safe", predictions, examples=examples, run_id=native)
    with client_for(tmp_path) as client:
        run_id = configured_run(client, tmp_path)
        assert client.get("/api/v1/runs/" + quote(run_id, safe="")).status_code == 200


def test_explicit_project_precedence(tmp_path, monkeypatch):
    """B1: явный project приоритетнее MLREVIEW_HOME; поиск из подкаталога работает."""
    from mlreview.cli.ui import resolve_workspace  # noqa: PLC0415

    chosen, _ = init_workspace(tmp_path / "chosen")
    other, _ = init_workspace(tmp_path / "other")
    monkeypatch.setenv("MLREVIEW_HOME", str(other.root))
    assert resolve_workspace(tmp_path / "chosen").root == chosen.root
    monkeypatch.delenv("MLREVIEW_HOME")
    nested = tmp_path / "chosen" / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert resolve_workspace(None).root == chosen.root


def test_mlflow_failure_does_not_block_files(tmp_path, monkeypatch):
    """E2: ошибка соединения MLflow возвращается JSON, локальные данные доступны."""
    import mlreview.adapters.web_discovery as discovery  # noqa: PLC0415

    def unavailable():
        raise RuntimeError("MLflow offline")

    monkeypatch.setattr(discovery, "_mlflow", unavailable)
    add_run(tmp_path)
    with client_for(tmp_path) as client:
        configured_run(client, tmp_path)
        result = client.post(
            "/api/v1/sources/test",
            json={"name": "remote", "kind": "mlflow", "tracking_uri": "http://tracker.invalid"},
            headers=headers(client),
        )
        assert result.status_code == 422
        assert "MLflow" in result.json()["detail"]
        assert client.get("/api/v1/runs").json()["total"] == 1
