"""Сервисы Web UI поверх общего API: источники, контракты, latest и задания."""

from __future__ import annotations

import base64
import json
import re
import shlex
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from mlreview import __version__
from mlreview.adapters.errors import AdapterError
from mlreview.adapters.files_adapter import run_id_of
from mlreview.adapters.web_discovery import discover, experiments
from mlreview.api import review_runs
from mlreview.core.store import Store
from mlreview.core.task_spec import load_task_spec, parse_task_spec
from mlreview.metrics.library import spec_metrics
from mlreview.web.state import State, WebError, atomic_write, digest, redact, safe_path

MAX_PENDING = 16


def prohibit_custom(spec):
    """Web не загружает произвольный Python из спецификации."""
    if spec.metrics and spec.metrics.custom:
        raise WebError(
            "Web P0 не исполняет пользовательские Python-метрики; используйте CLI",
            422,
            "custom_metric_disabled",
        )
    return spec


def namespaced(source_id: str, native_id: str) -> str:
    """Нативный ID экранируется внутри namespace, включая зарезервированные суффиксы."""
    native = str(native_id)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", native):
        native = "~" + base64.urlsafe_b64encode(native.encode()).decode().rstrip("=")
    return source_id + ":" + native


def timestamp() -> str:
    """UTC timestamp для состояния операций."""
    return datetime.now(timezone.utc).isoformat()


class Service:
    """Один сервер, одна ограниченная очередь, общее предметное хранилище."""

    def __init__(self, state: State):
        self.state = state
        self.workspace = state.workspace
        self.root = self.workspace.root.resolve().parent
        self.store = Store(self.workspace.store_path)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlreview-web")
        self.pending = 0
        package = Path(__file__).parent.parent
        self.analysis_version = digest(
            "".join(
                path.read_text(encoding="utf-8")
                for directory in ("analyzers", "metrics", "playbooks", "core")
                for path in sorted((package / directory).rglob("*"))
                if path.suffix in {".py", ".yaml"}
            )
        )

    def close(self) -> None:
        """Завершает уже принятые задания до остановки сервера."""
        self.worker.shutdown(wait=True)

    def task_path(self) -> Path:
        value = self.state.config().get("task_path", str(self.root / "task.yaml"))
        # Разрешён именно выбранный файл, подмена его или родителя symlink запрещена.
        path = Path(value)
        if path.is_absolute() and path != path.resolve():
            raise WebError("Путь спецификации изменён через symlink", 403, "unsafe_path")
        return path.resolve()

    def task(self) -> dict:
        path = self.task_path()
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        result = {"path": str(path), "text": text, "etag": digest(text), "exists": path.is_file()}
        try:
            result["parsed"] = parse_task_spec(text, base_dir=path.parent).model_dump()
            result["valid"] = True
        except ValueError as exc:
            result.update(valid=False, error=str(exc))
        return result

    def spec(self):
        return prohibit_custom(load_task_spec(self.task_path()))

    def save_task(self, text: str, etag: str) -> dict:
        with self.state.lock:
            current = self.task()
            if etag != current["etag"]:
                raise WebError(
                    "Файл изменился. Перезагрузите и сравните изменения", 409, "task_conflict"
                )
            parse_task_spec(text, base_dir=self.task_path().parent)
            if current["exists"]:
                atomic_write(Path(current["path"] + ".bak"), current["text"])
            atomic_write(self.task_path(), text)
            return self.task()

    def select_task(self, value: str) -> dict:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        if path.suffix not in (".yaml", ".yml") or not path.parent.resolve().is_dir():
            raise WebError("Выберите YAML-файл в существующем каталоге")
        # Выбор конкретного файла — явное разрешение, а не произвольный file API.
        if path.is_symlink() or not path.name:
            raise WebError("Символическая ссылка недопустима", 403, "unsafe_path")
        self.state.update_config({"task_path": str(path.resolve())})
        return self.task()

    def test_source(self, source: dict) -> dict:
        if source["kind"] == "files":
            path = Path(source["path"])
            if not path.is_absolute():
                path = self.root / path
            if not path.is_dir():
                raise WebError("Каталог источника не найден")
            if path.is_symlink():
                raise WebError("Выберите реальный каталог источника", 403, "unsafe_path")
            return {"path": str(path.resolve()), "available": True}
        uri = source.get("tracking_uri", "")
        parsed = urlsplit(uri)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise WebError(
                "URI не должен содержать секреты или query; используйте окружение сервера"
            )
        if parsed.scheme not in ("http", "https", "file", "sqlite"):
            raise WebError("Допустимы http, https, file и sqlite URI")
        return {"available": True, "experiments": experiments(uri)}

    def add_source(self, source: dict) -> dict:
        tested = self.test_source(source)
        record = {**source, "id": str(uuid.uuid4()), "available": True}
        if record["kind"] == "files":
            record["path"] = tested["path"]
        with self.state.lock:
            sources = self.state.config().get("sources", [])
            if any(
                (item["kind"], item.get("path"), item.get("tracking_uri"))
                == (record["kind"], record.get("path"), record.get("tracking_uri"))
                for item in sources
            ):
                raise WebError("Источник уже подключён")
            self.state.update_config({"sources": [*sources, record]})
        return record

    def edit_source(self, source_id: str, source: dict) -> dict:
        tested = self.test_source(source)
        with self.state.lock:
            if self.pending:
                raise WebError("Дождитесь завершения заданий перед изменением источника", 409)
            self.state.source(source_id)
            record = {**source, "id": source_id, "available": True}
            if source["kind"] == "files":
                record["path"] = tested["path"]
            sources = self.state.config().get("sources", [])
            self.state.update_config(
                {"sources": [record if item["id"] == source_id else item for item in sources]}
            )
            with self.state.connect() as db:
                db.execute("DELETE FROM catalog WHERE source=?", (source_id,))
        return record

    def source_status(self, source_id: str, **values) -> None:
        with self.state.lock:
            sources = self.state.config().get("sources", [])
            self.state.update_config(
                {
                    "sources": [
                        dict(item, **values) if item["id"] == source_id else item
                        for item in sources
                    ]
                }
            )

    def _scan_files(self, source: dict) -> set[str]:
        source_id = source["id"]
        seen = set()
        root = Path(source["path"])
        if not root.is_dir():
            raise WebError("Каталог источника недоступен", 503, "source_unavailable")
        safe_path(str(root), [root])
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            try:
                safe_path(str(directory), [root])
                for candidate in directory.iterdir():
                    if candidate.is_symlink():
                        safe_path(str(candidate), [root])
                native = run_id_of(directory)
                if namespaced(source_id, native) in seen:
                    raise WebError("Два каталога источника имеют одинаковый native ID")
                record = {
                    "native_id": native,
                    "name": directory.name,
                    "ref": str(directory.resolve()),
                    "status": "finished" if (directory / "run.yaml").exists() else "unknown",
                    "experiment": source["name"],
                }
                # Только маленькие metadata; таблицы читаются при открытии run.
                for stem in ("run", "config"):
                    for suffix in ("json", "yaml", "yml"):
                        path = directory / f"{stem}.{suffix}"
                        if path.is_file():
                            safe_path(str(path), [root])
                meta_path = next(
                    (
                        directory / name
                        for name in ("run.yaml", "run.json")
                        if (directory / name).is_file()
                    ),
                    None,
                )
                if meta_path:
                    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                    record["data_version"] = meta.get("data_version")
                    record["model"] = meta.get("model")
                self._index(source, record)
                seen.add(namespaced(source_id, native))
            except (ValueError, OSError, AdapterError) as exc:
                record = {
                    "native_id": directory.name,
                    "name": directory.name,
                    "ref": str(directory),
                    "status": "invalid",
                    "error": str(exc),
                    "experiment": source["name"],
                }
                self._index(source, record)
                seen.add(namespaced(source_id, directory.name))
        return seen

    def scan(self, source_id: str) -> dict:
        source = self.state.source(source_id)
        seen = set()
        if source["kind"] == "files":
            seen = self._scan_files(source)
        else:
            for record in discover(source["tracking_uri"], source.get("experiments", [])):
                self._index(source, record)
                seen.add(namespaced(source_id, record["native_id"]))
        with self.state.connect() as db:
            old = db.execute("SELECT id FROM catalog WHERE source=?", (source_id,)).fetchall()
            for row in old:
                if row[0] not in seen:
                    db.execute("DELETE FROM catalog WHERE id=?", (row[0],))
        return {"count": len(seen), "scanned_at": timestamp()}

    def _index(self, source: dict, record: dict) -> None:
        self.state.put_run(
            {
                **record,
                "source": source["id"],
                "source_name": source["name"],
                "id": namespaced(source["id"], record["native_id"]),
            }
        )

    def resolve(self, run_id: str) -> tuple[dict, dict]:
        run = self.state.run(run_id)
        source = self.state.source(run["source"])
        if source["kind"] == "files":
            directory = safe_path(run["ref"], [Path(source["path"])])
            if not directory.is_dir():
                raise WebError("Прогон недоступен в источнике", 404, "run_not_found")
            for path in directory.iterdir():
                if path.is_symlink():
                    safe_path(str(path), [Path(source["path"])])
        return run, source

    def contract(self, run_id: str, spec=None):
        from mlreview.api import load_run  # noqa: PLC0415 — общий API

        run, source = self.resolve(run_id)
        return load_run(spec or self.spec(), run["ref"], tracking_uri=source.get("tracking_uri"))

    def detail(self, run_id: str) -> dict:
        run, source = self.resolve(run_id)
        try:
            task_hash = self.task()["etag"]
            spec = self.spec()
            contract = self.contract(run_id, spec)
            from mlreview.analyzers.evaluation import prepare  # noqa: PLC0415

            available_metrics = {}
            # Значения вычисляются ядром; исходные labels/predictions не выходят в JSON.
            for metric in spec_metrics(spec):
                try:
                    available_metrics[metric.name] = {
                        "value": metric.metric(prepare(spec.task, contract.predictions)),
                        "higher_is_better": metric.higher_is_better,
                    }
                except (ValueError, TypeError):
                    available_metrics[metric.name] = {
                        "value": None,
                        "higher_is_better": metric.higher_is_better,
                    }
            run.update(
                metrics=available_metrics,
                metrics_task_hash=task_hash,
                data_version=contract.data_version,
                model=None if contract.model is None else contract.model.to_dict(),
            )
            self.state.put_run(run)
            return redact(
                {
                    **run,
                    "parameters": contract.config,
                    "metrics": available_metrics,
                    "data_version": contract.data_version,
                    "data_hash": contract.data_hash,
                    "model": None if contract.model is None else contract.model.to_dict(),
                    "contract": {
                        "status": "partial" if contract.gaps else "valid",
                        "gaps": [asdict(gap) for gap in contract.gaps],
                    },
                    "curves": []
                    if contract.curves is None
                    else contract.curves[["name", "step", "value"]].to_dict(orient="records"),
                    "artifacts": (
                        "В P0 доступны metadata контракта; "
                        "скачивание сырых артефактов не поддерживается"
                    ),
                    "source_kind": source["kind"],
                }
            )
        except (ValueError, OSError, AdapterError) as exc:
            return {**redact(run), "contract": {"status": "invalid", "error": str(exc)}}

    def context(self, request: dict) -> str:
        return digest(json.dumps(request, sort_keys=True))

    def signature(self, request: dict, task_hash: str | None = None) -> str:
        parts: list[Any] = [
            task_hash or self.task()["etag"],
            __version__,
            self.analysis_version,
            request,
        ]
        for run_id in [
            request["run_id"],
            request.get("baseline"),
            *request.get("seeds", []),
            *request.get("baseline_seeds", []),
        ]:
            if not run_id:
                continue
            run, source = self.resolve(run_id)
            parts.append([run["ref"], source.get("tracking_uri")])
            if source["kind"] == "files":
                root = Path(run["ref"])
                parts.append(
                    [
                        (item.name, item.stat().st_mtime_ns, item.stat().st_size)
                        for item in sorted(root.iterdir())
                        if item.is_file()
                    ]
                )
            else:
                # Контракт нужен для достоверной проверки свежести удалённых данных.
                contract = self.contract(run_id)
                parts.append([contract.data_hash, contract.config, contract.data_version])
        return digest(json.dumps(parts, sort_keys=True, default=str))

    def review(self, request: dict) -> dict:
        run, source = self.resolve(request["run_id"])
        baseline = request.get("baseline")
        related = [baseline, *request.get("seeds", []), *request.get("baseline_seeds", [])]
        refs = {}
        for run_id in related:
            if run_id:
                item, other_source = self.resolve(run_id)
                if other_source["kind"] == "mlflow" and other_source.get(
                    "tracking_uri"
                ) != source.get("tracking_uri"):
                    raise WebError("Сравнение разных MLflow tracking URI в P0 не поддерживается")
                refs[run_id] = item["ref"]
        task_snapshot = self.task()
        spec = prohibit_custom(
            parse_task_spec(task_snapshot["text"], base_dir=self.task_path().parent)
        )
        signature = self.signature(request, task_snapshot["etag"])
        result = review_runs(
            spec,
            run["ref"],
            refs.get(baseline),
            seeds=[refs[item] for item in request.get("seeds", [])],
            baseline_seeds=[refs[item] for item in request.get("baseline_seeds", [])],
            groups=request.get("groups")
            or ("data", "segments", "metrics", "model", "curves", "process"),
            tracking_uri=source.get("tracking_uri"),
            workspace=self.workspace,
        )
        payload = {
            "request": request,
            "signature": signature,
            "created_at": timestamp(),
            "analysis_version": self.analysis_version,
            "task_hash": task_snapshot["etag"],
            "report": result.to_dict(),
            "cli": shlex.join(
                [
                    "mlreview",
                    "review",
                    "--spec",
                    str(self.task_path()),
                    run["ref"],
                    *[refs[item] for item in request.get("seeds", [])],
                    *(
                        [
                            "--baseline",
                            refs[baseline],
                            *[refs[item] for item in request.get("baseline_seeds", [])],
                        ]
                        if baseline
                        else []
                    ),
                    *(
                        ["--tracking-uri", source["tracking_uri"]]
                        if source.get("tracking_uri")
                        else []
                    ),
                    *(["--groups", *request["groups"]] if request.get("groups") else []),
                ]
            ),
        }
        with self.state.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO reports VALUES (?,?,?)",
                (
                    self.context(request),
                    request["run_id"],
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
        return {"run_id": request["run_id"], "context": self.context(request)}

    def latest(self, run_id: str, context: str | None = None) -> dict:
        with self.state.connect() as db:
            rows = db.execute(
                "SELECT context,payload FROM reports WHERE run=? ORDER BY rowid DESC", (run_id,)
            ).fetchall()
        if context:
            rows = [row for row in rows if row[0] == context]
        if not rows:
            return {
                "status": "not_run",
                "explanation": "Полный отчёт отсутствует. Находки CLI доступны в общем списке",
            }
        result = json.loads(rows[0][1])
        try:
            result["stale"] = result["signature"] != self.signature(result["request"])
        except (ValueError, OSError):
            result["stale"] = True
        for finding in result["report"].get("findings", []):
            if finding.get("id"):
                current = self.store.finding(finding["id"])
                finding.update(status=current.status, reason=current.reason)
        return redact(
            {
                **result,
                "context": rows[0][0],
                "contexts": [row[0] for row in rows],
                "status": "completed",
            }
        )

    def submit(self, kind: str, request: dict) -> dict:
        with self.state.lock:
            if self.pending >= MAX_PENDING:
                raise WebError("Очередь заполнена, повторите позже", 429, "queue_full")
            self.pending += 1
            job_id = str(uuid.uuid4())
            payload = {"kind": kind, "request": request, "created_at": timestamp()}
            self.state.save_job(job_id, "queued", payload)

        def execute():
            self.state.save_job(job_id, "running", payload)
            try:
                result = (
                    self.review(request) if kind == "review" else self.scan(request["source_id"])
                )
                if kind == "scan":
                    self.source_status(
                        request["source_id"],
                        available=True,
                        last_scan=result["scanned_at"],
                        count=result["count"],
                        error=None,
                    )
                self.state.save_job(job_id, "completed", {**payload, "result": result})
            except Exception as exc:  # ошибка операции не уничтожает worker и другие источники
                self.state.save_job(job_id, "failed", {**payload, "error": str(exc)})
                if kind == "scan":
                    # Терминальный job сохранён даже при ошибке записи config.
                    with suppress(Exception):
                        self.source_status(request["source_id"], available=False, error=str(exc))
            finally:
                with self.state.lock:
                    self.pending -= 1

        self.worker.submit(execute)
        return self.state.job(job_id)
