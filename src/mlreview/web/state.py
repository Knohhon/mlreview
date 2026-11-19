"""Восстанавливаемый индекс, задания и последние отчёты Web UI."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from mlreview.core.workspace import Workspace


class WebError(ValueError):
    """Ошибка web-операции с HTTP-статусом."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def digest(text: str) -> str:
    """Отпечаток текста или сериализованного контекста."""
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    """Атомарная замена файла в том же каталоге."""
    temporary = path.with_name(path.name + ".web-tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def safe_path(value: str, roots: list[Path]) -> Path:
    """Путь внутри явно разрешённого корня, включая проверку symlink."""
    path = Path(value).resolve()
    if any(root.absolute() != root.resolve() for root in roots):
        raise WebError("Разрешённый корень заменён символической ссылкой", 403, "unsafe_path")
    if not any(path == root.resolve() or root.resolve() in path.parents for root in roots):
        raise WebError("Путь вне разрешённых каталогов", 403, "unsafe_path")
    return path


def redact(value: Any) -> Any:
    """Редактирование секретных полей вложенных metadata."""
    if isinstance(value, dict):
        return {
            key: "[скрыто]"
            if any(
                part in key.lower()
                for part in ("token", "secret", "password", "api_key", "authorization")
            )
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    return value


class State:
    """Состояние UI отдельно от предметных findings/journal."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.lock = threading.RLock()
        self.path = workspace.root / "web.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS catalog (
                    id TEXT PRIMARY KEY, source TEXT, native TEXT, name TEXT,
                    experiment TEXT, status TEXT, ref TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT, payload TEXT);
                CREATE TABLE IF NOT EXISTS reports (
                    context TEXT PRIMARY KEY, run TEXT, payload TEXT);
            """)
            db.execute("UPDATE jobs SET status='interrupted' WHERE status IN ('queued','running')")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def config(self) -> dict:
        return self.workspace.config().get("web", {})

    def update_config(self, changes: dict) -> dict:
        with self.lock:
            config = self.workspace.config()
            config["web"] = {**config.get("web", {}), **changes}
            atomic_write(
                self.workspace.config_path,
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            )
            return config["web"]

    def source(self, source_id: str) -> dict:
        for source in self.config().get("sources", []):
            if source["id"] == source_id:
                return source
        raise WebError("Источник не найден", 404, "source_not_found")

    def run(self, run_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM catalog WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise WebError("Прогон не найден; обновите источник", 404, "run_not_found")
        return json.loads(row[0])

    def list_runs(  # noqa: PLR0913 — фильтры списка
        self,
        *,
        q: str = "",
        source: str = "",
        status: str = "",
        experiment: str = "",
        sort: str = "name",
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        clauses, values = [], []
        if q:
            clauses.append("(instr(lower(name), lower(?))>0 OR instr(lower(id), lower(?))>0)")
            values += [q, q]
        for column, value in (("source", source), ("status", status), ("experiment", experiment)):
            if value:
                clauses.append(f"{column}=?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        column = sort.lstrip("-")
        column = column if column in ("name", "id", "source", "status", "experiment") else "name"
        direction = "DESC" if sort.startswith("-") else "ASC"
        with self.connect() as db:
            total = db.execute("SELECT count(*) FROM catalog" + where, values).fetchone()[0]
            rows = db.execute(
                f"SELECT payload FROM catalog{where} ORDER BY {column} {direction}, id "
                "LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
        return {
            "items": [json.loads(row[0]) for row in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def put_run(self, run: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO catalog VALUES (?,?,?,?,?,?,?,?)",
                (
                    run["id"],
                    run["source"],
                    run["native_id"],
                    run["name"],
                    run.get("experiment", ""),
                    run.get("status", "unknown"),
                    run["ref"],
                    json.dumps(run, ensure_ascii=False),
                ),
            )

    def job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT status,payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise WebError("Задание не найдено", 404, "job_not_found")
        return {**json.loads(row[1]), "id": job_id, "status": row[0]}

    def save_job(self, job_id: str, status: str, payload: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?,?,?)",
                (job_id, status, json.dumps(payload, ensure_ascii=False)),
            )
