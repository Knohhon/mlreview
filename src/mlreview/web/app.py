"""HTTP API v1 и bundled SPA локального интерфейса."""

from __future__ import annotations

import math
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mlreview import __version__
from mlreview.adapters.errors import AdapterError
from mlreview.core.store import StoreError
from mlreview.core.task_spec import parse_task_spec
from mlreview.core.workspace import Workspace, WorkspaceError
from mlreview.web.service import Service
from mlreview.web.state import State, WebError, redact

STATIC = Path(__file__).parent / "static"
FRONTEND_ROUTES = {
    "overview",
    "setup",
    "runs",
    "findings",
    "journal",
    "task",
    "sources",
    "settings",
    "not-found",
}


class DTO(BaseModel):
    """HTTP-ввод не принимает неописанные поля."""

    model_config = ConfigDict(extra="forbid")


class SourceInput(DTO):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["files", "mlflow"]
    path: str = ""
    tracking_uri: str = ""
    experiments: list[str] = []


class TaskInput(DTO):
    text: str = Field(max_length=500_000)
    etag: str = ""


class PathInput(DTO):
    path: str = Field(min_length=1, max_length=4096)


class ReviewInput(DTO):
    run_id: str
    baseline: str | None = None
    seeds: list[str] = Field(default_factory=list, max_length=20)
    baseline_seeds: list[str] = Field(default_factory=list, max_length=20)
    groups: list[Literal["data", "segments", "metrics", "model", "curves", "process"]] = []


class StatusInput(DTO):
    status: Literal["new", "accepted", "dismissed", "fixed"]
    reason: str | None = Field(default=None, max_length=10_000)


class JournalInput(DTO):
    kind: Literal["hypothesis", "decision", "note"]
    text: str = Field(min_length=1, max_length=100_000)
    runs: list[str] = []
    findings: list[str] = []
    hypothesis: str | None = None
    outcome: Literal["accept", "reject"] | None = None


def finite(value):
    """NaN/Infinity отсутствуют в JSON, отсутствие значения сохраняется как null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def response(value, status=200):
    return JSONResponse(finite(redact(value)), status_code=status)


def create_app(workspace: Workspace) -> FastAPI:  # noqa: PLR0915 — регистрация маршрутов
    """Приложение для одного явно выбранного workspace."""
    service = Service(State(workspace))
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        service.close()

    app = FastAPI(
        title="MLReview Local API", version="1", lifespan=lifespan, docs_url=None, redoc_url=None
    )
    app.state.service = service

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        host = request.headers.get("host", "")
        try:
            parsed = urlsplit("http://" + host)
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
        ):
            return response(
                {"code": "invalid_host", "detail": "Разрешён только loopback Host"}, 403
            )
        origin = request.headers.get("origin")
        if origin and origin != "http://" + host:
            return response(
                {"code": "invalid_origin", "detail": "Origin не совпадает с сервером"}, 403
            )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return response({"code": "cross_site", "detail": "Межсайтовый запрос запрещён"}, 403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if not secrets.compare_digest(
                request.headers.get("x-csrf-token", "").encode(), token.encode()
            ):
                return response(
                    {"code": "csrf", "detail": "Обновите страницу: неверный CSRF token"}, 403
                )
            if not origin:
                return response({"code": "origin_required", "detail": "Требуется Origin"}, 403)
        result = await call_next(request)
        result.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        result.headers["X-Content-Type-Options"] = "nosniff"
        result.headers["Referrer-Policy"] = "no-referrer"
        result.headers["Cache-Control"] = "no-store"
        return result

    @app.exception_handler(WebError)
    async def web_error(_request, exc):
        return response({"code": exc.code, "detail": str(exc)}, exc.status)

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    @app.exception_handler(WorkspaceError)
    @app.exception_handler(StoreError)
    @app.exception_handler(AdapterError)
    async def domain_error(_request, exc):
        return response({"code": "operation_failed", "detail": str(exc)}, 422)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        return response(
            {
                "code": "validation_error",
                "detail": "Некорректные параметры запроса",
                "fields": [item["loc"] for item in exc.errors()],
            },
            422,
        )

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/workspace")
    def workspace_info():
        task = service.task()
        return {
            "root": str(service.root),
            "csrf_token": token,
            "version": __version__,
            "ready": task["valid"] and bool(service.state.config().get("sources")),
            "task_valid": task["valid"],
        }

    @app.get("/api/v1/overview")
    def overview():
        findings = service.store.findings(min_severity="info")
        with service.state.connect() as db:
            reviewed = db.execute("SELECT count(DISTINCT run) FROM reports").fetchone()[0]
        runs = service.state.list_runs(limit=8)
        return response(
            {
                "runs": runs["total"],
                "reviewed": reviewed,
                "open_critical": sum(
                    item.status == "new" and item.severity == "critical" for item in findings
                ),
                "open_high": sum(
                    item.status == "new" and item.severity == "high" for item in findings
                ),
                "recent_runs": runs["items"],
                "findings": [item.to_dict() for item in findings if item.status == "new"][:8],
                "decisions": [entry.to_dict() for entry in service.store.entries("decision")][-5:],
            }
        )

    @app.get("/api/v1/sources")
    def sources():
        return response(service.state.config().get("sources", []))

    @app.post("/api/v1/sources/test")
    def source_test(source: SourceInput):
        return response(service.test_source(source.model_dump()))

    @app.post("/api/v1/sources")
    def source_add(source: SourceInput):
        result = service.add_source(source.model_dump())
        job = service.submit("scan", {"source_id": result["id"]})
        return response({**result, "job": job})

    @app.put("/api/v1/sources/{source_id}")
    def source_edit(source_id: str, source: SourceInput):
        result = service.edit_source(source_id, source.model_dump())
        return response({**result, "job": service.submit("scan", {"source_id": source_id})})

    @app.post("/api/v1/sources/{source_id}/refresh")
    def source_refresh(source_id: str):
        service.state.source(source_id)
        return response(service.submit("scan", {"source_id": source_id}))

    @app.delete("/api/v1/sources/{source_id}")
    def source_remove(source_id: str):
        with service.state.lock:
            if service.pending:
                raise WebError("Дождитесь завершения заданий", 409)
            sources = service.state.config().get("sources", [])
            service.state.update_config(
                {"sources": [item for item in sources if item["id"] != source_id]}
            )
            with service.state.connect() as db:
                db.execute("DELETE FROM catalog WHERE source=?", (source_id,))
        return {"removed": True}

    @app.get("/api/v1/runs")
    def runs(  # noqa: PLR0913 — фильтры HTTP
        *,
        q: str = "",
        source: str = "",
        status: str = "",
        experiment: str = "",
        sort: str = "name",
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
    ):
        page = service.state.list_runs(
            q=q,
            source=source,
            status=status,
            experiment=experiment,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        task_hash = service.task()["etag"]
        for run in page["items"]:
            if run.get("metrics_task_hash") and run["metrics_task_hash"] != task_hash:
                run.pop("metrics", None)
        return response(page)

    @app.post("/api/v1/runs/{run_id}/contract-check")
    def contract_check(run_id: str):
        return response(service.detail(run_id)["contract"])

    @app.post("/api/v1/reviews")
    def review(request: ReviewInput):
        service.spec()
        for run_id in [request.run_id, request.baseline, *request.seeds, *request.baseline_seeds]:
            if run_id:
                service.resolve(run_id)
        return response(service.submit("review", request.model_dump()))

    @app.get("/api/v1/runs/{run_id}/review")
    def latest_review(run_id: str, context: str | None = None):
        return response(service.latest(run_id, context))

    @app.get("/api/v1/runs/{run_id}")
    def run_detail(run_id: str):
        return response(service.detail(run_id))

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str):
        return response(service.state.job(job_id))

    @app.get("/api/v1/jobs")
    def jobs():
        with service.state.connect() as db:
            ids = db.execute("SELECT id FROM jobs ORDER BY rowid DESC LIMIT 30").fetchall()
        return response([service.state.job(row[0]) for row in ids])

    @app.get("/api/v1/findings")
    def findings(  # noqa: PLR0913 — фильтры HTTP
        *,
        status: str = "",
        severity: str = "",
        run: str = "",
        q: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
    ):
        native = service.state.run(run)["native_id"] if run else None
        items = [
            item.to_dict()
            for item in service.store.findings(min_severity="info")
            if (not status or item.status in status.split(","))
            and (not severity or item.severity in severity.split(","))
            and (not native or any(occ.run_id == native for occ in item.occurrences))
            and (not q or q.lower() in item.message.lower())
        ]
        return response(
            {
                "items": items[offset : offset + limit],
                "total": len(items),
                "offset": offset,
                "limit": limit,
            }
        )

    @app.get("/api/v1/findings/{finding_id}")
    def finding(finding_id: str):
        item = service.store.finding(finding_id)
        with service.state.connect() as db:
            ids = db.execute(
                "SELECT id FROM catalog WHERE native=?", (item.last.run_id,)
            ).fetchall()
        return response(
            {
                **item.to_dict(),
                "history": [asdict(event) for event in item.history],
                "run_ids": [row[0] for row in ids],
            }
        )

    @app.post("/api/v1/findings/{finding_id}/status")
    def finding_status(finding_id: str, value: StatusInput):
        service.store.set_status(finding_id, value.status, reason=value.reason)
        return response(service.store.finding(finding_id).to_dict())

    @app.get("/api/v1/journal")
    def journal(q: str = "", kind: str = ""):
        entries = service.store.entries(kind or None)
        if q:
            matched = {hit.entry.id for hit in service.store.search(q).hits}
            entries = [item for item in entries if item.id in matched]
        return response([item.to_dict() for item in reversed(entries)])

    @app.post("/api/v1/journal")
    def journal_add(value: JournalInput):
        data = value.model_dump()
        data["runs"] = [
            service.state.run(item)["native_id"] if ":" in item else item for item in data["runs"]
        ]
        return response(service.store.add_entry(**data).to_dict())

    @app.get("/api/v1/journal/{entry_id}")
    def journal_entry(entry_id: str):
        return response(service.store.entry(entry_id).to_dict())

    @app.get("/api/v1/task")
    def task():
        return response(service.task())

    @app.post("/api/v1/task/select")
    def task_select(value: PathInput):
        return response(service.select_task(value.path))

    @app.post("/api/v1/task/validate")
    def task_validate(value: TaskInput):
        return response(
            {
                "valid": True,
                "parsed": parse_task_spec(
                    value.text, base_dir=service.task_path().parent
                ).model_dump(),
            }
        )

    @app.put("/api/v1/task")
    def task_save(value: TaskInput):
        return response(service.save_task(value.text, value.etag))

    @app.get("/api/v1/settings")
    def settings():
        config = workspace.config()
        llm = config.get("llm", {})
        return response(
            {
                "workspace": str(service.root),
                "store": str(workspace.store_path),
                "task_path": str(service.task_path()),
                "version": __version__,
                "llm": {
                    "provider": llm.get("provider", "none"),
                    "model": llm.get("model"),
                    "web_available": False,
                    "configured": bool(
                        os.environ.get(llm.get("api_key_env") or "ANTHROPIC_API_KEY")
                    ),
                    "explanation": "LLM появится в P1; детерминированное ревью доступно",
                },
                "watch": {
                    "status": "unknown",
                    "explanation": "Сервер не управляет внешним watch-процессом",
                },
            }
        )

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/") or path == "api":
            return response({"code": "not_found", "detail": "API маршрут не найден"}, 404)
        candidate = (STATIC / path).resolve()
        if STATIC.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        if not path or path.split("/", maxsplit=1)[0] in FRONTEND_ROUTES:
            if (STATIC / "index.html").exists():
                return FileResponse(STATIC / "index.html")
            return response(
                {"code": "assets_missing", "detail": "Соберите frontend: npm ci && npm run build"},
                503,
            )
        return response({"code": "not_found", "detail": "Страница не найдена"}, 404)

    return app
