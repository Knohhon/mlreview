"""Хранилище находок, прогонов и журнала решений в SQLite (FR-C05, FR-C06, FR-R05).

Находка хранилища — отпечаток проверки и её объекта (`fingerprint`) со статусом, историей
и вхождениями по прогонам. Отклонённая находка скрывается, пока условия (версия и хеш
данных, величина эффекта) не изменились существенно (S17). Поведение зафиксировано
в `docs/specs/findings-store.md` и `docs/specs/journal.md`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlreview.core.findings import (
    SEVERITIES,
    Finding,
    Report,
    fingerprint,
    magnitude,
    severity_rank,
)
from mlreview.core.run_contract import RunContract

STATUSES = ("new", "accepted", "dismissed", "fixed")
STATUS_NAMES = {
    "new": "новая",
    "accepted": "принята",
    "dismissed": "отклонена",
    "fixed": "исправлена",
}
KINDS = {"hypothesis": "H", "decision": "D", "note": "N"}
OUTCOMES = ("accept", "reject")
REASON_REQUIRED = "отклонение critical-находки требует причины (--reason)"
GROWTH = 2.0
STEM = 5
MIN_WORD = 3
_STOPWORDS_TEXT = (
    "пробовал пробовала пробовали ли что как уже это или для был была были мне меня почему "
    "где когда какой какие про при над под без чем его она они все ещё еще было есть "
    "the and did was have with what why how"
)
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, source TEXT, data_version TEXT, data_hash TEXT,
    config TEXT, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    num INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT UNIQUE, check_id TEXT,
    requirement TEXT, status TEXT, reason TEXT, conditions TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS occurrences (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, finding_num INTEGER, run_id TEXT,
    baseline_id TEXT, severity TEXT, message TEXT, facts TEXT, evidence TEXT,
    suggestion TEXT, data_version TEXT, data_hash TEXT, reviewed_at TEXT, hidden INTEGER
);
CREATE TABLE IF NOT EXISTS history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, finding_num INTEGER, at TEXT, status TEXT,
    reason TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS journal (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, kind TEXT, text TEXT,
    runs TEXT, findings TEXT, hypothesis TEXT, outcome TEXT, status TEXT,
    data_versions TEXT, created_at TEXT
);
"""


class StoreError(RuntimeError):
    """Операция с хранилищем невозможна; сообщение объясняет причину."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


@dataclass(frozen=True)
class RunRecord:
    """Прогон, прошедший ревью: источник, версия и хеш данных, конфиг."""

    run_id: str
    source: str
    data_version: str | None
    data_hash: str | None
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_contract(cls, contract: RunContract, source: str) -> RunRecord:
        return cls(
            contract.run_id, source, contract.data_version, contract.data_hash, contract.config
        )


@dataclass(frozen=True)
class Entry:
    """Находка отчёта, сопоставленная с хранилищем."""

    finding: Finding
    id: str
    status: str
    occurrence: int
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "occurrences": self.occurrence,
            "note": self.note,
            **self.finding.to_dict(),
        }


@dataclass(frozen=True)
class ReviewRecord:
    """Результат записи ревью: показанные и скрытые (отклонённые) находки."""

    entries: list[Entry]
    hidden: list[Entry]
    store_path: str


@dataclass(frozen=True)
class Occurrence:
    run_id: str
    baseline_id: str | None
    severity: str
    message: str
    facts: dict[str, Any]
    evidence: tuple[str, ...]
    suggestion: str | None
    data_version: str | None
    data_hash: str | None
    reviewed_at: str
    hidden: bool


@dataclass(frozen=True)
class HistoryItem:
    at: str
    status: str
    reason: str | None
    note: str | None


@dataclass(frozen=True)
class StoredFinding:
    """Находка хранилища: последнее вхождение, статус, история."""

    id: str
    fingerprint: str
    check: str
    requirement: str
    status: str
    reason: str | None
    occurrences: list[Occurrence]
    history: list[HistoryItem]

    @property
    def last(self) -> Occurrence:
        return self.occurrences[-1]

    @property
    def severity(self) -> str:
        return self.last.severity

    @property
    def message(self) -> str:
        return self.last.message

    def to_dict(self) -> dict[str, Any]:
        last = self.last
        return {
            "id": self.id,
            "check": self.check,
            "requirement": self.requirement,
            "status": self.status,
            "reason": self.reason,
            "severity": last.severity,
            "message": last.message,
            "facts": last.facts,
            "evidence": list(last.evidence),
            "suggestion": last.suggestion,
            "occurrences": len(self.occurrences),
            "last_run": last.run_id,
        }


@dataclass(frozen=True)
class JournalEntry:
    """Запись журнала: гипотеза, решение или заметка (FR-C06)."""

    id: str
    kind: str
    text: str
    runs: tuple[str, ...]
    findings: tuple[str, ...]
    hypothesis: str | None
    outcome: str | None
    status: str | None
    data_versions: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "runs": list(self.runs),
            "findings": list(self.findings),
            "hypothesis": self.hypothesis,
            "outcome": self.outcome,
            "status": self.status,
            "data_versions": list(self.data_versions),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SearchHit:
    entry: JournalEntry
    score: int
    linked: list[JournalEntry]
    stale: str | None


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    config_matches: list[tuple[str, str, str]]


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _stem(word: str) -> str:
    return word[:STEM] if len(word) > STEM else word


def query_stems(query: str) -> list[str]:
    """Основы слов запроса: от 3 букв, без служебных слов (B3)."""
    stems = [_stem(w) for w in _words(query) if len(w) >= MIN_WORD and w not in STOPWORDS]
    return list(dict.fromkeys(stems))


def _matched(stems: Sequence[str], text: str) -> int:
    present = {_stem(word) for word in _words(text)}
    return sum(stem in present for stem in stems)


def _conditions(report: Report, finding: Finding) -> dict[str, Any]:
    return {
        "data_version": report.data_version,
        "data_hash": report.data_hash,
        "magnitude": magnitude(finding),
    }


def _changes(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Существенные изменения условий отклонённой находки (B5)."""
    changes = []
    if old.get("data_version") != new.get("data_version"):
        changes.append(
            f"версия данных {old.get('data_version') or 'нет'} → {new.get('data_version') or 'нет'}"
        )
    if old.get("data_hash") != new.get("data_hash"):
        changes.append(f"хеш данных {old.get('data_hash')} → {new.get('data_hash')}")
    before, after = old.get("magnitude"), new.get("magnitude")
    grown = before is not None and after is not None and abs(after) > 0
    if grown and abs(after) >= GROWTH * abs(before) - 1e-12:
        changes.append(f"эффект вырос: {before:g} → {after:g}")
    return changes


def _finding_id(num: int) -> str:
    return f"F-{num:04d}"


def _finding_num(finding_id: str) -> int | None:
    match = re.fullmatch(r"F-(\d+)", finding_id.strip())
    return int(match.group(1)) if match else None


class Store:
    """Хранилище рабочего пространства (`.mlreview/store.sqlite`)."""

    def __init__(self, path: str | Path, *, now: Callable[[], datetime] = utc_now) -> None:
        self.path = Path(path)
        self._now = now

    def _timestamp(self) -> str:
        return self._now().isoformat(timespec="seconds")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        except (OSError, sqlite3.Error) as exc:
            raise StoreError(f"хранилище {self.path} недоступно: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.executescript(_SCHEMA)
                yield connection
        except sqlite3.DatabaseError as exc:
            raise StoreError(f"хранилище {self.path} повреждено или не SQLite: {exc}") from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        """Создаёт пустую базу со схемой."""
        with self._connect():
            pass

    # Прогоны

    def record_runs(self, runs: Iterable[RunRecord]) -> None:
        with self._connect() as db:
            self._upsert_runs(db, runs)

    def _upsert_runs(self, db: sqlite3.Connection, runs: Iterable[RunRecord]) -> None:
        at = self._timestamp()
        for run in runs:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
                "source=excluded.source, data_version=excluded.data_version, "
                "data_hash=excluded.data_hash, config=excluded.config, "
                "reviewed_at=excluded.reviewed_at",
                (
                    run.run_id,
                    run.source,
                    run.data_version,
                    run.data_hash,
                    _dumps(dict(run.config)),
                    at,
                ),
            )

    @staticmethod
    def _run_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            row["run_id"],
            row["source"],
            row["data_version"],
            row["data_hash"],
            json.loads(row["config"] or "{}"),
        )

    def run(self, run_id: str) -> RunRecord | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else self._run_record(row)

    def runs(self) -> list[RunRecord]:
        """Прогоны, прошедшие ревью, от последнего к первому."""
        with self._connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY reviewed_at DESC, rowid DESC").fetchall()
        return [self._run_record(row) for row in rows]

    def count_runs(self, data_hash: str, exclude: Sequence[str] = ()) -> int:
        """Сколько прогонов прошло ревью на данных с этим хешем (FR-D06)."""
        with self._connect() as db:
            rows = db.execute("SELECT run_id FROM runs WHERE data_hash = ?", (data_hash,))
            return sum(row["run_id"] not in exclude for row in rows)

    # Находки

    def record_review(self, report: Report, *, runs: Sequence[RunRecord] = ()) -> ReviewRecord:
        """Записывает находки отчёта: новые, повторные, скрытые отклонённые (B1–B6)."""
        entries, hidden = [], []
        unique: dict[str, Finding] = {}
        for finding in report.findings:
            key = fingerprint(finding)
            kept = unique.get(key)
            if kept is None or severity_rank(finding.severity) < severity_rank(kept.severity):
                unique[key] = finding
        at = self._timestamp()
        with self._connect() as db:
            self._upsert_runs(db, runs)
            for key, finding in unique.items():
                entry, is_hidden = self._record_finding(db, report, key, finding, at)
                (hidden if is_hidden else entries).append(entry)
        return ReviewRecord(entries, hidden, str(self.path))

    def _record_finding(
        self, db: sqlite3.Connection, report: Report, key: str, finding: Finding, at: str
    ) -> tuple[Entry, bool]:
        row = db.execute("SELECT * FROM findings WHERE fingerprint = ?", (key,)).fetchone()
        current = _conditions(report, finding)
        note = None
        hidden = False
        if row is None:
            cursor = db.execute(
                "INSERT INTO findings (fingerprint, check_id, requirement, status, reason, "
                "conditions, created_at) VALUES (?, ?, ?, 'new', NULL, NULL, ?)",
                (key, finding.check, finding.requirement, at),
            )
            num = int(cursor.lastrowid)
            self._history(db, num, HistoryItem(at, "new", None, None))
            status = "new"
        else:
            num, status = int(row["num"]), row["status"]
            if status == "dismissed":
                changes = _changes(json.loads(row["conditions"] or "{}"), current)
                if changes:
                    note = f"отклонялась ранее: {row['reason'] or 'без причины'}; изменилось: "
                    note += "; ".join(changes)
                    status = self._reopen(db, num, at, note)
                else:
                    hidden = True
            elif status == "fixed":
                note = "отмечалась исправленной; появилась снова"
                status = self._reopen(db, num, at, note)
        db.execute(
            "INSERT INTO occurrences (finding_num, run_id, baseline_id, severity, message, facts, "
            "evidence, suggestion, data_version, data_hash, reviewed_at, hidden) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                num,
                report.run_id,
                report.baseline_id,
                finding.severity,
                finding.message,
                _dumps(dict(finding.facts)),
                _dumps(list(finding.evidence)),
                finding.suggestion,
                report.data_version,
                report.data_hash,
                at,
                int(hidden),
            ),
        )
        (count,) = db.execute(
            "SELECT COUNT(*) FROM occurrences WHERE finding_num = ?", (num,)
        ).fetchone()
        return Entry(finding, _finding_id(num), status, int(count), note), hidden

    def _reopen(self, db: sqlite3.Connection, num: int, at: str, note: str) -> str:
        db.execute(
            "UPDATE findings SET status = 'new', reason = NULL, conditions = NULL WHERE num = ?",
            (num,),
        )
        self._history(db, num, HistoryItem(at, "new", None, note))
        return "new"

    @staticmethod
    def _history(db: sqlite3.Connection, num: int, item: HistoryItem) -> None:
        db.execute(
            "INSERT INTO history (finding_num, at, status, reason, note) VALUES (?, ?, ?, ?, ?)",
            (num, item.at, item.status, item.reason, item.note),
        )

    def set_status(self, finding_id: str, status: str, *, reason: str | None = None) -> None:
        """Меняет статус находки; critical отклоняется только с причиной (X4)."""
        if status not in STATUSES:
            raise StoreError(f"неизвестный статус {status!r}; допустимо: {', '.join(STATUSES)}")
        reason = reason.strip() if reason and reason.strip() else None
        stored = self.finding(finding_id)
        if status == "dismissed" and stored.severity == "critical" and reason is None:
            raise StoreError(REASON_REQUIRED)
        conditions = None
        if status in ("dismissed", "fixed"):
            last = stored.last
            finding = Finding(stored.check, last.severity, last.message, "", facts=last.facts)
            conditions = _dumps(
                {
                    "data_version": last.data_version,
                    "data_hash": last.data_hash,
                    "magnitude": magnitude(finding),
                }
            )
        num = _finding_num(stored.id)
        at = self._timestamp()
        with self._connect() as db:
            db.execute(
                "UPDATE findings SET status = ?, reason = ?, conditions = ? WHERE num = ?",
                (status, reason, conditions, num),
            )
            self._history(db, int(num), HistoryItem(at, status, reason, None))

    def finding(self, finding_id: str) -> StoredFinding:
        num = _finding_num(finding_id)
        with self._connect() as db:
            row = None
            if num is not None:
                row = db.execute("SELECT * FROM findings WHERE num = ?", (num,)).fetchone()
            if row is None:
                raise StoreError(f"находка {finding_id} не найдена")
            return self._stored(db, row)

    def _stored(self, db: sqlite3.Connection, row: sqlite3.Row) -> StoredFinding:
        num = row["num"]
        occurrences = [
            Occurrence(
                o["run_id"],
                o["baseline_id"],
                o["severity"],
                o["message"],
                json.loads(o["facts"]),
                tuple(json.loads(o["evidence"])),
                o["suggestion"],
                o["data_version"],
                o["data_hash"],
                o["reviewed_at"],
                bool(o["hidden"]),
            )
            for o in db.execute(
                "SELECT * FROM occurrences WHERE finding_num = ? ORDER BY seq", (num,)
            )
        ]
        history = [
            HistoryItem(h["at"], h["status"], h["reason"], h["note"])
            for h in db.execute("SELECT * FROM history WHERE finding_num = ? ORDER BY seq", (num,))
        ]
        return StoredFinding(
            _finding_id(num),
            row["fingerprint"],
            row["check_id"],
            row["requirement"],
            row["status"],
            row["reason"],
            occurrences,
            history,
        )

    def findings(
        self,
        *,
        status: str | None = None,
        min_severity: str = "low",
        check: str | None = None,
    ) -> list[StoredFinding]:
        """Находки по порядку id; по умолчанию без `info` (B7)."""
        if min_severity not in SEVERITIES:
            raise StoreError(f"неизвестная важность {min_severity!r}")
        with self._connect() as db:
            rows = db.execute("SELECT * FROM findings ORDER BY num").fetchall()
            stored = [self._stored(db, row) for row in rows]
        limit = severity_rank(min_severity)
        return [
            item
            for item in stored
            if item.occurrences
            and severity_rank(item.severity) <= limit
            and (status is None or item.status == status)
            and (check is None or item.check == check)
        ]

    # Журнал

    def add_entry(  # noqa: PLR0913 — поля записи журнала
        self,
        kind: str,
        text: str,
        *,
        runs: Sequence[str] = (),
        findings: Sequence[str] = (),
        hypothesis: str | None = None,
        outcome: str | None = None,
        data_version: str | None = None,
    ) -> JournalEntry:
        """Добавляет запись журнала; решение меняет статус своей гипотезы (B1, B2)."""
        if kind not in KINDS:
            raise StoreError(f"неизвестный вид записи {kind!r}; допустимо: {', '.join(KINDS)}")
        if not text.strip():
            raise StoreError("пустой текст записи")
        if outcome is not None and kind != "decision":
            raise StoreError("исход (--outcome) задаётся только у решения")
        if outcome is not None and outcome not in OUTCOMES:
            raise StoreError(f"исход {outcome!r} недопустим; допустимо: {', '.join(OUTCOMES)}")
        if hypothesis is not None and kind != "decision":
            raise StoreError("гипотеза (--hypothesis) указывается только у решения")
        with self._connect() as db:
            parent = None
            if hypothesis is not None:
                parent = self._entry_row(db, hypothesis)
                if parent is None or parent["kind"] != "hypothesis":
                    raise StoreError(f"запись {hypothesis} не найдена")
            for finding_id in findings:
                num = _finding_num(finding_id)
                found = (
                    num is not None
                    and db.execute("SELECT 1 FROM findings WHERE num = ?", (num,)).fetchone()
                )
                if not found:
                    raise StoreError(f"запись {finding_id} не найдена")
            versions = self._versions(db, runs, data_version, parent)
            (count,) = db.execute("SELECT COUNT(*) FROM journal WHERE kind = ?", (kind,)).fetchone()
            entry_id = f"{KINDS[kind]}-{count + 1:04d}"
            status = "open" if kind == "hypothesis" else None
            db.execute(
                "INSERT INTO journal (id, kind, text, runs, findings, hypothesis, outcome, status, "
                "data_versions, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    kind,
                    text.strip(),
                    _dumps(list(runs)),
                    _dumps(list(findings)),
                    hypothesis,
                    outcome,
                    status,
                    _dumps(list(versions)),
                    self._timestamp(),
                ),
            )
            if parent is not None and outcome is not None:
                new_status = "confirmed" if outcome == "accept" else "rejected"
                db.execute("UPDATE journal SET status = ? WHERE id = ?", (new_status, hypothesis))
            row = self._entry_row(db, entry_id)
        return self._journal_entry(row)

    @staticmethod
    def _versions(
        db: sqlite3.Connection,
        runs: Sequence[str],
        data_version: str | None,
        parent: sqlite3.Row | None,
    ) -> list[str]:
        if data_version is not None:
            return [str(data_version)]
        versions = []
        for run_id in runs:
            row = db.execute("SELECT data_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is not None and row["data_version"] is not None:
                versions.append(row["data_version"])
        if not versions and parent is not None:
            versions = json.loads(parent["data_versions"] or "[]")
        return list(dict.fromkeys(versions))

    @staticmethod
    def _entry_row(db: sqlite3.Connection, entry_id: str) -> sqlite3.Row | None:
        return db.execute("SELECT * FROM journal WHERE id = ?", (entry_id.strip(),)).fetchone()

    @staticmethod
    def _journal_entry(row: sqlite3.Row) -> JournalEntry:
        return JournalEntry(
            row["id"],
            row["kind"],
            row["text"],
            tuple(json.loads(row["runs"])),
            tuple(json.loads(row["findings"])),
            row["hypothesis"],
            row["outcome"],
            row["status"],
            tuple(json.loads(row["data_versions"])),
            row["created_at"],
        )

    def entry(self, entry_id: str) -> JournalEntry:
        with self._connect() as db:
            row = self._entry_row(db, entry_id)
        if row is None:
            raise StoreError(f"запись {entry_id} не найдена")
        return self._journal_entry(row)

    def entries(self, kind: str | None = None) -> list[JournalEntry]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM journal ORDER BY seq").fetchall()
        return [self._journal_entry(r) for r in rows if kind is None or r["kind"] == kind]

    def search(self, query: str, *, current_version: str | None = None) -> SearchResult:
        """Поиск по журналу; без записей — по конфигам прогонов (B3–B6)."""
        stems = query_stems(query)
        if not stems:
            return SearchResult([], [])
        entries = self.entries()
        if current_version is None:
            current_version = next(
                (run.data_version for run in self.runs() if run.data_version), None
            )
        scored = [(entry, _matched(stems, entry.text)) for entry in entries]
        order = {entry.id: position for position, entry in enumerate(entries)}
        scored = [(entry, score) for entry, score in scored if score]
        scored.sort(key=lambda item: (-item[1], -order[item[0].id]))
        hits = [
            SearchHit(entry, score, self._linked(entry, entries), _stale(entry, current_version))
            for entry, score in scored
        ]
        matches = []
        if not hits:
            for run in sorted(self.runs(), key=lambda run: run.run_id):
                for key, value in sorted(run.config.items()):
                    if _matched(stems, f"{key} {value}"):
                        matches.append((run.run_id, key, str(value)))
        return SearchResult(hits, matches)

    @staticmethod
    def _linked(entry: JournalEntry, entries: Sequence[JournalEntry]) -> list[JournalEntry]:
        if entry.kind == "hypothesis":
            return [other for other in entries if other.hypothesis == entry.id]
        if entry.kind == "decision" and entry.hypothesis:
            return [other for other in entries if other.id == entry.hypothesis]
        return []

    def reference_run(self) -> str | None:
        """Референс проактивного ревью — первый прогон последнего принятого решения (B8)."""
        for entry in reversed(self.entries("decision")):
            if entry.outcome == "accept" and entry.runs:
                return entry.runs[0]
        return None


def _stale(entry: JournalEntry, current: str | None) -> str | None:
    if current is None or not entry.data_versions or current in entry.data_versions:
        return None
    before = ", ".join(entry.data_versions)
    return f"данные с тех пор сменились ({before} → {current}), вывод мог устареть"
