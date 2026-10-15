"""Хранилище находок: запись ревью, статусы, дедупликация, отклонение (FR-C05, FR-R05).

Спецификация: docs/specs/findings-store.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlreview.core.findings import Finding, Report, fingerprint, magnitude
from mlreview.core.store import RunRecord, Store, StoreError

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


def _finding(severity="high", delta=-0.05, value="smooth", check="slices.segment_regression"):
    facts = {"segment": "texture", "value": value, "metric": "roc_auc", "delta": delta}
    return Finding(check, severity, f"регрессия {value}: Δ {delta}", "FR-D02", ("e1",), facts)


def _report(run_id, findings, *, version="v1", data_hash="sha256:aaaa"):
    return Report(
        run_id=run_id,
        findings=findings,
        skipped=[],
        data_version=version,
        data_hash=data_hash,
    )


def _run(run_id, version="v1", data_hash="sha256:aaaa", config=None):
    return RunRecord(run_id, f"files:/runs/{run_id}", version, data_hash, config or {})


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store.sqlite", now=Clock())


def _record(store, run_id, findings, **kwargs):
    report = _report(run_id, findings, **kwargs)
    version = kwargs.get("version", "v1")
    data_hash = kwargs.get("data_hash", "sha256:aaaa")
    return store.record_review(report, runs=[_run(run_id, version, data_hash)])


def test_fingerprint_uses_object_not_numbers():
    """Отпечаток — проверка и объект; величина — из MAGNITUDE."""
    first, second = _finding(delta=-0.05), _finding(delta=-0.2, severity="critical")
    assert fingerprint(first) == fingerprint(second)
    assert fingerprint(first) != fingerprint(_finding(value="rough"))
    assert magnitude(second) == -0.2
    temporal = Finding(
        "data.temporal_leak",
        "critical",
        "m",
        "FR-D03",
        facts={"earlier_split": "train", "later_split": "test", "boundary": "2024-01-02"},
    )
    moved = Finding(
        "data.temporal_leak",
        "critical",
        "m",
        "FR-D03",
        facts={"earlier_split": "train", "later_split": "test", "boundary": "2024-03-01"},
    )
    assert fingerprint(temporal) == fingerprint(moved)


def test_record_creates_and_reuses_findings(store):
    """B1: новая находка — новый id; известный отпечаток — новое вхождение."""
    first = _record(store, "run-1", [_finding()])
    assert [(e.id, e.status, e.occurrence) for e in first.entries] == [("F-0001", "new", 1)]
    second = _record(store, "run-2", [_finding(delta=-0.06), _finding(value="rough")])
    assert [(e.id, e.occurrence) for e in second.entries] == [("F-0001", 2), ("F-0002", 1)]
    stored = store.finding("F-0001")
    assert [o.run_id for o in stored.occurrences] == ["run-1", "run-2"]
    assert stored.message == "регрессия smooth: Δ -0.06"
    assert store.run("run-2").source == "files:/runs/run-2"


def test_duplicate_fingerprints_in_one_report_are_merged(store):
    """B2: дубли в одном отчёте → одна находка с наибольшей severity."""
    record = _record(store, "run-1", [_finding(severity="critical"), _finding(severity="low")])
    assert [(e.id, e.finding.severity) for e in record.entries] == [("F-0001", "critical")]


def test_dismiss_stores_reason_and_conditions(store):
    """B3: отклонение — статус, причина, история."""
    _record(store, "run-1", [_finding()])
    store.set_status("F-0001", "dismissed", reason="сегмент не важен")
    stored = store.finding("F-0001")
    assert stored.status == "dismissed"
    assert stored.reason == "сегмент не важен"
    assert [(h.status, h.reason) for h in stored.history] == [
        ("new", None),
        ("dismissed", "сегмент не важен"),
    ]


def test_dismissed_finding_hidden_for_ten_runs(store):
    """B4 (S17): 10 следующих ревью при неизменных условиях — находки нет."""
    _record(store, "run-0", [_finding()])
    store.set_status("F-0001", "dismissed", reason="не важно")
    for i in range(1, 11):
        record = _record(store, f"run-{i}", [_finding(delta=-0.05 - i * 0.001)])
        assert record.entries == []
        assert [e.id for e in record.hidden] == ["F-0001"]
    stored = store.finding("F-0001")
    assert stored.status == "dismissed"
    assert len(stored.occurrences) == 11
    assert sum(o.hidden for o in stored.occurrences) == 10


@pytest.mark.parametrize(
    ("kwargs", "delta", "change"),
    [
        ({"version": "v2"}, -0.05, "версия данных v1 → v2"),
        ({"data_hash": "sha256:bbbb"}, -0.05, "хеш данных"),
        ({}, -0.1, "эффект вырос"),
    ],
)
def test_dismissed_finding_returns_when_conditions_change(store, kwargs, delta, change):
    """B5: смена версии, хеша данных или рост эффекта вдвое — находка возвращается."""
    _record(store, "run-0", [_finding()])
    store.set_status("F-0001", "dismissed", reason="не важно")
    record = _record(store, "run-1", [_finding(delta=delta)], **kwargs)
    (entry,) = record.entries
    assert entry.status == "new"
    assert entry.note.startswith("отклонялась ранее: не важно; изменилось: ")
    assert change in entry.note
    assert store.finding("F-0001").status == "new"


def test_fixed_reopens_and_accepted_is_kept(store):
    """B6: fixed появилась снова → new; accepted — показывается как принята."""
    _record(store, "run-0", [_finding(), _finding(value="rough")])
    store.set_status("F-0001", "fixed")
    store.set_status("F-0002", "accepted")
    record = _record(store, "run-1", [_finding(), _finding(value="rough")])
    by_id = {e.id: e for e in record.entries}
    assert by_id["F-0001"].status == "new"
    assert by_id["F-0001"].note == "отмечалась исправленной; появилась снова"
    assert by_id["F-0002"].status == "accepted"


def test_status_changes_and_listing(store):
    """B7: смена статусов, фильтры списка (по умолчанию без info)."""
    info = Finding("metrics.value", "info", "auc = 0.9", "FR-D01", facts={"metric": "auc"})
    _record(store, "run-0", [_finding(), info, _finding(value="rough", severity="low")])
    assert [f.id for f in store.findings()] == ["F-0001", "F-0003"]
    assert [f.id for f in store.findings(min_severity="info")] == ["F-0001", "F-0002", "F-0003"]
    store.set_status("F-0001", "accepted", reason="ok")
    store.set_status("F-0001", "new")
    assert [f.id for f in store.findings(status="new", min_severity="high")] == ["F-0001"]
    assert [f.id for f in store.findings(check="metrics.value", min_severity="info")] == ["F-0002"]
    assert [h.status for h in store.finding("F-0001").history] == ["new", "accepted", "new"]


def test_critical_dismiss_requires_reason(store):
    """E1 (X4): critical без причины не отклоняется."""
    _record(store, "run-0", [_finding(severity="critical")])
    for reason in (None, "  "):
        with pytest.raises(StoreError, match="требует причины"):
            store.set_status("F-0001", "dismissed", reason=reason)
    assert store.finding("F-0001").status == "new"


def test_unknown_finding(store):
    """E2: неизвестный id."""
    with pytest.raises(StoreError, match="находка F-0042 не найдена"):
        store.set_status("F-0042", "accepted")
    with pytest.raises(StoreError, match="не найдена"):
        store.finding("F-0042")


def test_corrupted_database(tmp_path):
    """E3: повреждённая база → StoreError с путём."""
    path = tmp_path / "store.sqlite"
    path.write_bytes(b"definitely not sqlite" * 100)
    with pytest.raises(StoreError, match=r"store\.sqlite"):
        Store(path).findings()
