"""Журнал гипотез и решений (FR-C06, S05).

Спецификация: docs/specs/journal.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlreview.cli import main
from mlreview.core.store import RunRecord, Store, StoreError


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "store.sqlite", now=Clock())
    store.record_runs(
        [
            RunRecord("exp-017", "mlflow:x", "ds-v1", "sha256:1", {"loss": "focal", "lr": "0.01"}),
            RunRecord("exp-041", "mlflow:x", "ds-v3", "sha256:3", {"loss": "ce", "lr": "0.01"}),
        ]
    )
    return store


def test_add_hypothesis_takes_run_data_version(store):
    """B1: гипотеза — H-0001, open, версия данных прогона."""
    entry = store.add_entry(
        "hypothesis", "focal loss поднимет recall редких классов", runs=["exp-017"]
    )
    assert (entry.id, entry.status, entry.data_versions) == ("H-0001", "open", ("ds-v1",))


def test_decision_updates_hypothesis_status(store):
    """B2: решение с исходом reject → гипотеза rejected; accept → confirmed."""
    store.add_entry("hypothesis", "focal loss поможет", runs=["exp-017"])
    decision = store.add_entry(
        "decision", "откатить focal loss", hypothesis="H-0001", runs=["exp-017"], outcome="reject"
    )
    assert decision.id == "D-0001"
    assert store.entry("H-0001").status == "rejected"
    store.add_entry("hypothesis", "больше данных de", runs=["exp-041"])
    store.add_entry("decision", "принять", hypothesis="H-0002", runs=["exp-041"], outcome="accept")
    assert store.entry("H-0002").status == "confirmed"


def test_search_matches_words_and_links(store):
    """B3, B4, B5 (S05): поиск находит гипотезу, её решение, прогоны и устаревание."""
    store.add_entry("hypothesis", "Focal loss поднимет recall редких классов", runs=["exp-017"])
    store.add_entry(
        "decision", "Откатить: эффект в пределах шума", hypothesis="H-0001", outcome="reject"
    )
    store.add_entry("note", "Посмотреть на сегмент de", runs=["exp-041"])
    result = store.search("пробовал ли я focal loss?", current_version="ds-v3")
    assert [hit.entry.id for hit in result.hits] == ["H-0001"]
    hit = result.hits[0]
    assert [linked.id for linked in hit.linked] == ["D-0001"]
    assert hit.entry.runs == ("exp-017",)
    assert hit.stale == "данные с тех пор сменились (ds-v1 → ds-v3), вывод мог устареть"


def test_search_uses_stems(store):
    """B3: слово длиннее 5 букв сравнивается по первым 5 буквам."""
    store.add_entry("note", "Сегменты с малым числом примеров нестабильны")
    assert [h.entry.id for h in store.search("сегмент").hits] == ["N-0001"]
    assert store.search("ли что").hits == []


def test_search_falls_back_to_run_configs(store):
    """B6: в журнале нет, в конфигах есть."""
    result = store.search("focal")
    assert result.hits == []
    assert result.config_matches == [("exp-017", "loss", "focal")]


def test_reference_run_is_last_accepted(store):
    """B8: референс — первый прогон последнего принятого решения."""
    assert store.reference_run() is None
    store.add_entry("decision", "принять", runs=["exp-017"], outcome="accept")
    store.add_entry("decision", "принять новую", runs=["exp-041", "exp-017"], outcome="accept")
    store.add_entry("decision", "отклонить", runs=["exp-099"], outcome="reject")
    assert store.reference_run() == "exp-041"


@pytest.mark.parametrize(
    ("kind", "kwargs", "message"),
    [
        ("decision", {"hypothesis": "H-0042"}, "запись H-0042 не найдена"),
        ("note", {"findings": ["F-0042"]}, "запись F-0042 не найдена"),
        ("hypothesis", {"outcome": "accept"}, "исход"),
        ("note", {"hypothesis": "H-0001"}, "гипотеза"),
        ("note", {}, "пустой"),
    ],
)
def test_entry_errors(store, kind, kwargs, message):
    """E1–E3: ссылки на неизвестные id, неуместные поля, пустой текст."""
    store.add_entry("hypothesis", "есть")
    text = "" if message == "пустой" else "текст"
    with pytest.raises(StoreError, match=message):
        store.add_entry(kind, text, **kwargs)


def test_journal_cli(tmp_path, monkeypatch, capsys):
    """CLI: add, list, show, search; без рабочего пространства — подсказка (E4)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLREVIEW_HOME", raising=False)
    assert main(["journal", "list"]) == 1
    assert "mlreview init" in capsys.readouterr().err
    main(["init"])
    capsys.readouterr()
    assert main(["journal", "add", "hypothesis", "focal loss поможет", "--run", "exp-017"]) == 0
    assert capsys.readouterr().out.strip() == "H-0001"
    args = ["journal", "add", "decision", "откатить", "--hypothesis", "H-0001"]
    assert main([*args, "--outcome", "reject", "--data-version", "ds-v1"]) == 0
    assert capsys.readouterr().out.strip() == "D-0001"
    assert main(["journal", "list"]) == 0
    listed = capsys.readouterr().out
    assert "H-0001" in listed
    assert "D-0001" in listed
    assert main(["journal", "show", "H-0001"]) == 0
    assert "статус: rejected" in capsys.readouterr().out
    spec = tmp_path / "task.yaml"
    spec.write_text(
        "task: {type: regression, output: values}\ndata: {version: ds-v3}\n", encoding="utf-8"
    )
    assert main(["journal", "search", "focal loss", "--spec", str(spec)]) == 0
    found = capsys.readouterr().out
    assert "H-0001" in found
    assert "D-0001" in found
    assert main(["journal", "search", "трансформер"]) == 0
    assert "ничего не найдено" in capsys.readouterr().out
    assert main(["journal", "add", "decision", "x", "--hypothesis", "H-0042"]) == 1
    assert "запись H-0042 не найдена" in capsys.readouterr().err
