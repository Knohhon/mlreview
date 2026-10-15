"""Рабочее пространство и команды init, findings, dismiss; ревью с хранилищем (FR-C05, FR-R05).

Спецификация: docs/specs/findings-store.md.
"""

from __future__ import annotations

import json

import pytest

from _factories import BINARY_SPEC_TEXT, binary_tables
from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Каталог проекта: спецификация и прогоны в файлах; текущий каталог — проект."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLREVIEW_HOME", raising=False)
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    for i in range(12):
        predictions, examples = binary_tables(seed=i + 1, duplicates=10)
        save_run_files(
            tmp_path / "runs" / f"run-{i}", predictions, examples=examples, data_version="v1"
        )
    return tmp_path


def _review(*args):
    return main(["review", *args, "--spec", "task.yaml"])


def test_init_creates_workspace(project, capsys):
    """Рабочее пространство: init создаёт .mlreview, повтор не перезаписывает."""
    assert main(["init"]) == 0
    assert (project / ".mlreview" / "config.yaml").read_text(encoding="utf-8").startswith("#")
    assert (project / ".mlreview" / "store.sqlite").exists()
    assert "создано" in capsys.readouterr().out
    assert main(["init"]) == 0
    assert "уже инициализировано" in capsys.readouterr().out


def test_review_without_workspace_does_not_store(project, capsys):
    """E4: без рабочего пространства ревью работает и подсказывает init."""
    assert _review("runs/run-0") == 0
    assert "  хранилище находок: не подключено — mlreview init" in capsys.readouterr().out
    assert not (project / ".mlreview").exists()
    assert main(["findings", "list"]) == 1
    assert "mlreview init" in capsys.readouterr().err


@pytest.mark.usefixtures("project")
def test_review_records_and_annotates(capsys):
    """B1, B8: ревью записывает находки и показывает id и статус."""
    main(["init"])
    capsys.readouterr()
    assert _review("runs/run-0") == 0
    out = capsys.readouterr().out
    assert "  1. [critical] F-0001 data.split_duplicates — " in out
    assert "     статус: новая" in out
    assert _review("runs/run-1") == 0
    assert "     статус: повторяется: 2-е вхождение" in capsys.readouterr().out
    assert _review("runs/run-2", "--format", "json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["store"].endswith("store.sqlite")
    first = data["findings"][0]
    assert (first["id"], first["status"], first["occurrences"]) == ("F-0001", "new", 3)


@pytest.mark.usefixtures("project")
def test_dismiss_flow_hides_finding_for_ten_runs(capsys):
    """B3, B4, E1 (S17, X4): dismiss critical требует причины; потом 10 прогонов без неё."""
    main(["init"])
    _review("runs/run-0")
    capsys.readouterr()
    assert main(["dismiss", "F-0001"]) == 1
    assert "отклонение critical-находки требует причины (--reason)" in capsys.readouterr().err
    assert main(["dismiss", "F-0001", "--reason", "дубли — это аугментации"]) == 0
    assert "F-0001: отклонена" in capsys.readouterr().out
    for i in range(1, 11):
        assert _review(f"runs/run-{i}") == 0
        out = capsys.readouterr().out
        assert "F-0001" not in out
        assert "скрыто отклонённых: 1" in out


@pytest.mark.usefixtures("project")
def test_findings_commands(capsys):
    """B7, E2: list, show, accept/fix/reopen, неизвестный id."""
    main(["init"])
    _review("runs/run-0")
    capsys.readouterr()
    assert main(["findings", "list"]) == 0
    out = capsys.readouterr().out
    assert "F-0001" in out
    assert "metrics.value" not in out  # info по умолчанию скрыты
    assert main(["findings", "list", "--min-severity", "info"]) == 0
    assert "metrics.value" in capsys.readouterr().out
    assert main(["findings", "accept", "F-0001", "--reason", "знаем"]) == 0
    assert main(["findings", "show", "F-0001"]) == 0
    shown = capsys.readouterr().out
    assert "статус: принята (знаем)" in shown
    assert "история:" in shown
    assert "вхождения: 1" in shown
    assert main(["findings", "fix", "F-0001"]) == 0
    assert main(["findings", "reopen", "F-0001"]) == 0
    capsys.readouterr()
    assert main(["findings", "show", "F-0099"]) == 1
    assert "находка F-0099 не найдена" in capsys.readouterr().err


@pytest.mark.usefixtures("project")
def test_min_severity_and_limit(capsys):
    """B9: фильтр важности и лимит."""
    main(["init"])
    capsys.readouterr()
    assert _review("runs/run-0", "--min-severity", "high", "--no-store") == 0
    out = capsys.readouterr().out
    assert "[info]" not in out
    assert "скрыто: " in out
    assert _review("runs/run-0", "--limit", "1", "--no-store") == 0
    out = capsys.readouterr().out
    assert "находки: 1" in out
    assert "  2. " not in out


@pytest.mark.usefixtures("project")
def test_no_store_flag(capsys):
    """B10: --no-store ничего не записывает."""
    main(["init"])
    assert _review("runs/run-0", "--no-store") == 0
    capsys.readouterr()
    assert main(["findings", "list", "--min-severity", "info"]) == 0
    assert "находок нет" in capsys.readouterr().out


@pytest.mark.usefixtures("project")
def test_mlreview_home_points_to_workspace(tmp_path_factory, monkeypatch, capsys):
    """Рабочее пространство через MLREVIEW_HOME."""
    other = tmp_path_factory.mktemp("elsewhere")
    assert main(["init", str(other)]) == 0
    monkeypatch.setenv("MLREVIEW_HOME", str(other / ".mlreview"))
    capsys.readouterr()
    assert _review("runs/run-0") == 0
    assert str(other / ".mlreview" / "store.sqlite") in capsys.readouterr().out
