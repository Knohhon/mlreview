"""Python API: ревью из кода и ноутбука, вызов в конце обучения (FR-R03, FR-R04).

Спецификация: docs/specs/api.md.
"""

from __future__ import annotations

import json

import pytest

from _factories import BINARY_SPEC_TEXT, binary_tables
from mlreview import api
from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main
from mlreview.core.task_spec import TaskSpecError


@pytest.fixture
def project(tmp_path):
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    for i in range(4):
        predictions, examples = binary_tables(seed=i + 1, duplicates=10)
        save_run_files(tmp_path / f"run-{i}", predictions, examples=examples, data_version="v1")
    return tmp_path


@pytest.mark.usefixtures("project")
def test_review_runs_with_store_and_seeds():
    """B1: ревью с базовым прогоном и сидами, запись в хранилище."""
    main(["init"])
    spec = api.load_spec("task.yaml")
    result = api.review_runs(
        spec, "run-0", baseline="run-1", seeds=["run-2"], baseline_seeds=["run-3"]
    )
    assert result.report.run_id == "run-0"
    assert result.report.baseline_id == "run-1"
    assert result.report.candidate_seeds == ("run-2",)
    assert result.record is not None
    assert result.store.endswith("store.sqlite")
    assert result.record.entries[0].id == "F-0001"


@pytest.mark.usefixtures("project")
def test_matches_cli_output(capsys):
    """B2: to_dict и to_text совпадают с CLI."""
    spec = api.load_spec("task.yaml")
    result = api.review_runs(spec, "run-0", baseline="run-1", store=False)
    capsys.readouterr()
    args = ["review", "run-0", "--spec", "task.yaml", "--baseline", "run-1", "--no-store"]
    assert main([*args, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == result.to_dict()
    assert main([*args, "--limit", "2"]) == 0
    assert capsys.readouterr().out.rstrip("\n") == result.to_text(limit=2)


@pytest.mark.usefixtures("project")
def test_notebook_markdown():
    """B3: таблица находок в ноутбуке."""
    result = api.review_runs(api.load_spec("task.yaml"), "run-0", store=False)
    markdown = result._repr_markdown_()
    assert markdown.startswith("**ревью прогона run-0** · версия данных: v1")
    assert "| важность | id | проверка | сообщение |" in markdown
    assert "| critical |  | data.split_duplicates |" in markdown
    assert "не проверено:" in markdown


@pytest.mark.usefixtures("project")
def test_without_workspace():
    """B4: без рабочего пространства — без записи."""
    result = api.review_runs(api.load_spec("task.yaml"), "run-0")
    assert result.record is None
    assert result.store is None


def test_on_run_end_uses_reference(project, capsys):
    """B5: референс из журнала, ≤ 5 находок, отчёт в файле."""
    main(["init"])
    api.review_runs(api.load_spec("task.yaml"), "run-0")
    main(["journal", "add", "decision", "принять run-0", "--run", "run-0", "--outcome", "accept"])
    capsys.readouterr()
    result = api.on_run_end("run-1", spec="task.yaml")
    out = capsys.readouterr().out
    assert result.report.baseline_id == "run-0"
    assert "базовый прогон: run-0" in out
    report_file = project / ".mlreview" / "reports" / "run-1.txt"
    assert report_file.read_text(encoding="utf-8").rstrip("\n") == out.rstrip("\n")
    numbered = [line for line in out.splitlines() if line[:5].strip().rstrip(".").isdigit()]
    assert 0 < len(numbered) <= 5


@pytest.mark.usefixtures("project")
def test_errors_are_library_exceptions(tmp_path):
    """E1: исключения библиотеки, а не SystemExit."""
    (tmp_path / "bad.yaml").write_text("task: {}\n", encoding="utf-8")
    with pytest.raises(TaskSpecError):
        api.load_spec(tmp_path / "bad.yaml")


@pytest.mark.usefixtures("project")
def test_on_run_end_never_crashes_training(capsys):
    """E2: ошибка в on_run_end — сообщение и None."""
    assert api.on_run_end("missing-run", spec="task.yaml") is None
    assert "mlreview: ревью не выполнено: " in capsys.readouterr().err
