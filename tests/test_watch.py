"""Проактивное ревью: mlreview watch по каталогу прогонов и эксперименту MLflow (FR-R04).

Спецификация: docs/specs/proactive.md.
"""

from __future__ import annotations

import pytest

from _factories import BINARY_SPEC_TEXT, binary_tables
from mlreview import proactive
from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import build_parser, main
from mlreview.core.store import Store


@pytest.fixture
def project(tmp_path):
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    for i in range(2):
        predictions, examples = binary_tables(seed=i + 1, duplicates=10)
        save_run_files(
            tmp_path / "runs" / f"run-{i}", predictions, examples=examples, data_version="v1"
        )
    main(["init"])
    return tmp_path


def _watch(*args):
    return main(["watch", "--spec", "task.yaml", "--runs-dir", "runs", "--once", *args])


def test_new_runs_are_reviewed_once(project, capsys):
    """B1, B2, B6: завершённые прогоны — ревью и отчёты; повтор — ничего нового."""
    predictions, examples = binary_tables(seed=9)
    running = save_run_files(project / "runs" / "run-9", predictions, examples=examples)
    (running / "run.yaml").unlink()  # обучение ещё идёт
    capsys.readouterr()
    assert _watch() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("прогон run-0: находок ")
    assert "critical 1" in lines[0]
    assert lines[0].endswith(str(project / ".mlreview" / "reports" / "run-0.txt"))
    assert lines[1].startswith("прогон run-1: ")
    assert len(lines) == 2
    assert (project / ".mlreview" / "reports" / "run-1.txt").exists()
    assert {run.run_id for run in Store(project / ".mlreview" / "store.sqlite").runs()} == {
        "run-0",
        "run-1",
    }
    assert _watch() == 0
    assert capsys.readouterr().out.strip() == "новых завершённых прогонов нет"


def test_reference_from_journal(project, capsys):
    """B3: новый прогон сравнивается с принятым."""
    assert _watch() == 0
    main(["journal", "add", "decision", "принять", "--run", "run-0", "--outcome", "accept"])
    predictions, examples = binary_tables(seed=5, duplicates=10)
    save_run_files(project / "runs" / "run-5", predictions, examples=examples, data_version="v1")
    capsys.readouterr()
    assert _watch() == 0
    report = (project / ".mlreview" / "reports" / "run-5.txt").read_text(encoding="utf-8")
    assert "базовый прогон: run-0" in report


def test_mlflow_experiment(mlflow_store, tmp_path, capsys):
    """B4: из эксперимента MLflow — только завершённые прогоны."""
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    main(["init"])
    predictions, examples = binary_tables(duplicates=10)
    finished = mlflow_store.run(
        artifacts={"predictions.parquet": predictions, "examples.parquet": examples}
    )
    mlflow_store.client.create_run(mlflow_store.experiment_id)  # ещё идёт
    capsys.readouterr()
    args = ["watch", "--spec", "task.yaml", "--experiment", "tests"]
    assert main([*args, "--tracking-uri", mlflow_store.uri, "--once"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"прогон {finished}: ")


def test_report_is_limited(project, capsys):
    """B5: в отчёте не больше лимита (по умолчанию 5) находок, в хранилище — все."""
    assert build_parser().parse_args(["watch", "--spec", "s", "--runs-dir", "r"]).limit == 5
    capsys.readouterr()
    assert _watch("--min-severity", "info", "--limit", "2") == 0
    report = (project / ".mlreview" / "reports" / "run-0.txt").read_text(encoding="utf-8")
    assert "  2. " in report
    assert "  3. " not in report
    assert "скрыто: 3 (ниже порога или сверх лимита)" in report
    store = Store(project / ".mlreview" / "store.sqlite")
    stored = store.findings(min_severity="info")
    assert sum(o.run_id == "run-0" for item in stored for o in item.occurrences) == 5


def test_loop_sleeps_between_scans(project):
    """B6: цикл — пауза между просмотрами."""
    pauses = []
    spec = proactive.load_spec_for_watch(project / "task.yaml")
    source = proactive.DirectorySource(project / "runs")
    proactive.watch(
        spec, source, interval=7, sleep=pauses.append, max_scans=3, echo=lambda _line: None
    )
    assert pauses == [7, 7]


def test_errors(tmp_path, capsys):
    """E1, E3, E4: нет рабочего пространства; нет источника; нет эксперимента."""
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    (tmp_path / "runs").mkdir()
    assert _watch() == 1
    assert "mlreview init" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["watch", "--spec", "task.yaml", "--once"])
    assert exc.value.code == 2


def test_invalid_run_is_reported_and_others_continue(project, capsys):
    """E2: невалидный прогон — строка с причиной, остальные проверяются."""
    predictions, _ = binary_tables()
    save_run_files(project / "runs" / "run-bad", predictions.drop(columns="label"))
    capsys.readouterr()
    assert _watch() == 0
    out = capsys.readouterr().out
    assert "прогон run-bad: не проверен — " in out
    assert "прогон run-0: находок" in out


def test_unknown_experiment(mlflow_store, tmp_path, capsys):
    """E4: эксперимента нет."""
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    main(["init"])
    capsys.readouterr()
    args = ["watch", "--spec", "task.yaml", "--experiment", "nope"]
    assert main([*args, "--tracking-uri", mlflow_store.uri, "--once"]) == 1
    assert "эксперимент nope не найден" in capsys.readouterr().err
