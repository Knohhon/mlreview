import sys

import pandas as pd
import pytest

from mlreview import __version__
from mlreview.cli import COMMANDS, main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_lists_commands(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


def test_help_lists_spec_command(capsys):
    assert main([]) == 0
    assert "spec" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["review", "metrics", "diagnose", "ask"])
def test_unimplemented_command_fails_explicitly(command, capsys):
    assert main([command]) == 1
    assert "не реализована" in capsys.readouterr().err


def test_spec_validate_valid(tmp_path, capsys):
    """C1: валидная спецификация — код 0 и сводка в stdout."""
    path = tmp_path / "task.yaml"
    text = "task: {type: classification/binary, output: probabilities}\n"
    path.write_text(text, encoding="utf-8")

    assert main(["spec", "validate", str(path)]) == 0

    out = capsys.readouterr().out
    assert "спецификация валидна" in out
    assert "task.yaml" in out
    assert "classification/binary" in out
    assert "probabilities" in out


def test_spec_validate_invalid(tmp_path, capsys):
    """C2: невалидная спецификация — код 1 и все проблемы с путями в stderr."""
    path = tmp_path / "task.yaml"
    path.write_text("task: {output: probabilities, segmets: [a]}\n", encoding="utf-8")

    assert main(["spec", "validate", str(path)]) == 1

    err = capsys.readouterr().err
    assert "task.type" in err
    assert "task.segmets" in err


def test_spec_validate_missing_file(tmp_path, capsys):
    """C2: нет файла — код 1 и понятное сообщение."""
    assert main(["spec", "validate", str(tmp_path / "nope.yaml")]) == 1
    assert "не найден" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["spec"], ["spec", "validate"]])
def test_spec_usage_errors(argv):
    """C3: без подкоманды или без пути — ошибка использования, код 2."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def _spec_file(tmp_path, text="task: {type: classification/binary, output: probabilities}\n"):
    path = tmp_path / "task.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _predictions():
    return pd.DataFrame(
        {"example_id": ["a1", "a2", "a3"], "label": [0, 1, 1], "prediction": [0.1, 0.8, 0.4]}
    )


def test_contract_check_valid(tmp_path, mlflow_store, capsys):
    """C1: валидный контракт — код 0, сводка и пробелы в stdout."""
    examples = pd.DataFrame(
        {
            "example_id": ["t1", "a1", "a2", "a3"],
            "split": ["train", "test", "test", "test"],
            "language": ["en", "en", "de", "en"],
        }
    )
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions(), "examples.parquet": examples},
        params={"C": "1.0"},
        metrics={"train_loss": [0.7, 0.5], "val_loss": [0.8, 0.6]},
        tags={"mlreview.data_version": "ds-v3"},
    )
    text = (
        "task: {type: classification/binary, output: probabilities, segments: [language, source]}\n"
    )
    spec = _spec_file(tmp_path, text)

    argv = ["contract", "check", run_id, "--spec", str(spec), "--tracking-uri", mlflow_store.uri]
    assert main(argv) == 0

    out = capsys.readouterr().out
    assert f"контракт прогона валиден: {run_id}" in out
    assert "предсказаний: 3" in out
    assert "примеров: 4, сплиты: test, train" in out
    assert "кривых: 2" in out
    assert "параметров: 1" in out
    assert "версия данных: ds-v3" in out
    assert "нет examples.source — недоступно: срез по сегменту source (FR-D02)" in out


def test_contract_check_invalid_spec(tmp_path, capsys):
    """C2: невалидная спецификация — код 1, проблемы в stderr."""
    spec = _spec_file(tmp_path, "task: {output: probabilities}\n")

    assert main(["contract", "check", "run-1", "--spec", str(spec)]) == 1
    assert "task.type" in capsys.readouterr().err


def test_contract_check_unknown_run(tmp_path, mlflow_store, capsys):
    """C2: ошибка адаптера — код 1, причина в stderr."""
    argv = ["contract", "check", "0" * 32, "--spec", str(_spec_file(tmp_path))]

    assert main([*argv, "--tracking-uri", mlflow_store.uri]) == 1
    assert "0" * 32 in capsys.readouterr().err


def test_contract_check_invalid_contract(tmp_path, mlflow_store, capsys):
    """C2: невалидный контракт — код 1, проблемы с путями в stderr."""
    run_id = mlflow_store.run(
        artifacts={"predictions.parquet": _predictions().drop(columns="label")}
    )
    argv = ["contract", "check", run_id, "--spec", str(_spec_file(tmp_path))]

    assert main([*argv, "--tracking-uri", mlflow_store.uri]) == 1
    assert "predictions.label" in capsys.readouterr().err


def test_contract_check_without_mlflow(tmp_path, monkeypatch, capsys):
    """C2, E1: без пакета mlflow — код 1 и подсказка установки."""
    monkeypatch.setitem(sys.modules, "mlflow", None)

    assert main(["contract", "check", "run-1", "--spec", str(_spec_file(tmp_path))]) == 1
    assert 'pip install "mlreview[mlflow]"' in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [["contract"], ["contract", "check"], ["contract", "check", "run-1"]]
)
def test_contract_usage_errors(argv):
    """C3: без действия, id прогона или --spec — ошибка использования, код 2."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
