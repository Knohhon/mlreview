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
