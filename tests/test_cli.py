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


@pytest.mark.parametrize("command", list(COMMANDS))
def test_unimplemented_command_fails_explicitly(command, capsys):
    assert main([command]) == 1
    assert "не реализована" in capsys.readouterr().err
