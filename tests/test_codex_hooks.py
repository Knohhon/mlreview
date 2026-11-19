"""Проверки хуков Codex: docs/specs/codex-setup.md."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".codex" / "hooks" / "checks.py"


@pytest.fixture
def hooks():
    spec = importlib.util.spec_from_file_location("codex_checks", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".codex-state/\n.env\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_session_baseline_skips_existing_dirty_code(hooks, project, monkeypatch):
    """B1: начальное грязное дерево не считается правкой агента."""
    calls = []
    monkeypatch.setattr(hooks, "run_check", lambda *args: calls.append(args) or None)
    hooks.handle("start", {"session_id": "s"}, project)
    assert hooks.handle("stop", {"session_id": "s"}, project) is None
    assert calls == []


def test_snapshot_never_reads_dialogs_or_ignored_files(hooks, project, monkeypatch):
    """B1: исключение применяется до чтения содержимого файлов."""
    dialogs = project / "docs" / "dialogs"
    dialogs.mkdir(parents=True)
    (dialogs / "private.md").write_text("private", encoding="utf-8")
    (project / ".env").write_text("secret", encoding="utf-8")
    original = Path.read_bytes

    def read(path):
        assert "dialogs" not in path.parts
        assert path.name != ".env"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    snapshot = hooks.snapshot(project)
    assert "src/a.py" in snapshot
    assert "docs/dialogs/private.md" not in snapshot
    assert ".env" not in snapshot


def test_file_checks_cache_content_and_skip_deletions(hooks, project, monkeypatch):
    """B2: проверять только изменённые версии существующих Python-файлов."""
    calls = []
    monkeypatch.setattr(hooks, "run_check", lambda *args: calls.append(args) or None)
    payload = {"session_id": "s"}
    hooks.handle("start", payload, project)
    path = project / "src" / "a.py"
    path.write_text("x = 2\n", encoding="utf-8")
    hooks.handle("file", payload, project)
    assert len(calls) == 2
    assert calls[0][1] == ("ruff", "check", "src/a.py")
    hooks.handle("file", payload, project)
    assert len(calls) == 2
    path.unlink()
    hooks.handle("file", payload, project)
    assert len(calls) == 2


@pytest.mark.parametrize("relative", ["src/a.py", "src/playbook.yaml", "pyproject.toml"])
def test_stop_checks_code_and_config_once(hooks, project, monkeypatch, relative):
    """B3: три проверки кэшируются до следующего изменения содержимого."""
    calls = []
    monkeypatch.setattr(hooks, "run_check", lambda *args: calls.append(args) or None)
    payload = {"session_id": "s"}
    hooks.handle("start", payload, project)
    (project / relative).write_text("# change\n", encoding="utf-8")
    hooks.handle("stop", payload, project)
    assert [call[1][0] for call in calls] == ["ruff", "ruff", "pytest"]
    hooks.handle("stop", payload, project)
    assert len(calls) == 3
    (project / relative).write_text("# another change\n", encoding="utf-8")
    hooks.handle("stop", payload, project)
    assert len(calls) == 6


def test_docs_only_returns_context_without_tests(hooks, project, monkeypatch):
    """B4: документация вызывает напоминание, а не полный pytest."""
    calls = []
    monkeypatch.setattr(hooks, "run_check", lambda *args: calls.append(args) or None)
    payload = {"session_id": "s"}
    hooks.handle("start", payload, project)
    (project / "README.md").write_text("# project\n", encoding="utf-8")
    result = hooks.handle("stop", payload, project)
    assert result["decision"] == "block"
    assert "mlreview-docs-sync" in result["reason"]
    assert calls == []
    assert hooks.handle("stop", payload, project) is None


def test_state_is_session_scoped_and_safe(hooks, project):
    """B5: session_id не позволяет записать состояние вне служебного каталога."""
    first = hooks.state_path(project, {"session_id": "../../escape"})
    second = hooks.state_path(project, {"session_id": "other"})
    assert first.parent == project / ".codex-state"
    assert first != second


def test_failed_checks_all_run_and_are_not_cached(hooks, project, monkeypatch):
    """E1: ошибки всех команд видны и не превращаются в успешный кэш."""
    calls = []

    def fail(*args):
        calls.append(args)
        return "failed check"

    monkeypatch.setattr(hooks, "run_check", fail)
    result = hooks.handle("stop", {"session_id": "s"}, project)
    assert result["decision"] == "block"
    assert len(calls) == 3
    hooks.handle("stop", {"session_id": "s"}, project)
    assert len(calls) == 6


def test_run_check_reports_missing_module_and_timeout(hooks, project, monkeypatch):
    """E1: пропавшая зависимость и таймаут — ошибки с командой."""
    # Git snapshot uses subprocess too; this test calls only the check runner.
    monkeypatch.setattr(
        hooks.subprocess,
        "run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args, 1, "", "No module named ruff"),
    )
    assert "No module named ruff" in hooks.run_check(project, ("ruff", "check", "."))

    def timeout(*args, **_kwargs):
        raise subprocess.TimeoutExpired(args[0], 240)

    monkeypatch.setattr(hooks.subprocess, "run", timeout)
    assert "240" in hooks.run_check(project, ("pytest", "-q"))


def test_repeat_stop_does_not_loop(hooks, project, monkeypatch):
    """E2: повторный Stop не запускает команды и не блокирует завершение."""
    monkeypatch.setattr(hooks, "run_check", lambda *_args: pytest.fail("unexpected check"))
    assert hooks.handle("stop", {"stop_hook_active": True}, project) is None


@pytest.mark.parametrize("content", ["{broken", "[]", '{"observed": []}'])
def test_corrupt_state_falls_back_to_checks(hooks, project, monkeypatch, content):
    """E3: повреждённый снимок не маскирует существующий код."""
    path = hooks.state_path(project, {"session_id": "s"})
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")
    calls = []
    monkeypatch.setattr(hooks, "run_check", lambda *args: calls.append(args) or None)
    hooks.handle("stop", {"session_id": "s"}, project)
    assert len(calls) == 3


def test_cli_from_subdirectory_and_invalid_payload(project):
    """B5/E4: cwd не влияет на корень, повреждённый stdin не означает успех."""
    hook_path = project / ".codex" / "hooks" / "checks.py"
    hook_path.parent.mkdir(parents=True)
    hook_path.write_bytes(HOOK.read_bytes())
    proc = subprocess.run(
        ["python", str(hook_path), "start"],
        cwd=project / "src",
        input=json.dumps({"session_id": "s"}),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert list((project / ".codex-state").glob("*.json"))
    proc = subprocess.run(
        ["python", str(hook_path), "stop"],
        input="{broken",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "JSON" in proc.stderr
