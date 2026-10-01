"""Проверки качества для хуков Claude Code.

Режимы:
  file  — PostToolUse (Edit|Write): ruff check и ruff format --check для изменённого .py-файла.
  stop  — Stop: ruff check, ruff format --check и pytest по всему проекту, если менялся код.

Код выхода 2 возвращает вывод Claude, чтобы он исправил проблему.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
CODE_SUFFIXES = (".py", "pyproject.toml")
MAX_OUTPUT = 4000


def run(*args: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def tools_missing() -> bool:
    ok_ruff, _ = run("ruff", "--version")
    ok_pytest, _ = run("pytest", "--version")
    if ok_ruff and ok_pytest:
        return False
    message = 'ruff или pytest не установлены в окружении: выполните pip install -e ".[dev]"'
    print(json.dumps({"systemMessage": f"Проверки mlreview пропущены: {message}"}))
    return True


def fail(title: str, failures: list[tuple[str, str]]) -> None:
    report = "\n\n".join(f"$ {cmd}\n{out[-MAX_OUTPUT:]}" for cmd, out in failures)
    print(f"{title}\n\n{report}", file=sys.stderr)
    sys.exit(2)


def check_file(payload: dict) -> None:
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or ""
    if not file_path.endswith(".py"):
        return
    path = Path(file_path).resolve()
    if not path.is_file() or PROJECT not in path.parents or tools_missing():
        return
    relative = str(path.relative_to(PROJECT))
    failures = []
    for args in (("ruff", "check", relative), ("ruff", "format", "--check", "--diff", relative)):
        ok, out = run(*args)
        if not ok:
            failures.append((" ".join(args), out))
    if failures:
        fail(f"Линтер или формат не проходят для {relative}:", failures)


def code_changed() -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return True
    return any(line.rstrip().endswith(CODE_SUFFIXES) for line in proc.stdout.splitlines())


def check_project(payload: dict) -> None:
    if payload.get("stop_hook_active") or not code_changed() or tools_missing():
        return
    failures = []
    for args in (("ruff", "check", "."), ("ruff", "format", "--check", "."), ("pytest", "-q")):
        ok, out = run(*args)
        if not ok:
            failures.append((" ".join(args), out))
    if failures:
        fail(
            "Проверки после разработки не прошли. Исправьте их или явно объясните пользователю, "
            "почему они красные (например, этап «тесты написаны, реализации ещё нет»).",
            failures,
        )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    if mode == "file":
        check_file(payload)
    elif mode == "stop":
        check_project(payload)


if __name__ == "__main__":
    main()
