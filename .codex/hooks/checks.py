"""Проверки локального Codex; спецификация: docs/specs/codex-setup.md.

JSON stdin/stdout соответствует событиям SessionStart, PostToolUse и Stop.
Снимки учитывают содержимое, а не начальный git diff; диалоги не читаются.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CHECK_TIMEOUT = 240
MAX_OUTPUT = 4000
MAX_FILES = 30
CONFIG_FILES = frozenset(
    {"pyproject.toml", "requirements.txt", ".pre-commit-config.yaml", ".github/workflows/ci.yml"}
)
EXCLUDED = ("docs/dialogs/", ".codex-state/", ".git/")
SUFFIXES = frozenset({".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt"})


def snapshot(root: Path) -> dict[str, str]:
    """Хеши неигнорируемых файлов без чтения диалогов и внешних symlink."""
    proc = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=20,
    )
    result = {}
    for relative in sorted(set(proc.stdout.decode("utf-8").split("\0"))):
        if not relative or relative.startswith(EXCLUDED):
            continue
        path = root / relative
        resolved = path.resolve()
        if (
            path.suffix not in SUFFIXES
            or path.is_symlink()
            or root not in resolved.parents
            or resolved.relative_to(root).as_posix().startswith(EXCLUDED)
            or not path.is_file()
        ):
            continue
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def state_path(root: Path, payload: dict) -> Path:
    session = str(payload.get("session_id", "manual"))
    name = hashlib.sha256(session.encode("utf-8")).hexdigest()
    return root / ".codex-state" / f"{name}.json"


def load_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict):
        return {}
    for key in ("observed", "checked", "reported"):
        values = state.get(key, {})
        if not isinstance(values, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()
        ):
            return {}
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def changed(old: dict[str, str], new: dict[str, str]) -> list[str]:
    return sorted(p for p in old.keys() | new.keys() if old.get(p) != new.get(p))


def needs_checks(relative: str) -> bool:
    return (
        relative.endswith(".py")
        or relative in CONFIG_FILES
        or (relative.startswith("src/") and relative.endswith((".yaml", ".yml")))
    )


def run_check(root: Path, args: tuple[str, ...]) -> str | None:
    """Возвращает ошибку, включая отсутствующий модуль, либо None при успехе."""
    command = (sys.executable, "-m", *args)
    title = "$ python -m " + " ".join(args)
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"{title}\nТаймаут {CHECK_TIMEOUT} с."
    except OSError as exc:
        return f"{title}\n{type(exc).__name__}: {exc}"
    if proc.returncode:
        output = (proc.stdout + proc.stderr).strip()
        return f"{title}\n{output[-MAX_OUTPUT:]}"
    return None


def commands(mode: str, modified: list[str], current: dict) -> list[tuple[str, ...]]:
    if mode == "file":
        python_files = [p for p in modified if p.endswith(".py") and p in current]
        if not python_files:
            return []
        return [
            ("ruff", "check", *python_files),
            ("ruff", "format", "--check", *python_files),
        ]
    if any(needs_checks(p) for p in modified):
        return [("ruff", "check", "."), ("ruff", "format", "--check", "."), ("pytest", "-q")]
    return []


def handle(mode: str, payload: dict, root: Path = PROJECT) -> dict | None:
    if mode not in {"start", "file", "stop"}:
        raise ValueError("ожидается режим start, file или stop")
    if mode == "stop" and payload.get("stop_hook_active"):
        return None
    path = state_path(root, payload)
    current = snapshot(root)
    state = load_state(path)
    if mode == "start":
        # Повторный start после compact/resume сохраняет непроверенные правки.
        if not state:
            save_state(path, {"observed": current, "checked": current, "reported": current})
        return None
    previous = state.get("observed" if mode == "file" else "checked", {})
    modified = changed(previous, current)
    failures = [
        failure for args in commands(mode, modified, current) if (failure := run_check(root, args))
    ]
    if not failures:
        state["observed" if mode == "file" else "checked"] = current
    documentation = changed(state.get("reported", {}), current) if mode == "stop" else []
    if documentation:
        state["reported"] = current
    save_state(path, state)
    if failures:
        return {
            "decision": "block",
            "reason": (
                "Проверки MLReview не прошли. Исправь ошибки или явно объясни результат "
                "пользователю. При отсутствии инструментов нужно активировать окружение "
                'разработки (python -m pip install -e ".[dev]").\n\n' + "\n\n".join(failures)
            ),
        }
    if documentation:
        listed = "\n".join(documentation[:MAX_FILES])
        return {
            "decision": "block",
            "reason": (
                "Проверь документацию по скиллу mlreview-docs-sync и заверши задачу. "
                "Не читай docs/dialogs/. Изменившиеся файлы "
                f"({len(documentation)}; показано до {MAX_FILES}):\n{listed}"
            ),
        }
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("stdin должен содержать JSON-объект")
        result = handle(sys.argv[1] if len(sys.argv) > 1 else "", payload)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Хук MLReview: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if result:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
