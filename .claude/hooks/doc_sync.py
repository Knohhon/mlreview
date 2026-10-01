"""Актуализация документации после изменений (хук Stop и скилл docs-sync).

Спецификация: docs/specs/doc-sync.md.

Режимы:
  stop    — хук Stop: если файлы изменились с прошлого снимка, возвращает Claude (код 2)
            список изменений, автоматические находки и инструкцию по актуализации.
  report  — напечатать автоматические находки по всем markdown (битые ссылки, неизвестные FR/NFR).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
STATE = Path(".claude") / "state" / "doc-sync.json"
ANALYSIS = Path("docs") / "MLReview-analysis.md"

EXCLUDED_PREFIXES = ("docs/dialogs/", ".claude/state/", ".git/")
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
NOT_PATH_CHARS = set("<>*{}…$ ")
INLINE_CODE = re.compile(r"(`+)(.+?)\1")
REQUIREMENT = re.compile(r"\b(FR-[A-Z]\d{2}|NFR-\d{2})\b")
DEFINED_REQUIREMENT = re.compile(r"^\|\s*(FR-[A-Z]\d{2}|NFR-\d{2})\s*\|", re.MULTILINE)
MAX_LISTED = 50

IsIgnored = Callable[[str], bool]


def _excluded(relative: str) -> bool:
    return relative.startswith(EXCLUDED_PREFIXES)


def _list_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and Path(root, ".git").exists():
        return [p for p in proc.stdout.split("\0") if p]
    return [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not EXCLUDED_DIRS.intersection(path.relative_to(root).parts)
    ]


def snapshot(root: Path, exclude: Iterable[str] = ()) -> dict[str, str]:
    """Снимок проекта: путь → sha256 содержимого (без docs/dialogs/ и служебных файлов)."""
    skip = set(exclude)
    result = {}
    for relative in _list_files(root):
        path = root / relative
        if _excluded(relative) or relative in skip or not path.is_file():
            continue
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def changed(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Пути, добавленные, изменённые или удалённые между снимками."""
    return sorted(path for path in old.keys() | new.keys() if old.get(path) != new.get(path))


def _markdown_files(root: Path) -> list[str]:
    candidates = [root / "CLAUDE.md", root / "README.md"]
    candidates += (root / "docs").rglob("*.md")
    candidates += (root / ".claude" / "skills").rglob("*.md")
    relatives = {p.relative_to(root).as_posix() for p in candidates if p.is_file()}
    return sorted(r for r in relatives if not _excluded(r))


def _inline_tokens(text: str) -> Iterable[tuple[int, str]]:
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in INLINE_CODE.finditer(line):
            yield number, match.group(2).strip().strip("`").strip()


def _as_path(token: str) -> str | None:
    if (
        "/" not in token
        or "://" in token
        or token.startswith(("/", "-", "~"))
        or NOT_PATH_CHARS.intersection(token)
    ):
        return None
    path = token.split("::", 1)[0]
    return re.sub(r":\d+$", "", path)


def git_ignored(root: Path) -> IsIgnored:
    def is_ignored(relative: str) -> bool:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=root, capture_output=True, check=False
        )
        return proc.returncode == 0

    return is_ignored


def broken_links(root: Path, is_ignored: IsIgnored) -> list[str]:
    """Ссылки на несуществующие пути в обратных кавычках."""
    findings = []
    for relative in _markdown_files(root):
        text = (root / relative).read_text(encoding="utf-8")
        for number, token in _inline_tokens(text):
            path = _as_path(token)
            if path and not (root / path).exists() and not is_ignored(path):
                findings.append(f"{relative}:{number}: битая ссылка `{token}`")
    return findings


def unknown_requirements(root: Path) -> list[str]:
    """FR/NFR, упомянутые в документации Claude, спецификациях и коде, но не определённые."""
    analysis = root / ANALYSIS
    defined = set(DEFINED_REQUIREMENT.findall(analysis.read_text(encoding="utf-8")))
    sources = [r for r in _markdown_files(root) if not r.startswith("docs/") or "/specs/" in r]
    sources += [p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")]
    findings = []
    for relative in sorted(sources):
        text = (root / relative).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            findings.extend(
                f"{relative}:{number}: неизвестное требование {rid}"
                for rid in REQUIREMENT.findall(line)
                if rid not in defined
            )
    return findings


def findings(root: Path, is_ignored: IsIgnored) -> list[str]:
    result = broken_links(root, is_ignored)
    if (root / ANALYSIS).is_file():
        result += unknown_requirements(root)
    return result


def _load(state: Path) -> dict[str, str] | None:
    try:
        data = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _save(state: Path, snap: dict[str, str]) -> None:
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(snap, indent=0, sort_keys=True), encoding="utf-8")


def _message(paths: list[str], found: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in paths[:MAX_LISTED])
    if len(paths) > MAX_LISTED:
        listed += f"\n- … и ещё {len(paths) - MAX_LISTED}"
    auto = "\n".join(f"- {f}" for f in found) or "- нет"
    return (
        "Изменились файлы с прошлой актуализации документации:\n"
        f"{listed}\n\n"
        "До завершения хода (скилл docs-sync):\n"
        "1. Внутренние файлы Claude — CLAUDE.md, .claude/skills/*/SKILL.md, "
        ".claude/settings.json: сверь с изменениями и актуализируй сам, если описанное устарело.\n"
        "2. Остальные markdown — README.md, docs/*.md, docs/specs/*.md (docs/dialogs/ не читать): "
        "проверь на несоответствия изменениям. Сам не правь: перечисли пользователю расхождения "
        "и предложи конкретные правки.\n"
        "3. Если расхождений нет — коротко скажи об этом.\n\n"
        f"Автоматические находки:\n{auto}"
    )


def run_stop(payload: dict, root: Path, state: Path, is_ignored: IsIgnored) -> tuple[int, str]:
    exclude = []
    if state.resolve().is_relative_to(root.resolve()):
        exclude.append(state.resolve().relative_to(root.resolve()).as_posix())
    current = snapshot(root, exclude)
    previous = _load(state)
    if payload.get("stop_hook_active") or previous is None:
        _save(state, current)
        return 0, ""
    paths = changed(previous, current)
    if not paths:
        return 0, ""
    return 2, _message(paths, findings(root, is_ignored))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "report":
        print("\n".join(findings(PROJECT, git_ignored(PROJECT))) or "Расхождений не найдено.")
        return
    if mode == "stop":
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError:
            payload = {}
        code, message = run_stop(payload, PROJECT, PROJECT / STATE, git_ignored(PROJECT))
        if message:
            print(message, file=sys.stderr)
        sys.exit(code)


if __name__ == "__main__":
    main()
