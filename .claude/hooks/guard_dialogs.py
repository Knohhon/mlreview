"""PreToolUse (Bash): не даёт Claude читать docs/dialogs/ через оболочку.

Разрешены операции, не раскрывающие содержимое: ls, mv, rm, mkdir, git add/mv/rm/status.
Рекурсивный grep без исключения папки dialogs тоже блокируется.
Эвристика, а не граница безопасности: основная защита — правило в CLAUDE.md.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

SAFE = {"ls", "mv", "rm", "mkdir", "touch"}
SAFE_GIT = {"add", "mv", "rm", "status"}
SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;&\n]")
SCOPED_PATHS = ("src", "tests", ".claude", ".github", "scripts", "pyproject.toml")


def words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def touches_dialogs(argv: list[str]) -> bool:
    mentioned = " ".join(a for a in argv if not a.startswith("--exclude"))
    if not argv or "dialogs" not in mentioned:
        return False
    if argv[0] in SAFE:
        return False
    return not (argv[0] == "git" and len(argv) > 1 and argv[1] in SAFE_GIT)


def unscoped_recursive_grep(argv: list[str]) -> bool:
    if not argv or argv[0] not in {"grep", "egrep", "fgrep"}:
        return False
    flags = [a for a in argv[1:] if a.startswith("-")]
    recursive = any(
        f in {"--recursive", "--dereference-recursive"}
        or (not f.startswith("--") and ("r" in f or "R" in f))
        for f in flags
    )
    if not recursive or any("--exclude-dir" in f and "dialogs" in f for f in flags):
        return False
    paths = [a for a in argv[1:] if not a.startswith("-")][1:]
    return not paths or not all(p.lstrip("./").startswith(SCOPED_PATHS) for p in paths)


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            ensure_ascii=False,
        )
    )
    sys.exit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    command = (payload.get("tool_input") or {}).get("command") or ""
    for segment in SEGMENT_SPLIT.split(command):
        argv = words(segment.strip())
        if touches_dialogs(argv):
            deny(
                "docs/dialogs/ содержит сохранённые диалоги: читать их запрещено (CLAUDE.md). "
                "Разрешены только ls, mv, rm, mkdir и git add/mv/rm/status."
            )
        if unscoped_recursive_grep(argv):
            deny(
                "Рекурсивный grep может прочитать docs/dialogs/. Добавьте --exclude-dir=dialogs "
                "или ограничьте поиск путями src/, tests/ и т. п.; "
                "либо используйте инструмент Grep."
            )


if __name__ == "__main__":
    main()
