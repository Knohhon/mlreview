"""Сохраняет текущий диалог с Claude в docs/dialogs/ (хуки Stop и SessionEnd).

Пишет реплики пользователя и ответы Claude в Markdown; вызовы инструментов — одной строкой,
результаты инструментов и размышления не сохраняются. Один файл на сессию, перезаписывается.
Ничего не печатает: содержимое диалога не должно возвращаться в контекст Claude.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
DIALOGS = PROJECT / "docs" / "dialogs"
SYSTEM_TAGS = re.compile(
    r"<(system-reminder|local-command-stdout|local-command-caveat)>.*?</\1>", re.DOTALL
)


def clean(text: str) -> str:
    return SYSTEM_TAGS.sub("", text).strip()


def tool_line(block: dict) -> str:
    tool_input = block.get("input") or {}
    detail = (
        tool_input.get("description")
        or tool_input.get("file_path")
        or tool_input.get("skill")
        or tool_input.get("pattern")
        or ""
    )
    detail = str(detail).splitlines()[0][:120] if detail else ""
    return f"- `{block.get('name', '?')}`" + (f": {detail}" if detail else "")


def render(entries: list[dict]) -> str:
    parts: list[str] = []
    last_role = None
    for entry in entries:
        if entry.get("isSidechain") or entry.get("isMeta"):
            continue
        role = entry.get("type")
        content = (entry.get("message") or {}).get("content")
        if role == "user" and isinstance(content, str):
            text = clean(content)
            if text:
                parts.append(f"## Пользователь\n\n{text}")
                last_role = "user"
        elif role == "assistant" and isinstance(content, list):
            for block in content:
                if block.get("type") == "text" and clean(block.get("text", "")):
                    header = "" if last_role == "assistant" else "## Claude\n\n"
                    parts.append(header + clean(block["text"]))
                    last_role = "assistant"
                elif block.get("type") == "tool_use":
                    header = "" if last_role == "assistant" else "## Claude\n\n"
                    parts.append(header + tool_line(block))
                    last_role = "assistant"
    return "\n\n".join(parts)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    transcript = payload.get("transcript_path")
    session_id = str(payload.get("session_id") or "unknown")
    if not transcript or not Path(transcript).is_file():
        return
    entries = []
    with Path(transcript).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    body = render(entries)
    if not body:
        return
    started = next((e["timestamp"][:10] for e in entries if e.get("timestamp")), "unknown")
    DIALOGS.mkdir(parents=True, exist_ok=True)
    target = DIALOGS / f"claude-{started}-{session_id[:8]}.md"
    header = f"# Диалог с Claude · {started} · сессия {session_id}\n\n"
    target.write_text(header + body + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # сохранение диалога не должно ломать работу Claude
        sys.exit(0)
