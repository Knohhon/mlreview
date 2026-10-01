"""Актуализация документации после изменений.

Спецификация: docs/specs/doc-sync.md.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "doc_sync.py"
_spec = importlib.util.spec_from_file_location("doc_sync", HOOK)
doc_sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doc_sync)

ANALYSIS = "| ID | Требование |\n| --- | --- |\n| FR-C01 | Спецификация |\n| NFR-01 | Ядро |\n"


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(root: Path) -> Path:
    _write(root, "docs/MLReview-analysis.md", ANALYSIS)
    _write(root, "src/mlreview/cli.py", '"""CLI (FR-C01)."""\n')
    _write(root, "README.md", "Код в `src/mlreview/cli.py`.\n")
    return root


def _no_ignore(_path: str) -> bool:
    return False


# Снимок и изменения


def test_changed_lists_added_modified_and_removed():
    """B1: изменённые, добавленные и удалённые пути, отсортированные."""
    old = {"a.py": "1", "b.md": "2", "c.md": "3"}
    new = {"a.py": "1", "b.md": "changed", "d.py": "4"}
    assert doc_sync.changed(old, new) == ["b.md", "c.md", "d.py"]


def test_snapshot_skips_dialogs(tmp_path):
    """B2: docs/dialogs/ не входит в снимок."""
    _project(tmp_path)
    _write(tmp_path, "docs/dialogs/claude-x.md", "диалог")
    snap = doc_sync.snapshot(tmp_path)
    assert "README.md" in snap
    assert not any(path.startswith("docs/dialogs/") for path in snap)


def test_snapshot_changes_when_content_changes(tmp_path):
    """B1: снимок отражает содержимое файлов."""
    _project(tmp_path)
    before = doc_sync.snapshot(tmp_path)
    _write(tmp_path, "README.md", "Новый текст.\n")
    assert doc_sync.changed(before, doc_sync.snapshot(tmp_path)) == ["README.md"]


# Битые ссылки


def test_broken_path_reference_is_reported(tmp_path):
    """B3: несуществующий путь в обратных кавычках — с файлом и строкой."""
    _project(tmp_path)
    _write(tmp_path, "docs/specs/x.md", "# X\n\nТест: `tests/test_x.py::test_y`.\n")
    assert doc_sync.broken_links(tmp_path, _no_ignore) == [
        "docs/specs/x.md:3: битая ссылка `tests/test_x.py::test_y`"
    ]


def test_existing_and_ignored_paths_are_fine(tmp_path):
    """B3: существующие и игнорируемые git пути не сообщаются."""
    _project(tmp_path)
    _write(tmp_path, "CLAUDE.md", "См. `src/mlreview/` и `.claude/settings.local.json`.\n")
    assert doc_sync.broken_links(tmp_path, lambda p: p.endswith("settings.local.json")) == []


def test_non_path_tokens_are_skipped(tmp_path):
    """B4: код-блоки, пробелы, URL, абсолютные пути, флаги, шаблоны, токены без «/»."""
    _project(tmp_path)
    text = (
        "```bash\ncat missing/file.py\n```\n"
        "`pip install -e x/y` `https://a.b/c` `/etc/x` `--x=a/b` "
        "`docs/specs/<имя>.md` `tests/*.py` `a/{b}` `a/…` `$X/y` `metrics.py`\n"
    )
    _write(tmp_path, "docs/notes.md", text)
    assert doc_sync.broken_links(tmp_path, _no_ignore) == []


def test_checked_markdown_set(tmp_path):
    """B6: проверяются CLAUDE.md, README.md, docs/** кроме dialogs, скиллы."""
    _project(tmp_path)
    for relative in (
        "CLAUDE.md",
        "docs/a/b.md",
        ".claude/skills/s/SKILL.md",
        "docs/dialogs/d.md",
        "other/x.md",
    ):
        _write(tmp_path, relative, "`missing/path.py`\n")
    reported = {line.split(":")[0] for line in doc_sync.broken_links(tmp_path, _no_ignore)}
    assert reported == {"CLAUDE.md", "docs/a/b.md", ".claude/skills/s/SKILL.md"}


# Неизвестные требования


def test_unknown_requirement_ids_are_reported(tmp_path):
    """B5: FR/NFR, которых нет в таблицах анализа, — с файлом и строкой."""
    _project(tmp_path)
    _write(tmp_path, "CLAUDE.md", "Опора: FR-C01, NFR-01.\nИ ещё FR-Z99.\n")
    _write(tmp_path, "src/mlreview/core/x.py", '"""Ядро (NFR-42)."""\n')
    assert doc_sync.unknown_requirements(tmp_path) == [
        "CLAUDE.md:2: неизвестное требование FR-Z99",
        "src/mlreview/core/x.py:1: неизвестное требование NFR-42",
    ]


# Хук целиком


def test_first_run_only_saves_snapshot(tmp_path):
    """B9: без снимка — только сохранить его."""
    _project(tmp_path)
    state = tmp_path / "state.json"
    assert doc_sync.run_stop({}, tmp_path, state, _no_ignore) == (0, "")
    assert json.loads(state.read_text(encoding="utf-8"))


def test_corrupted_state_is_treated_as_missing(tmp_path):
    """E2: повреждённый снимок — как первый запуск."""
    _project(tmp_path)
    state = tmp_path / "state.json"
    state.write_text("{не json", encoding="utf-8")
    assert doc_sync.run_stop({}, tmp_path, state, _no_ignore) == (0, "")


def test_no_changes_is_silent(tmp_path):
    """B7: без изменений хук молчит."""
    _project(tmp_path)
    state = tmp_path / "state.json"
    doc_sync.run_stop({}, tmp_path, state, _no_ignore)
    assert doc_sync.run_stop({}, tmp_path, state, _no_ignore) == (0, "")


def test_changes_return_instructions_and_findings(tmp_path):
    """B7: изменения → код 2, список файлов, находки и инструкция."""
    _project(tmp_path)
    state = tmp_path / "state.json"
    doc_sync.run_stop({}, tmp_path, state, _no_ignore)
    _write(tmp_path, "src/mlreview/cli.py", '"""CLI (FR-Q77)."""\n')
    code, message = doc_sync.run_stop({}, tmp_path, state, _no_ignore)
    assert code == 2
    assert "src/mlreview/cli.py" in message
    assert "FR-Q77" in message
    assert "CLAUDE.md" in message
    assert "предложи" in message


def test_followup_stop_saves_snapshot_silently(tmp_path):
    """B8: повторный вызов в том же ходе запоминает снимок и молчит."""
    _project(tmp_path)
    state = tmp_path / "state.json"
    doc_sync.run_stop({}, tmp_path, state, _no_ignore)
    _write(tmp_path, "CLAUDE.md", "Актуализировано.\n")
    assert doc_sync.run_stop({"stop_hook_active": True}, tmp_path, state, _no_ignore) == (0, "")
    assert doc_sync.run_stop({}, tmp_path, state, _no_ignore) == (0, "")
