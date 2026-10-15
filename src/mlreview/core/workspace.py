"""Рабочее пространство проекта `.mlreview/`: настройки, хранилище, отчёты (FR-C05, FR-L01).

Каталог ищется от текущего вверх, как `.git`; `MLREVIEW_HOME` задаёт его явно. Данные
не покидают машину (NFR-04). Поведение зафиксировано в `docs/specs/findings-store.md`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

WORKSPACE_DIR = ".mlreview"
ENV_HOME = "MLREVIEW_HOME"
INIT_HINT = "хранилище находок не подключено — mlreview init"

CONFIG_TEMPLATE = """\
# Настройки MLReview для проекта (docs/quickstart.md).
# LLM выключен: все детерминированные находки работают без него (NFR-03).
# Пример для Anthropic:   llm: {provider: anthropic, model: claude-sonnet-5-5}
# Пример для Ollama:      llm: {provider: openai, base_url: "http://localhost:11434/v1",
#                               model: llama3.1, api_key_env: null}
llm:
  provider: none
"""


class WorkspaceError(RuntimeError):
    """Рабочее пространство не найдено или настройки не читаются."""


@dataclass(frozen=True)
class Workspace:
    """Каталог `.mlreview` и пути внутри него."""

    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    @property
    def store_path(self) -> Path:
        return self.root / "store.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def llm_log_path(self) -> Path:
        return self.root / "llm-log.jsonl"

    def config(self) -> dict[str, Any]:
        """Настройки из `config.yaml`; нет файла — пустой словарь."""
        if not self.config_path.is_file():
            return {}
        try:
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise WorkspaceError(f"не удалось прочитать {self.config_path}: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise WorkspaceError(f"{self.config_path}: ожидается словарь настроек")
        return dict(data)


def find_workspace(start: Path | None = None) -> Workspace | None:
    """Рабочее пространство из `MLREVIEW_HOME` или ближайшее вверх от `start`."""
    home = os.environ.get(ENV_HOME)
    if home:
        return Workspace(Path(home))
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / WORKSPACE_DIR
        if candidate.is_dir():
            return Workspace(candidate)
    return None


def init_workspace(directory: Path | None = None) -> tuple[Workspace, bool]:
    """Создаёт `.mlreview/` с шаблоном настроек; второе значение — создано ли сейчас."""
    root = (directory or Path.cwd()) / WORKSPACE_DIR
    created = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(root)
    if not workspace.config_path.exists():
        workspace.config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return workspace, created
