"""Плейбуки: пакеты знаний по семействам моделей и типам задач — метрики, сбои, проверки.

Манифест YAML и Python-проверки (FR-E02, внутренний вариант), проверки и подсказки по
семейству модели (FR-D11). Поведение зафиксировано в `docs/specs/playbooks.md`.
"""

from mlreview.playbooks import loader
from mlreview.playbooks.loader import Manifest, Playbook, PlaybookError, load_playbooks
from mlreview.playbooks.resolve import model_params, resolve_family


def builtin_playbooks() -> tuple[Playbook, ...]:
    """Встроенные плейбуки пакета (подменяются в тестах через `loader.builtin_playbooks`)."""
    return loader.builtin_playbooks()


__all__ = [
    "Manifest",
    "Playbook",
    "PlaybookError",
    "builtin_playbooks",
    "load_playbooks",
    "loader",
    "model_params",
    "resolve_family",
]
