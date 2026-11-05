"""Манифесты плейбуков и их загрузка из каталога (FR-E02, внутренний вариант).

Плейбук — каталог с `playbook.yaml` (данные) и необязательным `checks.py` с функцией
`run(ctx) -> CheckResults`. Поведение зафиксировано в `docs/specs/playbooks.md`.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mlreview.core.findings import CheckResults
from mlreview.core.model_info import FAMILIES
from mlreview.core.task_spec import TASK_TYPES

MANIFEST = "playbook.yaml"
CHECKS = "checks.py"
BUILTIN_ROOT = Path(__file__).parent


class PlaybookError(ValueError):
    """Плейбук не загружен: сообщение с путём файла и полем."""


def _known_families(families: list[str]) -> list[str]:
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(
            f"неизвестные семейства {', '.join(unknown)}; допустимо: {', '.join(FAMILIES)}"
        )
    return families


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppliesTo(_Strict):
    families: list[str]
    task_types: list[str] = Field(default_factory=lambda: list(TASK_TYPES))

    _families = field_validator("families")(_known_families)

    @field_validator("task_types")
    @classmethod
    def _known_task_types(cls, task_types: list[str]) -> list[str]:
        unknown = [task_type for task_type in task_types if task_type not in TASK_TYPES]
        if unknown:
            raise ValueError(f"неизвестные типы задач {', '.join(unknown)}")
        return task_types


class ParamSpec(_Strict):
    aliases: list[str]
    unbounded: list[Any] = Field(default_factory=list)


class CheckText(_Strict):
    suggestion: str


class Manifest(_Strict):
    """Манифест плейбука: применимость, распознавание, гиперпараметры, пороги, тексты."""

    id: str
    title: str
    requirement: str
    applies_to: AppliesTo
    detect: dict[str, list[str]] = Field(default_factory=dict)
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    thresholds: dict[str, int | float] = Field(default_factory=dict)
    checks: dict[str, CheckText] = Field(default_factory=dict)
    overrides: dict[str, str] = Field(default_factory=dict)

    @field_validator("detect")
    @classmethod
    def _detect_families(cls, detect: dict[str, list[str]]) -> dict[str, list[str]]:
        _known_families(list(detect))
        return detect


@dataclass(frozen=True)
class Playbook:
    """Загруженный плейбук: манифест и функция проверок (или `None`)."""

    manifest: Manifest
    run: Callable[[Any], CheckResults] | None = None

    @property
    def id(self) -> str:
        return self.manifest.id

    def families_of(self, short_name: str) -> list[str]:
        """Семейства, шаблоны `detect` которых распознают короткое имя класса."""
        return [
            family
            for family, patterns in self.manifest.detect.items()
            if any(fnmatchcase(short_name, pattern) for pattern in patterns)
        ]

    def applies(self, family: str | None, task_type: str) -> bool:
        applies_to = self.manifest.applies_to
        return family in applies_to.families and task_type in applies_to.task_types


def _manifest(path: Path, expected_id: str) -> Manifest:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PlaybookError(f"{path}: не удалось прочитать: {exc}") from exc
    if not isinstance(data, dict):
        raise PlaybookError(f"{path}: ожидается словарь")
    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as exc:
        error = exc.errors()[0]
        field = ".".join(str(part) for part in error["loc"])
        message = "неизвестное поле" if error["type"] == "extra_forbidden" else error["msg"]
        raise PlaybookError(f"{path}: {field}: {message}") from None
    if manifest.id != expected_id:
        raise PlaybookError(f"{path}: id: ожидается «{expected_id}» — имя каталога плейбука")
    return manifest


def _checks(path: Path, playbook_id: str) -> Callable[[Any], CheckResults] | None:
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"_mlreview_playbook_{playbook_id}", path)
    if spec is None or spec.loader is None:
        raise PlaybookError(f"{path}: не удалось загрузить модуль")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise PlaybookError(f"{path}: нет функции run")
    return run


def load_playbooks(root: str | Path) -> tuple[Playbook, ...]:
    """Плейбуки каталога: по подкаталогу с `playbook.yaml`, в порядке имён (B1)."""
    root = Path(root)
    playbooks = []
    for directory in sorted(path for path in root.iterdir() if (path / MANIFEST).is_file()):
        manifest = _manifest(directory / MANIFEST, directory.name)
        playbooks.append(Playbook(manifest, _checks(directory / CHECKS, manifest.id)))
    return tuple(playbooks)


@functools.cache
def builtin_playbooks() -> tuple[Playbook, ...]:
    """Встроенные плейбуки пакета; загружаются один раз (B2)."""
    return load_playbooks(BUILTIN_ROOT)
