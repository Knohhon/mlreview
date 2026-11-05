"""Контекст проверок плейбука (FR-D11): ревью, плейбук, семейство, гиперпараметры, пороги."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlreview.analyzers.registry import ReviewContext
from mlreview.playbooks.loader import Playbook


@dataclass(frozen=True)
class PlaybookContext:
    """Что получает `run` плейбука: контекст ревью и нормализованные сведения о модели."""

    review: ReviewContext
    playbook: Playbook
    family: str
    params: Mapping[str, Any]

    @property
    def thresholds(self) -> Mapping[str, float]:
        return self.playbook.manifest.thresholds
