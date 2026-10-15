"""Находки анализаторов и отчёт ревью (FR-D01–D03, FR-R01).

Находка — детерминированный факт с важностью и доказательствами; LLM-слой (L1) только
синтезирует гипотезы поверх находок и берёт числа из `facts` (FR-L02). Хранилище находок
со статусами — FR-C05, L1. Поведение зафиксировано в `docs/specs/review.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["critical", "high", "medium", "low", "info"]
SEVERITIES: tuple[Severity, ...] = ("critical", "high", "medium", "low", "info")
EVIDENCE_LIMIT = 10


@dataclass(frozen=True)
class Finding:
    """Находка: проверка, важность, сообщение, опора и доказательства."""

    check: str
    severity: Severity
    message: str
    requirement: str
    evidence: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "requirement": self.requirement,
            "evidence": list(self.evidence),
            "facts": dict(self.facts),
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class Skipped:
    """Невыполненная проверка и причина (X1)."""

    check: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "reason": self.reason}


@dataclass
class CheckResults:
    """Находки и пропуски одной группы проверок."""

    findings: list[Finding] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)

    def skip(self, check: str, reason: str) -> None:
        self.skipped.append(Skipped(check, reason))

    def extend(self, other: CheckResults) -> None:
        self.findings += other.findings
        self.skipped += other.skipped


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Сортирует находки: важность (critical первыми), затем проверка и сообщение."""
    return sorted(findings, key=lambda f: (SEVERITIES.index(f.severity), f.check, f.message))


def evidence_ids(ids: Iterable[Any]) -> tuple[str, ...]:
    """Первые `EVIDENCE_LIMIT` id примеров как строки."""
    result = []
    for value in ids:
        if len(result) == EVIDENCE_LIMIT:
            break
        result.append(str(value))
    return tuple(result)


@dataclass(frozen=True)
class Report:
    """Отчёт ревью прогона: находки по важности и что не проверено."""

    run_id: str
    findings: list[Finding]
    skipped: list[Skipped]
    baseline_id: str | None = None
    data_version: str | None = None
    baseline_data_version: str | None = None
    compared_examples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "baseline_id": self.baseline_id,
            "data_version": self.data_version,
            "baseline_data_version": self.baseline_data_version,
            "compared_examples": self.compared_examples,
            "findings": [finding.to_dict() for finding in self.findings],
            "skipped": [skipped.to_dict() for skipped in self.skipped],
            "llm": None,
        }
