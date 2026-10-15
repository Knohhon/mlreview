"""Находки анализаторов и отчёт ревью (FR-D01–D03, FR-R01, FR-R05).

Находка — детерминированный факт с важностью и доказательствами; LLM-слой только
синтезирует гипотезы поверх находок и берёт числа из `facts` (FR-L02). Отпечаток находки
для хранилища и отметки «не важно» (FR-C05, FR-R05) — `fingerprint` и `magnitude`.
Поведение зафиксировано в `docs/specs/review.md` и `docs/specs/findings-store.md`.
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


# Ключи объекта находки в `facts`: отпечаток не зависит от чисел (docs/specs/findings-store.md).
IDENTITY: dict[str, tuple[str, ...]] = {
    "data.temporal_leak": ("earlier_split", "later_split"),
}
# Факт с величиной эффекта: рост модуля вдвое — существенное изменение условий.
MAGNITUDE: dict[str, str] = {
    "data.split_duplicates": "share",
    "data.temporal_leak": "share",
    "data.strong_feature": "value",
    "slices.segment_worse": "delta",
    "slices.segment_regression": "delta",
    "comparison.metric_delta": "delta",
}


def fingerprint(finding: Finding) -> str:
    """Отпечаток находки: проверка и её объект (строковые факты или ключи из `IDENTITY`)."""
    keys = IDENTITY.get(finding.check)
    if keys is None:
        keys = tuple(sorted(k for k, v in finding.facts.items() if isinstance(v, str)))
    parts = [f"{key}={finding.facts.get(key)}" for key in keys]
    return "|".join([finding.check, *parts])


def magnitude(finding: Finding) -> float | None:
    """Величина эффекта находки или `None`, если для проверки она не задана."""
    key = MAGNITUDE.get(finding.check)
    value = finding.facts.get(key) if key else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def severity_rank(severity: str) -> int:
    """Номер важности: 0 — critical, 4 — info."""
    return SEVERITIES.index(severity)  # type: ignore[arg-type]


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
    data_hash: str | None = None
    baseline_data_hash: str | None = None
    candidate_seeds: tuple[str, ...] = ()
    baseline_seeds: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "baseline_id": self.baseline_id,
            "data_version": self.data_version,
            "baseline_data_version": self.baseline_data_version,
            "data_hash": self.data_hash,
            "baseline_data_hash": self.baseline_data_hash,
            "compared_examples": self.compared_examples,
            "candidate_seeds": list(self.candidate_seeds),
            "baseline_seeds": list(self.baseline_seeds),
            "findings": [finding.to_dict() for finding in self.findings],
            "skipped": [skipped.to_dict() for skipped in self.skipped],
            "llm": None,
        }
