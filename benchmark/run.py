"""Прогон бенчмарка: ревью каждой версии проекта, recall дефектов (Q1) и ложные critical (Q2).

Спецификация — `docs/specs/benchmark.md`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark.projects import PROJECTS as PROJECT_DEFINITIONS
from benchmark.projects import VERSIONS, Project, build_runs
from mlreview.analyzers.evaluation import is_worse, prepare, primary_metric
from mlreview.analyzers.review import review
from mlreview.core.findings import Finding, Report

PROJECTS = tuple(PROJECT_DEFINITIONS)
GATE_RECALL = 0.5
GATE_FALSE_CRITICAL = 3
INSTALL_HINT = "бенчмарку нужен scikit-learn: pip install -r requirements.txt"

__all__ = ["PROJECTS", "VERSIONS", "BenchmarkResult", "VersionResult", "main", "run_benchmark"]


@dataclass(frozen=True)
class VersionResult:
    """Результат одной версии проекта; `found` — `None` для чистой версии."""

    project: str
    version: str
    found: bool | None
    critical: int
    overall_improvement: float
    report: Report


@dataclass(frozen=True)
class BenchmarkResult:
    versions: list[VersionResult]

    @property
    def recall(self) -> float:
        defects = [v for v in self.versions if v.found is not None]
        return sum(bool(v.found) for v in defects) / len(defects) if defects else 0.0

    @property
    def false_critical(self) -> float:
        clean = [v for v in self.versions if v.found is None]
        return sum(v.critical for v in clean) / len(clean) if clean else 0.0

    @property
    def passed(self) -> bool:
        return self.recall >= GATE_RECALL and self.false_critical <= GATE_FALSE_CRITICAL


def _is_found(project: Project, version: str, findings: list[Finding]) -> bool | None:
    if version == "clean":
        return None
    if version == "B12":
        return any(
            f.check == "slices.segment_regression"
            and f.facts["segment"] == project.segment
            and f.facts["value"] == project.corrupted_value
            for f in findings
        )
    check = {"B01": "data.split_duplicates", "B02": "data.temporal_leak"}[version]
    return any(f.check == check and f.severity == "critical" for f in findings)


def run_version(project: Project, version: str) -> VersionResult:
    runs = build_runs(project, version)
    report = review(runs.spec, runs.candidate, runs.baseline)
    metric = primary_metric(runs.spec.task)
    delta = metric(prepare(runs.spec.task, runs.candidate.predictions)) - metric(
        prepare(runs.spec.task, runs.baseline.predictions)
    )
    improvement = -abs(delta) if is_worse(metric, delta) else abs(delta)
    return VersionResult(
        project=project.name,
        version=version,
        found=_is_found(project, version, report.findings),
        critical=sum(f.severity == "critical" for f in report.findings),
        overall_improvement=improvement,
        report=report,
    )


def run_benchmark(projects: Sequence[str] = PROJECTS) -> BenchmarkResult:
    """Ревью всех версий выбранных проектов."""
    return BenchmarkResult(
        [
            run_version(PROJECT_DEFINITIONS[name], version)
            for name in projects
            for version in VERSIONS
        ]
    )


def _describe(result: VersionResult) -> str:
    if result.found is None:
        return f"чистая версия, critical: {result.critical}"
    return "найден" if result.found else "НЕ НАЙДЕН"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__)
    parser.add_argument(
        "--project", action="append", choices=PROJECTS, help="проект (можно несколько)"
    )
    args = parser.parse_args(argv)
    try:
        import sklearn  # noqa: F401, PLC0415 — проверка зависимости бенчмарка
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    result = run_benchmark(tuple(args.project or PROJECTS))
    width = max(len(name) for name in PROJECTS)
    print(f"{'проект':<{width}}  версия  результат")
    for version in result.versions:
        print(f"{version.project:<{width}}  {version.version:<6}  {_describe(version)}")
    defects = [v for v in result.versions if v.found is not None]
    found = sum(bool(v.found) for v in defects)
    gate = "пройден" if result.passed else "НЕ пройден"
    print(
        f"Q1 (recall дефектов): {result.recall:.0%} ({found}/{len(defects)}) · "
        f"Q2 (critical на чистый проект): {result.false_critical:.1f} · гейт L0: {gate}"
    )
    return 0 if result.passed else 1
