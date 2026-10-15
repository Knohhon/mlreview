"""Прогон бенчмарка: ревью каждой версии проекта, recall дефектов (Q1) и ложные critical (Q2).

`--perf` — замер производительности ревью на синтетическом прогоне (NFR-06).

Спецификация — `docs/specs/benchmark.md`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark.perf import DEFAULT_N, measure
from benchmark.projects import PROJECTS as PROJECT_DEFINITIONS
from benchmark.projects import Project, build_runs, versions_for
from mlreview.analyzers.evaluation import is_worse, prepare
from mlreview.analyzers.review import review
from mlreview.core.findings import Finding, Report
from mlreview.metrics.library import primary

PROJECTS = tuple(PROJECT_DEFINITIONS)
GATE_RECALL = 0.7  # гейт L1 (L0 — 0,5)
GATE_FALSE_CRITICAL = 1  # гейт L1 (L0 — 3)
INSTALL_HINT = "бенчмарку нужен scikit-learn: pip install -r requirements.txt"
FOUND_BY: dict[str, tuple[str, str | tuple[str, ...] | None]] = {
    "B01": ("data.split_duplicates", "critical"),
    "B02": ("data.temporal_leak", "critical"),
    "B07": ("process.data_version_mismatch", None),
    "B08": ("seeds.within_noise", ("high", "medium")),
    "B11": ("process.multiple_changes", None),
}

__all__ = ["PROJECTS", "BenchmarkResult", "VersionResult", "main", "run_benchmark"]


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
    if version == "B03":
        return any(
            f.check == "data.strong_feature" and f.facts["feature"] == "leak" for f in findings
        )
    if version == "B05":
        return any(
            f.check == "metrics.unsuitable" and f.facts["metric"] == "accuracy" for f in findings
        )
    check, severity = FOUND_BY[version]
    allowed = (severity,) if isinstance(severity, str) else severity
    return any(f.check == check and (allowed is None or f.severity in allowed) for f in findings)


def run_version(project: Project, version: str) -> VersionResult:
    runs = build_runs(project, version)
    report = review(
        runs.spec,
        runs.candidate,
        runs.baseline,
        candidate_seeds=runs.candidate_seeds,
        baseline_seeds=runs.baseline_seeds,
    )
    metric = primary(runs.spec)
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
            for version in versions_for(name)
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
    parser.add_argument("--perf", action="store_true", help="замер производительности (NFR-06)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="примеров test в замере")
    args = parser.parse_args(argv)
    if args.perf:
        result = measure(args.n)
        print(result.describe())
        return 0 if result.passed else 1
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
        f"Q2 (critical на чистый проект): {result.false_critical:.1f} · гейт L1: {gate}"
    )
    return 0 if result.passed else 1
