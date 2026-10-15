"""Встроенные анализаторы в реестре (FR-E01): данные, срезы, метрики, кривые, процесс.

Порядок регистрации задаёт порядок находок и пропусков внутри группы.
"""

from __future__ import annotations

from mlreview.analyzers.calibration import calibration_findings
from mlreview.analyzers.comparison import compare_runs, run_metrics
from mlreview.analyzers.curves import curve_findings
from mlreview.analyzers.data_checks import data_findings
from mlreview.analyzers.process import process_findings
from mlreview.analyzers.registry import Analyzer, ReviewContext, register
from mlreview.analyzers.seeds import seed_findings
from mlreview.analyzers.slices import segment_findings
from mlreview.core.findings import CheckResults


def _metrics(ctx: ReviewContext) -> CheckResults:
    if ctx.baseline is None:
        return run_metrics(ctx.spec, ctx.contract)
    return compare_runs(ctx.spec, ctx.contract, ctx.baseline)


BUILTIN = (
    Analyzer("data_checks", "data", "FR-D03", lambda ctx: data_findings(ctx.spec, ctx.contract)),
    Analyzer(
        "slices",
        "segments",
        "FR-D02",
        lambda ctx: segment_findings(ctx.spec, ctx.contract, ctx.baseline),
    ),
    Analyzer("comparison", "metrics", "FR-D01", _metrics),
    Analyzer(
        "seeds",
        "metrics",
        "FR-D05",
        lambda ctx: seed_findings(ctx.spec, ctx.candidates, ctx.baselines or None),
    ),
    Analyzer(
        "calibration",
        "metrics",
        "FR-D08",
        lambda ctx: calibration_findings(ctx.spec, ctx.contract, ctx.baseline),
    ),
    Analyzer("curves", "curves", "FR-D04", lambda ctx: curve_findings(ctx.contract.curves)),
    Analyzer("process", "process", "FR-D06", process_findings),
)

for _analyzer in BUILTIN:
    register(_analyzer)
