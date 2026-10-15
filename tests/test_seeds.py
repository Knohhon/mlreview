"""Разброс по сидам и «улучшение в пределах шума» (FR-D05, B08).

Спецификация: docs/specs/seeds.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlreview.analyzers.seeds import seed_findings
from mlreview.core.run_contract import build_run_contract
from mlreview.core.task_spec import parse_task_spec

SPEC = parse_task_spec("task: {type: regression, output: values}\nmetrics: {primary: mae}\n")
N = 200


def _run(run_id: str, mae: float, *, label_shift: float = 0.0):
    """Регрессия, у которой MAE ровно `mae`: ошибка ±mae по знаку из фиксированного узора."""
    rng = np.random.default_rng(0)
    labels = rng.normal(0, 1, N) + label_shift
    signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    predictions = pd.DataFrame(
        {
            "example_id": [f"e{i:03d}" for i in range(N)],
            "label": labels,
            "prediction": labels + signs * mae,
        }
    )
    return build_run_contract(SPEC, run_id=run_id, predictions=predictions)


def _side(prefix, values):
    return [_run(f"{prefix}{i}", value) for i, value in enumerate(values, start=1)]


def test_improvement_within_noise():
    """B1: разница меньше разброса по сидам — high."""
    results = seed_findings(SPEC, _side("c", [0.98, 1.08, 0.88]), _side("b", [1.0, 1.1, 0.9]))
    (finding,) = results.findings
    assert (finding.check, finding.severity) == ("seeds.within_noise", "high")
    facts = finding.facts
    assert facts["metric"] == "mae"
    assert facts["baseline_mean"] == 1.0
    assert facts["candidate_mean"] == 0.98
    assert facts["delta"] == -0.02
    assert facts["baseline_std"] == 0.1
    assert facts["candidate_std"] == 0.1
    assert facts["baseline_seeds"] == 3
    assert facts["candidate_seeds"] == 3
    assert facts["p_value"] > 0.05
    assert facts["seeds_needed"] == 393
    assert finding.message.startswith(
        "улучшение в пределах шума по сидам: mae 1.000 → 0.980 (Δ -0.020), σ по сидам 0.100"
    )
    assert "(Уэлч, 3 и 3 сида)" in finding.message
    assert finding.message.endswith("для эффекта такого размера нужно ≈ 393 сида на конфигурацию")
    assert "393" in finding.suggestion
    assert finding.requirement == "FR-D05, B08"


def test_decline_within_noise_is_info():
    """B1: ухудшение в пределах шума — info."""
    results = seed_findings(SPEC, _side("c", [1.02, 1.12, 0.92]), _side("b", [1.0, 1.1, 0.9]))
    (finding,) = results.findings
    assert (finding.check, finding.severity) == ("seeds.within_noise", "info")


@pytest.mark.parametrize(
    ("candidate", "severity", "start"),
    [
        ([0.50, 0.52, 0.48], "info", "улучшение устойчиво по сидам"),
        ([1.50, 1.52, 1.48], "high", "ухудшение устойчиво по сидам"),
    ],
)
def test_significant_difference(candidate, severity, start):
    """B2: значимая разница по сидам."""
    results = seed_findings(SPEC, _side("c", candidate), _side("b", [1.0, 1.1, 0.9]))
    (finding,) = results.findings
    assert (finding.check, finding.severity) == ("seeds.significant", severity)
    assert finding.message.startswith(start)


def test_single_seed_is_not_evaluated():
    """B3: у кандидата один прогон."""
    results = seed_findings(SPEC, _side("c", [0.9]), _side("b", [1.0]))
    (finding,) = results.findings
    assert (finding.check, finding.severity) == ("seeds.not_evaluated", "low")
    assert finding.message == (
        "разброс по сидам не оценён: у кандидата 1 прогон, у базового 1 — интервал только"
        " по примерам; нужно хотя бы 3 сида на конфигурацию"
    )


def test_spread_without_baseline():
    """B4: разброс трёх сидов без базового прогона."""
    results = seed_findings(SPEC, _side("c", [1.0, 1.1, 0.9]), None)
    (finding,) = results.findings
    assert (finding.check, finding.severity) == ("seeds.spread", "info")
    assert finding.message == "разброс по 3 сидам: mae 1.000 ± 0.100 (min 0.900, max 1.100)"


def test_facts_are_reproducible():
    """B5: одинаковые входы — одинаковые факты."""
    first = seed_findings(SPEC, _side("c", [0.98, 1.08, 0.88]), _side("b", [1.0, 1.1, 0.9]))
    second = seed_findings(SPEC, _side("c", [0.98, 1.08, 0.88]), _side("b", [1.0, 1.1, 0.9]))
    assert [f.to_dict() for f in first.findings] == [f.to_dict() for f in second.findings]


def test_single_run_without_baseline_is_skipped():
    """E1: один прогон без базового."""
    results = seed_findings(SPEC, _side("c", [1.0]), None)
    assert results.findings == []
    assert [(s.check, s.reason) for s in results.skipped] == [
        ("seeds.spread", "один прогон — разброс по сидам не оценён")
    ]


def test_seeds_on_different_data_are_skipped():
    """E2: сид на других данных."""
    candidates = [*_side("c", [1.0, 1.1]), _run("c3", 0.9, label_shift=1.0)]
    results = seed_findings(SPEC, candidates, _side("b", [1.0, 1.1, 0.9]))
    assert results.findings == []
    (skipped,) = results.skipped
    assert skipped.reason.startswith("сиды оценены на разных данных")
    assert "c3" in skipped.reason


def test_undefined_metric_is_skipped():
    """E3: метрика не определена на сиде (один класс) — пропуск только для неё."""
    spec = parse_task_spec("task: {type: classification/binary, output: probabilities}\n")

    def run(run_id, shift):
        predictions = pd.DataFrame(
            {
                "example_id": [f"e{i}" for i in range(50)],
                "label": [0] * 50,
                "prediction": np.linspace(0.1, 0.4, 50) + shift,
            }
        )
        return build_run_contract(spec, run_id=run_id, predictions=predictions)

    results = seed_findings(spec, [run("c1", 0), run("c2", 0.1)], [run("b1", 0), run("b2", 0.2)])
    assert "roc_auc не определена на прогоне c1" in results.skipped[0].reason
    assert {f.facts["metric"] for f in results.findings} == {"log_loss", "brier"}


def test_every_task_metric_is_checked():
    """B6: по находке на метрику; дополнительная — medium; нулевая разница — info."""
    spec = parse_task_spec(
        "task: {type: regression, output: values}\nmetrics: {primary: mae, secondary: [rmse]}\n"
    )
    results = seed_findings(spec, _side("c", [0.98, 1.08, 0.88]), _side("b", [1.0, 1.1, 0.9]))
    by_metric = {f.facts["metric"]: (f.check, f.severity) for f in results.findings}
    assert by_metric == {
        "mae": ("seeds.within_noise", "high"),
        "rmse": ("seeds.within_noise", "medium"),
    }
    same = seed_findings(spec, _side("c", [1.0, 1.1, 0.9]), _side("b", [1.0, 1.1, 0.9]))
    for finding in same.findings:
        assert finding.severity == "info"
        assert finding.message.startswith("разницы по сидам нет: ")
