"""Раздел metrics: спецификации задачи (FR-C01, FR-M01–M05).

Спецификация: docs/specs/metrics.md (пункты B1, E1).
"""

from __future__ import annotations

import pytest

from mlreview.core.task_spec import TaskSpecError, load_task_spec, parse_task_spec

TASK = "task: {type: classification/binary, output: probabilities}\n"


def test_valid_metrics_section():
    """B1: допустимые метрики и контекст диалога."""
    spec = parse_task_spec(
        TASK
        + "metrics:\n"
        + "  primary: pr_auc\n"
        + "  secondary: [roc_auc, recall_at_k@50, fbeta@2, ece]\n"
        + "  threshold: 0.3\n"
        + "  context: {decision: ranking, budget_k: 50, cost_ratio: 20, calibration: true}\n"
    )
    assert spec.metrics.primary == "pr_auc"
    assert spec.metrics.secondary == ["roc_auc", "recall_at_k@50", "fbeta@2", "ece"]
    assert spec.metrics.threshold == 0.3
    assert spec.metrics.context.budget_k == 50
    assert spec.metrics.names == ["pr_auc", "roc_auc", "recall_at_k@50", "fbeta@2", "ece"]


def test_spec_without_metrics_section():
    """B1: раздела нет — None."""
    assert parse_task_spec(TASK).metrics is None


@pytest.mark.parametrize(
    ("section", "path", "message"),
    [
        ("{primary: nope}", "metrics.primary", "неизвестная метрика 'nope'"),
        ("{primary: mae}", "metrics.primary", "не подходит к classification/binary"),
        ("{primary: roc_auc, secondary: [f1, precision_at_k@0]}", "metrics.secondary.1", "k"),
        ("{primary: roc_auc, secondary: [roc_auc]}", "metrics.secondary.0", "повтор"),
        ("{primary: roc_auc, threshold: 1.5}", "metrics.threshold", "(0, 1)"),
    ],
)
def test_invalid_metrics(section, path, message):
    """E1: неизвестная, неподходящая, плохой параметр, повтор, порог."""
    with pytest.raises(TaskSpecError) as error:
        parse_task_spec(TASK + f"metrics: {section}\n")
    problems = {problem.path: problem.message for problem in error.value.problems}
    assert path in problems
    assert message in problems[path]


def test_regression_parameters():
    """E1: pinball@τ — τ в (0, 1)."""
    task = "task: {type: regression, output: values}\n"
    assert parse_task_spec(task + "metrics: {primary: pinball@0.9}\n").metrics.primary
    with pytest.raises(TaskSpecError, match="τ"):
        parse_task_spec(task + "metrics: {primary: pinball@2}\n")


def test_custom_metric_paths_are_resolved(tmp_path):
    """Пользовательская метрика: путь — относительно файла спецификации."""
    spec_path = tmp_path / "specs" / "task.yaml"
    spec_path.parent.mkdir()
    spec_path.write_text(
        TASK
        + "metrics:\n"
        + "  primary: top50\n"
        + "  custom: [{name: top50, path: my_metrics.py, function: top50}]\n",
        encoding="utf-8",
    )
    spec = load_task_spec(spec_path)
    (custom,) = spec.metrics.custom
    assert custom.path == str((tmp_path / "specs" / "my_metrics.py").resolve())
    assert custom.higher_is_better is True
    with pytest.raises(TaskSpecError, match="повтор"):
        parse_task_spec(
            TASK
            + "metrics: {primary: roc_auc, custom: [{name: roc_auc, path: a.py, function: f}]}\n"
        )
