"""Команды metrics: init (диалог), test, export, list; пользовательская метрика (FR-M01–M05).

Спецификация: docs/specs/metrics.md (пункты B5–B10, E2, E4).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main
from mlreview.core.task_spec import load_task_spec
from mlreview.metrics.library import metric_by_name

SPEC_TEXT = """\
# Модерация: бинарная классификация
task:
  type: classification/binary
  output: probabilities
  error_costs: {false_negative: high, false_positive: low}
  segments: [source]
data: {version: v1}  # версия датасета
"""


class _Tty(io.StringIO):
    def isatty(self):
        return True


def _run(path, labels, scores):
    n = len(labels)
    predictions = pd.DataFrame(
        {"example_id": [f"e{i:03d}" for i in range(n)], "label": labels, "prediction": scores}
    )
    examples = pd.DataFrame(
        {
            "example_id": predictions["example_id"],
            "split": "test",
            "source": ["forum", "chat"] * (n // 2),
        }
    )
    save_run_files(path, predictions, examples=examples, data_version="v1")


@pytest.fixture
def project(tmp_path):
    (tmp_path / "task.yaml").write_text(SPEC_TEXT, encoding="utf-8")
    rng = np.random.default_rng(0)
    n = 500
    labels = np.zeros(n, dtype=int)
    labels[::50] = 1
    good = np.clip(labels * 0.6 + rng.uniform(0, 0.4, n), 0, 1)
    _run(tmp_path / "good", labels, good)
    _run(tmp_path / "noise", labels, rng.uniform(0, 1, n))
    return tmp_path


def _with_metrics(project, section):
    path = project / "task.yaml"
    path.write_text(SPEC_TEXT + section, encoding="utf-8")
    return path


def test_dialog_writes_metrics_section(project, monkeypatch, capsys):
    """B5: диалог в терминале; --write сохраняет остальной текст и комментарии."""
    monkeypatch.setattr(sys, "stdin", _Tty("2\n50\n\nда\n\n"))
    assert main(["metrics", "init", "--spec", "task.yaml", "--run", "good", "--write"]) == 0
    out = capsys.readouterr().out
    assert "Как используется предсказание?" in out
    assert "по умолчанию 10" in out  # error_costs high/low → 10
    assert "precision_at_k@50" in out
    assert "ловушки:" in out
    text = (project / "task.yaml").read_text(encoding="utf-8")
    assert text.startswith("# Модерация: бинарная классификация\n")
    assert "data: {version: v1}  # версия датасета" in text
    spec = load_task_spec(project / "task.yaml")
    assert spec.metrics.primary == "precision_at_k@50"
    assert "pr_auc" in spec.metrics.secondary
    assert "ece" in spec.metrics.secondary
    assert spec.metrics.context.cost_ratio == 10
    assert spec.metrics.context.critical_segments == ["source"]
    # повторный запуск заменяет раздел, а не дописывает второй
    answers = project / "answers.yaml"
    answers.write_text("decision: probability\n", encoding="utf-8")
    args = ["metrics", "init", "--spec", "task.yaml", "--answers", str(answers), "--write"]
    assert main(args) == 0
    text = (project / "task.yaml").read_text(encoding="utf-8")
    assert text.count("\nmetrics:") == 1
    assert load_task_spec(project / "task.yaml").metrics.primary == "log_loss"


@pytest.mark.usefixtures("project")
def test_dialog_without_terminal_uses_defaults(monkeypatch, capsys):
    """B5: без терминала — ответы по умолчанию с пометкой."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["metrics", "init", "--spec", "task.yaml"]) == 0
    out = capsys.readouterr().out
    assert "ответы по умолчанию" in out
    assert "metrics:" in out


def test_degenerate_accuracy(project, capsys):
    """B6: accuracy при 1:50 — провал «вырожденное решение»; roc_auc — пройден."""
    _with_metrics(project, "metrics: {primary: roc_auc, secondary: [accuracy]}\n")
    assert main(["metrics", "test", "good", "--spec", "task.yaml"]) == 0
    out = capsys.readouterr().out
    accuracy = out[out.index("accuracy") :]
    assert "✗ вырожденное решение: константа мажоритарного класса получает 0.980" in accuracy
    roc = out[out.index("roc_auc") : out.index("accuracy")]
    assert "✓ вырожденное решение" in roc
    assert "95% CI" in roc
    assert "MDE" in roc


def test_distinguishes_model(project, capsys):
    """B7: шум не отличается от константы; хорошая модель — отличается."""
    _with_metrics(project, "metrics: {primary: roc_auc}\n")
    assert main(["metrics", "test", "noise", "--spec", "task.yaml"]) == 0
    assert "✗ отличает модель" in capsys.readouterr().out
    assert main(["metrics", "test", "good", "--spec", "task.yaml"]) == 0
    assert "✓ отличает модель" in capsys.readouterr().out


CUSTOM = """\
import numpy as np


def top10_share(labels, predictions):
    order = np.argsort(-predictions, kind="mergesort")[:10]
    return float(np.mean(labels[order] == 1))
"""


def test_custom_metric_in_tests_review_and_export(project, capsys):
    """B8: пользовательская метрика — в тестах, ревью и экспорте."""
    (project / "my_metrics.py").write_text(CUSTOM, encoding="utf-8")
    _with_metrics(
        project,
        "metrics:\n"
        "  primary: roc_auc\n"
        "  secondary: [top10]\n"
        "  custom: [{name: top10, path: my_metrics.py, function: top10_share, bounded: high}]\n",
    )
    assert main(["metrics", "test", "good", "--spec", "task.yaml"]) == 0
    assert "top10" in capsys.readouterr().out
    assert main(["review", "good", "--spec", "task.yaml", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "top10" in [f["facts"].get("metric") for f in data["findings"]]
    assert main(["metrics", "export", "--spec", "task.yaml", "-o", "out.py"]) == 0
    assert "top10" in (project / "out.py").read_text(encoding="utf-8")


def test_keep_failing_metric_goes_to_journal(project, capsys):
    """B9: --keep с причиной — решение в журнале."""
    _with_metrics(project, "metrics: {primary: roc_auc, secondary: [accuracy]}\n")
    args = ["metrics", "test", "good", "--spec", "task.yaml", "--keep", "accuracy"]
    assert main([*args, "--reason", "нужна для отчёта"]) == 1  # без рабочего пространства
    assert "mlreview init" in capsys.readouterr().err
    main(["init"])
    assert main([*args, "--reason", "нужна для отчёта"]) == 0
    assert "D-0001" in capsys.readouterr().out
    assert main(["journal", "show", "D-0001"]) == 0
    shown = capsys.readouterr().out
    assert "оставить метрику accuracy, несмотря на непройденные тесты: нужна для отчёта" in shown
    assert "прогоны: good" in shown


def test_export_runs_without_mlreview(project, capsys):
    """B10: экспорт исполняется без mlreview и совпадает с библиотекой."""
    _with_metrics(
        project,
        "metrics: {primary: pr_auc, secondary: [roc_auc, recall_at_k@20, fbeta@2, ece, brier],"
        " threshold: 0.4}\n",
    )
    assert main(["metrics", "export", "--spec", "task.yaml", "-o", "metrics_out.py"]) == 0
    capsys.readouterr()
    code = (project / "metrics_out.py").read_text(encoding="utf-8")
    assert "mlreview" not in code.split('"""', 2)[-1]  # кроме докстринга
    predictions = pd.read_parquet(project / "good" / "predictions.parquet")
    script = (
        "import json, sys, runpy\n"
        "sys.modules['mlreview'] = None\n"
        "import pandas as pd\n"
        f"ns = runpy.run_path({str(project / 'metrics_out.py')!r})\n"
        f"df = pd.read_parquet({str(project / 'good' / 'predictions.parquet')!r})\n"
        "print(json.dumps(ns['compute'](df['label'].to_numpy(), df['prediction'].to_numpy())))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    exported = json.loads(result.stdout)
    spec = load_task_spec(project / "task.yaml")
    labels = predictions["label"].to_numpy()
    values = predictions["prediction"].to_numpy()
    for name in spec.metrics.names:
        ours = metric_by_name(spec, name).compute(labels, values)
        assert exported[name] == pytest.approx(ours, abs=1e-12)


def test_custom_metric_errors(project, capsys):
    """E2: файла нет; функции нет."""
    _with_metrics(
        project,
        "metrics: {primary: roc_auc, secondary: [x],"
        " custom: [{name: x, path: missing.py, function: f}]}\n",
    )
    assert main(["metrics", "test", "good", "--spec", "task.yaml"]) == 1
    assert "missing.py" in capsys.readouterr().err
    (project / "has.py").write_text("def other(a, b):\n    return 1.0\n", encoding="utf-8")
    _with_metrics(
        project,
        "metrics: {primary: roc_auc, secondary: [x],"
        " custom: [{name: x, path: has.py, function: f}]}\n",
    )
    assert main(["metrics", "test", "good", "--spec", "task.yaml"]) == 1
    assert "нет функции f" in capsys.readouterr().err


def test_bad_answers(project, capsys):
    """E4: неизвестный ключ или плохое значение в --answers."""
    answers = project / "answers.yaml"
    answers.write_text(yaml.safe_dump({"decision": "guess"}), encoding="utf-8")
    assert main(["metrics", "init", "--spec", "task.yaml", "--answers", str(answers)]) == 1
    assert "decision" in capsys.readouterr().err
    answers.write_text(yaml.safe_dump({"colour": "red"}), encoding="utf-8")
    assert main(["metrics", "init", "--spec", "task.yaml", "--answers", str(answers)]) == 1
    assert "colour" in capsys.readouterr().err


def test_metrics_list(capsys):
    """Каталог метрик: обоснование и ловушки."""
    args = ["metrics", "list", "--task", "classification/binary", "--output", "labels"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "roc_auc" not in out
    assert "ловушки:" in out
