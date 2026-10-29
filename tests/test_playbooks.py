"""Плейбуки: формат, загрузка, семейство модели, группа model (FR-E02, FR-D11).

Спецификация: docs/specs/playbooks.md.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from _factories import BINARY_SPEC_TEXT, binary_contract, binary_spec, binary_tables
from mlreview.adapters.files_adapter import save_run_files
from mlreview.analyzers.process import process_findings
from mlreview.analyzers.registry import Analyzer, ReviewContext, register, unregister
from mlreview.analyzers.review import review
from mlreview.cli import main
from mlreview.core.findings import CheckResults, Finding
from mlreview.core.model_info import ModelInfo
from mlreview.llm.context import run_dict
from mlreview.playbooks import (
    PlaybookError,
    builtin_playbooks,
    load_playbooks,
    loader,
    model_params,
    resolve_family,
)

DEMO_MANIFEST = """\
id: demo
title: Демо
requirement: FR-D11
applies_to:
  families: [neighbors]
  task_types: [classification/binary]
params:
  n_neighbors: {aliases: [n_neighbors, k]}
thresholds: {small_k: 2}
checks:
  demo.check: {suggestion: увеличить k}
overrides:
  custom.check: подсказка плейбука
"""
DEMO_CHECKS = """\
from mlreview.core.findings import CheckResults, Finding


def run(ctx):
    results = CheckResults()
    k = ctx.params.get("n_neighbors")
    message = f"семейство {ctx.family}, k={k}, порог {ctx.thresholds['small_k']}"
    suggestion = ctx.playbook.manifest.checks["demo.check"].suggestion
    finding = Finding("demo.check", "low", message, "FR-D11", facts={"k": k}, suggestion=suggestion)
    results.findings.append(finding)
    results.skip("demo.skipped", "нет данных")
    return results
"""
KNN = {"class": "sklearn.neighbors.KNeighborsClassifier"}


def _write(root, name, manifest, checks=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "playbook.yaml").write_text(textwrap.dedent(manifest), encoding="utf-8")
    if checks is not None:
        (directory / "checks.py").write_text(checks, encoding="utf-8")
    return directory


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """Встроенные плейбуки и демо-плейбук для neighbors с проверкой и переопределением."""
    _write(tmp_path / "extra", "demo", DEMO_MANIFEST, DEMO_CHECKS)
    playbooks = (*builtin_playbooks(), *load_playbooks(tmp_path / "extra"))
    monkeypatch.setattr(loader, "builtin_playbooks", lambda: playbooks)
    return playbooks


@pytest.fixture
def custom_check():
    """Общая проверка группы data с подсказкой, которую переопределяет демо-плейбук."""

    def run(_ctx):
        results = CheckResults()
        results.findings.append(
            Finding("custom.check", "low", "общая", "FR-D03", suggestion="старая")
        )
        return results

    analyzer = register(Analyzer("custom", "data", "FR-D03", run))
    yield analyzer
    unregister(analyzer.name)


def _model_group(report):
    return [f for f in report.findings if f.check.startswith("demo.")], [
        s for s in report.skipped if s.check.startswith(("demo.", "model", "playbook."))
    ]


# --- Манифест и загрузка -------------------------------------------------------------------------


def test_load_playbooks_from_directory(tmp_path):
    """B1: плейбуки каталога — по имени подкаталога; run — из checks.py или None."""
    _write(tmp_path, "b", DEMO_MANIFEST.replace("id: demo", "id: b"), DEMO_CHECKS)
    _write(tmp_path, "a", "id: a\ntitle: А\nrequirement: FR-D11\napplies_to: {families: [tree]}\n")

    playbooks = load_playbooks(tmp_path)

    assert [playbook.id for playbook in playbooks] == ["a", "b"]
    assert playbooks[0].run is None
    assert callable(playbooks[1].run)
    assert playbooks[1].manifest.thresholds == {"small_k": 2}


KNOWN_CLASSES = [
    "LogisticRegression",
    "LinearRegression",
    "Ridge",
    "SGDClassifier",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "RadiusNeighborsClassifier",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "BaggingClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "AdaBoostClassifier",
    "LGBMClassifier",
    "LGBMRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
]


def test_builtin_playbooks_do_not_conflict():
    """B2: встроенные linear, neighbors, trees; класс не распознаётся в два семейства."""
    playbooks = builtin_playbooks()
    assert [playbook.id for playbook in playbooks] == ["linear", "neighbors", "trees"]
    for name in KNOWN_CLASSES:
        families = {family for playbook in playbooks for family in playbook.families_of(name)}
        assert len(families) == 1, (name, families)


# --- Семейство модели ----------------------------------------------------------------------------


def test_explicit_family_wins():
    """B3: явное семейство важнее распознавания по классу."""
    model = ModelInfo(class_name="DecisionTreeClassifier", family="other")
    assert resolve_family(model) == "other"


@pytest.mark.parametrize(
    ("class_name", "family"),
    [
        ("sklearn.neighbors.KNeighborsClassifier", "neighbors"),
        ("KNeighborsRegressor", "neighbors"),
        ("sklearn.tree._classes.DecisionTreeClassifier", "tree"),
        ("ExtraTreeRegressor", "tree"),
        ("RandomForestClassifier", "ensemble/bagging"),
        ("ExtraTreesRegressor", "ensemble/bagging"),
        ("LGBMClassifier", "ensemble/boosting"),
        ("xgboost.XGBRegressor", "ensemble/boosting"),
        ("CatBoostClassifier", "ensemble/boosting"),
        ("HistGradientBoostingRegressor", "ensemble/boosting"),
        ("LogisticRegression", "linear"),
        ("Ridge", "linear"),
    ],
)
def test_family_from_class(class_name, family):
    """B4: семейство по короткому имени класса."""
    assert resolve_family(ModelInfo(class_name=class_name)) == family


def test_unknown_model_has_no_family():
    """B5: модели нет или класс неизвестен — None."""
    assert resolve_family(None) is None
    assert resolve_family(ModelInfo(class_name="MyNet")) is None


# --- Гиперпараметры ------------------------------------------------------------------------------


def _playbook(playbook_id):
    return next(playbook for playbook in builtin_playbooks() if playbook.id == playbook_id)


@pytest.mark.parametrize(
    "config",
    [
        {"n_neighbors": "3"},
        {"kneighborsclassifier__n_neighbors": "3"},
        {"model": {"k": 3}},
    ],
)
def test_params_by_alias(config):
    """B6: алиас, имя шага Pipeline и вложенный ключ."""
    assert model_params(config, _playbook("neighbors"))["n_neighbors"] == 3


def test_param_values_are_parsed():
    """B7: строки MLflow разбираются; unbounded — None; отсутствующего ключа нет."""
    trees = _playbook("trees")
    params = model_params(
        {
            "max_depth": "-1",
            "min_samples_leaf": "None",
            "learning_rate": "0.5",
            "criterion": "gini",
            "bootstrap": "True",
            "max_leaf_nodes": "null",
        },
        trees,
    )
    assert params["max_depth"] is None
    assert params["min_samples_leaf"] is None
    assert params["learning_rate"] == 0.5
    assert params["max_leaf_nodes"] is None
    assert "n_estimators" not in params
    neighbors = model_params({"weights": "distance", "p": "2", "k": True}, _playbook("neighbors"))
    assert neighbors == {"weights": "distance", "p": 2, "n_neighbors": True}


def test_exact_name_wins_over_pipeline_step():
    """B8: точное имя важнее шага Pipeline; среди шагов — первый по имени ключа."""
    trees = _playbook("trees")
    assert model_params({"est__max_depth": "3", "max_depth": "5"}, trees)["max_depth"] == 5
    assert model_params({"b__max_depth": "3", "a__max_depth": "4"}, trees)["max_depth"] == 4


# --- Ревью: группа model -------------------------------------------------------------------------


@pytest.mark.usefixtures("demo")
def test_playbook_checks_run_for_matching_family():
    """B9: плейбук применимого семейства запускается; других семейств — нет."""
    knn = binary_contract(model=KNN, config={"n_neighbors": "1"})
    tree = binary_contract(model={"class": "DecisionTreeClassifier"})

    findings, skipped = _model_group(review(binary_spec(), knn, groups=("model",)))
    assert [f.message for f in findings] == ["семейство neighbors, k=1, порог 2"]
    assert findings[0].suggestion == "увеличить k"
    assert [s.check for s in skipped] == ["demo.skipped"]

    assert _model_group(review(binary_spec(), tree, groups=("model",))) == ([], [])


@pytest.mark.usefixtures("demo")
def test_family_without_checks_adds_nothing():
    """B10: linear — семейство определено, проверок нет, группа model пуста."""
    contract = binary_contract(model={"class": "LogisticRegression"})
    report = review(binary_spec(), contract, groups=("model",))
    assert (report.findings, report.skipped) == ([], [])


@pytest.mark.usefixtures("demo", "custom_check")
def test_overrides_replace_suggestions():
    """B11: подсказка общей проверки заменяется для семейства плейбука."""
    knn = review(binary_spec(), binary_contract(model=KNN), groups=("data",))
    tree = review(binary_spec(), binary_contract(model={"family": "tree"}), groups=("data",))

    assert next(f for f in knn.findings if f.check == "custom.check").suggestion == (
        "подсказка плейбука"
    )
    assert next(f for f in tree.findings if f.check == "custom.check").suggestion == "старая"


def test_report_carries_model():
    """B12: Report.model — класс и определённое семейство; в JSON и контексте LLM."""
    report = review(binary_spec(), binary_contract(model=KNN), groups=("data",))
    expected = {"class": "sklearn.neighbors.KNeighborsClassifier", "family": "neighbors"}

    assert report.model == expected
    assert report.to_dict()["model"] == expected
    assert run_dict(report)["model"] == expected
    assert "n_neighbors" not in json.dumps(run_dict(report))
    assert review(binary_spec(), binary_contract(), groups=("data",)).model is None


@pytest.mark.usefixtures("demo")
def test_diagnose_model_command(tmp_path, capsys, monkeypatch):
    """B13: diagnose model — только группа model, заголовок «диагностика модели прогона»."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    predictions, examples = binary_tables()
    save_run_files(tmp_path / "run", predictions, examples=examples, model=KNN, config={"k": 1})

    assert main(["diagnose", "model", "run", "--spec", "task.yaml"]) == 0
    assert "диагностика модели прогона" in capsys.readouterr().out
    assert main(["diagnose", "model", "run", "--spec", "task.yaml", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [f["check"] for f in data["findings"]] == ["demo.check"]
    assert [s["check"] for s in data["skipped"]] == ["demo.skipped"]


def test_playbooks_list_command(capsys):
    """B14: playbooks list — id, название, семейства, типы задач, проверки."""
    assert main(["playbooks", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    linear = next(line for line in lines if line.startswith("linear"))
    assert "семейства: linear" in linear
    assert "типы задач: все" in linear
    assert "только распознавание" in linear
    trees = next(line for line in lines if line.startswith("trees"))
    assert "семейства: tree, ensemble/bagging, ensemble/boosting" in trees


# --- Смена модели в сравнении (B11) --------------------------------------------------------------


def _changes(candidate, baseline):
    results = process_findings(ReviewContext(binary_spec(), candidate, baseline))
    return [f for f in results.findings if f.check == "process.multiple_changes"]


def test_family_change_is_one_change():
    """B15: смена семейства — одно изменение; ключи одной стороны и алиасы — к ней."""
    baseline = binary_contract(
        "base", model={"class": "LogisticRegression"}, config={"C": "1.0", "features": "10"}
    )
    tree_only = binary_contract(
        "tree",
        model={"class": "DecisionTreeClassifier"},
        config={"max_depth": "4", "min_samples_leaf": "5", "features": "10"},
    )
    assert _changes(tree_only, baseline) == []

    tree_features = binary_contract(
        "tree2",
        model={"class": "DecisionTreeClassifier"},
        config={"max_depth": "4", "features": "all"},
    )
    [finding] = _changes(tree_features, baseline)
    assert finding.message.startswith(
        "в эксперименте изменено сразу 2 параметра: features (10 → all), model (linear → tree)"
    )
    assert finding.facts == {"changed": 2, "keys": "features, model"}


# --- Деградация и ошибки -------------------------------------------------------------------------


def test_missing_model_is_skipped():
    """E1: нет model — пропуск model с подсказкой."""
    report = review(binary_spec(), binary_contract(), groups=("model",))
    assert [(s.check, s.reason) for s in report.skipped] == [
        (
            "model",
            "в контракте нет сведений о модели — проверки по семейству не выполнены;"
            " укажите model в run.yaml или теги mlreview.model.*",
        )
    ]


def test_unknown_class_is_skipped():
    """E2: класс есть, семейство не определено — пропуск с подсказкой."""
    report = review(binary_spec(), binary_contract(model={"class": "MyNet"}), groups=("model",))
    assert [(s.check, s.reason) for s in report.skipped] == [
        (
            "model",
            "семейство модели не определено для класса MyNet; укажите model.family (run.yaml)"
            " или тег mlreview.model.family",
        )
    ]


@pytest.mark.parametrize(
    ("manifest", "field"),
    [
        (DEMO_MANIFEST + "extra: 1\n", "extra"),
        (DEMO_MANIFEST.replace("families: [neighbors]", "families: [forest]"), "families"),
        (DEMO_MANIFEST + "detect: {forest: [X*]}\n", "detect"),
        (DEMO_MANIFEST.replace("id: demo", "id: other"), "id"),
        ("id: [demo\n", "playbook.yaml"),
    ],
)
def test_bad_manifest(tmp_path, manifest, field):
    """E3: неизвестное поле, семейство, чужой id, плохой YAML — PlaybookError с путём."""
    _write(tmp_path, "demo", manifest)
    with pytest.raises(PlaybookError) as exc:
        load_playbooks(tmp_path)
    assert "playbook.yaml" in str(exc.value)
    assert field in str(exc.value)


def test_checks_without_run(tmp_path):
    """E4: checks.py без run — PlaybookError."""
    _write(tmp_path, "demo", DEMO_MANIFEST, "VALUE = 1\n")
    with pytest.raises(PlaybookError, match=r"checks\.py: нет функции run"):
        load_playbooks(tmp_path)


def test_failing_playbook_is_skipped(tmp_path, monkeypatch):
    """E5: исключение в run плейбука — пропуск playbook.<id>, ревью не падает."""
    _write(tmp_path, "demo", DEMO_MANIFEST, "def run(ctx):\n    raise RuntimeError('сломано')\n")
    playbooks = (*builtin_playbooks(), *load_playbooks(tmp_path))
    monkeypatch.setattr(loader, "builtin_playbooks", lambda: playbooks)

    report = review(binary_spec(), binary_contract(model=KNN), groups=("model",))

    assert [(s.check, s.reason) for s in report.skipped] == [
        ("playbook.demo", "ошибка анализатора: RuntimeError: сломано")
    ]
