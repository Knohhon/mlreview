"""Тесты контракта прогона (docs/specs/run-contract.md)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from mlreview.core.run_contract import (
    ContractGap,
    RunContract,
    RunContractError,
    build_run_contract,
)
from mlreview.core.task_spec import TaskSpec, parse_task_spec


def _spec(
    task_type: str = "classification/binary",
    output: str = "probabilities",
    *,
    segments: tuple[str, ...] = (),
    time_column: str | None = None,
) -> TaskSpec:
    lines = [f"task: {{type: {task_type}, output: {output}, segments: [{', '.join(segments)}]}}"]
    if time_column:
        lines.append(f"data: {{split_strategy: temporal, time_column: {time_column}}}")
    return parse_task_spec("\n".join(lines) + "\n")


def _binary_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a1", "a2", "a3"],
            "label": [0, 1, 1],
            "prediction": [0.1, 0.8, 0.4],
        }
    )


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["t1", "t2", "a1", "a2", "a3"],
            "split": ["train", "train", "test", "test", "test"],
            "language": ["en", "de", "en", "de", "en"],
            "created_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02", "2026-02-03"]
            ),
        }
    )


def _curves() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["train_loss", "train_loss", "val_loss", "val_loss"],
            "step": [0, 1, 0, 1],
            "value": [0.7, 0.5, 0.72, math.nan],
        }
    )


def _build(spec: TaskSpec | None = None, **overrides) -> RunContract:
    kwargs = {"run_id": "exp-1", "predictions": _binary_predictions()}
    kwargs.update(overrides)
    return build_run_contract(spec or _spec(), **kwargs)


def _errors(spec: TaskSpec | None = None, **overrides) -> RunContractError:
    with pytest.raises(RunContractError) as exc:
        _build(spec, **overrides)
    return exc.value


def _paths(error: RunContractError) -> list[str]:
    return [problem.path for problem in error.problems]


def _message_for(error: RunContractError, path: str) -> str:
    messages = [problem.message for problem in error.problems if problem.path == path]
    assert messages, f"нет проблемы с путём {path}: {_paths(error)}"
    return messages[0]


def _full_kwargs() -> dict:
    return {
        "run_id": "exp-043",
        "predictions": _binary_predictions(),
        "examples": _examples(),
        "config": {"C": 1.0, "penalty": "l2"},
        "curves": _curves(),
        "data_version": "ds-v3",
    }


# --- Ожидаемое поведение -------------------------------------------------------------------------


def test_full_contract_keeps_all_fields():
    """B1: все поля валидного контракта попадают в RunContract, пробелов нет."""
    spec = _spec(segments=("language",), time_column="created_at")

    contract = build_run_contract(spec, **_full_kwargs())

    assert isinstance(contract, RunContract)
    assert contract.run_id == "exp-043"
    assert contract.config == {"C": 1.0, "penalty": "l2"}
    assert contract.data_version == "ds-v3"
    pd.testing.assert_frame_equal(contract.predictions, _binary_predictions())
    assert contract.examples is not None
    pd.testing.assert_frame_equal(contract.examples, _examples())
    assert contract.curves is not None
    pd.testing.assert_frame_equal(contract.curves, _curves())
    assert contract.gaps == []


def test_tables_get_default_index():
    """B1: индекс таблиц сбрасывается в 0…n−1."""
    predictions = _binary_predictions().set_index(pd.Index([10, 20, 30]))

    contract = _build(predictions=predictions)

    assert list(contract.predictions.index) == [0, 1, 2]


@pytest.mark.parametrize(
    ("task_type", "output", "columns"),
    [
        ("regression", "values", {"label": [1.5, 2.0, -3.0], "prediction": [1.4, 2.2, -2.5]}),
        ("classification/binary", "labels", {"label": [0, 1, 1], "prediction": [0, 1, 0]}),
        (
            "classification/binary",
            "labels",
            {"label": [False, True, True], "prediction": [False, True, False]},
        ),
        (
            "classification/binary",
            "probabilities",
            {"label": [0, 1, 1], "prediction": [0.0, 1.0, 0.4]},
        ),
        ("classification/binary", "scores", {"label": [0, 1, 1], "prediction": [-2.0, 3.1, 0.0]}),
        (
            "classification/multiclass",
            "labels",
            {"label": ["cat", "dog", "fox"], "prediction": ["cat", "fox", "fox"]},
        ),
        (
            "classification/multiclass",
            "probabilities",
            {
                "label": ["cat", "dog", "fox"],
                "prediction:cat": [0.7, 0.2, 0.1],
                "prediction:dog": [0.2, 0.5, 0.3],
                "prediction:fox": [0.1, 0.3, 0.6],
            },
        ),
        (
            "classification/multiclass",
            "scores",
            {
                "label": ["cat", "dog", "cat"],
                "prediction:cat": [2.0, -1.0, 0.5],
                "prediction:dog": [-2.0, 1.0, 0.1],
            },
        ),
    ],
)
def test_supported_prediction_formats(task_type, output, columns):
    """B2: каждая поддерживаемая пара тип · выход принимает свой формат предсказаний."""
    predictions = pd.DataFrame({"example_id": ["a1", "a2", "a3"], **columns})

    contract = _build(_spec(task_type, output), predictions=predictions)

    pd.testing.assert_frame_equal(contract.predictions, predictions)


def test_minimal_contract_has_defaults():
    """B3: обязательны только run_id и predictions."""
    contract = _build()

    assert contract.examples is None
    assert contract.curves is None
    assert contract.config == {}
    assert contract.data_version is None


def test_contract_keeps_copies():
    """B4: изменения исходных таблиц и конфига после сборки не меняют контракт."""
    kwargs = _full_kwargs()
    kwargs["config"] = {"C": 1.0, "grid": [0.1, 1.0]}
    contract = build_run_contract(_spec(), **kwargs)

    kwargs["predictions"].loc[0, "prediction"] = 0.99
    kwargs["examples"].loc[0, "language"] = "fr"
    kwargs["curves"].loc[0, "value"] = 100.0
    kwargs["config"]["C"] = 2.0
    kwargs["config"]["grid"].append(10.0)

    assert contract.predictions.loc[0, "prediction"] == 0.1
    assert contract.examples is not None
    assert contract.examples.loc[0, "language"] == "en"
    assert contract.curves is not None
    assert contract.curves.loc[0, "value"] == 0.7
    assert contract.config == {"C": 1.0, "grid": [0.1, 1.0]}


def test_numeric_data_version_becomes_text():
    """B5: числовая версия данных приводится к строке."""
    assert _build(data_version=3).data_version == "3"


# --- Пробелы контракта (X1) ----------------------------------------------------------------------


def _gap_fields(contract: RunContract) -> list[str]:
    return [gap.field for gap in contract.gaps]


def test_missing_examples_is_one_gap():
    """G1: нет examples — один пробел examples про срезы и проверки данных."""
    spec = _spec(segments=("language",), time_column="created_at")
    contract = build_run_contract(
        spec, **{**_full_kwargs(), "examples": None, "predictions": _binary_predictions()}
    )

    assert contract.gaps == [
        ContractGap(
            "examples",
            "срезы по сегментам (FR-D02), проверки данных между сплитами (FR-D03)",
        )
    ]


def test_missing_segment_column_is_gap():
    """G2: в examples нет колонки сегмента из спецификации — пробел на сегмент."""
    spec = _spec(segments=("language", "source"))

    contract = build_run_contract(spec, **_full_kwargs())

    assert contract.gaps == [ContractGap("examples.source", "срез по сегменту source (FR-D02)")]


def test_missing_time_column_is_gap():
    """G3: в examples нет колонки времени из спецификации — пробел про временную утечку."""
    spec = _spec(time_column="event_time")

    contract = build_run_contract(spec, **_full_kwargs())

    assert contract.gaps == [
        ContractGap("examples.event_time", "проверка временной утечки (FR-D03, B02)")
    ]


def test_single_split_is_gap():
    """G4: в examples один сплит — пробел про дубли между сплитами."""
    examples = _examples().iloc[2:]

    contract = build_run_contract(_spec(), **{**_full_kwargs(), "examples": examples})

    assert contract.gaps == [
        ContractGap(
            "examples.split",
            "нужно не меньше двух сплитов: проверка дублей между сплитами (FR-D03, B01)",
        )
    ]


@pytest.mark.parametrize(
    ("override", "gap"),
    [
        ({"curves": None}, ContractGap("curves", "анализ кривых обучения (FR-D04)")),
        ({"config": None}, ContractGap("config", "логические проверки процесса (FR-D06)")),
        ({"config": {}}, ContractGap("config", "логические проверки процесса (FR-D06)")),
        (
            {"data_version": None},
            ContractGap("data_version", "сверка версий данных между прогонами (FR-C04, FR-D06)"),
        ),
    ],
)
def test_missing_optional_fields_are_gaps(override, gap):
    """G5: нет кривых, конфига или версии данных — пробел с назначением."""
    contract = build_run_contract(_spec(), **{**_full_kwargs(), **override})

    assert contract.gaps == [gap]


def test_gaps_order():
    """G6: пробелы идут в фиксированном порядке."""
    spec = _spec(segments=("region", "source"), time_column="event_time")
    examples = _examples().iloc[2:]

    contract = _build(spec, examples=examples)

    assert _gap_fields(contract) == [
        "examples.region",
        "examples.source",
        "examples.event_time",
        "examples.split",
        "curves",
        "config",
        "data_version",
    ]


# --- Деградация и ошибки -------------------------------------------------------------------------


def test_error_header_names_run():
    """E*: сообщение начинается с id прогона и «контракт прогона невалиден»."""
    error = _errors(predictions=None)

    assert str(error).startswith("прогон exp-1: контракт прогона невалиден")


@pytest.mark.parametrize("run_id", ["", "   ", None, 42])
def test_bad_run_id(run_id):
    """E1: пустой или нестроковый run_id — ошибка с путём run_id."""
    error = _errors(run_id=run_id)

    assert "run_id" in _paths(error)


@pytest.mark.parametrize(
    "predictions",
    [None, [{"example_id": "a1"}], pd.DataFrame(columns=["example_id", "label", "prediction"])],
)
def test_bad_predictions_table(predictions):
    """E2: predictions не DataFrame или пуст — ошибка с путём predictions."""
    error = _errors(predictions=predictions)

    assert "predictions" in _paths(error)


@pytest.mark.parametrize("column", ["example_id", "label", "prediction"])
def test_required_prediction_columns(column):
    """E3: нет обязательной колонки — ошибка «обязательная колонка» с путём колонки."""
    error = _errors(predictions=_binary_predictions().drop(columns=column))

    assert "обязательная колонка" in _message_for(error, f"predictions.{column}")


def test_unknown_prediction_column_points_to_examples():
    """E3: лишняя колонка — «неизвестная колонка» с подсказкой про examples."""
    predictions = _binary_predictions().assign(language=["en", "de", "en"])

    message = _message_for(_errors(predictions=predictions), "predictions.language")

    assert "неизвестная колонка" in message
    assert "examples" in message


def test_duplicate_prediction_ids():
    """E4: повтор example_id в predictions — ошибка с числом и примерами id."""
    predictions = _binary_predictions().assign(example_id=["a1", "a1", "a2"])

    message = _message_for(_errors(predictions=predictions), "predictions.example_id")

    assert "a1" in message
    assert "1" in message


def test_missing_prediction_ids():
    """E4: пропуски example_id в predictions — ошибка с путём колонки."""
    predictions = _binary_predictions().assign(example_id=["a1", None, "a3"])

    message = _message_for(_errors(predictions=predictions), "predictions.example_id")

    assert "пропуск" in message


def test_duplicate_example_ids():
    """E4: повтор example_id в examples — ошибка с путём examples.example_id."""
    examples = pd.concat([_examples(), _examples().iloc[[0]]], ignore_index=True)

    message = _message_for(_errors(examples=examples), "examples.example_id")

    assert "t1" in message


@pytest.mark.parametrize("column", ["label", "prediction"])
def test_missing_values_in_predictions(column):
    """E5: пропуски в метке или предсказании — ошибка с числом строк и примерами id."""
    predictions = _binary_predictions()
    predictions[column] = (
        predictions[column].astype("float").where(predictions["example_id"] != "a2")
    )

    message = _message_for(_errors(predictions=predictions), f"predictions.{column}")

    assert "a2" in message


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame({"example_id": ["a1", "a2", "a3"], **columns})


@pytest.mark.parametrize(
    ("task_type", "output", "predictions", "path", "evidence"),
    [
        (
            "regression",
            "values",
            _frame(label=[1.0, 2.0, 3.0], prediction=["x", "y", "z"]),
            "predictions.prediction",
            "числ",
        ),
        (
            "regression",
            "values",
            _frame(label=[1.0, 2.0, 3.0], prediction=[1.0, math.inf, 3.0]),
            "predictions.prediction",
            "a2",
        ),
        (
            "regression",
            "values",
            _frame(label=["a", "b", "c"], prediction=[1.0, 2.0, 3.0]),
            "predictions.label",
            "числ",
        ),
        (
            "classification/binary",
            "probabilities",
            _frame(label=[0, 2, 1], prediction=[0.1, 0.2, 0.3]),
            "predictions.label",
            "a2",
        ),
        (
            "classification/binary",
            "probabilities",
            _frame(label=["no", "yes", "yes"], prediction=[0.1, 0.2, 0.3]),
            "predictions.label",
            "0 или 1",
        ),
        (
            "classification/binary",
            "labels",
            _frame(label=[0, 1, 1], prediction=[0, 0.5, 1]),
            "predictions.prediction",
            "a2",
        ),
        (
            "classification/binary",
            "probabilities",
            _frame(label=[0, 1, 1], prediction=[0.1, 1.2, -0.1]),
            "predictions.prediction",
            "[0, 1]",
        ),
        (
            "classification/binary",
            "scores",
            _frame(label=[0, 1, 1], prediction=[0.1, -math.inf, 0.3]),
            "predictions.prediction",
            "a2",
        ),
        (
            "classification/multiclass",
            "probabilities",
            _frame(
                label=["cat", "dog", "cat"],
                **{"prediction:cat": [0.9, 0.5, 0.5], "prediction:dog": [0.1, 0.4, 0.5]},
            ),
            "predictions",
            "a2",
        ),
        (
            "classification/multiclass",
            "probabilities",
            _frame(
                label=["cat", "dog", "cat"],
                **{"prediction:cat": [1.1, 0.5, 0.5], "prediction:dog": [-0.1, 0.5, 0.5]},
            ),
            "predictions.prediction:cat",
            "a1",
        ),
        (
            "classification/multiclass",
            "scores",
            _frame(
                label=["cat", "fox", "cat"],
                **{"prediction:cat": [1.0, 0.0, 2.0], "prediction:dog": [0.0, 1.0, 0.0]},
            ),
            "predictions.label",
            "fox",
        ),
        (
            "classification/multiclass",
            "scores",
            _frame(label=["cat", "cat", "cat"], **{"prediction:cat": [1.0, 0.0, 2.0]}),
            "predictions",
            "двух",
        ),
        (
            "classification/multiclass",
            "probabilities",
            _frame(label=["cat", "dog", "cat"]),
            "predictions",
            "двух",
        ),
    ],
)
def test_values_must_match_format(task_type, output, predictions, path, evidence):
    """E6: значения не по формату — ошибка с путём, ожидаемым форматом и примерами id."""
    error = _errors(_spec(task_type, output), predictions=predictions)

    assert evidence in _message_for(error, path)


def test_multilabel_is_explicitly_unsupported():
    """E7: multilabel на L0 — явный отказ с путём predictions."""
    error = _errors(_spec("classification/multilabel", "probabilities"))

    assert "multilabel" in _message_for(error, "predictions")


def test_examples_required_columns():
    """E8: examples без split — ошибка «обязательная колонка» с путём examples.split."""
    error = _errors(examples=_examples().drop(columns="split"))

    assert "обязательная колонка" in _message_for(error, "examples.split")


def test_examples_require_example_id():
    """E8: examples без example_id — ошибка с путём examples.example_id."""
    error = _errors(examples=_examples().drop(columns="example_id"))

    assert "обязательная колонка" in _message_for(error, "examples.example_id")


@pytest.mark.parametrize("bad", ["", None, "  "])
def test_examples_empty_split(bad):
    """E8: пустой split — ошибка с путём examples.split и примером id."""
    examples = _examples()
    examples.loc[1, "split"] = bad

    assert "t2" in _message_for(_errors(examples=examples), "examples.split")


def test_prediction_ids_must_be_in_examples():
    """E8: id из predictions, которых нет в examples, — ошибка с примерами id."""
    examples = _examples().iloc[:3]

    message = _message_for(_errors(examples=examples), "predictions.example_id")

    assert "a2" in message
    assert "a3" in message
    assert "examples" in message


@pytest.mark.parametrize("column", ["name", "step", "value"])
def test_curves_required_columns(column):
    """E9: в кривых нет обязательной колонки — ошибка с путём колонки."""
    error = _errors(curves=_curves().drop(columns=column))

    assert "обязательная колонка" in _message_for(error, f"curves.{column}")


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("step", [0, -1, 0, 1]),
        ("step", [0, 1.5, 0, 1]),
        ("value", ["a", 0.5, 0.7, 0.6]),
    ],
)
def test_curves_bad_values(column, values):
    """E9: отрицательный или нецелый шаг, нечисловое значение — ошибка с путём колонки."""
    curves = _curves()
    curves[column] = values

    assert f"curves.{column}" in _paths(_errors(curves=curves))


def test_curves_duplicate_steps():
    """E9: повтор пары (name, step) — ошибка с путём curves."""
    curves = _curves()
    curves.loc[1, "step"] = 0

    assert "train_loss" in _message_for(_errors(curves=curves), "curves")


@pytest.mark.parametrize("config", [["C", 1.0], {1: "a"}])
def test_bad_config(config):
    """E10: конфиг не словарь или с нестроковыми ключами — ошибка с путём config."""
    assert "config" in _paths(_errors(config=config))


def test_all_problems_reported_together():
    """E11: несколько ошибок сразу перечислены в одной RunContractError."""
    predictions = _binary_predictions()
    predictions["label"] = [0, None, 1]

    error = _errors(run_id="", predictions=predictions, examples=_examples().iloc[:3])

    paths = _paths(error)
    assert "run_id" in paths
    assert "predictions.label" in paths
    assert "predictions.example_id" in paths
