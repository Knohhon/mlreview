"""Тесты спецификации задачи (docs/specs/task-spec.md)."""

from __future__ import annotations

import textwrap

import pytest

from mlreview.core.task_spec import TaskSpec, TaskSpecError, load_task_spec, parse_task_spec

FULL_SPEC = textwrap.dedent(
    """\
    task:
      type: classification/multilabel
      output: probabilities
      decision: "модератор проверяет топ-K сообщений очереди"
      error_costs: {false_negative: high, false_positive: low}
      segments: [language, source, text_length]
    data:
      split_strategy: temporal
      time_column: created_at
      version: ds-v3
      known_issues: ["шумные метки в source=forum"]
    """
)

MINIMAL_SPEC = "task: {type: regression, output: values}\n"


def _errors(text: str) -> TaskSpecError:
    with pytest.raises(TaskSpecError) as exc:
        parse_task_spec(text)
    return exc.value


def _paths(error: TaskSpecError) -> list[str]:
    return [problem.path for problem in error.problems]


def _message_for(error: TaskSpecError, path: str) -> str:
    messages = [problem.message for problem in error.problems if problem.path == path]
    assert messages, f"нет проблемы с путём {path}: {_paths(error)}"
    return messages[0]


def test_full_spec_loads_from_file(tmp_path):
    """B1: все поля из файла попадают в TaskSpec."""
    path = tmp_path / "task.yaml"
    path.write_text(FULL_SPEC, encoding="utf-8")

    spec = load_task_spec(path)

    assert isinstance(spec, TaskSpec)
    assert spec.task.type == "classification/multilabel"
    assert spec.task.output == "probabilities"
    assert spec.task.decision == "модератор проверяет топ-K сообщений очереди"
    assert spec.task.error_costs == {"false_negative": "high", "false_positive": "low"}
    assert spec.task.segments == ["language", "source", "text_length"]
    assert spec.data is not None
    assert spec.data.split_strategy == "temporal"
    assert spec.data.time_column == "created_at"
    assert spec.data.version == "ds-v3"
    assert spec.data.known_issues == ["шумные метки в source=forum"]


def test_minimal_spec_has_defaults():
    """B2: обязательны только task.type и task.output, остальное по умолчанию пусто."""
    spec = parse_task_spec(MINIMAL_SPEC)

    assert spec.task.type == "regression"
    assert spec.task.output == "values"
    assert spec.task.decision is None
    assert spec.task.segments == []
    assert spec.task.error_costs == {}
    assert spec.data is None


@pytest.mark.parametrize(
    ("task_type", "output"),
    [
        ("classification/binary", "labels"),
        ("classification/binary", "probabilities"),
        ("classification/multiclass", "scores"),
        ("classification/multilabel", "probabilities"),
        ("regression", "values"),
    ],
)
def test_supported_task_types(task_type, output):
    """B3: каждый тип задачи валиден с допустимым для него выходом."""
    spec = parse_task_spec(f"task: {{type: {task_type}, output: {output}}}\n")

    assert spec.task.type == task_type
    assert spec.task.output == output


def test_parse_matches_load(tmp_path):
    """B4: разбор строки даёт тот же результат, что загрузка файла."""
    path = tmp_path / "task.yaml"
    path.write_text(FULL_SPEC, encoding="utf-8")

    assert parse_task_spec(FULL_SPEC) == load_task_spec(path)


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        ("task: {output: values}\n", "task.type"),
        ("task: {type: regression}\n", "task.output"),
        ("data: {split_strategy: random}\n", "task"),
    ],
)
def test_required_fields(text, missing):
    """E1: отсутствие обязательного поля — ошибка с путём поля."""
    error = _errors(text)

    assert "обязательное поле" in _message_for(error, missing)


def test_unknown_task_type_lists_allowed():
    """E2: неизвестный тип задачи — ошибка с перечнем допустимых типов."""
    error = _errors("task: {type: ranking, output: scores}\n")

    message = _message_for(error, "task.type")
    assert "ranking" in message
    for allowed in (
        "classification/binary",
        "classification/multiclass",
        "classification/multilabel",
        "regression",
    ):
        assert allowed in message


@pytest.mark.parametrize(
    ("task_type", "output", "allowed"),
    [
        ("regression", "probabilities", ["values"]),
        ("classification/binary", "values", ["labels", "probabilities", "scores"]),
    ],
)
def test_output_must_match_task_type(task_type, output, allowed):
    """E3: выход, недопустимый для типа задачи, — ошибка с перечнем допустимых."""
    error = _errors(f"task: {{type: {task_type}, output: {output}}}\n")

    message = _message_for(error, "task.output")
    for value in allowed:
        assert value in message


@pytest.mark.parametrize(
    ("text", "path"),
    [
        ("task: {type: regression, output: values, segmets: [a]}\n", "task.segmets"),
        (
            "task: {type: regression, output: values}\ndata: {split_strategy: random, ver: 1}\n",
            "data.ver",
        ),
        ("task: {type: regression, output: values}\nmodel: linear\n", "model"),
    ],
)
def test_unknown_field_is_error(text, path):
    """E4: неизвестное поле на любом уровне — ошибка, а не молчаливый пропуск."""
    error = _errors(text)

    assert "неизвестное поле" in _message_for(error, path)


def test_temporal_split_requires_time_column():
    """E5: temporal без time_column — ошибка с путём data.time_column."""
    error = _errors("task: {type: regression, output: values}\ndata: {split_strategy: temporal}\n")

    assert "temporal" in _message_for(error, "data.time_column")


def test_unknown_split_strategy():
    """E5: split_strategy вне random | stratified | temporal — ошибка с перечнем."""
    error = _errors("task: {type: regression, output: values}\ndata: {split_strategy: group}\n")

    message = _message_for(error, "data.split_strategy")
    for allowed in ("random", "stratified", "temporal"):
        assert allowed in message


@pytest.mark.parametrize(
    ("text", "path"),
    [
        (
            "task: {type: classification/binary, output: labels, "
            "error_costs: {underestimation: high}}\n",
            "task.error_costs.underestimation",
        ),
        (
            "task: {type: regression, output: values, error_costs: {false_positive: high}}\n",
            "task.error_costs.false_positive",
        ),
        (
            "task: {type: regression, output: values, error_costs: {overestimation: huge}}\n",
            "task.error_costs.overestimation",
        ),
    ],
)
def test_error_costs_validation(text, path):
    """E6: ключ цены ошибок не по типу задачи или уровень вне low/medium/high — ошибка."""
    error = _errors(text)

    assert _message_for(error, path)


@pytest.mark.parametrize("segments", ["[language, language]", "[language, '']"])
def test_bad_segments(segments):
    """E7: повторяющиеся или пустые имена сегментов — ошибка с путём task.segments."""
    error = _errors(f"task: {{type: regression, output: values, segments: {segments}}}\n")

    assert any(path.startswith("task.segments") for path in _paths(error))


def test_all_problems_reported_at_once():
    """E8: несколько проблем перечисляются в одной ошибке."""
    error = _errors("task: {output: values, segmets: [a]}\n")

    paths = _paths(error)
    assert "task.type" in paths
    assert "task.segmets" in paths
    text = str(error)
    assert "task.type" in text
    assert "task.segmets" in text


def test_consistency_problems_reported_with_field_problems():
    """E8: ошибки полей не скрывают проверки согласованности (выход, цена ошибок, сегменты)."""
    error = _errors(
        "task:\n"
        "  type: regression\n"
        "  output: probabilities\n"
        "  segmets: [a]\n"
        "  error_costs: {false_positive: huge}\n"
        "  segments: [x, x]\n"
        "data: {split_strategy: temporal, known_issues: oops}\n"
    )

    paths = _paths(error)
    for path in (
        "task.segmets",
        "task.error_costs.false_positive",
        "task.output",
        "task.segments",
        "data.known_issues",
        "data.time_column",
    ):
        assert path in paths
    pairs = [(problem.path, problem.message) for problem in error.problems]
    assert len(pairs) == len(set(pairs)), "проблемы не должны дублироваться"


def test_missing_file(tmp_path):
    """E9: файла нет — понятная ошибка с путём файла."""
    path = tmp_path / "nope.yaml"

    with pytest.raises(TaskSpecError) as exc:
        load_task_spec(path)

    assert "не найден" in str(exc.value)
    assert "nope.yaml" in str(exc.value)


def test_yaml_syntax_error_has_line():
    """E9: синтаксическая ошибка YAML — сообщение с номером строки."""
    error = _errors("task:\n  type: regression\n  output: [values\n")

    assert "строк" in str(error)


@pytest.mark.parametrize("text", ["", "# только комментарий\n", "- a\n- b\n", "просто строка\n"])
def test_not_a_mapping(text):
    """E9: пустой файл или верхний уровень не словарь — понятная ошибка."""
    error = _errors(text)

    assert "словар" in str(error)


def test_error_names_source_file(tmp_path):
    """E*: сообщение об ошибке указывает источник — имя файла."""
    path = tmp_path / "broken.yaml"
    path.write_text("task: {type: ranking, output: values}\n", encoding="utf-8")

    with pytest.raises(TaskSpecError) as exc:
        load_task_spec(path)

    assert "broken.yaml" in str(exc.value)
