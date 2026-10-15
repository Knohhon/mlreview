"""Сверка ответа LLM с фактами и ссылками (FR-L02, FR-L03, Q4, Q5, X3).

Спецификация: docs/specs/llm.md (пункты B4, B5).
"""

from __future__ import annotations

import pytest

from mlreview.llm.grounding import Grounding, check_numbers, validate_answer, validate_hypotheses

FINDINGS = [
    {
        "id": "F-0001",
        "message": "регрессия сегмента language=de: 0.913 → 0.850 (n = 312)",
        "facts": {"delta": 0.041, "ci_low": 0.412345, "x": -0.063, "n": 312, "share": 0.412},
    },
    {"id": "F-0002", "message": "версии v1 → v2", "facts": {"boundary": "2024-01-02"}},
]


@pytest.fixture
def grounding():
    return Grounding.build(FINDINGS, texts=["почему хуже на 10 процентов?"])


@pytest.mark.parametrize(
    "text",
    [
        "рост на 4.1%",
        "интервал от 0,412",
        "дельта −0.063",
        "примерно 0.41",
        "на 312 примерах",
        "с 0.913 до 0.850",
        "находка F-0003 и гипотеза H1, прогон exp-041, данные ds-v3",
        "граница 2024-01-02",
        "вопрос про 10 процентов",
    ],
)
def test_grounded_numbers(grounding, text):
    """B5: проценты, запятая, минус, округление, числа сообщений и вопроса."""
    assert check_numbers(text, grounding) == []


@pytest.mark.parametrize(("text", "bad"), [("примерно 0.43", "0.43"), ("ровно 77 штук", "77")])
def test_ungrounded_numbers(grounding, text, bad):
    """B4, B5: число вне фактов."""
    assert check_numbers(text, grounding) == [bad]


def _valid():
    return {
        "hypotheses": [
            {
                "id": "H1",
                "statement": "сегмент хуже на 0.063",
                "findings": ["F-0001"],
                "alternative": "шум",
            },
            {
                "id": "H2",
                "statement": "данные сменились",
                "findings": ["F-0002"],
                "alternative": "модель",
            },
        ],
        "experiment": {
            "description": "обучить старую модель на новых данных",
            "findings": ["F-0001"],
        },
    }


def test_valid_hypotheses(grounding):
    """B3: корректный ответ проходит."""
    assert validate_hypotheses(_valid(), {"F-0001", "F-0002"}, grounding) == []


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda d: d["hypotheses"][0].update(statement="хуже на 0.77"), "число 0.77 не найдено"),
        (lambda d: d["hypotheses"][1].update(findings=[]), "H2 без ссылки на находку"),
        (lambda d: d["hypotheses"][1].update(findings=["F-0099"]), "F-0099"),
        (lambda d: d["hypotheses"].pop(), "нужно не меньше двух гипотез"),
        (lambda d: d["hypotheses"][0].pop("alternative"), "H1 без альтернативного объяснения"),
        (lambda d: d.pop("experiment"), "нет различающего эксперимента"),
        (lambda d: d["experiment"].update(findings=[]), "эксперимент без ссылки на находку"),
    ],
)
def test_invalid_hypotheses(grounding, mutate, problem):
    """B4 (X3): блокирующие нарушения."""
    data = _valid()
    mutate(data)
    problems = validate_hypotheses(data, {"F-0001", "F-0002"}, grounding)
    assert any(problem in item for item in problems), problems


def test_answer_validation(grounding):
    """B4: ответ на вопрос — непустой, со ссылками из контекста, числа из фактов."""
    ids = {"F-0001", "H-0001"}
    ok = {"answer": "пробовали в H-0001, эффект 0.041", "references": ["H-0001"]}
    assert validate_answer(ok, ids, grounding) == []
    assert validate_answer({"answer": "да", "references": []}, ids, grounding)
    assert validate_answer({"answer": "эффект 0.5", "references": ["H-0001"]}, ids, grounding)
    assert validate_answer({"answer": "", "references": ["H-0001"]}, ids, grounding)
