"""Сверка ответа LLM с фактами: числа — из фактов, утверждения — со ссылками (FR-L02, FR-L03).

Ответ, не прошедший сверку, блокируется (X3, Q4, Q5). Правила зафиксированы
в `docs/specs/llm.md`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Число: необязательный знак, дробная часть через точку или запятую, суффикс процентов.
# Цифры внутри идентификаторов (F-0003, H1, exp-041, ds-v3) числами не считаются.
NUMBER = re.compile(r"(?<![\w.\-−/])([-+−]?\d+(?:[.,]\d+)?)(%|pp|п\.\s?п\.)?(?!\w)")
MIN_HYPOTHESES = 2


def numbers_in(text: str) -> list[str]:
    """Числа в тексте как написаны (без суффикса процентов)."""
    return [match.group(1) for match in NUMBER.finditer(text)]


def _value(token: str) -> tuple[float, int]:
    normal = token.replace("−", "-").replace(",", ".")
    decimals = len(normal.split(".", 1)[1]) if "." in normal else 0
    return abs(float(normal)), decimals


@dataclass(frozen=True)
class Grounding:
    """Допустимые числа: факты находок и числа текстов контекста."""

    values: tuple[float, ...]

    @classmethod
    def build(cls, findings: Sequence[Mapping[str, Any]], texts: Iterable[str] = ()) -> Grounding:
        values: list[float] = []
        strings: list[str] = list(texts)
        for finding in findings:
            for value in (finding.get("facts") or {}).values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    values.append(abs(float(value)))
                elif isinstance(value, str):
                    strings.append(value)
            strings += [finding.get("message") or "", finding.get("suggestion") or ""]
        for text in strings:
            values += [_value(token)[0] for token in numbers_in(text)]
        return cls(tuple(values))

    def allows(self, token: str) -> bool:
        """Число — округление модуля факта (или ×100 — проценты) до показанных знаков."""
        number, decimals = _value(token)
        tolerance = 0.5 * 10.0**-decimals + 1e-9
        return any(
            abs(value - number) <= tolerance or abs(value * 100 - number) <= tolerance
            for value in self.values
        )


def check_numbers(text: str, grounding: Grounding) -> list[str]:
    """Числа текста, которых нет в фактах."""
    return [token for token in numbers_in(text) if not grounding.allows(token)]


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _references(label: str, refs: Any, ids: set[str]) -> list[str]:
    if not isinstance(refs, list) or not refs:
        return [f"{label} без ссылки на находку"]
    return [f"{label}: ссылка на неизвестную находку {ref}" for ref in refs if ref not in ids]


def _numbers(label: str, texts: Iterable[str], grounding: Grounding) -> list[str]:
    return [
        f"{label}: число {token} не найдено в фактах"
        for text in texts
        for token in check_numbers(text, grounding)
    ]


def validate_hypotheses(data: Any, ids: set[str], grounding: Grounding) -> list[str]:
    """Причины блокировки гипотез; пусто — ответ прошёл сверку."""
    if not isinstance(data, Mapping) or not isinstance(data.get("hypotheses"), list):
        return ["ответ без списка hypotheses"]
    hypotheses = data["hypotheses"]
    problems = []
    if len(hypotheses) < MIN_HYPOTHESES:
        problems.append(f"нужно не меньше двух гипотез, получено {len(hypotheses)}")
    for index, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, Mapping):
            problems.append(f"гипотеза {index} — не объект")
            continue
        label = _text(hypothesis.get("id")) or f"H{index}"
        statement, alternative = (
            _text(hypothesis.get("statement")),
            _text(hypothesis.get("alternative")),
        )
        if not statement:
            problems.append(f"{label} без утверждения")
        if not alternative:
            problems.append(f"{label} без альтернативного объяснения")
        problems += _references(label, hypothesis.get("findings"), ids)
        problems += _numbers(label, [statement, alternative], grounding)
    experiment = data.get("experiment")
    if not isinstance(experiment, Mapping) or not _text(experiment.get("description")):
        problems.append("нет различающего эксперимента")
    else:
        problems += _references("эксперимент", experiment.get("findings"), ids)
        problems += _numbers("эксперимент", [_text(experiment.get("description"))], grounding)
    return problems


def validate_answer(data: Any, ids: set[str], grounding: Grounding) -> list[str]:
    """Причины блокировки ответа на вопрос."""
    if not isinstance(data, Mapping) or not _text(data.get("answer")):
        return ["пустой ответ"]
    problems = _references("ответ", data.get("references"), ids)
    return problems + _numbers("ответ", [_text(data.get("answer"))], grounding)
