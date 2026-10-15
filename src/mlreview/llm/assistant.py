"""Синтез поверх находок: гипотезы с различающим экспериментом и ответы на вопросы (FR-D07).

Ответ проходит сверку с фактами (FR-L02, FR-L03); стоимость показывается и пишется
в журнал запросов `.mlreview/llm-log.jsonl` (NFR-08). Поведение — `docs/specs/llm.md`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlreview.llm import transport as transport_module
from mlreview.llm.config import LLMConfig
from mlreview.llm.providers import LLMUnavailableError, make_provider

MARKER = "Контекст (JSON):\n"
DISABLED = "в настройках llm.provider: none"

SYSTEM = (
    "Ты — критик ML-экспериментов. Опирайся только на факты из контекста JSON: находки"
    " детерминированных анализаторов, записи журнала, описание задачи. Каждое число в ответе"
    " бери из фактов или сообщений находок дословно (можно округлить или записать в процентах);"
    " не вычисляй новых чисел. Каждое утверждение ссылается на id из контекста. Ты не применяешь"
    " изменения сам — только предлагаешь эксперимент. Отвечай по-русски, строго JSON по схеме."
)
HYPOTHESES_TASK = (
    "Предложи не меньше двух гипотез, объясняющих находки. У каждой гипотезы — ссылки на id"
    " находок (поле findings) и альтернативное объяснение тех же фактов (поле alternative,"
    " роль «адвоката дьявола»). Затем предложи один различающий эксперимент: такой, чей исход"
    " отличает гипотезы друг от друга, со ссылками на находки."
)
ANSWER_TASK = (
    "Ответь на вопрос пользователя по контексту. В поле references перечисли id находок,"
    " записей журнала или прогонов, на которых основан ответ. Если данные с тех пор сменились"
    " (пометка stale), скажи об этом. Если в контексте нет ответа, так и скажи."
)
_IDS = {"type": "array", "items": {"type": "string"}}
HYPOTHESES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "findings": _IDS,
                    "alternative": {"type": "string"},
                },
                "required": ["id", "statement", "findings", "alternative"],
                "additionalProperties": False,
            },
        },
        "experiment": {
            "type": "object",
            "properties": {"description": {"type": "string"}, "findings": _IDS},
            "required": ["description", "findings"],
            "additionalProperties": False,
        },
    },
    "required": ["hypotheses", "experiment"],
    "additionalProperties": False,
}
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "references": _IDS},
    "required": ["answer", "references"],
    "additionalProperties": False,
}


@dataclass
class LLMOutcome:
    """Итог обращения к LLM: статус, данные ответа, причины блокировки, токены, стоимость."""

    status: str  # ok | blocked | unavailable | disabled | dry_run
    label: str | None = None
    data: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)
    reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    payload: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "model": self.label,
            "problems": self.problems,
            "reason": self.reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost,
            "notes": self.notes,
        }
        if self.data is not None:
            result.update(self.data)
        if self.payload is not None:
            result["request"] = self.payload
        return result

    def cost_line(self) -> str:
        cost = f"${self.cost:.4f}" if self.cost is not None else "неизвестна (задайте llm.price)"
        tokens = f"{self.input_tokens} → {self.output_tokens}"
        return f"LLM: {self.label} · токены {tokens} · стоимость {cost}"


def user_message(task: str, context: Mapping[str, Any]) -> str:
    body = json.dumps(context, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    return f"{task}\n\n{MARKER}{body}"


def parse_json(text: str) -> Any:
    """JSON ответа; модели иногда оборачивают его в ```json … ```."""
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


class Assistant:
    """Вызов LLM по настройкам: запрос, сверка, журнал стоимости."""

    def __init__(self, config: LLMConfig, log_path: Path | None = None) -> None:
        self.config = config
        self.log_path = log_path

    def _log(self, purpose: str, outcome: LLMOutcome) -> None:
        if self.log_path is None or outcome.status in ("disabled", "dry_run"):
            return
        record = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": self.config.provider,
            "model": outcome.label,
            "purpose": purpose,
            "status": outcome.status,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "cost_usd": outcome.cost,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # журнал стоимости не должен ронять ответ; стоимость уже показана

    def run(  # noqa: PLR0913 — назначение, задача, контекст, схема, сверка, режим
        self,
        purpose: str,
        task: str,
        context: Mapping[str, Any],
        schema: dict[str, Any],
        validate: Callable[[Any], list[str]],
        *,
        dry_run: bool = False,
    ) -> LLMOutcome:
        config = self.config
        if not config.enabled:
            return LLMOutcome("disabled", reason=DISABLED)
        user = user_message(task, context)
        if dry_run:
            # без ключа: в напечатанный запрос заголовки и ключ не попадают
            preview = make_provider(replace(config, api_key_env=None), _NoTransport())
            _, _, payload = preview.request(SYSTEM, user, schema)
            return LLMOutcome("dry_run", label=config.label, payload=payload)
        provider = make_provider(config, transport_module.get_transport())
        try:
            completion = provider.complete(SYSTEM, user, schema)
        except LLMUnavailableError as exc:
            outcome = LLMOutcome("unavailable", label=config.label, reason=str(exc))
            self._log(purpose, outcome)
            return outcome
        outcome = LLMOutcome(
            "ok",
            label=f"{config.provider}/{completion.model}",
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost=completion.cost,
        )
        try:
            data = parse_json(completion.text)
        except ValueError:
            outcome.status, outcome.reason = "unavailable", "ответ модели не JSON"
        else:
            outcome.problems = validate(data)
            outcome.status = "blocked" if outcome.problems else "ok"
            outcome.data = data if isinstance(data, dict) and not outcome.problems else None
        self._log(purpose, outcome)
        return outcome


class _NoTransport:
    def post(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise LLMUnavailableError("запрос не отправляется (--dry-run)")
