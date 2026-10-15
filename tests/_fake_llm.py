"""Поддельный транспорт LLM: записывает запросы и отвечает по контексту (NFR-04, тесты)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

MARKER = "Контекст (JSON):\n"


def context_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Контекст из сообщения пользователя запроса (Anthropic или OpenAI-совместимого)."""
    messages = [m for m in payload["messages"] if m["role"] == "user"]
    return json.loads(messages[-1]["content"].split(MARKER, 1)[1])


def _first_number(finding: dict[str, Any]) -> str:
    for value in finding["facts"].values():
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{value:g}"
    return ""


def hypotheses_from(context: dict[str, Any], *, extra: str = "") -> dict[str, Any]:
    findings = context["findings"]
    first, last = findings[0], findings[-1]
    return {
        "hypotheses": [
            {
                "id": "H1",
                "statement": f"дубли завышают метрику ({_first_number(first)}){extra}",
                "findings": [first["id"]],
                "alternative": "метрика выросла из-за модели",
            },
            {
                "id": "H2",
                "statement": "разница объясняется данными",
                "findings": [last["id"]],
                "alternative": "шум бутстрапа",
            },
        ],
        "experiment": {
            "description": "убрать дубли и переоценить модель",
            "findings": [first["id"]],
        },
    }


class FakeTransport:
    """Транспорт без сети: `respond(payload) -> dict` строит тело JSON-ответа модели."""

    def __init__(
        self, respond: Callable[[dict[str, Any]], dict[str, Any]], *, provider="anthropic"
    ):
        self.respond = respond
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def post(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        text = json.dumps(self.respond(payload), ensure_ascii=False)
        if self.provider == "openai":
            return {
                "model": payload["model"],
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
            }
        return {
            "model": payload["model"],
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }

    @property
    def sent(self) -> str:
        """Всё, что ушло наружу, одной строкой (для аудита приватности)."""
        return json.dumps([call["payload"] for call in self.calls], ensure_ascii=False)
