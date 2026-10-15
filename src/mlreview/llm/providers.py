"""Адаптеры провайдеров LLM: Anthropic Messages API и OpenAI-совместимый Chat Completions.

OpenAI-совместимый адаптер покрывает локальные модели (Ollama, vLLM, LM Studio) — FR-L01,
NFR-02. Поведение и форма запросов зафиксированы в `docs/specs/llm.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from mlreview.llm.config import LLMConfig
from mlreview.llm.transport import Transport, TransportError

ANTHROPIC_VERSION = "2023-06-01"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailableError(RuntimeError):
    """LLM не ответил: нет ключа, сеть, HTTP-ошибка, отказ модели (X2)."""


@dataclass(frozen=True)
class Completion:
    """Ответ модели: текст, модель, токены и стоимость (NFR-08)."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float | None


def _key(config: LLMConfig) -> str | None:
    if config.api_key_env is None:
        return None
    key = os.environ.get(config.api_key_env)
    if not key:
        raise LLMUnavailableError(f"нет ключа: переменная {config.api_key_env} не задана")
    return key


def _cost(config: LLMConfig, model: str, tokens_in: int, tokens_out: int) -> float | None:
    price = config.price_for(model)
    if price is None:
        return None
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000


class Provider:
    """Провайдер: собирает запрос, отправляет через транспорт, разбирает ответ."""

    def __init__(self, config: LLMConfig, transport: Transport) -> None:
        self.config = config
        self.transport = transport

    def request(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, dict, dict]:
        raise NotImplementedError

    def parse(self, body: dict[str, Any]) -> tuple[str, str, int, int]:
        raise NotImplementedError

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> Completion:
        url, headers, payload = self.request(system, user, schema)
        try:
            body = self.transport.post(url, headers, payload, self.config.timeout)
        except TransportError as exc:
            raise LLMUnavailableError(f"запрос к LLM не выполнен: {exc}") from exc
        try:
            text, model, tokens_in, tokens_out = self.parse(body)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"неожиданный формат ответа: {exc!r}") from exc
        cost = _cost(self.config, model, tokens_in, tokens_out)
        return Completion(text, model, tokens_in, tokens_out, cost)


class AnthropicProvider(Provider):
    """Anthropic Messages API: `POST /v1/messages`."""

    def request(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, dict, dict]:
        config = self.config
        headers = {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        key = _key(config)
        if key is not None:
            headers["x-api-key"] = key
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if config.effort:
            output_config["effort"] = config.effort
        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if config.fallbacks:
            headers["anthropic-beta"] = FALLBACK_BETA
            payload["fallbacks"] = "default"
        return f"{config.base_url}/v1/messages", headers, payload

    def parse(self, body: dict[str, Any]) -> tuple[str, str, int, int]:
        if body.get("stop_reason") == "refusal":
            details = body.get("stop_details") or {}
            category = details.get("category") if isinstance(details, dict) else None
            raise LLMUnavailableError(f"отказ модели (категория: {category or 'не указана'})")
        text = "".join(
            block.get("text", "") for block in body["content"] if block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        model = body.get("model") or self.config.model
        return text, model, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


class OpenAIProvider(Provider):
    """OpenAI-совместимый Chat Completions: `POST {base_url}/chat/completions`."""

    def request(self, system: str, user: str, schema: dict[str, Any]) -> tuple[str, dict, dict]:
        del schema  # схема описана в промпте: json_schema поддерживают не все серверы
        config = self.config
        headers = {"content-type": "application/json"}
        key = _key(config)
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        return f"{config.base_url}/chat/completions", headers, payload

    def parse(self, body: dict[str, Any]) -> tuple[str, str, int, int]:
        choice = body["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            raise LLMUnavailableError("отказ модели (content_filter)")
        usage = body.get("usage") or {}
        model = body.get("model") or self.config.model
        tokens_in = int(usage.get("prompt_tokens", 0))
        return (
            choice["message"]["content"] or "",
            model,
            tokens_in,
            int(usage.get("completion_tokens", 0)),
        )


def make_provider(config: LLMConfig, transport: Transport) -> Provider:
    if config.provider == "anthropic":
        return AnthropicProvider(config, transport)
    if config.provider == "openai":
        return OpenAIProvider(config, transport)
    raise LLMUnavailableError("LLM выключен: llm.provider: none")
