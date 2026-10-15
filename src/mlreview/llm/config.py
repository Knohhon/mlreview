"""Настройки LLM из `.mlreview/config.yaml`: провайдер, модель, адрес, ключ, цены (FR-L01).

Смена провайдера — правкой раздела `llm` (NFR-02). Поведение зафиксировано
в `docs/specs/llm.md`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

PROVIDERS = ("none", "anthropic", "openai")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_URLS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1"}
DEFAULT_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
# $ за 1 млн токенов (вход, выход): первые лица Anthropic API.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
KEYS = frozenset(
    {"provider", "model", "base_url", "api_key_env", "max_tokens", "timeout", "effort"}
    | {"fallbacks", "price"}
)


class LLMConfigError(ValueError):
    """Раздел `llm` настроек невалиден; сообщение содержит путь ключа."""


@dataclass(frozen=True)
class LLMConfig:
    """Настройки провайдера LLM."""

    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 16000
    timeout: float = 300.0
    effort: str | None = None
    fallbacks: bool = True
    price: tuple[float, float] | None = None

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    @property
    def is_local(self) -> bool:
        return (urlparse(self.base_url or "").hostname or "") in LOCAL_HOSTS

    def price_for(self, model: str | None) -> tuple[float, float] | None:
        """Цена за 1 млн токенов: из настроек, для локальной модели — 0, иначе из таблицы."""
        if self.price is not None:
            return self.price
        if self.is_local:
            return (0.0, 0.0)
        return PRICES.get(model or "") or PRICES.get(self.model or "")

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


def _number(section: Mapping[str, Any], key: str, kind: type, default: Any) -> Any:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise LLMConfigError(f"llm.{key}: ожидается положительное число")
    return kind(value)


def _price(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"input_per_mtok", "output_per_mtok"}:
        raise LLMConfigError("llm.price: ожидается {input_per_mtok: <$>, output_per_mtok: <$>}")
    try:
        return float(value["input_per_mtok"]), float(value["output_per_mtok"])
    except (TypeError, ValueError):
        raise LLMConfigError("llm.price: цены — числа") from None


def load_llm_config(settings: Mapping[str, Any]) -> LLMConfig:
    """Раздел `llm` настроек → `LLMConfig`; `llm: none` или нет раздела — выключено."""
    section = settings.get("llm")
    if section is None or section == "none":
        return LLMConfig()
    if not isinstance(section, Mapping):
        raise LLMConfigError("llm: ожидается словарь настроек или none")
    unknown = sorted(set(section) - KEYS)
    if unknown:
        raise LLMConfigError(
            f"llm.{unknown[0]}: неизвестный ключ; допустимо: {', '.join(sorted(KEYS))}"
        )
    provider = section.get("provider", "none")
    if provider not in PROVIDERS:
        raise LLMConfigError(f"llm.provider: {provider!r}; допустимо: {', '.join(PROVIDERS)}")
    if provider == "none":
        return LLMConfig()
    model = section.get("model", DEFAULT_MODEL if provider == "anthropic" else None)
    if not isinstance(model, str) or not model.strip():
        raise LLMConfigError(f"llm.model: обязателен для provider: {provider}")
    effort = section.get("effort")
    if effort is not None and effort not in EFFORTS:
        raise LLMConfigError(f"llm.effort: {effort!r}; допустимо: {', '.join(EFFORTS)}")
    key_env = section.get("api_key_env", DEFAULT_KEYS[provider])
    if key_env is not None and not isinstance(key_env, str):
        raise LLMConfigError("llm.api_key_env: имя переменной окружения или null")
    fallbacks = section.get("fallbacks", True)
    if not isinstance(fallbacks, bool):
        raise LLMConfigError("llm.fallbacks: true или false")
    base_url = section.get("base_url", DEFAULT_URLS[provider])
    if not isinstance(base_url, str):
        raise LLMConfigError("llm.base_url: ожидается адрес")
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key_env=key_env,
        max_tokens=_number(section, "max_tokens", int, 16000),
        timeout=_number(section, "timeout", float, 300),
        effort=effort,
        fallbacks=fallbacks,
        price=_price(section.get("price")),
    )
