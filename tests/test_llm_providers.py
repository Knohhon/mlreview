"""Провайдеры LLM и настройки (FR-L01, NFR-02, NFR-08).

Спецификация: docs/specs/llm.md (пункты B1, B2, E2, E3).
"""

from __future__ import annotations

import pytest

from _fake_llm import FakeTransport
from mlreview.llm.config import LLMConfigError, load_llm_config
from mlreview.llm.providers import LLMUnavailableError, make_provider
from mlreview.llm.transport import UrllibTransport

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def _anthropic(**overrides):
    return load_llm_config({"llm": {"provider": "anthropic", **overrides}})


def test_anthropic_request_and_response(monkeypatch):
    """B1: адрес, заголовки, тело; текст, токены, модель, стоимость."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    transport = FakeTransport(lambda _payload: {"answer": "ok"})
    completion = make_provider(_anthropic(), transport).complete("система", "вопрос", SCHEMA)
    (call,) = transport.calls
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == "secret-key"
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert call["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    payload = call["payload"]
    assert payload["model"] == "claude-opus-5-5"
    assert payload["max_tokens"] == 16000
    assert payload["system"] == "система"
    assert payload["messages"] == [{"role": "user", "content": "вопрос"}]
    assert payload["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert payload["fallbacks"] == "default"
    assert call["timeout"] == 300
    assert completion.text == '{"answer": "ok"}'
    assert (completion.input_tokens, completion.output_tokens) == (1000, 200)
    assert completion.model == "claude-opus-5-5"
    assert completion.cost == pytest.approx(0.008)


def test_anthropic_options(monkeypatch):
    """B1: effort в output_config; fallbacks выключаются; своя цена."""
    monkeypatch.setenv("MY_KEY", "k")
    config = _anthropic(
        effort="high",
        fallbacks=False,
        api_key_env="MY_KEY",
        model="claude-sonnet-5-5",
        price={"input_per_mtok": 1, "output_per_mtok": 2},
    )
    transport = FakeTransport(lambda _payload: {"answer": "ok"})
    completion = make_provider(config, transport).complete("s", "u", SCHEMA)
    call = transport.calls[0]
    assert "anthropic-beta" not in call["headers"]
    assert "fallbacks" not in call["payload"]
    assert call["payload"]["output_config"]["effort"] == "high"
    assert completion.cost == pytest.approx(1000 * 1 / 1e6 + 200 * 2 / 1e6)


def test_openai_compatible_local_model():
    """B2: локальная модель — chat/completions без ключа, стоимость 0."""
    config = load_llm_config(
        {
            "llm": {
                "provider": "openai",
                "model": "llama3.1",
                "base_url": "http://localhost:11434/v1",
                "api_key_env": None,
            }
        }
    )
    transport = FakeTransport(lambda _payload: {"answer": "ok"}, provider="openai")
    completion = make_provider(config, transport).complete("система", "вопрос", SCHEMA)
    call = transport.calls[0]
    assert call["url"] == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in call["headers"]
    assert call["payload"]["messages"] == [
        {"role": "system", "content": "система"},
        {"role": "user", "content": "вопрос"},
    ]
    assert call["payload"]["response_format"] == {"type": "json_object"}
    assert completion.text == '{"answer": "ok"}'
    assert completion.cost == 0.0


def test_openai_cloud_uses_bearer_key(monkeypatch):
    """B2: облачный OpenAI-совместимый API — ключ в Authorization, цена неизвестна."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = load_llm_config({"llm": {"provider": "openai", "model": "gpt-x"}})
    transport = FakeTransport(lambda _payload: {"answer": "ok"}, provider="openai")
    completion = make_provider(config, transport).complete("s", "u", SCHEMA)
    assert transport.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert completion.cost is None


def test_refusal_and_missing_key(monkeypatch):
    """E2: отказ модели; нет переменной с ключом."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    class Refusing(FakeTransport):
        def post(self, url, headers, payload, timeout):
            body = super().post(url, headers, payload, timeout)
            return {**body, "stop_reason": "refusal", "content": [], "stop_details": None}

    with pytest.raises(LLMUnavailableError, match="отказ модели"):
        make_provider(_anthropic(), Refusing(lambda _p: {})).complete("s", "u", SCHEMA)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
        make_provider(_anthropic(), FakeTransport(lambda _p: {})).complete("s", "u", SCHEMA)


def test_network_is_unavailable_offline(monkeypatch):
    """E2: настоящий транспорт без сети — LLMUnavailableError, а не падение."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(LLMUnavailableError, match="запрос к LLM не выполнен"):
        make_provider(_anthropic(), UrllibTransport()).complete("s", "u", SCHEMA)


@pytest.mark.parametrize(
    ("section", "path"),
    [
        ({"provider": "magic"}, "llm.provider"),
        ({"provider": "openai"}, "llm.model"),
        ({"provider": "anthropic", "colour": "red"}, "llm.colour"),
        ({"provider": "anthropic", "effort": "huge"}, "llm.effort"),
    ],
)
def test_config_errors(section, path):
    """E3: неверные настройки — ошибка с путём ключа."""
    with pytest.raises(LLMConfigError, match=path.replace(".", r"\.")):
        load_llm_config({"llm": section})


def test_disabled_by_default():
    """E1: без раздела llm и с llm: none — выключено."""
    assert load_llm_config({}).provider == "none"
    assert load_llm_config({"llm": "none"}).provider == "none"
