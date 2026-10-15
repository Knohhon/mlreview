"""Единственная точка выхода в сеть для LLM: POST JSON через стандартный urllib (NFR-04).

Тесты подменяют `get_transport` и видят каждый исходящий запрос (аудит приватности).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol


class TransportError(RuntimeError):
    """Запрос не выполнен: сеть, HTTP-ошибка, не JSON."""


class Transport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]: ...


class UrllibTransport:
    """POST JSON → JSON через `urllib.request`."""

    def post(
        self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
    ) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise TransportError(f"HTTP {exc.code} от {url}: {detail}") from exc
        except Exception as exc:  # сеть, DNS, таймаут, TLS — у них разные типы ошибок
            raise TransportError(f"{url}: {type(exc).__name__}: {exc}") from exc
        try:
            result = json.loads(body)
        except ValueError as exc:
            raise TransportError(f"{url}: ответ не JSON") from exc
        if not isinstance(result, dict):
            raise TransportError(f"{url}: ответ не объект JSON")
        return result


def get_transport() -> Transport:
    return UrllibTransport()
