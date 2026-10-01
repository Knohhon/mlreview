"""Общие фикстуры тестов."""

from __future__ import annotations

import socket

import pytest

_NETWORK_FAMILIES = frozenset({socket.AF_INET, socket.AF_INET6})


def _blocked(*_args, **_kwargs):
    raise RuntimeError("сеть в тестах запрещена: тесты должны проходить офлайн (NFR-03)")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Запрещает TCP/UDP-соединения; Unix-сокеты остаются доступны."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(self, address):
        if self.family in _NETWORK_FAMILIES:
            _blocked()
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in _NETWORK_FAMILIES:
            _blocked()
        return original_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
