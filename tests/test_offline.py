"""Тесты работают офлайн (NFR-03, NFR-04, S16).

Спецификация: docs/specs/invariant-guards.md.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("family", "address"),
    [(socket.AF_INET, ("127.0.0.1", 9)), (socket.AF_INET6, ("::1", 9))],
)
def test_network_connection_is_blocked(family, address):
    """B3: сетевое соединение в тестах запрещено."""
    with (
        socket.socket(family, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match="сеть"),
    ):
        sock.connect(address)


def test_create_connection_is_blocked():
    """B3: высокоуровневый ``create_connection`` тоже запрещён."""
    with pytest.raises(RuntimeError, match="сеть"):
        socket.create_connection(("example.com", 443), timeout=1)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="нет Unix-сокетов")
def test_unix_sockets_are_allowed():
    """B4: локальные Unix-сокеты не блокируются."""
    left, right = socket.socketpair(socket.AF_UNIX)
    with left, right:
        left.sendall(b"ok")
        assert right.recv(2) == b"ok"


def test_tests_are_isolated_from_developer_workspace(tmp_path):
    """B5: текущий каталог — временный каталог теста, MLREVIEW_HOME не задан."""
    assert Path.cwd() == tmp_path
    assert "MLREVIEW_HOME" not in os.environ
