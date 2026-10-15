"""LLM в командах: гипотезы, ask, приватность, деградация (FR-L01–L03, FR-D07, NFR-04, NFR-08).

Спецификация: docs/specs/llm.md (пункты B3, B4, B6–B9, E1, E2, E4).
"""

from __future__ import annotations

import json

import pytest

from _factories import BINARY_SPEC_TEXT, binary_tables
from _fake_llm import FakeTransport, context_of, hypotheses_from
from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main

CONFIG = "llm: {provider: anthropic}\n"


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    predictions, examples = binary_tables(duplicates=10)
    save_run_files(tmp_path / "run", predictions, examples=examples, data_version="v1")
    main(["init"])
    (tmp_path / ".mlreview" / "config.yaml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return tmp_path


def _install(monkeypatch, respond, **kwargs):
    transport = FakeTransport(respond, **kwargs)
    monkeypatch.setattr("mlreview.llm.transport.get_transport", lambda: transport)
    return transport


def _review(*args):
    return main(["review", "run", "--spec", "task.yaml", "--hypotheses", *args])


def test_hypotheses_are_shown_with_cost(project, monkeypatch, capsys):
    """B3: гипотезы, альтернативы, эксперимент, стоимость; журнал запросов."""
    capsys.readouterr()
    transport = _install(monkeypatch, lambda payload: hypotheses_from(context_of(payload)))
    assert _review() == 0
    out = capsys.readouterr().out
    assert "гипотезы (LLM: anthropic/claude-opus-5-5):" in out
    assert "  H1. дубли завышают метрику" in out
    assert "      альтернатива: метрика выросла из-за модели" in out
    assert "  различающий эксперимент: убрать дубли и переоценить модель [F-0001]" in out
    assert out.rstrip().endswith(
        "LLM: anthropic/claude-opus-5-5 · токены 1000 → 200 · стоимость $0.0080"
    )
    assert len(transport.calls) == 1
    log = (project / ".mlreview" / "llm-log.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(log[0])
    assert entry["purpose"] == "hypotheses"
    assert entry["status"] == "ok"
    assert entry["cost_usd"] == pytest.approx(0.008)

    assert _review("--format", "json", "--no-store") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["llm"]["status"] == "ok"
    assert len(data["llm"]["hypotheses"]) == 2
    assert data["llm"]["input_tokens"] == 1000


@pytest.mark.usefixtures("project")
def test_ungrounded_answer_is_blocked(monkeypatch, capsys):
    """B4 (X3): число вне фактов — ответ заблокирован, находки на месте."""
    capsys.readouterr()
    _install(monkeypatch, lambda p: hypotheses_from(context_of(p), extra=" на 0.777"))
    assert _review() == 0
    out = capsys.readouterr().out
    assert "data.split_duplicates" in out
    assert "LLM: ответ заблокирован сверкой с фактами (FR-L02): H1: число 0.777 не найдено" in out
    assert "  H1." not in out


@pytest.mark.usefixtures("project")
def test_ask_uses_journal(monkeypatch, capsys):
    """B6 (S05): ask — записи журнала в контексте, ответ со ссылками."""
    main(["journal", "add", "hypothesis", "focal loss поднимет recall", "--data-version", "v1"])
    args = ["journal", "add", "decision", "откатить focal loss", "--hypothesis", "H-0001"]
    main([*args, "--outcome", "reject"])
    capsys.readouterr()

    def respond(payload):
        context = context_of(payload)
        assert [entry["id"] for entry in context["journal"]] == ["H-0001", "D-0001"]
        return {
            "answer": "да, в H-0001; решение D-0001 — откатить",
            "references": ["H-0001", "D-0001"],
        }

    _install(monkeypatch, respond)
    assert main(["ask", "пробовал ли я focal loss?"]) == 0
    out = capsys.readouterr().out
    assert "да, в H-0001; решение D-0001 — откатить" in out
    assert "ссылки: H-0001, D-0001" in out
    assert "стоимость $0.0080" in out


@pytest.mark.usefixtures("project")
def test_privacy_audit(monkeypatch, capsys):
    """B7 (NFR-04, X6): без явного выбора примеры и признаки не уходят."""
    capsys.readouterr()
    transport = _install(monkeypatch, lambda p: hypotheses_from(context_of(p)))
    _, examples = binary_tables(duplicates=10)
    assert _review("--no-store") == 0
    sent = transport.sent
    assert "te-0" not in sent  # id примеров из доказательств
    assert "tr-0" not in sent
    for value in examples["f0"].head(20):
        assert repr(float(value))[:8] not in sent
    assert _review("--no-store", "--example", "te-001", "--yes") == 0
    payload = transport.calls[-1]["payload"]
    shown = context_of(payload)["examples"]
    assert [row["example_id"] for row in shown] == ["te-001"]
    assert "f0" in shown[0]
    assert "label" in shown[0]


@pytest.mark.usefixtures("project")
def test_examples_need_consent(monkeypatch, capsys):
    """B8: без --yes и без терминала примеры не уходят."""
    capsys.readouterr()
    transport = _install(monkeypatch, lambda p: hypotheses_from(context_of(p)))
    assert _review("--no-store", "--example", "te-001") == 0
    assert "примеры не отправлены: нет согласия (--yes)" in capsys.readouterr().out
    assert "examples" not in context_of(transport.calls[-1]["payload"])


@pytest.mark.usefixtures("project")
def test_dry_run_sends_nothing(monkeypatch, capsys):
    """B9: --dry-run печатает запрос и не отправляет его."""
    capsys.readouterr()
    transport = _install(monkeypatch, lambda p: hypotheses_from(context_of(p)))
    assert _review("--no-store", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "запрос к LLM (не отправлен):" in out
    assert '"model": "claude-opus-5-5"' in out
    assert transport.calls == []


def test_disabled_llm(tmp_path, capsys):
    """E1 (S16): llm.provider: none — отчёт детерминированный; ask — код 1 и поиск."""
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    predictions, examples = binary_tables(duplicates=10)
    save_run_files(tmp_path / "run", predictions, examples=examples)
    assert _review("--no-store") == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("LLM: гипотезы требуют LLM — в настройках llm.provider: none")
    main(["init"])
    main(["journal", "add", "note", "focal loss не помог"])
    capsys.readouterr()
    assert main(["ask", "focal loss?"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("mlreview ask: требует LLM — в настройках llm.provider: none")
    assert "N-0001" in captured.out


@pytest.mark.usefixtures("project")
def test_unavailable_llm(capsys):
    """E2 (X2): сети нет — детерминированный отчёт и причина; ask — код 1."""
    capsys.readouterr()
    assert _review() == 0
    out = capsys.readouterr().out
    assert "data.split_duplicates" in out
    assert "LLM: недоступен — запрос к LLM не выполнен" in out
    main(["journal", "add", "note", "заметка"])
    capsys.readouterr()
    assert main(["ask", "заметка?"]) == 1
    assert "LLM: недоступен" in capsys.readouterr().err


@pytest.mark.usefixtures("project")
def test_ask_without_facts(monkeypatch, capsys):
    """E4: нечего передать в LLM — запрос не отправляется."""
    capsys.readouterr()
    transport = _install(monkeypatch, lambda _p: {"answer": "x", "references": []})
    assert main(["ask", "что нового?"]) == 1
    assert "нет фактов для ответа" in capsys.readouterr().err
    assert transport.calls == []
