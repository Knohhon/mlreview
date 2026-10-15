"""MCP-сервер с операциями CLI (FR-R02).

Спецификация: docs/specs/mcp-server.md.
"""

from __future__ import annotations

import sys

import pytest

from _factories import BINARY_SPEC_TEXT, binary_tables
from mlreview import api
from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main
from mlreview.cli.common import CliError
from mlreview.mcp_server import tools

EXPECTED = [
    "review",
    "contract_check",
    "metrics_recommend",
    "metrics_test",
    "findings_list",
    "finding_show",
    "finding_set_status",
    "journal_add",
    "journal_search",
    "ask",
]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
    for i in range(2):
        predictions, examples = binary_tables(seed=i + 1, duplicates=10)
        save_run_files(tmp_path / f"run-{i}", predictions, examples=examples, data_version="v1")
    return tmp_path


def test_tool_table():
    """B1: инструменты и описания."""
    assert [tool.name for tool in tools.TOOLS] == EXPECTED
    for tool in tools.TOOLS:
        assert tool.description


@pytest.mark.usefixtures("project")
def test_review_tool_matches_api():
    """B2: review = api.review_runs(...).to_dict(); groups — как diagnose."""
    result = tools.review(run="run-0", spec="task.yaml", baseline="run-1", store=False)
    expected = api.review_runs(api.load_spec("task.yaml"), "run-0", baseline="run-1", store=False)
    assert result == expected.to_dict()
    data = tools.review(run="run-0", spec="task.yaml", groups=["data"], store=False)
    assert {f["check"].split(".")[0] for f in data["findings"]} == {"data"}


@pytest.mark.usefixtures("project")
def test_store_and_journal_tools():
    """B2, E2: хранилище и журнал через инструменты."""
    with pytest.raises(CliError, match="mlreview init"):
        tools.findings_list()
    main(["init"])
    tools.review(run="run-0", spec="task.yaml")
    listed = tools.findings_list()
    assert listed["findings"][0]["id"] == "F-0001"
    assert tools.finding_show(finding_id="F-0001")["status"] == "new"
    with pytest.raises(Exception, match="требует причины"):
        tools.finding_set_status(finding_id="F-0001", status="dismissed")
    changed = tools.finding_set_status(finding_id="F-0001", status="dismissed", reason="ок")
    assert changed["status"] == "dismissed"
    entry = tools.journal_add(kind="hypothesis", text="focal loss поможет", runs=["run-0"])
    assert entry["id"] == "H-0001"
    found = tools.journal_search(query="focal loss")
    assert found["hits"][0]["entry"]["id"] == "H-0001"
    assert tools.contract_check(run="run-0", spec="task.yaml")["run_id"] == "run-0"
    plan = tools.metrics_recommend(spec="task.yaml", answers={"decision": "probability"})
    assert plan["primary"] == "log_loss"
    tested = tools.metrics_test(run="run-0", spec="task.yaml")
    assert tested["metrics"][0]["name"] == "roc_auc"
    assert tested["metrics"][0]["tests"]


@pytest.mark.usefixtures("project")
def test_ask_without_llm_is_not_an_error():
    """B5: ask без LLM — статус disabled и поиск по журналу."""
    main(["init"])
    tools.journal_add(kind="note", text="focal loss не помог")
    result = tools.ask(question="focal loss?")
    assert result["status"] == "disabled"
    assert "N-0001" in "\n".join(result["journal_search"])


@pytest.mark.usefixtures("project")
def test_protocol_roundtrip():
    """B1, B3, B4: клиент MCP в процессе — список, вызов, ошибка с сообщением."""
    pytest.importorskip("mcp")
    import anyio  # noqa: PLC0415 — зависимость mcp
    from mcp import Client  # noqa: PLC0415

    from mlreview.mcp_server.server import build_server  # noqa: PLC0415

    async def scenario():
        async with Client(build_server()) as client:
            listed = await client.list_tools()
            names = [tool.name for tool in listed.tools]
            ok = await client.call_tool(
                "review", {"run": "run-0", "spec": "task.yaml", "store": False}
            )
            bad = await client.call_tool("review", {"run": "nope", "spec": "task.yaml"})
            return names, ok, bad

    names, ok, bad = anyio.run(scenario)
    assert names == EXPECTED
    assert not ok.is_error
    assert '"run_id": "run-0"' in ok.content[0].text
    assert bad.is_error
    assert "nope" in bad.content[0].text


def test_missing_extra(monkeypatch, capsys):
    """E1: без пакета mcp — подсказка установки."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    assert main(["mcp"]) == 1
    assert 'pip install "mlreview[mcp]"' in capsys.readouterr().err
