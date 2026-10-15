# MCP-сервер

Опора: FR-R02 (MCP-сервер с теми же операциями, что CLI — L1 · Must), S03 (агент в IDE вызывает `diagnose` через MCP), FR-R01, NFR-03, NFR-04.

`mlreview mcp` запускает MCP-сервер по stdio на официальном SDK `mcp` (extra `mcp`: `pip install "mlreview[mcp]"`, версия 2.x, класс `MCPServer`). Инструменты — функции `mlreview.mcp_server.tools` поверх `mlreview.api`; они возвращают словари JSON, как `--format json` в CLI. Подключение к агенту, например в Claude Code: `claude mcp add mlreview -- mlreview mcp` (запускать из каталога проекта, где `.mlreview/`).

| Инструмент | Команда CLI |
| --- | --- |
| `review(run, spec, baseline=None, seeds=[], baseline_seeds=[], groups=[], min_severity="info", limit=0, store=True, tracking_uri=None, hypotheses=False, focus=None)` | `review`, `diagnose <группа>` (через `groups`) |
| `contract_check(run, spec, tracking_uri=None)` | `contract check` |
| `metrics_recommend(spec, answers={}, run=None)` | `metrics init` (без записи в файл) |
| `metrics_test(run, spec)` | `metrics test` |
| `findings_list(status=None, min_severity="low", check=None)` | `findings list` |
| `finding_show(finding_id)` | `findings show` |
| `finding_set_status(finding_id, status, reason=None)` | `findings accept/fix/reopen`, `dismiss` |
| `journal_add(kind, text, runs=[], findings=[], hypothesis=None, outcome=None, data_version=None)` | `journal add` |
| `journal_search(query, spec=None)` | `journal search` |
| `ask(question, spec=None, run=None, baseline=None)` | `ask` |

Примеры (`--example`) через MCP не отправляются: в LLM уходят только агрегаты (NFR-04).

## Ожидаемое поведение

- B1. Дано: сервер · тогда список инструментов — таблица выше, у каждого описание по-русски и схема аргументов.
- B2. Дано: `review` по каталогу прогона · тогда результат равен `api.review_runs(...).to_dict()` на тех же входах; `groups=["data"]` — как `diagnose data`.
- B3. Дано: ошибка операции (невалидная спецификация, нет прогона, неизвестная находка, отклонение critical без причины) · тогда ошибка инструмента с тем же сообщением, что в CLI.
- B4. Дано: клиент MCP в том же процессе · тогда `list_tools` и `call_tool("review", …)` работают по протоколу.
- B5. Дано: `ask` без LLM · тогда результат `{"status": "disabled", …}` с детерминированной выдачей поиска по журналу, а не ошибка.

## Деградация и ошибки

- E1. Дано: extra `mcp` не установлен · тогда `mlreview mcp` — код 1 и «MCP-сервер требует пакет mcp: pip install "mlreview[mcp]"»; остальные команды работают.
- E2. Дано: инструменты хранилища и журнала без рабочего пространства · тогда ошибка с подсказкой `mlreview init`.

## Вне рамок

- Транспорты SSE и Streamable HTTP, авторизация, ресурсы и промпты MCP.

## Тесты

- B1–B5, E1, E2 → `tests/test_mcp_server.py` (протокольный тест B4 пропускается без пакета `mcp`)
