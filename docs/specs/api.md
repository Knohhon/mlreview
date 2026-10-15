# Python API и работа из ноутбука

Опора: FR-R03 (Python API и работа из ноутбука — L1 · Should), FR-R04 (вызов из скрипта обучения — `docs/specs/proactive.md`), FR-R01, S02, S15 (API для пайплайнов), NFR-03.

Модуль `mlreview.api` — те же операции, что CLI, но с объектами вместо текста. CLI, MCP-сервер и проактивный режим построены на нём.

```python
from mlreview import api

spec = api.load_spec("task.yaml")
result = api.review_runs(spec, "runs/exp-043", baseline="runs/exp-041")
result.report  # Report: находки, пропуски, версии и хеши данных
result.to_dict()  # то же, что `mlreview review … --format json`
print(result.to_text())
result  # в ноутбуке — таблица находок (Markdown)

api.on_run_end("runs/exp-043", spec="task.yaml")  # в конце скрипта обучения
```

## Ожидаемое поведение

- B1. Дано: `review_runs(spec, run, baseline=…, seeds=…, baseline_seeds=…, groups=…, tracking_uri=…, store=True)` · тогда `ReviewResult` с `report`, `record` (запись в хранилище, если есть рабочее пространство и `store=True`) и `store` (путь базы или `None`); прогоны — каталоги или id MLflow, сиды — списки.
- B2. Дано: результат · тогда `to_dict(min_severity=…, limit=…)` и `to_text(...)` совпадают с `--format json` и текстом CLI на тех же входах.
- B3. Дано: результат в ноутбуке · тогда `_repr_markdown_()` — заголовок с прогоном и версией данных, таблица «важность · id · проверка · сообщение» и число непроверенного.
- B4. Дано: нет рабочего пространства · тогда ревью работает, `record` и `store` — `None`.
- B5. Дано: `on_run_end(run, spec=…)` · тогда ревью против референса (первый прогон последнего принятого решения журнала, `docs/specs/journal.md`; без него — без базового прогона), запись в хранилище, отчёт с не больше чем 5 находками по важности (не ниже `low`) печатается и сохраняется в `.mlreview/reports/<run_id>.txt`; возвращается `ReviewResult`.

## Деградация и ошибки

- E1. Дано: невалидная спецификация, контракт или ошибка адаптера в `review_runs` · тогда исключения библиотеки (`TaskSpecError`, `RunContractError`, `AdapterError`), а не `SystemExit`.
- E2. Дано: ошибка в `on_run_end` · тогда скрипт обучения не падает: в stderr «mlreview: ревью не выполнено: <причина>», возвращается `None`.

## Вне рамок

- Асинхронный API, виджеты ноутбука, графики.

## Тесты

- B1–B5, E1, E2 → `tests/test_api.py`
