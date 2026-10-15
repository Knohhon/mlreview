# Находки, ревью прогона и команды CLI

Опора: FR-R01 (реактивный режим в CLI: review, metrics, diagnose, ask — L0 · Must), FR-D01–D03 (анализаторы ревью), FR-C03 (прогоны читаются из MLflow), S02 (ревью после прогона), S04 (`diagnose data`), S15 (машиночитаемый вывод), X1 (неполный контракт → перечислить, что не проверено), X2 (без LLM — детерминированный отчёт), NFR-03, NFR-05.

## Находка и отчёт

`mlreview.core.findings`:

- `Finding` — `check` (id проверки, например `data.split_duplicates`), `severity` (`critical` > `high` > `medium` > `low` > `info`), `message` (одна строка по-русски с числами), `requirement` (опора, например `FR-D03, B01`), `evidence` (до 10 id примеров), `facts` (словарь чисел и строк, на которых построено сообщение; дробные числа округлены до 6 знаков, а числа в сообщении выводятся из этих фактов; основа будущей сверки чисел LLM, FR-L02), `suggestion` (предложение изменения или эксперимента; инструмент его не применяет).
- `Skipped` — `check` и `reason`: что не проверено и почему (X1).
- `Report` — `run_id`, `baseline_id`, `data_version`, `baseline_data_version`, `compared_examples` (число общих примеров при сравнении), `findings`, `skipped`; `to_dict()` — словарь для JSON.

## Ревью

`mlreview.analyzers.review.review(spec, contract, baseline=None, groups=…)` запускает группы проверок:

| Группа | Проверки | Без базового прогона | С базовым прогоном |
| --- | --- | --- | --- |
| `data` | `data.split_duplicates`, `data.temporal_leak`, `data.strong_feature` (`docs/specs/data-checks.md`) | по кандидату | по кандидату |
| `segments` | `slices.segment_worse`, `slices.segment_regression` (`docs/specs/slices.md`) | только `segment_worse` | обе |
| `metrics` | `metrics.value`, `comparison.metric_delta` (`docs/specs/run-comparison.md`) | `metrics.value` | `comparison.metric_delta` |

## Ожидаемое поведение

- B1. Дано: находки разной важности · когда собирается отчёт · тогда находки отсортированы по severity (`critical` первыми), затем по `check`, затем по сообщению; `skipped` — в порядке групп `data`, `segments`, `metrics`.
- B2. Дано: прогон с дублями train/test, сегментом хуже остальных и без базового прогона · когда `review(spec, contract)` · тогда в отчёте находки групп `data`, `segments`, `metrics`; `baseline_id` и `compared_examples` — `None`.
- B3. Дано: `groups=("data",)` · тогда в отчёте только находки и пропуски группы `data`.
- B4. Дано: базовый прогон · тогда `baseline_id`, `baseline_data_version` и `compared_examples` заполнены; есть находки `comparison.metric_delta`.
- B5. Дано: одинаковые входы · тогда `to_dict()` двух запусков равны (NFR-05).

### Команды

Прогоны читаются из MLflow адаптером (`docs/specs/mlflow-adapter.md`). Общие аргументы: `<run_id>`, `--spec <путь>` (обязателен), `--tracking-uri <адрес>`, `--format text|json` (по умолчанию `text`).

- C1. `mlreview review <run_id> --spec … [--baseline <run_id>]` · тогда код 0; в stdout — отчёт всех групп.
- C2. `mlreview diagnose data <run_id> --spec …` — только группа `data`; `mlreview diagnose segments <run_id> --spec … [--baseline <run_id>]` — только группа `segments`. Код 0.
- C3. Текстовый отчёт:

  ```text
  ревью прогона <run_id>                      (diagnose: «диагностика данных|сегментов прогона <run_id>»)
    базовый прогон: <baseline_id>, общих примеров: <n>     (только с базовым)
    версия данных: <версия> | <база> → <кандидат> | нет
  находки: <k>                                 (или «находок нет»)
    1. [critical] data.split_duplicates — <сообщение> (<опора>)
       примеры: <id>, <id>, …                  (если есть доказательства)
       предложение: <предложение>              (если есть)
  не проверено: <m>                            (если есть пропуски)
    - <check>: <причина>
  LLM: не используется — отчёт детерминированный
  ```

- C4. `--format json` · тогда в stdout — `Report.to_dict()` как JSON (UTF-8, кириллица без экранирования) с полем `"llm": null`.
- C5. Дано: невалидная спецификация, ошибка адаптера или невалидный контракт кандидата или базового прогона · тогда код 1, сообщение в stderr.
- C6. Дано: нет `--spec`, нет id прогона, `diagnose` без группы · тогда ошибка использования argparse, код 2.
- C7. `mlreview ask <вопрос>` · тогда код 1 и в stderr «mlreview ask: требует LLM — LLM-слой не подключён (появится на L1, FR-L01); детерминированные находки без LLM: mlreview review <run_id> --spec <путь>» (X2: явный отказ, а не падение).
- C8. `mlreview metrics` · тогда код 1 и «команда пока не реализована: проектирование метрик (FR-M01–M06) — уровень L1».

## Деградация и ошибки

- E1. Дано: контракт без `examples` · тогда отчёт строится; проверки групп `data` и `segments` в `skipped` с причинами; находки `metrics` на месте (X1).
- E2. Дано: команда без LLM · тогда последняя строка текстового отчёта — «LLM: не используется — отчёт детерминированный» (X2, NFR-03).

## Вне рамок

- Хранилище находок, статусы, `dismiss` и дедупликация (FR-C05, FR-R05 — L1), лимит в 5 находок для проактивного ревью (S02 — L1).
- Выбор базового прогона из журнала (S02 — L1), MCP (FR-R02 — L1), `--fail-on` для CI (S15 — предложение).
- Ревью прогонов из локальных файлов (FR-C03, L1) — пока только Python API `review()` с контрактом из `build_run_contract`.

## Тесты

- B1–B5, E1 → `tests/test_review.py`
- C1–C8, E2 → `tests/test_review_cli.py`
