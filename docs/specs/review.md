# Находки, ревью прогона и команды CLI

Опора: FR-R01 (реактивный режим в CLI: review, metrics, diagnose, ask — L0 · Must), FR-D01–D03 (анализаторы ревью), FR-C03 (прогоны читаются из MLflow), S02 (ревью после прогона), S04 (`diagnose data`), S15 (машиночитаемый вывод), X1 (неполный контракт → перечислить, что не проверено), X2 (без LLM — детерминированный отчёт), NFR-03, NFR-05.

## Находка и отчёт

`mlreview.core.findings`:

- `Finding` — `check` (id проверки, например `data.split_duplicates`), `severity` (`critical` > `high` > `medium` > `low` > `info`), `message` (одна строка по-русски с числами), `requirement` (опора, например `FR-D03, B01`), `evidence` (до 10 id примеров), `facts` (словарь чисел и строк, на которых построено сообщение; дробные числа округлены до 6 знаков, а числа в сообщении выводятся из этих фактов; основа будущей сверки чисел LLM, FR-L02), `suggestion` (предложение изменения или эксперимента; инструмент его не применяет).
- `Skipped` — `check` и `reason`: что не проверено и почему (X1).
- `Report` — `run_id`, `baseline_id`, `data_version`, `baseline_data_version`, `compared_examples` (число общих примеров при сравнении), `findings`, `skipped`; `to_dict()` — словарь для JSON.

## Ревью

`mlreview.analyzers.review.review(spec, contract, baseline=None, groups=…, *, candidate_seeds=(), baseline_seeds=(), prior_runs=None)` запускает анализаторы групп (интерфейс анализаторов и сиды — `docs/specs/analyzers.md`):

| Группа | Проверки | Без базового прогона | С базовым прогоном |
| --- | --- | --- | --- |
| `data` | `data.split_duplicates`, `data.temporal_leak`, `data.strong_feature` (`docs/specs/data-checks.md`) | по кандидату | по кандидату |
| `segments` | `slices.segment_worse`, `slices.segment_regression` (`docs/specs/slices.md`) | только `segment_worse` | обе |
| `metrics` | `metrics.value`, `comparison.metric_delta` (`docs/specs/run-comparison.md`); `metrics.unsuitable` (`docs/specs/metrics.md`); `seeds.*` (`docs/specs/seeds.md`); `calibration.*`, `comparison.disagreement` (`docs/specs/calibration.md`) | `metrics.value`, `seeds.spread`, `calibration.ece` | `comparison.metric_delta`, `seeds.*`, `calibration.ece_delta`, `comparison.disagreement` |
| `curves` | `curves.*` (`docs/specs/learning-curves.md`) | по кандидату | по кандидату |
| `process` | `process.*` (`docs/specs/process-checks.md`) | `seed_config_mismatch`, `validation_reuse` | все |

## Ожидаемое поведение

- B1. Дано: находки разной важности · когда собирается отчёт · тогда находки отсортированы по severity (`critical` первыми), затем по `check`, затем по сообщению; `skipped` — в порядке групп `data`, `segments`, `metrics`, `curves`, `process`.
- B2. Дано: прогон с дублями train/test, сегментом хуже остальных и без базового прогона · когда `review(spec, contract)` · тогда в отчёте находки групп `data`, `segments`, `metrics`; `baseline_id` и `compared_examples` — `None`.
- B3. Дано: `groups=("data",)` · тогда в отчёте только находки и пропуски группы `data`.
- B4. Дано: базовый прогон · тогда `baseline_id`, `baseline_data_version` и `compared_examples` заполнены; есть находки `comparison.metric_delta`.
- B5. Дано: одинаковые входы · тогда `to_dict()` двух запусков равны (NFR-05).

### Команды

Прогоны читаются из MLflow адаптером (`docs/specs/mlflow-adapter.md`). Общие аргументы: `<run_id>`, `--spec <путь>` (обязателен), `--tracking-uri <адрес>`, `--format text|json` (по умолчанию `text`).

- C1. `mlreview review <run_id> [<сиды>…] --spec … [--baseline <run_id> [<сиды>…]]` · тогда код 0; в stdout — отчёт всех групп.
- C2. `mlreview diagnose data <run_id> --spec …` — только группа `data`; `mlreview diagnose segments|curves|process <run_id> --spec … [--baseline <run_id>]` — только эта группа. Код 0.
- C3. Текстовый отчёт:

  ```text
  ревью прогона <run_id>                      (diagnose: «диагностика данных|сегментов прогона <run_id>»)
    базовый прогон: <baseline_id>, общих примеров: <n>     (только с базовым)
    версия данных: <версия> (<хеш>) | <база> (<хеш>) → <кандидат> (<хеш>) | нет (<хеш>)
    хранилище находок: <путь> | не подключено — mlreview init | не используется (--no-store)
  находки: <k>                                 (или «находок нет»)
    1. [critical] F-0001 data.split_duplicates — <сообщение> (<опора>)   (id — с хранилищем)
       статус: новая | повторяется: <n>-е вхождение | принята             (с хранилищем)
       пометка: отклонялась ранее: …           (если находка вернулась)
       примеры: <id>, <id>, …                  (если есть доказательства)
       предложение: <предложение>              (если есть)
  скрыто отклонённых: <n>                      (если есть)
  скрыто: <n> (ниже порога или сверх лимита)   (при --min-severity / --limit)
  не проверено: <m>                            (если есть пропуски)
    - <check>: <причина>
  LLM: не используется — отчёт детерминированный
  ```

  Хеш данных — FR-C04 (`docs/specs/data-version.md`); хранилище, статусы, `--min-severity`, `--limit`, `--no-store` — `docs/specs/findings-store.md`.

- C4. `--format json` · тогда в stdout — `Report.to_dict()` как JSON (UTF-8, кириллица без экранирования) с полем `"llm": null`; находки — после фильтров, с полями хранилища; поля `hidden_dismissed`, `hidden_by_filter`, `store` (`null` без хранилища).
- C5. Дано: невалидная спецификация, ошибка адаптера или невалидный контракт кандидата или базового прогона · тогда код 1, сообщение в stderr.
- C6. Дано: нет `--spec`, нет id прогона, `diagnose` без группы · тогда ошибка использования argparse, код 2.
- C7. `mlreview ask <вопрос>` без LLM · тогда код 1 и в stderr «mlreview ask: требует LLM — в настройках llm.provider: none (.mlreview/config.yaml); детерминированно без LLM: mlreview journal search <запрос>, mlreview review <run_id> --spec <путь>» (X2: явный отказ, а не падение); с LLM — `docs/specs/llm.md`.
- C8. `mlreview metrics` без действия · тогда ошибка использования argparse, код 2; действия `init`, `test`, `export`, `list` — `docs/specs/metrics.md`.

## Деградация и ошибки

- E1. Дано: контракт без `examples` · тогда отчёт строится; проверки групп `data` и `segments` в `skipped` с причинами первыми, затем пропуски остальных групп; находки `metrics` на месте (X1).
- E2. Дано: команда без LLM · тогда последняя строка текстового отчёта — «LLM: не используется — отчёт детерминированный» (X2, NFR-03).

## Вне рамок

- `--fail-on` для CI (S15 — предложение).

## Тесты

- B1–B5, E1 → `tests/test_review.py`
- C1–C8, E2 → `tests/test_review_cli.py`
