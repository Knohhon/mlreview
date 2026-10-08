# Спецификация задачи (YAML)

Опора: FR-C01, FR-R01 (команда проверки); B02 (временная утечка ловится по `split_strategy`), B12 и FR-D02 (сегменты); сквозной пример и S04 из `docs/User-cases-MLReview.md`; X1–X6 (явная деградация); NFR-01, NFR-03.

Спецификация задачи — YAML-файл, описывающий задачу, а не модель. Модуль `mlreview.core.task_spec` читает файл, проверяет его по схеме и возвращает объект `TaskSpec`; при ошибках поднимает `TaskSpecError` со списком всех найденных проблем.

## Формат

```yaml
task:
  type: classification/binary      # обязательно: classification/binary | classification/multiclass |
                                   #   classification/multilabel | regression
  output: probabilities            # обязательно: для классификации labels | probabilities | scores,
                                   #   для регрессии values
  decision: "модератор проверяет топ-K"  # необязательно: как используются предсказания
  error_costs:                     # необязательно: уровни low | medium | high
    false_negative: high           #   классификация: false_positive, false_negative
    false_positive: low            #   регрессия: underestimation, overestimation
  segments: [language, source]     # необязательно: имена колонок метаданных, уникальные
data:                              # необязательно
  split_strategy: temporal         # random | stratified | temporal
  time_column: created_at          # обязательно при split_strategy: temporal
  version: ds-v3                   # необязательно
  known_issues: ["шумные метки в source=forum"]  # необязательно
```

## Ожидаемое поведение

- B1. Дано: YAML со всеми полями из примера выше · когда `load_task_spec(path)` · тогда возвращается `TaskSpec`, значения полей совпадают с файлом.
- B2. Дано: YAML только с `task.type: regression` и `task.output: values` · когда загрузка · тогда спецификация валидна, `data` равно `None`, `segments` — пустой список, `error_costs` — пустой словарь, `decision` — `None`.
- B3. Дано: каждый из четырёх типов задачи с допустимым для него `output` · когда загрузка · тогда спецификация валидна.
- B4. Дано: `parse_task_spec(text)` со строкой YAML · когда разбор · тогда результат тот же, что у `load_task_spec` для файла с этим текстом.

### Команда проверки

- C1. Дано: валидный файл · когда `mlreview spec validate <путь>` · тогда код возврата 0, в stdout — «спецификация валидна» с путём, типом задачи и выходом.
- C2. Дано: невалидный файл (любой случай E1–E9) · когда `mlreview spec validate <путь>` · тогда код возврата 1, в stderr — все проблемы, каждая с путём поля.
- C3. Дано: `mlreview spec` без подкоманды или `mlreview spec validate` без пути · тогда ошибка использования от argparse, код возврата 2.

## Деградация и ошибки

Все ошибки — `TaskSpecError`; сообщение по-русски, у каждой проблемы указан путь к полю (`task.output`) и источник (имя файла). Перечисляются все проблемы сразу, а не только первая.

- E1. Дано: нет `task.type` или `task.output` · тогда ошибка «обязательное поле» с путём поля.
- E2. Дано: `task.type: ranking` · тогда ошибка с путём `task.type` и перечнем допустимых типов.
- E3. Дано: `regression` с `output: probabilities` (или классификация с `output: values`) · тогда ошибка с путём `task.output` и перечнем допустимых для этого типа значений.
- E4. Дано: неизвестное поле (например, опечатка `segmets`) на любом уровне · тогда ошибка «неизвестное поле» с его путём, а не молчаливый пропуск.
- E5. Дано: `split_strategy: temporal` без `time_column` · тогда ошибка с путём `data.time_column`: без колонки времени нельзя проверить временную утечку (B02).
- E6. Дано: в `error_costs` ключ, не подходящий к типу задачи (`underestimation` для классификации), или уровень вне `low | medium | high` · тогда ошибка с путём ключа.
- E7. Дано: повторяющиеся или пустые имена в `segments` · тогда ошибка с путём `task.segments`.
- E8. Дано: несколько ошибок сразу (E1 + E4) · тогда все перечислены в одной `TaskSpecError`.
- E9. Дано: файла нет, YAML синтаксически неверен, файл пуст или верхний уровень не словарь · тогда `TaskSpecError` с понятной причиной (для синтаксиса — номер строки), а не трассировка из yaml/pydantic.

## Вне рамок

- Сверка `segments` и `time_column` с реальными колонками прогона — это контракт прогона (FR-C02) и анализаторы.
- Раздел `metrics:` (FR-M04, L1), `group_key` и сплит по группам (S21, B13 — предложение, не принято), площадки применения (S22).
- Числовая цена ошибок и матрица стоимостей для мультикласса.
- Версия формата и миграции (NFR-10, L2).
- Остальные подкоманды `spec` (создание шаблона, экспорт JSON Schema).

## Тесты

- B1–B4 → `tests/test_task_spec.py`
- E1–E9 → `tests/test_task_spec.py`
- C1–C3 → `tests/test_cli.py`
