# MLReview — полный справочник CLI

## 1. Установка и справка

Требуется **Python 3.10+**. Из исходников:

```bash
git clone https://github.com/Knohhon/mlreview.git
cd mlreview
pip install -e .                   # базовые команды, локальные файлы
pip install -e '.[mlflow,mcp]'      # если нужны MLflow и MCP

mlreview --help
mlreview --version
mlreview review --help
mlreview diagnose --help
mlreview diagnose model --help
mlreview findings --help
```

Вместо `mlreview` можно вызвать `python -m mlreview`.

**Обозначения:** `<run>` — путь к каталогу прогона (например, `runs/exp-002`) либо идентификатор прогона MLflow; `task.yaml` — спецификация задачи. Примеры предполагают запуск из каталога проекта. Для работы с хранилищем сначала запустите `mlreview init`.

## 2. Все команды — краткая карта

| Команда | Назначение |
|---|---|
| `mlreview init [directory]` | Создать `.mlreview/` и хранилище |
| `mlreview spec validate <path>` | Валидировать спецификацию задачи |
| `mlreview contract check <run> --spec <file>` | Проверить контракт прогона |
| `mlreview review <run> [<run> ...] --spec <file>` | Полное ревью и сравнение экспериментов |
| `mlreview diagnose data <run> --spec <file>` | Проверки качества и утечек данных |
| `mlreview diagnose segments <run> --spec <file>` | Анализ сегментов и регрессий качества |
| `mlreview diagnose model <run> --spec <file>` | Проверки плейбуков модели |
| `mlreview diagnose curves <run> --spec <file>` | Анализ кривых обучения |
| `mlreview diagnose process <run> --spec <file>` | Проверки корректности экспериментального процесса |
| `mlreview playbooks list` | Список встроенных модельных плейбуков |
| `mlreview metrics init --spec <file>` | Рекомендации по выбору метрик |
| `mlreview metrics test <run> --spec <file>` | Проверки пригодности метрик |
| `mlreview metrics export --spec <file>` | Экспорт метрик в Python |
| `mlreview metrics list` | Каталог доступных метрик |
| `mlreview findings list` | Список сохранённых находок |
| `mlreview findings show <id>` | Подробности находки |
| `mlreview findings accept <id>` | Принять находку |
| `mlreview findings fix <id>` | Пометить находку исправленной |
| `mlreview findings reopen <id>` | Повторно открыть находку |
| `mlreview findings dismiss <id>` | Отклонить находку |
| `mlreview dismiss <id>` | Короткий синоним отклонения находки |
| `mlreview journal add <kind> <text>` | Добавить запись в журнал |
| `mlreview journal list` | Просмотреть записи |
| `mlreview journal show <id>` | Детали записи |
| `mlreview journal search <query>` | Поиск по журналу и конфигам |
| `mlreview watch --spec <file> ...` | Следить за новыми прогонами |
| `mlreview ask <question>` | Вопрос по истории проекта через LLM |
| `mlreview mcp` | Запустить MCP-сервер по stdio |
| `mlreview --help`, `mlreview --version` | Справка и версия |

## 3. Инициализация и контракты

### `mlreview init [directory]`

Создаёт рабочее пространство `.mlreview/`, конфигурацию LLM и SQLite-хранилище находок и журнала. Если каталог не указан, используется текущий.

```bash
mlreview init
mlreview init /path/to/my-project
```

### `mlreview spec validate <path>`

Проверяет YAML-спецификацию задачи: тип, формат выхода и другие ограничения схемы. Сообщает тип задачи и тип выхода.

```bash
mlreview spec validate task.yaml
```

### `mlreview contract check <run> --spec <file>`

Загружает прогон и проверяет контракт: предсказания, примеры и сплиты, конфиг, кривые обучения, версию и хеш данных, сведения о модели и пробелы в артефактах.

```bash
mlreview contract check runs/exp-002 --spec task.yaml
mlreview contract check 0a1b2c3d --spec task.yaml --tracking-uri http://localhost:5000
```

Опции: `--spec` (обязателен), `--tracking-uri` (MLflow; иначе `MLFLOW_TRACKING_URI`).

## 4. Полное ревью

### `mlreview review <run> [<run> ...] --spec <file>`

Выполняет полный набор анализаторов; поддерживает сравнение с baseline, несколько прогонов с разными seed, сохранение находок, JSON-вывод и опциональные LLM-гипотезы.

```bash
# Один прогон
mlreview review runs/exp-002 --spec task.yaml

# Сравнение с baseline
mlreview review runs/exp-002 --spec task.yaml --baseline runs/exp-001

# Несколько seed в обеих группах
mlreview review runs/new-s1 runs/new-s2 runs/new-s3 \
  --spec task.yaml \
  --baseline runs/base-s1 runs/base-s2 runs/base-s3

# Для CI: JSON-отчёт без записи в хранилище
mlreview review runs/exp-002 --spec task.yaml \
  --format json --no-store > review.json

# Показывать не более 5 находок уровня high и выше
mlreview review runs/exp-002 --spec task.yaml --min-severity high --limit 5

# Попросить LLM сформулировать гипотезы
mlreview review runs/exp-002 --spec task.yaml \
  --hypotheses --focus 'просадка на region=north'
```

Опции:

| Параметр | Значение |
|---|---|
| `--spec FILE` | **Обязателен**; спецификация задачи |
| `--baseline RUN [RUN ...]` | Baseline-прогон и необязательные дополнительные seed |
| `--tracking-uri URI` | Адрес MLflow |
| `--format text\|json` | Вывод в текстовом формате (по умолчанию) или JSON |
| `--min-severity LEVEL` | Нижний порог важности (по умолчанию `info`) |
| `--limit N` | Максимум находок в отчёте; `0` означает все |
| `--no-store` | Не записывать находки в SQLite |
| `--hypotheses` | Дополнительные гипотезы и различающий эксперимент от LLM |
| `--focus TEXT` | Тема гипотез LLM |
| `--example ID` | ID примера для передачи LLM; опция повторяемая |
| `--yes` | Согласие на передачу выбранных примеров LLM |
| `--dry-run` | Показать запрос LLM, но не отправлять его |

**Важно:** значения `LEVEL` смотрите в `--help`; примеры `info`, `low`, `high` приведены в коде/документации. Все группы диагностики получают такие же флаги отчёта и LLM, но **не** поддерживают несколько seed как позиционные аргументы (`<run>` только один).

## 5. Выборочная диагностика `diagnose`

Каждая команда выполняет одну группу проверок и принимает `--spec`, `--tracking-uri`, `--format`, `--min-severity`, `--limit`, `--no-store`, `--hypotheses`, `--focus`, `--example`, `--yes`, `--dry-run`. Для всех групп **кроме `data`** разрешён один `--baseline <run>`.

### `mlreview diagnose data <run> --spec <file>`

Анализ данных: дубли между сплитами, временная утечка, подозрительно сильные признаки.

```bash
mlreview diagnose data runs/exp-002 --spec task.yaml
```

### `mlreview diagnose segments <run> --spec <file>`

Проверяет качество на срезах и деградацию отдельных сегментов относительно baseline.

```bash
mlreview diagnose segments runs/exp-002 --spec task.yaml \
  --baseline runs/exp-001
```

### `mlreview diagnose model <run> --spec <file>`

Модельно-специфичные проверки (плейбуки kNN, деревьев и распознавание других семейств в зависимости от сведений о модели).

```bash
mlreview diagnose model runs/tree-exp --spec task.yaml --format json
```

### `mlreview diagnose curves <run> --spec <file>`

Диагностирует переобучение, недообучение и нестабильность по логам кривых обучения.

```bash
mlreview diagnose curves runs/exp-002 --spec task.yaml
```

### `mlreview diagnose process <run> --spec <file>`

Проверяет дисциплину сравнения экспериментов: разные версии данных, много изменений за один прогон и повторную валидацию.

```bash
mlreview diagnose process runs/exp-002 --spec task.yaml \
  --baseline runs/exp-001
```

## 6. Плейбуки

### `mlreview playbooks list`

Выводит встроенные плейбуки, поддерживаемые семейства моделей, типы задач и проверки. В репозитории описаны `linear`, `neighbors`, `trees`; у `linear` может быть только распознавание семейства без собственных проверок.

```bash
mlreview playbooks list
```

## 7. Работа с метриками

### `mlreview metrics init --spec <file>`

Интерактивный подбор метрик на основе контекста задачи. Может загрузить ответы из YAML, использовать прогон для оценки доли положительного класса и записать секцию `metrics:` в спецификацию.

```bash
mlreview metrics init --spec task.yaml
mlreview metrics init --spec task.yaml --run runs/exp-002 --write
mlreview metrics init --spec task.yaml --answers metric_answers.yaml --write
```

Опции: `--spec` (**обязателен**), `--answers FILE`, `--run RUN`, `--tracking-uri URI`, `--write`.

### `mlreview metrics test <run> --spec <file>`

Тестирует выбранные метрики: устойчивость к тривиальным baseline, шум, минимальный обнаруживаемый эффект (MDE). Можно осознанно оставить провалившую тест метрику с объяснением.

```bash
mlreview metrics test runs/exp-002 --spec task.yaml
mlreview metrics test runs/exp-002 --spec task.yaml \
  --keep roc_auc --reason 'используем как вспомогательную'
```

Опции: `--spec` (**обязателен**), `--tracking-uri`, `--keep NAME`, `--reason TEXT`.

### `mlreview metrics export --spec <file>`

Экспортирует выбранные в спецификации метрики в исполняемый Python-модуль.

```bash
mlreview metrics export --spec task.yaml
mlreview metrics export --spec task.yaml -o generated_metrics.py
```

Опции: `--spec` (**обязателен**), `-o`/`--output FILE` (по умолчанию `metrics.py`).

### `mlreview metrics list`

Печатает каталог метрик, обоснования и распространённые ловушки. Есть фильтры по типу задачи и формату выхода.

```bash
mlreview metrics list
mlreview metrics list --task classification/binary --output probabilities
mlreview metrics list --task regression --output values
```

Опции: `--task TYPE`, `--output OUTPUT`. Разрешённые значения определяются схемой задачи и показываются в `--help`.

## 8. Хранилище находок

### `mlreview findings list`

Показывает находки из SQLite с учётом статусов, важности и типа проверки.

```bash
mlreview findings list
mlreview findings list --status new
mlreview findings list --min-severity high --check data.split_duplicates
```

Опции: `--status STATUS`, `--min-severity LEVEL` (по умолчанию `low`), `--check CHECK_ID`.

### `mlreview findings show <id>`

Детали находки: статус, важность, описание, факты, примеры, предложение, история появления.

```bash
mlreview findings show F-0003
```

### `mlreview findings accept <id>`

Подтвердить, что находка имеет смысл и принята в работу.

```bash
mlreview findings accept F-0003 --reason 'будем исправлять'
```

### `mlreview findings fix <id>`

Отметить находку исправленной.

```bash
mlreview findings fix F-0003 --reason 'исправили разделение train/test'
```

### `mlreview findings reopen <id>`

Вернуть находку в статус `new`.

```bash
mlreview findings reopen F-0003 --reason 'ошибка повторилась'
```

### `mlreview findings dismiss <id>`

Отклонить находку как неактуальную. Для критичных находок объяснение обязательно.

```bash
mlreview findings dismiss F-0003 --reason 'сегмент нерелевантен'
```

### `mlreview dismiss <id>`

Отдельная короткая команда с тем же действием отклонения, что и `findings dismiss`.

```bash
mlreview dismiss F-0003 --reason 'известное ограничение датасета'
```

Все операции изменения статуса принимают `--reason TEXT`; для отклонения critical-находок он обязателен.

## 9. Журнал экспериментов и решений

### `mlreview journal add <kind> <text>`

Создать гипотезу, решение или заметку с привязкой к прогонам и находкам.

```bash
mlreview journal add hypothesis 'больше данных по north повысит recall' --run exp-002
mlreview journal add decision 'принять exp-002' --run exp-002 --outcome accept
mlreview journal add note 'проверить разметку' --finding F-0003
mlreview journal add decision 'не менять порог' \
  --run exp-001 --run exp-002 --hypothesis H-0001
```

Опции: `--run RUN` и `--finding ID` (повторяемые), `--hypothesis ID`, `--outcome OUTCOME`, `--data-version VERSION`. Допустимые значения `kind` и `outcome` выводятся через `--help` (в примерах используются `hypothesis`, `decision`, `note` и `accept`).

### `mlreview journal list`

Показать все записи или отфильтровать по типу.

```bash
mlreview journal list
mlreview journal list --kind hypothesis
```

### `mlreview journal show <id>`

Показать полный текст и связи записи.

```bash
mlreview journal show H-0001
```

### `mlreview journal search <query>`

Поиск по журналу и конфигурациям прогонов. Опциональная спецификация задаёт актуальную версию данных.

```bash
mlreview journal search 'region north'
mlreview journal search 'focal loss' --spec task.yaml
```

## 10. Наблюдение за экспериментами

### `mlreview watch --spec <file> (--runs-dir <dir> | --experiment <name>)`

Проактивно отслеживает новые завершённые прогоны. Источники **взаимоисключающие**: каталог с прогонами либо эксперимент MLflow.

```bash
# Мониторинг локальной директории
mlreview watch --spec task.yaml --runs-dir runs

# Однократная проверка, полезно для cron/CI
mlreview watch --spec task.yaml --runs-dir runs --once

# MLflow с интервалом 30 секунд
mlreview watch --spec task.yaml --experiment my-experiment \
  --tracking-uri http://localhost:5000 --interval 30

# Меньше шума в отчётах
mlreview watch --spec task.yaml --runs-dir runs \
  --min-severity high --limit 3
```

Опции: `--spec` (**обязателен**), один из `--runs-dir DIR`/`--experiment NAME` (**обязателен**), `--tracking-uri URI`, `--interval SECONDS` (по умолчанию 60), `--once`, `--limit N` (по умолчанию 5), `--min-severity LEVEL` (по умолчанию `low`). Требуется инициализированный `.mlreview/`.

## 11. Вопросы к LLM

### `mlreview ask <question>`

Отвечает на вопросы по сохранённым находкам и истории проекта с опорой на факты; при наличии указанных прогонов может использовать контекст их ревью. Требует настроенного LLM-провайдера; без него сообщает об отсутствии LLM и предлагает детерминированный поиск.

```bash
mlreview ask 'Пробовали ли мы больше данных для north?'
mlreview ask 'Что изменилось относительно baseline?' \
  --spec task.yaml --run runs/exp-002 --baseline runs/exp-001

# Посмотреть, что отправилось бы в модель
mlreview ask 'Почему ухудшился recall?' --run runs/exp-002 \
  --spec task.yaml --dry-run

# Передать строго выбранный пример с явным согласием
mlreview ask 'Что происходит с этим примером?' \
  --run runs/exp-002 --spec task.yaml --example ex-123 --yes
```

Опции: `--spec FILE`, `--run RUN`, `--baseline RUN`, `--tracking-uri URI`, `--example ID` (повторяемый), `--yes`, `--dry-run`. **Внимание:** `--hypotheses` и `--focus` относятся к `review`/`diagnose`, но не к `ask`.

LLM настраивается в `.mlreview/config.yaml`; без настройки он по умолчанию выключен. Передача отдельных примеров требует согласия, `--dry-run` не отправляет запрос.

## 12. MCP-сервер

### `mlreview mcp`

Запускает MCP-сервер по `stdio` для IDE-агентов. Требуется extra `mcp`.

```bash
pip install -e '.[mcp]'
mlreview mcp

# Регистрация в Claude Code из каталога проекта
claude mcp add mlreview -- mlreview mcp
```

После регистрации агент может использовать инструменты MLReview (например, ревью, тесты метрик и работу с журналом). Сервер рассчитан на MCP-клиент; обычный терминал не является интерактивным чат-интерфейсом сервера.

## 13. Типовой сценарий от начала до конца

```bash
# 1. Инициализация
mlreview init

# 2. Проверка task.yaml и артефактов прогона
mlreview spec validate task.yaml
mlreview contract check runs/exp-002 --spec task.yaml

# 3. Посмотреть встроенные модельные проверки
mlreview playbooks list

# 4. Посмотреть каталог метрик и проверить выбранные
mlreview metrics list --task classification/binary
mlreview metrics test runs/exp-002 --spec task.yaml

# 5. Ревью относительно baseline
mlreview review runs/exp-002 --spec task.yaml --baseline runs/exp-001

# 6. Разобрать отдельные проблемы
mlreview diagnose segments runs/exp-002 --spec task.yaml --baseline runs/exp-001
mlreview diagnose model runs/exp-002 --spec task.yaml

# 7. Зафиксировать решение
mlreview findings list
mlreview findings show F-0001
mlreview journal add hypothesis 'просадка обусловлена shift данных' --run exp-002

# 8. Периодически сканировать новые прогоны
mlreview watch --spec task.yaml --runs-dir runs --once
```

## 14. Команды разработчика (не подкоманды MLReview)

Это команды для сопровождения **самого репозитория**, описанные в `CLAUDE.md`:

```bash
pip install -r requirements.txt
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
pytest tests/test_cli.py::test_version
python -m benchmark
python -m benchmark --perf
```

`python -m benchmark` проверяет приёмочный набор дефектов, а `--perf` — производительность на крупном прогоне. Эти команды не являются подкомандами `mlreview`.

## 15. Источники и границы проверки

Этот справочник собран по объявленным аргументам и обработчикам CLI в следующих файлах ветки `main` (состояние на дату проверки):

- [`cli/__init__.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/__init__.py) — все команды верхнего уровня.
- [`cli/review.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/review.py) — `spec`, `contract`, `review`, `diagnose`.
- [`cli/metrics.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/metrics.py) — все действия `metrics`.
- [`cli/store.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/store.py) — `init`, `findings`, `dismiss`.
- [`cli/journal.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/journal.py) — журнал.
- [`cli/llm.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/llm.py) — `ask` и общие LLM-флаги.
- [`cli/watch.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/watch.py) — `watch`, `mcp`.
- [`cli/playbooks.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/playbooks.py) — плейбуки.
- [`cli/common.py`](https://github.com/Knohhon/mlreview/blob/main/src/mlreview/cli/common.py) — общие флаги.
- [`docs/quickstart.md`](https://github.com/Knohhon/mlreview/blob/main/docs/quickstart.md) и [`CLAUDE.md`](https://github.com/Knohhon/mlreview/blob/main/CLAUDE.md).

Фактический вывод `--help` и результаты исполнения команд не проверялись, поэтому примеры следует рассматривать как образцы синтаксиса; наличие конкретных входных прогонов и ID в вашем проекте не предполагается.
