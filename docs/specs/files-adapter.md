# Адаптер локальных файлов

Опора: FR-C03 (адаптер локальных файлов parquet/csv, L1 · Must), FR-C02 (адаптер отдаёт контракт прогона; формат — `docs/specs/run-contract.md`), FR-R01 (CLI), FR-R04 (каталог прогонов для проактивного режима — `docs/specs/proactive.md`), S27 (адаптер сообщает, каких полей не хватает), X1, NFR-01 (адаптер не импортирует фреймворки и трекеры), NFR-03.

Адаптер `mlreview.adapters.files_adapter` читает прогон из каталога на диске и собирает контракт через `build_run_contract`. Так работают без трекера: скрипт обучения сохраняет таблицы рядом с моделью, а `mlreview` читает их.

```python
from mlreview.adapters.files_adapter import load_files_contract, save_run_files

# в конце скрипта обучения:
save_run_files(
    "runs/exp-043",
    predictions,
    examples=examples,
    config={"lr": 0.01},
    curves=curves,
    data_version="ds-v3",
)

# при анализе:
contract = load_files_contract(spec, "runs/exp-043")
```

## Соглашение: каталог прогона

| Поле контракта | Файл в каталоге прогона |
| --- | --- |
| `predictions` | `predictions.parquet` или `predictions.csv` (обязателен) |
| `examples` | `examples.parquet` или `examples.csv` |
| `curves` | `curves.parquet` или `curves.csv` (колонки `name`, `step`, `value`) |
| `config` | `config.yaml` или `config.json` — словарь параметров |
| `run_id`, `data_version`, `model` | `run.yaml` или `run.json` — словарь с ключами `run_id`, `data_version` и `model` (`class`/`family`/`library`, `docs/specs/model-info.md`); без файла или ключа `run_id` — имя каталога, `data_version` и `model` — нет |
| `feature_importance` | `feature_importance.parquet` или `feature_importance.csv` (колонки `feature`, `importance`, необязательная `kind`) |

`run.yaml` пишется последним и служит признаком «прогон завершён» для проактивного режима. В CSV колонка `example_id` читается как строка, как в адаптере MLflow.

## Выбор адаптера в CLI

Ссылка на прогон в командах (`review`, `diagnose`, `contract check`, `metrics test`) — существующий каталог или id прогона MLflow: если по ссылке есть каталог, работает адаптер файлов, иначе — адаптер MLflow. `mlreview.adapters.base.load_run(spec, ref, tracking_uri=…)` делает этот выбор для CLI, MCP и Python API.

## Ожидаемое поведение

- B1. Дано: каталог со всеми файлами в parquet и `run.yaml` · когда `load_files_contract(spec, путь)` · тогда контракт с `run_id` и `data_version` из `run.yaml`, `config` из `config.yaml`, таблицами, равными сохранённым; `gaps` пуст.
- B2. Дано: те же таблицы в CSV и `config.json`, `run.json` · тогда контракт тот же; `example_id` — строки (`007` остаётся `007`).
- B3. Дано: в каталоге только `predictions.csv` · тогда `run_id` — имя каталога, `gaps` перечисляет `examples`, `curves`, `config`, `data_version`, `model`.
- B4. Дано: `save_run_files(путь, predictions, examples=…, curves=…, config=…, data_version=…, run_id=…, model=…, feature_importance=…)` · тогда каталог создан, таблицы — в parquet без индекса, `config.yaml` и `run.yaml` записаны; `load_files_contract` возвращает равный контракт.
- B5. Дано: ссылка — существующий каталог · когда `load_run(spec, ссылка)` · тогда работает адаптер файлов; иначе — адаптер MLflow с `tracking_uri`.
- B6. Дано: каталог прогона · когда `mlreview contract check <каталог> --spec …` · тогда код 0 и сводка контракта, как для MLflow.

## Деградация и ошибки

Ошибки чтения — `FilesAdapterError` (подкласс общего `AdapterError`, как и `MlflowAdapterError`) с путём каталога или файла.

- E1. Дано: каталога нет · тогда ошибка «каталог прогона не найден: <путь>».
- E2. Дано: нет файла предсказаний · тогда ошибка с обоими ожидаемыми именами `predictions.parquet` и `predictions.csv`.
- E3. Дано: таблица и в parquet, и в CSV (или `config.yaml` и `config.json`) · тогда ошибка неоднозначности с обоими путями.
- E4. Дано: файл не читается (повреждённый parquet, некорректный YAML) · тогда ошибка с путём файла и причиной.
- E5. Дано: `config` или `run` — не словарь · тогда ошибка «ожидается словарь» с путём файла.
- E6. Дано: таблицы не по формату контракта · тогда `RunContractError` из ядра без изменений.

## Вне рамок

- Каталоги с несколькими прогонами в одном файле, форматы кроме parquet/csv/yaml/json, удалённые хранилища (S3).

## Тесты

- B1–B5, E1–E6 → `tests/test_files_adapter.py`; модель и важности — `docs/specs/model-info.md` (F1–F2, C1)
- B6 → `tests/test_files_adapter.py::test_contract_check_command_reads_directory`
