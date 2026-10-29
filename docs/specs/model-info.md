# Сведения о модели в контракте

Опора: FR-C07 (сведения о модели: класс, библиотека, семейство, важности признаков; автоопределение по данным трекера), FR-C02 (контракт прогона — `docs/specs/run-contract.md`), FR-C03 (адаптеры MLflow и файлов), FR-D11 (потребитель: проверки по семейству модели через плейбуки — `docs/specs/playbooks.md`), X1 (неполный контракт), NFR-01 (ядро не импортирует фреймворки: модель описывается строками), NFR-04 (в контракте нет самой модели — только её описание).

Контракт получает два необязательных поля: `model` — что за модель (класс, библиотека, семейство) и `feature_importance` — важности признаков. Ядро не загружает и не исполняет модель: оно знает только её описание, которое передал скрипт обучения или прочитал адаптер. Семейство по имени класса определяют плейбуки (`docs/specs/playbooks.md`); контракт хранит только явно указанное семейство.

```python
contract = build_run_contract(
    spec,
    run_id="exp-044",
    predictions=predictions,
    examples=examples,
    model={"class": "sklearn.tree.DecisionTreeClassifier", "family": "tree"},
    feature_importance=pd.DataFrame({"feature": ["age", "income"], "importance": [0.7, 0.3]}),
)
contract.model.class_name  # "sklearn.tree.DecisionTreeClassifier"
contract.model.short_name  # "DecisionTreeClassifier"
contract.model.library  # "sklearn" — первая часть пути класса
```

## Формат

**`model`** — `ModelInfo` или словарь с ключами:

| Ключ | Поле `ModelInfo` | Содержимое |
| --- | --- | --- |
| `class` | `class_name` | имя класса модели, лучше с модулем: `sklearn.neighbors.KNeighborsClassifier`, `lightgbm.LGBMRegressor` |
| `family` | `family` | семейство: `linear`, `neighbors`, `tree`, `ensemble/bagging`, `ensemble/boosting`, `other` |
| `library` | `library` | библиотека; без ключа — первая часть пути класса (`sklearn`), у короткого имени — нет |

Нужен хотя бы `class` или `family`. `ModelInfo.source` — откуда взяты сведения: `explicit` (передано вызывающим кодом или в `run.yaml`), `tag` (теги `mlreview.model.*` MLflow), `autolog` (тег `estimator_class` автологирования sklearn), `pipeline` (последний шаг `Pipeline` из параметра `steps`).

**`feature_importance`** — таблица важностей признаков:

| Колонка | Обязательна | Содержимое |
| --- | --- | --- |
| `feature` | да | имя признака — колонка `examples` |
| `importance` | да | конечное число; отрицательное — только при `kind: permutation` |
| `kind` | нет | способ подсчёта: `impurity`, `gain`, `split`, `permutation` или другой непустой |

Пара (`feature`, `kind`) уникальна. На хеш данных (FR-C04) ни `model`, ни `feature_importance` не влияют.

## Ожидаемое поведение

- B1. Дано: `model={"class": "sklearn.tree.DecisionTreeClassifier", "family": "tree"}` · когда сборка контракта · тогда `contract.model` — `ModelInfo` с этим классом, семейством `tree`, библиотекой `sklearn`, `short_name` `DecisionTreeClassifier`, `source` `explicit`.
- B2. Дано: `model={"class": "KNeighborsClassifier"}` (без модуля) · тогда `library` — `None`, `family` — `None`; `model={"family": "neighbors"}` · тогда `class_name` — `None`. Оба контракта валидны.
- B3. Дано: `model=ModelInfo(...)` · тогда хранится как есть; `ModelInfo.to_dict()` возвращает `{"class", "family", "library", "source"}`.
- B4. Дано: `feature_importance` с колонками `feature`, `importance`, `kind` · тогда хранится копией с индексом 0…n−1; изменение исходной таблицы на контракт не влияет; `data_hash` тот же, что без неё.
- B5. Дано: контракт без `model` · тогда пробел `model` с назначением «проверки по семейству модели (FR-D11)» — последним в порядке пробелов (G7 к `docs/specs/run-contract.md`). Отсутствие `feature_importance` пробелом не считается: проверка, которой нужны важности, сама сообщает пропуск.

### Адаптер файлов

- F1. Дано: в `run.yaml` ключ `model: {class: …, family: …}` и файл `feature_importance.parquet` или `.csv` · когда `load_files_contract` · тогда `contract.model` и `contract.feature_importance` прочитаны; `source` — `explicit`.
- F2. Дано: `save_run_files(путь, predictions, model=…, feature_importance=…)` · тогда `model` записан в `run.yaml` словарём `class`/`family`/`library` (только заданные), важности — в `feature_importance.parquet`; `load_files_contract` возвращает равные поля.

### Адаптер MLflow

- M1. Дано: теги `mlreview.model.class`, `mlreview.model.family`, `mlreview.model.library` · тогда `model` из них, `source` — `tag`.
- M2. Дано: тег `estimator_class` (автологирование sklearn), например `sklearn.tree._classes.DecisionTreeRegressor` · тогда `class_name` — значение тега, `source` — `autolog`.
- M3. Дано: `estimator_class` — `sklearn.pipeline.Pipeline` и параметр `steps` вида `[('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=3))]` · тогда `class_name` — класс последнего шага (`KNeighborsClassifier`), `library` — `None`, `source` — `pipeline`. Если `steps` нет или не разбирается — `class_name` — `sklearn.pipeline.Pipeline`, `source` — `autolog`.
- M4. Дано: теги `mlreview.model.*` и `estimator_class` вместе · тогда поля из тегов `mlreview.model.*` важнее, недостающие берутся из автологирования; `source` — `tag`.
- M5. Дано: в каталоге артефактов `mlreview` есть `feature_importance.parquet` или `.csv` · тогда он становится `feature_importance`; в обоих форматах сразу — ошибка неоднозначности, как E4 адаптера.
- M6. Дано: активный прогон · когда `log_contract_artifacts(predictions, model=…, feature_importance=…)` · тогда `model` записан тегами `mlreview.model.*`, важности — артефактом `feature_importance.parquet` в каталоге `mlreview`; `load_mlflow_contract` возвращает равные поля.

### Команда проверки контракта

- C1. Дано: контракт с `model` · когда `mlreview contract check` · тогда в сводке строка «модель: <класс> (семейство: <семейство>, источник: <source>)»; семейство — явное, или определённое плейбуками по классу с пометкой «(по классу)», или «не определено» и «важностей признаков: <n>»; без `model` — «модель: не указана». В MCP `contract_check` — ключи `model` (словарь `ModelInfo.to_dict()`, где `family` — определённое семейство, или `null`) и `feature_importance` (число строк или `null`).

## Деградация и ошибки

Ошибки — `RunContractError` из ядра с путями полей, как в `docs/specs/run-contract.md`.

- E1. Дано: `model` не словарь и не `ModelInfo`, или словарь без `class` и `family` · тогда ошибка с путём `model`.
- E2. Дано: в словаре `model` неизвестный ключ (например `params`) · тогда ошибка «неизвестное поле» с путём `model.<ключ>`; гиперпараметры передаются в `config`.
- E3. Дано: `class` или `library` — не непустая строка · тогда ошибка с путём `model.class`/`model.library`; `family` не из списка · ошибка с путём `model.family` и перечнем допустимых.
- E4. Дано: `feature_importance` не DataFrame, без `feature` или `importance`, с лишней колонкой · тогда ошибка с путём колонки (`feature_importance.<колонка>`) или таблицы.
- E5. Дано: пустое или нестроковое имя признака, нечисловая или бесконечная важность, NaN, отрицательная важность не при `kind: permutation`, повтор пары (`feature`, `kind`) · тогда ошибка с путём колонки и примерами строк.
- E6. Дано: `examples` переданы, а признака из `feature_importance` нет среди их колонок · тогда ошибка с путём `feature_importance.feature` и примерами имён.

## Вне рамок

- Определение семейства по имени класса, нормализация гиперпараметров между библиотеками — плейбуки (`docs/specs/playbooks.md`).
- Загрузка самой модели, чтение `MLmodel` и flavors MLflow: библиотека берётся из пути класса.
- Гиперпараметры в `model`: они остаются в `config`.
- Несколько моделей в одном прогоне (стекинг) — описывается внешняя модель.

## Тесты

- B1–B5, E1–E6 → `tests/test_model_info.py`
- F1–F2 → `tests/test_files_adapter.py`
- M1–M6 → `tests/test_mlflow_adapter.py`
- C1 → `tests/test_files_adapter.py`, `tests/test_mcp_server.py`
