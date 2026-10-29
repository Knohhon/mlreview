# Плейбуки: формат, загрузка, семейство модели

Опора: FR-E02 (плейбуки как пакеты данных без изменения ядра — L2 · Must; здесь внутренний вариант), FR-D11 (проверки и подсказки по семейству модели), FR-C07 (сведения о модели — `docs/specs/model-info.md`), FR-E01 (единый интерфейс анализаторов — `docs/specs/analyzers.md`), FR-D06 и B11 (смена модели — одно изменение), FR-L02 (числа для LLM — только из фактов), X1, NFR-01, NFR-03.

Плейбук — пакет знаний о семействе моделей или типе задачи: к каким прогонам он применим, как распознать модель по имени класса, как называются гиперпараметры в разных библиотеках, пороги, тексты подсказок и детерминированные проверки. Формат — YAML-манифест плюс необязательный Python-модуль проверок:

```text
src/mlreview/playbooks/<id>/
  playbook.yaml   # манифест: данные
  checks.py       # необязательно: def run(ctx: PlaybookContext) -> CheckResults
```

```yaml
id: neighbors                      # совпадает с именем каталога
title: Метрические методы (kNN)
requirement: FR-D11                # опора находок плейбука
applies_to:
  families: [neighbors]            # семейства из docs/specs/model-info.md
  task_types: [classification/binary, classification/multiclass, regression]  # без ключа — все
detect:                            # семейство ← шаблоны короткого имени класса (fnmatch, регистр важен)
  neighbors: [KNeighbors*, RadiusNeighbors*, NearestCentroid]
params:                            # нормализованное имя ← алиасы в config разных библиотек
  n_neighbors: {aliases: [n_neighbors, k]}
  max_depth: {aliases: [max_depth, depth], unbounded: [null, -1]}   # пример из trees
thresholds: {small_k: 2}           # числа для проверок
checks:                            # тексты находок плейбука
  neighbors.small_k: {suggestion: "…"}
overrides:                         # подсказки общих проверок для этого семейства
  curves.overfitting: "…"
```

Встроенные плейбуки: `linear` (только распознавание линейных моделей), `neighbors` и `trees` (распознавание; проверки — `docs/specs/playbook-neighbors.md`, `docs/specs/playbook-trees.md`). Плейбук `trees` уже распознаёт ансамбли: `RandomForest*`, `ExtraTrees*`, `Bagging*` → `ensemble/bagging`; `GradientBoosting*`, `HistGradientBoosting*`, `AdaBoost*`, `LGBM*`, `XGB*`, `CatBoost*` → `ensemble/boosting` — и знает алиасы их гиперпараметров.

## API

```python
from mlreview.playbooks import builtin_playbooks, load_playbooks, resolve_family, model_params

playbooks = builtin_playbooks()  # кортеж Playbook по id; загружается один раз
resolve_family(contract.model)  # "neighbors" | None
model_params(contract.config, playbook)  # {"n_neighbors": 1, ...}
```

- `load_playbooks(root)` — плейбуки из каталога `root` (по подкаталогу на плейбук, в порядке имён); `builtin_playbooks()` — то же для `src/mlreview/playbooks/`.
- `Playbook`: `manifest` (модель манифеста), `run` (функция `checks.run` или `None`), свойство `id`.
- `PlaybookContext`: `review` (`ReviewContext`), `playbook`, `family`, `params` (нормализованные гиперпараметры кандидата) и `thresholds`.

## Ожидаемое поведение

### Манифест и загрузка

- B1. Дано: каталог с подкаталогами `a` и `b`, в каждом `playbook.yaml` (у `b` ещё `checks.py` с `run`) · когда `load_playbooks(каталог)` · тогда два плейбука в порядке `a`, `b`; у `a` `run` — `None`, у `b` — функция из `checks.py`.
- B2. Дано: встроенные плейбуки · тогда `builtin_playbooks()` — `linear`, `neighbors`, `trees`; ни один шаблон `detect` двух плейбуков не распознаёт один класс в разные семейства (на списке известных классов sklearn, LightGBM, XGBoost, CatBoost).

### Семейство модели

- B3. Дано: `model.family` задано явно · тогда `resolve_family` возвращает его, даже если класс распознаётся иначе.
- B4. Дано: класс без семейства: `sklearn.neighbors.KNeighborsClassifier`, `KNeighborsRegressor`, `sklearn.tree._classes.DecisionTreeClassifier`, `ExtraTreeRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `LGBMClassifier`, `XGBRegressor`, `CatBoostClassifier`, `HistGradientBoostingRegressor`, `LogisticRegression`, `Ridge` · тогда семейства `neighbors`, `neighbors`, `tree`, `tree`, `ensemble/bagging`, `ensemble/bagging`, `ensemble/boosting` ×4, `linear`, `linear`.
- B5. Дано: модели нет или класс неизвестен (`MyNet`) · тогда `None`.

### Гиперпараметры

- B6. Дано: config `{"n_neighbors": "3"}` (строки MLflow), `{"kneighborsclassifier__n_neighbors": "3"}` (параметры Pipeline), `{"model": {"k": 3}}` (вложенный) · тогда `model_params(...)["n_neighbors"] == 3`.
- B7. Дано: значения `"None"`, `"null"`, `"True"`, `"0.5"`, `"gini"` · тогда `None`, `None`, `True`, `0.5`, `"gini"`; значение из `unbounded` (`-1` у `max_depth`) · тогда `None` — «без ограничения»; гиперпараметра нет в config · тогда ключа нет в результате.
- B8. Дано: точное имя и имя шага Pipeline одновременно (`max_depth` и `est__max_depth`) · тогда берётся точное имя; несколько шагов — первый по имени ключа.

### Ревью: группа `model`

Группа `model` идёт после `metrics` (`data`, `segments`, `metrics`, `model`, `curves`, `process`).

- B9. Дано: плейбук с `run`, семейство кандидата в `applies_to.families`, тип задачи в `applies_to.task_types` · когда `review` · тогда `run` вызван с `PlaybookContext`, его находки и пропуски — в отчёте группы `model`. Плейбуки других семейств не вызываются и пропусков не дают.
- B10. Дано: семейство кандидата определено, но плейбука с проверками для него нет (`linear`) · тогда группа `model` ничего не добавляет.
- B11. Дано: семейство кандидата определено · тогда у находок общих проверок, чей `check` есть в `overrides` применимого плейбука, `suggestion` заменён текстом плейбука; у других семейств подсказки прежние.
- B12. Дано: отчёт ревью · тогда `Report.model` — `{"class", "family"}` кандидата с определённым семейством (или `None` без модели); в JSON-отчёте и в контексте LLM (`run_dict`) — ключ `model`. Гиперпараметры в LLM не уходят: числа — только через `facts` находок (FR-L02, NFR-04).
- B13. Дано: `mlreview diagnose model <прогон> --spec …` · тогда только находки и пропуски группы `model`; заголовок «диагностика модели прогона».
- B14. Дано: `mlreview playbooks list` · тогда по строке на плейбук: id, название, семейства, типы задач (или «все»), проверки (или «только распознавание»).

### Смена модели в сравнении (B11)

- B15. Дано: у базового и кандидата разные определённые семейства (`linear` → `tree`) · тогда в `process.multiple_changes` смена модели — одно изменение `model (linear → tree)`; ключи config, которые есть только у одной стороны, и ключи, распознанные алиасами плейбуков этих семейств, отнесены к смене модели и отдельно не считаются. Только смена модели — находки нет; смена модели и ещё одного общего ключа — находка «изменено сразу 2 параметра: model (linear → tree), features (10 → all)».

## Деградация и ошибки

- E1. Дано: в контракте нет `model` · тогда в группе `model` пропуск `model` «в контракте нет сведений о модели — проверки по семейству не выполнены; укажите model в run.yaml или теги mlreview.model.*».
- E2. Дано: класс есть, семейство не определено · тогда пропуск `model` «семейство модели не определено для класса <класс>; укажите model.family (run.yaml) или тег mlreview.model.family».
- E3. Дано: манифест с неизвестным полем, неизвестным семейством в `applies_to`/`detect`, `id` не совпадает с каталогом, некорректный YAML · тогда `PlaybookError` с путём файла и полем.
- E4. Дано: `checks.py` без функции `run` · тогда `PlaybookError` «<путь>: нет функции run».
- E5. Дано: `run` плейбука бросил исключение · тогда ревью не падает: пропуск «playbook.<id>: ошибка анализатора: …» (как B2 в `docs/specs/analyzers.md`).

## Вне рамок

- Внешние плейбуки из пакетов через entry points, шаблон контрибьюции (FR-E02, FR-E03 — L2, следующий шаг). Загрузчик принимает любой каталог, поэтому источник подменяется без изменения ядра.
- Плейбуки по типам задач (S10) — формат тот же, `applies_to.task_types`.
- Значения гиперпараметров по умолчанию у библиотек: если параметра нет в config, проверка, которой он нужен, сообщает пропуск.
- Проверки плейбуков `neighbors` и `trees` — в их спецификациях.

## Тесты

- B1–B11, B15, E1–E5 → `tests/test_playbooks.py`
- B12 → `tests/test_playbooks.py`
- B13–B14 → `tests/test_playbooks.py`
