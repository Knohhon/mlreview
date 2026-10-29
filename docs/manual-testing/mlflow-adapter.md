# Ручная проверка адаптера MLflow

Опора: FR-C03, FR-C02, FR-C07, FR-R01, FR-R04; спецификация поведения — `docs/specs/mlflow-adapter.md`.

Автотесты (`tests/test_mlflow_adapter.py`) проверяют адаптер на локальном sqlite-трекере при заблокированной сети. Этот путь закрывает то, что автотесты не видят:

- настоящий сервер трекинга по HTTP (`mlflow server`) с прокси артефактов — так MLflow разворачивают self-hosted;
- живой скрипт обучения (линейная модель sklearn), который логирует контракт через `log_contract_artifacts`;
- недоступный сервер трекинга и время до ошибки;
- установка без extra `mlflow`;
- проактивное ревью по эксперименту на настоящем сервере (`mlreview watch --experiment`);
- модель из автологирования sklearn: дерево и kNN в `Pipeline` (`docs/specs/model-info.md`).

Занимает 10–15 минут. Результаты записываются в журнал в конце файла.

## 0. Подготовка

Из корня репозитория — окружение L0 (пакет с extras `mlflow`, `dev` и scikit-learn):

```bash
pip install -r requirements.txt
```

Дальше всё делается во временном каталоге вне репозитория:

```bash
export WORK=$(mktemp -d) && cd "$WORK"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Создать два файла.

`task.yaml` — спецификация задачи (FR-C01):

```yaml
task:
  type: classification/binary
  output: probabilities
  error_costs: {false_negative: high, false_positive: low}
  segments: [size]
data:
  split_strategy: stratified
  version: breast-cancer-v1
```

`train.py` — прогон: SGD-логистическая регрессия на `breast_cancer` из sklearn (датасет поставляется с sklearn, сеть не нужна), 10 эпох с кривыми `train_loss` и `val_loss`, сегмент `size` по медиане `mean radius`. Флаги ломают прогон для проверки деградации.

```python
"""Прогон для ручной проверки адаптера MLflow: линейная модель на breast_cancer из sklearn."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlreview.adapters.mlflow_adapter import log_contract_artifacts

parser = argparse.ArgumentParser()
parser.add_argument("--no-examples", action="store_true", help="не логировать examples")
parser.add_argument("--no-label", action="store_true", help="сломать предсказания: без label")
parser.add_argument("--metric-without-step", action="store_true", help="метрика без step")
args = parser.parse_args()

data = load_breast_cancer(as_frame=True)
frame = data.frame.assign(example_id=[f"bc-{i:03d}" for i in range(len(data.frame))])
frame["size"] = np.where(frame["mean radius"] > frame["mean radius"].median(), "large", "small")
features = list(data.feature_names)
train, test = train_test_split(frame, test_size=0.3, random_state=0, stratify=frame["target"])
scaler = StandardScaler().fit(train[features])
x_train, x_test = scaler.transform(train[features]), scaler.transform(test[features])

mlflow.set_experiment("mlreview-manual")
with mlflow.start_run() as run:
    params = {"loss": "log_loss", "alpha": 1e-3, "epochs": 10, "random_state": 0}
    mlflow.log_params(params)
    model = SGDClassifier(loss="log_loss", alpha=1e-3, random_state=0)
    for epoch in range(10):
        model.partial_fit(x_train, train["target"], classes=[0, 1])
        train_loss = log_loss(train["target"], model.predict_proba(x_train))
        val_loss = log_loss(test["target"], model.predict_proba(x_test))
        if args.metric_without_step:
            mlflow.log_metric("train_loss", train_loss)
        else:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    predictions = pd.DataFrame(
        {
            "example_id": test["example_id"],
            "label": test["target"],
            "prediction": model.predict_proba(x_test)[:, 1],
        }
    )
    if args.no_label:
        predictions = predictions.drop(columns="label")
    examples = pd.concat([train.assign(split="train"), test.assign(split="test")])
    examples = examples[["example_id", "split", "size"]]
    log_contract_artifacts(
        predictions,
        examples=None if args.no_examples else examples,
        data_version="breast-cancer-v1",
        model={"class": "sklearn.linear_model.SGDClassifier", "family": "linear"},
    )

Path("run_id.txt").write_text(run.info.run_id, encoding="utf-8")
print(f"run_id: {run.info.run_id}")
```

Id прогона скрипт пишет в `run_id.txt`: MLflow 3 сам печатает в stdout ссылку на эксперимент в конце прогона, поэтому брать id из последней строки вывода нельзя.

## 1. Сервер трекинга

В отдельном терминале, в том же `$WORK`:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri "sqlite:///$WORK/mlflow.db" --artifacts-destination "$WORK/mlartifacts"
```

Готовность: `curl -s http://127.0.0.1:5000/health` отвечает `OK`.

## 2. Сценарии

Каждый сценарий: команда → ожидаемый результат. Id прогонов у каждого свои.

### M1. Полный контракт (B1, B6, C1)

```bash
python train.py && mlreview contract check "$(cat run_id.txt)" --spec task.yaml; echo "exit=$?"
```

Ожидается код 0 и сводка без пробелов:

```text
контракт прогона валиден: <run_id>
  предсказаний: 171
  примеров: 569, сплиты: test, train
  кривых: 2
  параметров: 4
  версия данных: breast-cancer-v1
  хеш данных: sha256:<16 hex>
  модель: sklearn.linear_model.SGDClassifier (семейство: linear, источник: tag)
exit=0
```

В UI (http://127.0.0.1:5000) у прогона: параметры `loss`, `alpha`, `epochs`, `random_state`; графики `train_loss` и `val_loss` по 10 шагов; теги `mlreview.data_version`, `mlreview.model.class`, `mlreview.model.family`; в артефактах каталог `mlreview` с файлами `predictions.parquet` и `examples.parquet`.

### M2. Сегмента нет в примерах (G2, X1)

```bash
sed 's/segments: \[size\]/segments: [size, clinic]/' task.yaml > task-extra-segment.yaml
mlreview contract check "$(cat run_id.txt)" --spec task-extra-segment.yaml; echo "exit=$?"
```

Ожидается код 0 и последняя строка сводки:

```text
  нет examples.clinic — недоступно: срез по сегменту clinic (FR-D02)
```

### M3. Без таблицы примеров (B4, G1, X1)

```bash
python train.py --no-examples && mlreview contract check "$(cat run_id.txt)" --spec task.yaml
```

Ожидается код 0, `примеров: нет` и строка:

```text
  нет examples — недоступно: срезы по сегментам (FR-D02), проверки данных между сплитами (FR-D03)
```

### M4. Метрика залогирована без step (E6)

```bash
python train.py --metric-without-step && mlreview contract check "$(cat run_id.txt)" --spec task.yaml
```

Ожидается код 1:

```text
прогон <run_id>: контракт прогона невалиден:
  - curves: пара (name, step) должна быть уникальна; повторяются: train_loss@0
```

### M5. Предсказания без метки (E6)

```bash
python train.py --no-label && mlreview contract check "$(cat run_id.txt)" --spec task.yaml
```

Ожидается код 1 и `predictions.label: обязательная колонка`.

### M6. Прогона нет (E2)

```bash
mlreview contract check 00000000000000000000000000000000 --spec task.yaml; echo "exit=$?"
```

Ожидается код 1 и `прогон 0000… не получен из http://127.0.0.1:5000: RESOURCE_DOES_NOT_EXIST: …`.

### M9. Проактивное ревью по эксперименту (FR-R04; `docs/specs/proactive.md`, `docs/specs/mlflow-adapter.md` B7)

Сервер ещё работает; в том же `$WORK`:

```bash
mlreview init
mlreview watch --spec task.yaml --experiment mlreview-manual --once; echo "exit=$?"
mlreview watch --spec task.yaml --experiment nope --once; echo "exit=$?"
```

Ожидается: по строке на каждый завершённый прогон эксперимента — «прогон <id>: находок <n> (critical …, high …) → <путь отчёта>» для валидных (M1, M3) и «прогон <id>: не проверен — … контракт прогона невалиден …» для M4 и M5, код 0; отчёты в `.mlreview/reports/`. Для несуществующего эксперимента — код 1 и «эксперимент nope не найден в http://127.0.0.1:5000».

### M7. Сервер трекинга недоступен (E2)

Остановить сервер (Ctrl+C в его терминале), затем:

```bash
time mlreview contract check "$(cat run_id.txt)" --spec task.yaml
time MLFLOW_HTTP_REQUEST_MAX_RETRIES=0 mlreview contract check "$(cat run_id.txt)" --spec task.yaml
```

Ожидается код 1 и `прогон <run_id> не получен из http://127.0.0.1:5000: … Max retries exceeded …`. Записать время обоих запусков: по умолчанию MLflow повторяет запрос с паузами, и ошибка приходит через минуты; с `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` — сразу.

### M8. Установка без extra mlflow (E1)

```bash
python -m venv bare && bare/bin/pip install -q <путь к репозиторию>
bare/bin/mlreview contract check abc --spec task.yaml; echo "exit=$?"
bare/bin/mlreview spec validate task.yaml; echo "exit=$?"
```

Ожидается: `contract check` — код 1 и `адаптер MLflow требует пакет mlflow: pip install "mlreview[mlflow]"`; `spec validate` — код 0 (остальные команды без mlflow работают).

### M10. Модель из автологирования sklearn (FR-C07; `docs/specs/model-info.md` M2, M3)

Сервер ещё работает; в том же `$WORK` — `autolog.py`:

```python
"""Прогоны с автологированием sklearn: дерево и kNN в Pipeline; таблицы — через mlreview."""

from pathlib import Path

import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlreview.adapters.mlflow_adapter import log_contract_artifacts

data = load_breast_cancer(as_frame=True)
frame = data.frame.assign(example_id=[f"bc-{i:03d}" for i in range(len(data.frame))])
features = list(data.feature_names)
train, test = train_test_split(frame, test_size=0.3, random_state=0, stratify=frame["target"])
mlflow.set_experiment("mlreview-autolog")
mlflow.sklearn.autolog(log_models=False, log_datasets=False)
for name, model in (
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
):
    with mlflow.start_run() as run:
        model.fit(train[features], train["target"])
        predictions = pd.DataFrame(
            {
                "example_id": test["example_id"],
                "label": test["target"],
                "prediction": model.predict_proba(test[features])[:, 1],
            }
        )
        log_contract_artifacts(predictions)
    Path(f"run_id_{name}.txt").write_text(run.info.run_id, encoding="utf-8")
```

```bash
python autolog.py
mlreview contract check "$(cat run_id_tree.txt)" --spec task.yaml | grep модель
mlreview contract check "$(cat run_id_knn.txt)" --spec task.yaml | grep модель
```

Ожидается: `модель: sklearn.tree._classes.DecisionTreeClassifier (семейство: не указано, источник: autolog)` и `модель: KNeighborsClassifier (семейство: не указано, источник: pipeline)`. Семейство по классу определяют плейбуки (`docs/specs/playbooks.md`); после их появления здесь будут `tree` и `neighbors`.

## 3. Уборка

```bash
cd ~ && rm -rf "$WORK"
```

## Журнал прогонов

| Дата | Python | MLflow | sklearn | M1 | M2 | M3 | M4 | M5 | M6 | M7 (по умолчанию / без повторов) | M8 | M9 | M10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-08 | 3.14.0 | 3.16.1 | 1.9.1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ 249 с / 1 с | ✓ | — | — |
| 2026-10-15 | 3.14.0 | 3.16.1 | 1.9.1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ 248 с / 2 с | ✓ | ✓ | — |
