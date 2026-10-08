# Ручная проверка адаптера MLflow

Опора: FR-C03, FR-C02, FR-R01; спецификация поведения — `docs/specs/mlflow-adapter.md`.

Автотесты (`tests/test_mlflow_adapter.py`) проверяют адаптер на локальном sqlite-трекере при заблокированной сети. Этот путь закрывает то, что автотесты не видят:

- настоящий сервер трекинга по HTTP (`mlflow server`) с прокси артефактов — так MLflow разворачивают self-hosted;
- живой скрипт обучения (линейная модель sklearn), который логирует контракт через `log_contract_artifacts`;
- недоступный сервер трекинга и время до ошибки;
- установка без extra `mlflow`.

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
exit=0
```

В UI (http://127.0.0.1:5000) у прогона: параметры `loss`, `alpha`, `epochs`, `random_state`; графики `train_loss` и `val_loss` по 10 шагов; тег `mlreview.data_version`; в артефактах каталог `mlreview` с файлами `predictions.parquet` и `examples.parquet`.

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

## 3. Уборка

```bash
cd ~ && rm -rf "$WORK"
```

## Журнал прогонов

| Дата | Python | MLflow | sklearn | M1 | M2 | M3 | M4 | M5 | M6 | M7 (по умолчанию / без повторов) | M8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-08 | 3.14.0 | 3.16.1 | 1.9.1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ 249 с / 1 с | ✓ |
