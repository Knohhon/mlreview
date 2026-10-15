# Быстрый старт MLReview

От установки до первого ревью — около 15 минут. Нужен Python 3.10+. LLM не нужен: все проверки детерминированные и работают офлайн; LLM подключается в конце для гипотез и вопросов.

## 1. Установка

```bash
git clone https://github.com/Knohhon/mlreview.git && cd mlreview
pip install -e ".[mlflow,mcp]"
mlreview --version
```

Extra `mlflow` нужен для прогонов из MLflow, `mcp` — для MCP-сервера. Для прогонов в файлах extras не нужны: `pip install -e .`. После публикации на PyPI — `pip install "mlreview[mlflow,mcp]"`.

## 2. Рабочее пространство проекта

В корне проекта:

```bash
mlreview init
```

Появится каталог `.mlreview/`: настройки `config.yaml`, хранилище находок и журнал `store.sqlite`, отчёты. Добавьте его в `.gitignore`.

## 3. Спецификация задачи

Файл `task.yaml` описывает задачу, а не модель:

```yaml
task:
  type: classification/binary        # classification/multiclass, regression
  output: probabilities              # labels, scores; для регрессии — values
  error_costs: {false_negative: high, false_positive: low}
  segments: [region]                 # колонки examples, по которым смотреть срезы
data:
  split_strategy: temporal           # random, stratified
  time_column: ts
  version: ds-v1
```

```bash
mlreview spec validate task.yaml
```

## 4. Прогон: что сохранить из скрипта обучения

Контракт прогона — таблицы, а не модель: предсказания (`example_id`, `label`, `prediction`; для multiclass — колонки `prediction:<класс>`), метаданные примеров (`example_id`, `split`, сегменты, время, признаки), конфиг, кривые обучения, версия данных. В конце скрипта обучения:

```python
from mlreview.adapters.files_adapter import save_run_files

save_run_files(
    "runs/exp-002",
    predictions,  # DataFrame: example_id, label, prediction — по test
    examples=examples,  # DataFrame: example_id, split, сегменты, время, признаки — train и test
    config={"C": 1.0, "features": "all"},
    curves=curves,  # DataFrame: name, step, value — например train_loss, val_loss
    data_version="ds-v1",
)
```

С MLflow — внутри `mlflow.start_run()`: `from mlreview.adapters.mlflow_adapter import log_contract_artifacts; log_contract_artifacts(predictions, examples=examples, data_version="ds-v1")`; параметры и метрики прогона MLflow становятся конфигом и кривыми. Проверка контракта:

```bash
mlreview contract check runs/exp-002 --spec task.yaml     # или id прогона MLflow
```

## 5. Первое ревью

```bash
mlreview review runs/exp-002 --spec task.yaml --baseline runs/exp-001
```

В отчёте: находки по важности (дубли между сплитами, временная утечка, сильные признаки, сегменты хуже и регрессии сегментов, дельты метрик с интервалами, калибровка, кривые обучения, разные данные, несколько изменений), у каждой — id в хранилище, доказательства и предложение эксперимента; затем — что не проверено и почему. Полезные флаги: `--min-severity high`, `--limit 5`, `--format json` (для CI).

Несколько сидов одной конфигурации — следующими аргументами: `mlreview review runs/a-s1 runs/a-s2 runs/a-s3 --spec task.yaml --baseline runs/b-s1 runs/b-s2 runs/b-s3` — так проверяется, не в пределах ли шума улучшение.

## 6. Находки и журнал

```bash
mlreview findings list                                  # открытые находки
mlreview dismiss F-0003 --reason "сегмент не важен"     # не показывать, пока условия те же
mlreview journal add hypothesis "больше данных по region=north поможет" --run exp-002
mlreview journal add decision "принять exp-002" --run exp-002 --outcome accept
mlreview journal search "region north"
```

Принятое решение делает прогон референсом для проактивного ревью.

## 7. Метрики

```bash
mlreview metrics init --spec task.yaml --run runs/exp-002 --write   # диалог → раздел metrics: в task.yaml
mlreview metrics test runs/exp-002 --spec task.yaml                 # тривиальные бейзлайны, шум, MDE
mlreview metrics export --spec task.yaml -o metrics.py              # метрики как код на numpy
```

## 8. Ревью после каждого прогона

```bash
mlreview watch --spec task.yaml --runs-dir runs          # или --experiment <имя> для MLflow
```

Либо в конце скрипта обучения: `from mlreview import api; api.on_run_end("runs/exp-003", spec="task.yaml")`. Отчёты (5 главных находок) — в `.mlreview/reports/`.

## 9. LLM (по желанию)

В `.mlreview/config.yaml`:

```yaml
llm: {provider: anthropic}              # ключ — в переменной ANTHROPIC_API_KEY
# локальная модель через Ollama:
# llm: {provider: openai, base_url: "http://localhost:11434/v1", model: llama3.1, api_key_env: null}
```

```bash
mlreview review runs/exp-002 --spec task.yaml --baseline runs/exp-001 --hypotheses
mlreview ask "пробовал ли я больше данных по north?"
```

Числа в ответах сверяются с фактами, утверждения — со ссылками на находки; ответ без опоры блокируется. В LLM уходят только агрегаты; примеры — только с `--example <id>` и согласием; `--dry-run` показывает запрос, ничего не отправляя. Стоимость показывается после ответа и пишется в `.mlreview/llm-log.jsonl`.

## 10. Агент в IDE (MCP)

```bash
claude mcp add mlreview -- mlreview mcp    # из каталога проекта
```

Агент получает те же операции: `review`, `metrics_test`, `findings_list`, `journal_search`, `ask` и другие.
