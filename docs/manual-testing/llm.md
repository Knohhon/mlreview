# Ручная проверка LLM-слоя на настоящих провайдерах

Опора: FR-L01, FR-L02, NFR-02 (один набор проверок на ≥ 2 провайдерах, включая локальную модель), NFR-08; спецификация — `docs/specs/llm.md`.

Автотесты (`tests/test_llm_*.py`, `tests/test_grounding.py`) проверяют форму запросов, сверку и деградацию с поддельным транспортом при заблокированной сети. Этот путь закрывает то, что автотесты не видят: настоящие ответы моделей (проходят ли они сверку с фактами), ключи, цены и время ответа. Каждый запуск тратит деньги облачного провайдера — порядка цента на запрос.

## 0. Подготовка

```bash
pip install -r requirements.txt
export WORK=$(mktemp -d) && cd "$WORK"
mlreview init
cat > task.yaml <<'YAML'
task: {type: classification/binary, output: probabilities, segments: [size]}
data: {split_strategy: stratified, version: v1}
YAML
python - <<'PY'
import numpy as np, pandas as pd
from mlreview.adapters.files_adapter import save_run_files
rng = np.random.default_rng(0); n = 300
examples = pd.DataFrame({"example_id": [f"e{i}" for i in range(2 * n)],
                         "split": ["train"] * n + ["test"] * n,
                         "size": ["small", "large"] * n, "f0": rng.normal(size=2 * n)})
examples.loc[:9, "f0"] = examples.loc[n:n + 9, "f0"].to_numpy()  # дубли train/test
for name, noise in (("cand", 1.0), ("base", 1.5)):
    label = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(np.where(label == 1, 1.5, -1.5) + rng.normal(0, noise, n))))
    preds = pd.DataFrame({"example_id": examples["example_id"][n:].to_numpy(), "label": label, "prediction": p})
    save_run_files(name, preds, examples=examples, data_version="v1")
PY
```

## 1. Провайдеры

Для каждого провайдера записать раздел `llm` в `.mlreview/config.yaml` и пройти шаги L1–L4.

- **Anthropic:** `llm: {provider: anthropic}` и `export ANTHROPIC_API_KEY=…` (модель по умолчанию — `claude-opus-5-5`).
- **Локальная модель (Ollama):** `ollama pull llama3.1 && ollama serve`, затем `llm: {provider: openai, base_url: "http://localhost:11434/v1", model: llama3.1, api_key_env: null}`.
- **Облачный OpenAI-совместимый API (по желанию):** `llm: {provider: openai, model: <модель>, price: {input_per_mtok: …, output_per_mtok: …}}` и `export OPENAI_API_KEY=…`.

## 2. Шаги

- **L1. Гипотезы.** `mlreview review cand --spec task.yaml --baseline base --hypotheses` — в конце отчёта гипотезы H1, H2 со ссылками на `F-…`, альтернативы, различающий эксперимент и строка стоимости; либо «ответ заблокирован сверкой…» с причиной (тоже корректный исход — записать причину).
- **L2. Журнал стоимости.** `tail -1 .mlreview/llm-log.jsonl` — строка с провайдером, моделью, токенами и стоимостью (для Ollama — 0).
- **L3. Вопрос.** `mlreview journal add hypothesis "focal loss поможет" --run cand`, затем `mlreview ask "пробовал ли я focal loss?"` — ответ со ссылкой на `H-0001`.
- **L4. Приватность.** `mlreview review cand --spec task.yaml --hypotheses --dry-run` — в напечатанном запросе нет id примеров (`e…`) и значений `f0`; с `--example e305 --dry-run --yes` — есть ровно строка `e305`.
- **L5. Недоступность.** Остановить Ollama (или указать неверный ключ) и повторить L1 — детерминированный отчёт и «LLM: недоступен — …», код 0.

## 3. Уборка

```bash
cd ~ && rm -rf "$WORK"
```

## Журнал прогонов

| Дата | Провайдер / модель | L1 (ok / blocked: причина) | L2 | L3 | L4 | L5 | Стоимость L1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
