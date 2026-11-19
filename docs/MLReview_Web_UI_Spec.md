# MLReview Web UI — продуктовая и техническая спецификация

**Статус:** предложение для разработки (Web UI не заявлен как реализованный компонент).  
**Версия:** 0.1 · 19 ноября 2026  
**Целевой путь в репозитории:** `docs/specs/web-ui.md` (или `docs/web-ui-design.md` до утверждения требований).  
**Пример UX:** MLflow Tracking UI — эксперименты, прогоны, сравнение, детализация; **отличие:** MLReview объясняет результаты и управляет находками/решениями, но не заменяет experiment tracker.

## 1. Цель, границы и источник истины

Веб-интерфейс позволяет ML-инженеру увидеть состояние экспериментов и сделать через визуальные сценарии то, что уже возможно в CLI: проверить контракт, запустить ревью и диагностику, сравнить прогоны, изучить факты, протестировать метрики, обработать находки, вести журнал и при наличии настроенного LLM задавать вопросы.

**Основа текущего проекта (по публичным README, CLAUDE.md и quickstart, проверенным 19.11.2026):**

- Реализован L1 (MVP alpha): `TaskSpec`, контракт прогона, адаптеры MLflow и файлов, анализаторы (`data`, `segments`, `metrics`, `model`, `curves`, `process`), хранилище SQLite, findings, journal, метрики, опциональный LLM, CLI, Python API, MCP и proactive review.
- Инициализация `mlreview init` создаёт `.mlreview/config.yaml`, `.mlreview/store.sqlite`, каталог отчётов; спецификация задачи может лежать отдельно в `task.yaml`.
- Ревью не обязательно запускается автоматически. Прогоны могут поступать из директории или из MLflow; внешний MLflow Tracking Server — отдельная система.
- Web UI, HTTP API, `mlreview ui` и описанные ниже маршруты **предлагаются**, не считаются существующими командами или эндпоинтами.
- Приоритет: **single-user localhost**, без облака, аккаунтов, командных workspace и обязательного LLM. W&B, SSO, коллаборация — вне MVP.

**Инварианты:** интерфейс не проводит обучение, не редактирует модель и не подменяет MLflow; числа и evidence поступают от детерминированного ядра; LLM-гипотезы визуально отделены от фактов; недостаточность данных показывается явно; действия в UI и CLI обращаются к одному хранилищу и должны согласованно отражаться друг в друге.

## 2. Ментальная модель и информационная архитектура

**MLflow:** Experiment → Runs → Parameters/Metrics/Artifacts.  
**MLReview:** Project (локальное рабочее пространство) → подключённые sources → Runs → Reviews → Findings → Decisions.

`Project` в MVP — одна рабочая директория с `.mlreview/`, не отдельный объект многопользовательской платформы. `Experiment` является группировкой **в источнике** (MLflow experiment или виртуальная группа локальных runs), а не новой сущностью хранения MLReview. `Review` — снимок результата проверки одного/нескольких прогонов с параметрами, версиями и временем; **для полноценной истории review потребуется определить persistency**, поскольку существование всех нужных исторических записей в текущем store не подтверждено. При отсутствии истории в MVP можно показывать текущий отчёт и сохранённые находки.

### Глобальный shell

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│ MLReview  ▾ Project: churn-model    ● Local   [Search ⌘K]    [Settings]      │
├──────────────────────┬─────────────────────────────────────────────────────────┤
│ Overview             │ Breadcrumb: Project / Experiments / churn / Run 42    │
│ Experiments          │ ─────────────────────────────────────────────────────── │
│ Runs                 │ Заголовок страницы                   [Primary action]  │
│ Reviews              │ Контекст / фильтры / tabs                               │
│ Findings         (3) │                                                         │
│ Metrics              │               Содержимое страницы                      │
│ Task specification   │                                                         │
│ Journal              │                                                         │
│ Playbooks            │                                                         │
│ ───────────────────  │                                                         │
│ Sources & Settings   │                                                         │
└──────────────────────┴─────────────────────────────────────────────────────────┘
```

**Паттерны MLflow:** фиксированная левая навигация, таблицы run со сортировкой/фильтрами, кликабельные experiment/run, карточка run с вкладками, compare выбранных runs, графики metrics. **Паттерны MLReview:** severity и evidence прежде всего; review coverage, причины невозможности проверок, действия acknowledge/dismiss/decision, воспроизводимость анализа. Не копировать UI MLflow буквально.

- Таблицы: search, sort, фильтрация по severity/status/source/time, мультивыбор, пагинация, фиксированные заголовки, сохранение фильтров в URL.
- На каждом review — **fact / interpretation / recommendation**, а не единый непрозрачный текст.
- Справа при необходимости открывается inspector drawer: run ID, версия данных, evidence, команды-эквиваленты CLI, raw JSON; URL должен оставаться пригодным для прямого входа.
- Критичные находки — заметные текстовые статусы + доступные для дальтонизма иконки, без опоры только на цвет.
- Empty/error/loading states проектируются наравне с заполненными. Desktop-first с корректным responsive layout; на мобильном анализ таблиц допускает горизонтальную прокрутку.

## 3. Как попасть в интерфейс после инициализации

### 3.1. Предлагаемые команды и жизненный цикл

```bash
# Внутри каталога проекта пользователя (после установки MLReview с web extra)
mlreview init                         # существует сейчас
mlreview ui                           # НОВАЯ команда, предложена этой спецификацией
# → MLReview UI: http://127.0.0.1:8765/
# → по умолчанию открывает браузер; --no-browser отключает автозапуск

mlreview ui --port 8766 --no-browser  # НОВЫЕ флаги
mlreview ui --host 127.0.0.1          # localhost по умолчанию
```

`mlreview init` **не должен запускать постоянно работающий сервер**: это создание workspace, а не запуск процесса. В успешном выводе `init` добавить подсказку `Next: mlreview ui`. Для удобства дополнительно возможно `mlreview ui --project /path/to/project` (предлагаемая опция), чтобы открыть workspace не из его cwd. Путь должен быть валидирован.

**Способ установки (предлагается):** `pip install -e ".[web,mlflow]"` для разработки и `pip install 'mlreview[web,mlflow]'` после публикации. `web` extra отделён от CLI. Точный стек web-сервера утверждается при реализации.

**Поведение `mlreview ui`:**

1. Находит корень workspace по явному `--project` либо начиная с текущей директории, поднимаясь к `.mlreview/`; явный путь приоритетнее.
2. Если `.mlreview/` не найден: сообщает `Workspace not initialized` с командой `mlreview init`; **не создаёт проект неявно**. Допускается явная кнопка "Initialize here" только после подтверждения пути и действия.
3. Проверяет доступность/схему SQLite, `config.yaml`, task spec, источников и extras. Неинициализированная task spec не блокирует просмотр UI.
4. Стартует HTTP server, слушает только `127.0.0.1`, выдаёт фактический URL (при занятом порте — понятная ошибка и предложение `--port`, без тихого переключения).
5. Запускает браузер, если TTY/GUI доступен и не указан `--no-browser` (headless/SSH — только URL); открывает `/`.
6. `/` делает client redirect на `/overview` при готовом workspace, на `/setup` при отсутствии минимальной конфигурации задачи/источников. **Отсутствие данных не означает ошибку:** dashboard c empty state тоже допустим.
7. При Ctrl+C останавливается сервер, уже выполненные операции и данные в store сохраняются.

**Дополнительная интеграция:** для CLI `mlreview review ...` можно напечатать `Open in UI: http://127.0.0.1:8765/reviews/<id>` **только если активный сервер действительно доступен и есть разрешаемый постоянный review ID**. Не выводить неработающую ссылку и не зависеть от Web UI для CLI.

### 3.2. First-run setup и сценарии

**A. Новый проект с локальными файлами**

```text
cd my-ml-project
mlreview init
mlreview ui
Browser → /setup
  1. Workspace: отображает безопасный локальный путь
  2. Task spec: выбрать task.yaml / создать через форму / пропустить
  3. Source: Local runs directory → ./runs
  4. Scan preview: найдено N run directories, N valid, N incomplete
  5. Finish → /overview
  6. Runs → выбрать run → Check contract → Run review
```

**B. Проект с MLflow**

```text
mlreview init && mlreview ui
Browser → /setup → Connect MLflow
  Tracking URI: file:///... или http://... (явно указанный)
  Test connection → перечень доступных experiments
  Select experiment(s) → /experiments → /experiments/<id>/runs
  Run → Review / Compare
```

Подключение к удалённому MLflow может требовать network access и секреты: сервер хранит секреты только разрешённым способом, не отображает их в API/browser, не помещает в URL.

**C. Работа из IDE/WSL2/headless**: `mlreview ui --no-browser`; пользователь вручную открывает показанный адрес, для WSL2 — localhost-переадресация при поддержке среды; если нет, документировать SSH tunnel вместо открытия `0.0.0.0` без защиты. Сервер должен указывать, на каком host и port слушает.

**D. Повторное открытие**: `cd project && mlreview ui`; сразу `/overview`, источники и решения прочитаны из `.mlreview/`. Закладки на `/runs/<id>` должны работать после рестарта (для локальных файлов — **стабильный namespace + ID**, а не индекс строки таблицы).

**E. Проект не настроен**: `/setup` показывает точный перечень отсутствующих вещей; пользователь может пропустить необязательное, а доступные разделы работают в degraded mode.

## 4. Клиентские маршруты (предлагаемая версия v1)

Статические маршруты перечислены прежде динамических, чтобы не конфликтовать с `:id`. Все пути начинаются от корня локального сервера; ID в URL — URL-encoded стабильные идентификаторы, **не файловые пути**.

| Path | Page | Назначение | Приоритет |
|---|---|---|---|
| `/` | Root redirect | `/overview` либо `/setup` | P0 |
| `/setup` | Setup wizard | task spec + источники | P0 |
| `/overview` | Dashboard | текущее состояние проекта | P0 |
| `/experiments` | Experiments | группировки runs из sources | P1 |
| `/experiments/:experimentId` | Experiment | сводка эксперимента и runs | P1 |
| `/runs` | Runs | общая таблица прогонов | P0 |
| `/runs/compare?ids=...` | Compare | несколько прогонов, baseline | P0 |
| `/runs/:runId` | Run detail | параметры, метрики, артефакты, контракты | P0 |
| `/runs/:runId/review` | Run review | последний review этого прогона/его запуск | P0 |
| `/reviews` | Reviews | отчёты проверок | P1 |
| `/reviews/:reviewId` | Review detail | snapshot review с фактами | P1* |
| `/findings` | Findings | единый backlog находок | P0 |
| `/findings/:findingId` | Finding detail | evidence, status, resolution | P0 |
| `/metrics` | Metrics | каталог + тестирование пригодности | P1 |
| `/metrics/test?run=:id` | Metric suitability | тест выбранных метрик | P1 |
| `/task` | Task specification | визуальный редактор task.yaml | P0 |
| `/journal` | Journal | поиск гипотез, экспериментов и решений | P0 |
| `/journal/:entryId` | Journal entry | детальная запись/связи | P1 |
| `/playbooks` | Playbooks | список встроенных checks | P1 |
| `/sources` | Data sources | файловые директории, MLflow URI/experiments | P0 |
| `/settings` | Settings | workspace, LLM, watch, безопасность | P0 |
| `/not-found` | Not found | диагностика несуществующей ссылки | P0 |

`P0` — минимальный веб-сценарий; `P1` — следующий этап UI. `*` — требует определённого долговременного review snapshot и ID; до этого `/runs/:runId/review` может показывать актуальный report без истории. Отдельная маршрутизация для model registry, deployment, teams или jobs намеренно отсутствует.

### URL для фильтров и shareable state

```text
/runs?source=mlflow&experiment=e-1&sort=-startedAt&status=finished
/findings?severity=critical,high&status=open&run=r-42
/runs/compare?ids=mlflow%3Ar1,mlflow%3Ar2&baseline=mlflow%3Ar1
/reviews?run=mlflow%3Ar42
```

Запрещено помещать в query строку токены, API keys, приватные примеры, raw SQL или путь к данным. URL содержит только безопасные ID и фильтры. Неизвестные параметры игнорируются или мягко нормализуются, а не обрывают страницу.

## 5. Страницы: состав и визуальный дизайн

### 5.1. `/overview` — Project overview

**Задача:** ответить за 10–15 секунд: "Что происходило? Где проблемы? Какой прогон требует внимания?"

```text
Project overview        Task: binary classification · Source: MLflow + files
[Runs 37] [Reviewed 25] [Open critical 2] [Open high 5] [Unreviewed 12]
──────────────────────────────────────────────────────────────────────────
Recent runs                              Findings requiring attention
Run       Metric   Δ baseline  Review     [CRITICAL] Possible leakage →
run-042   0.873    +0.012      3 high     [HIGH] Segment north regression →
run-041   0.861    -0.003      clean      [HIGH] Metric insensitive →
──────────────────────────────────────────────────────────────────────────
Recent decisions              Coverage / Unchecked checks
[Accepted run-039]            4 checks skipped: missing feature columns
```

- KPI: runs discovered, runs with review, findings by severity, last successful review; каждое число с определением и фильтром.
- Recent runs (ссылки на детализацию), review status и delta к **явно выбранному** baseline; если baseline нет — прочерк, не выдуманная delta.
- Active findings с кликом в evidence и возможностью фильтра.
- Coverage summary: выполнявшиеся и пропущенные анализаторы, причины.
- Recent journal decisions и принятый reference run.
- **Действия:** `Review latest run`, `Open findings`, `Connect source`, `Edit task`.
- Если runs нет: onboarding-подсказка `Connect files/MLflow → See runs`, без нулей, вводящих в заблуждение.

### 5.2. `/experiments`, `/experiments/:experimentId` — experiments в стиле MLflow

- Таблица: experiment/source, #runs, last updated, reviewed count, open criticals, last decision; поиск по имени.
- Experiment detail: header c source/URI, время, фильтры, таблица runs, mini chart выбранной метрики по завершённым runs, selected runs + `Compare`.
- Для файловых источников: "Local runs / <source name>" как **виртуальная группа**, без обязательного эксперимента MLflow.
- Drill-down: experiment → run → review; перейти в оригинальный MLflow URL можно только при известной и безопасной ссылке.

### 5.3. `/runs` — таблица прогонов (ключевой экран)

```text
Runs                                 [Filter] [Refresh] [Compare selected]
[Search by name/id] [Source ▾] [Status ▾] [Review ▾] [Date ▾]
□  Run         Experiment    Metric     Baseline Δ  Findings  Last review
□  run-042     churn-v2      0.873      +0.012      3 high    18:25
□  run-041     churn-v2      0.861      —           0         18:10
□  local-002   local-runs    0.820      —           not run   —
```

- Колонки: name/id, source, experiment/group, lifecycle status (если источник его сообщает), metric(s) с единицами, data version, model family, last review, finding counts.
- Пользователь выбирает отображаемую primary metric; несопоставимые метрики не агрегируются.
- Фильтр "contract complete/incomplete", сортировка, bulk selection, контекст `Check contract` и `Review`.
- `Compare selected` активируется для 2+ runs; проверяет совместимость task/data version и **не скрывает** несовместимость.
- `Refresh` обновляет index источников без автоматической перезаписи review.
- Run ID нормализуется как `<source-id>:<native-run-id>`; source-id хранится стабильно в config/store.

### 5.4. `/runs/:runId` — карточка прогона

**Header:** имя, run ID, source, experiment, status, data version, model family, время, кнопки `Run review`, `Check contract`, `Compare`, `Open in MLflow` (если применимо).

**Tabs (предпочтительно query `?tab=...`):**

1. **Summary** — primary metrics, параметры, data version, latest review и open findings.
2. **Metrics & curves** — таблица доступных метрик и графики learning curves; графики не рисуются, если нет данных.
3. **Parameters & model** — config, class/family, доступные feature importances; не обещать доступ к весам/объекту модели.
4. **Data & contract** — наличие predictions, examples, IDs, splits, segments, timestamps, dataset hash, список gaps и зависящие от них анализаторы.
5. **Findings** — фильтр по этому run; links evidence.
6. **Artifacts** — только доступные через адаптер артефакты, с metadata/безопасной ссылкой или объяснением "не поддерживается"; не дублировать полноценный MLflow artifact browser в P0.

**Контракт:** `Valid / Partial / Invalid` только на основании явно определённых правил; рядом — список отсутствующих полей и последствия для каждого анализатора. Результат `Check contract` можно запускать отдельно без полного review.

### 5.5. `/runs/compare` — сравнение, основной исследовательский flow

```text
Compare runs   [Baseline: run-040 ▾] [Metric: ROC AUC ▾] [Run review]
              run-040 (baseline)   run-041        run-042
ROC AUC       0.861                0.862 (+.001) 0.873 (+.012)
CI / p-value  source: analyzer     ...            ...
Data version  ds-v2                ds-v2          ds-v3 ⚠
Parameters    C=0.1                C=1.0         C=1.0
Segments      [interactive table of regressions / improvements]
Curves        [small multiples / overlay, if curves exist]
Findings      [diff in detected issues]
```

- Выбор baseline **обязателен** для интерпретации delta; при входе без baseline UI предлагает выбрать один, а не угадывает.
- Таблица parameters diff с подсветкой изменённых значений и warning при нескольких изменениях; split/data versions.
- Таблица deterministic comparison: effect size, CI, significance, sample/seed info, segment regressions. Рассчитанные значения показываются только если соответствующая операция ядра поддерживает сравнение.
- Для нескольких сидов UI позволяет определить **группы** seed runs; сравнение групп и отображение шумовой неопределённости только по доступному анализатору.
- `Run comparison review` = вызов существующей review-логики с выбранными runs и baseline; для клиента выполнить через backend, не вычислять статистику в JS.
- Кнопка `Save comparison` — P1 и **не реализовывать**, пока не утверждена сущность сохранённого comparison.

### 5.6. `/runs/:runId/review`, `/reviews/:reviewId` — Review report

Визуальная структура:

```text
Review: run-042 vs run-040      [Run again] [Download JSON] [Copy CLI]
Completed 18:32 · Spec hash ... · Data version ds-v3
Critical 1 · High 2 · Medium 4 · Low 1
───────────────────────────────────────────────────────────
Findings [severity ▼] [group ▼] [status ▼]
CRITICAL  F-0003  Possible temporal leakage          [Open evidence]
HIGH      F-0004  Region north metric regression    [Open evidence]
───────────────────────────────────────────────────────────
Facts / metrics / segment differences / curves / process
What was NOT checked: model check — missing hyperparameters
LLM hypotheses (optional, visually separate, linked to findings)
```

- Review context: run IDs, baseline, spec/version, analyzer versions (если доступны), параметры запуска, check coverage, status queued/running/succeeded/failed.
- Findings группируются по severity или analyzer; содержание finding: observed fact, evidence, affected segments/IDs, why it matters, suggested next experiment.
- `Hypotheses` доступна лишь при включённом LLM; **все числовые значения должны проверяться backend grounding**. Явный provenance и предупреждение, что hypothesis не доказанный факт.
- Секция `Not checked`: какие проверки не выполнились и какие поля добавить.
- История review (`/reviews/:reviewId`) P1: после утверждения схемы snapshots. Без неё не создавать фиктивный ID и не имитировать сохранение.
- `Download JSON` сериализует тот же report schema, что CLI `--format json`, с redaction приватных полей; `Copy CLI` показывает эквивалент без секретов.

### 5.7. `/findings` и `/findings/:findingId` — backlog и evidence

**List view:** status, severity, title, analyzer, run, first seen/last seen (если хранятся), evidence count, dismissal reason; группировка по status и severity. Статусы должны соответствовать имеющейся предметной модели store, а не UI-изобретению.

**Detail:**

- Текст проблемы, severity, affected run(s), semantic fingerprint/ID (если открыт API), analyzer и точные условия обнаружения.
- **Facts:** метрика, интервал, сравнение, IDs примеров и версии данных, исходные references без показа приватных строк по умолчанию.
- **Why it matters / How to investigate / Suggested experiment** — детерминированные рекомендации или отдельно подписанные LLM-гипотезы.
- Действия: `Dismiss` с обязательной причиной для critical (и по текущим правилам для других — как определит backend), `Open run`, `Add journal decision`.
- Подтверждение действий, отражение статуса и audit record; нельзя считать находку исправленной лишь по клику без доказательств.
- Обнаружение заново учитывает семантику fingerprint и исходный store; UI не реализует собственную suppression logic.

### 5.8. `/metrics`, `/metrics/test` — выбор и валидация метрик

- Каталог по типу задачи: название, направление оптимизации, требования к входу, интерпретация, известные ловушки, сценарии использования.
- `Recommend metrics`: форма, использующая task spec и существующий advisor; объяснение trade-offs.
- `Metric test`: выбрать run/spec/metric и запустить тесты against trivial baseline, sensitivity, noise, MDE; таблица результатов с условиями валидности и предупреждениями о недостатке данных.
- `Export`: подтверждённый preview кода и явный путь записи `metrics.py` через backend; никаких file writes от браузера к произвольным путям.
- `Init metrics` — управляемый wizard с dry-run diff YAML до сохранения. Уметь отменить и не перезаписывать без подтверждения.

### 5.9. `/task` — визуальный редактор task specification

- Верхняя часть: выбранный файл спецификации, последнее изменение, status `Valid / Invalid`, `Validate`, `Save`, `View YAML diff`.
- Формы: task type/output, error costs, segments, split strategy/time column, dataset version, known issues и блок `metrics`.
- Синхронно показывать YAML preview; schema validation выполняет Python backend, не независимая клиентская бизнес-логика.
- `Save` атомарно обновляет **явно выбранный локальный файл** в пределах контролируемого workspace с резервной копией / confirm diff и защитой от конфликтов (mtime/etag).
- Если `task.yaml` отсутствует: создание через wizard; не выдавать синтетическую спецификацию за реальную.

### 5.10. `/journal`, `/journal/:entryId` — научная память проекта

- Таймлайн и таблица: time, type (`hypothesis`, `experiment`, `decision`, другие поддерживаемые store типы), text, linked run(s), outcome, author/source (если доступны).
- Search + filters. `Add entry`: type, text, linked run, outcome — согласовано с CLI `journal add`.
- Детальная запись со ссылками на исходные run и findings. Accepted decision может определять reference run для proactive review согласно правилам ядра.
- Не вводить полноценные approvals/workflows или shared editing в single-user MVP.

### 5.11. `/playbooks` — объяснимость проверок

- Список (`linear`, `neighbors`, `trees` при наличии): family, detection rules, active checks, необходимые поля и пороги.
- Details в drawer: что проверяет, как применимость определяется по `model.class`, связь с analyzer `model`, варианты missing metadata.
- P0/P1: **read-only**; редактор YAML/Python checks через web browser не нужен, чтобы не создавать исполнение произвольного кода.

### 5.12. `/sources` — подключение и диагностика источников

- Cards: Local files (path, last scan, runs count, errors), MLflow (tracking URI, experiments, test connection, status).
- Add/Edit source: validated local directory или MLflow tracking URI; test connection; выбор experiments; исключение невалидных runs и вывод причины.
- Settings хранятся в `.mlreview/config.yaml`/store по утверждённой схеме; не писать секреты в браузерное localStorage.
- `Refresh runs` = re-index; не удалять историю/находки при кратковременной недоступности источника.
- Отдельно показать `Configured source`, `Available now`, `Run discovered`, `Run has full contract`.

### 5.13. `/settings` — настройки и environment

Разделы:

- **Workspace:** project root, store path (read-only в MVP), task spec path, UI host/port, версия MLReview.
- **Analysis defaults:** selected spec, baseline policy (только явно настроенное правило), severity threshold, output preferences.
- **LLM (optional):** provider, model, endpoint, env var status (только `configured/not configured`), privacy warning, dry-run preview. Не показывать токен ни в логах, ни в response.
- **Proactive watch:** status и конфигурация; P1 — start/stop только для процесса, которым управляет текущий сервер. Нельзя показывать `Running` без проверяемого процесса.
- **Diagnostics:** health checks, доступность MLflow и зависимостей, ошибки импортов/конфигурации, копирование safe diagnostic bundle.

### 5.14. `/setup`, errors и empty states

- Setup: сверху progress, справа preview результата каждого шага; возможность пропустить MLflow или LLM.
- `No runs yet`: инструкции для `save_run_files()` и `log_contract_artifacts()`, ссылка на quickstart.
- `Incomplete contract`: список реально отсутствующих полей и отключённых checks.
- `MLflow unreachable`: не падать целиком; сохранить доступ к локальным sources и store, `Retry`.
- `LLM disabled`: deterministic review доступно; hypotheses/ask показывают пояснение, а не ошибку.
- `No findings`: отличать "проверено, проблем не найдено" от "ревью ещё не запускалось" и "анализаторы пропущены".
- `Invalid run ID`: 404 с путём возврата в runs; `Store migration needed` — отдельная диагностическая страница без автоматической разрушительной миграции.

## 6. CLI ↔ UI: функциональное соответствие

| CLI (существующая семья команд) | Web view / действие | Правило |
|---|---|---|
| `mlreview init` | `/setup` | Создание workspace остаётся явным CLI-действием; web setup может конфигурировать подготовленный workspace |
| `mlreview spec validate` | `/task` → Validate | То же backend validation |
| `mlreview contract check` | `/runs/:runId?tab=contract` → Check | Причины gaps и coverage |
| `mlreview review` | Run/Compare → Review | Same core, same result semantics |
| `mlreview diagnose data\|segments\|model\|curves\|process` | Review → analyzer filter/tab | Не повторять расчёт в браузере |
| `mlreview findings list` | `/findings` | Shared store, same statuses |
| `mlreview dismiss` | Finding → Dismiss | Причины и ограничения backend |
| `mlreview journal add/search` | `/journal` | Shared store |
| `mlreview metrics init/test/export/list` | `/metrics` | Preview/validation/export |
| `mlreview playbooks list` | `/playbooks` | Read-only listing |
| `mlreview ask` | P1: contextual `Ask` drawer на run/review/findings | Требует configured LLM, grounded output |
| `mlreview watch` | `/settings` → Watch status | Не обещать самозапуск/управление внешним процессом |
| `mlreview mcp` | Нет отдельного web клона | Агент может выдавать стабильные deep links |
| `mlreview ui` | Запускает сервер | **Новая предлагаемая команда** |

**Важно:** таблица показывает намеренное соответствие концепций, а не утверждает, что веб-страницы уже реализованы или что сигнатуры Python API полностью совпадают с CLI.

## 7. Навигация и пользовательские сценарии (acceptance flows)

**Flow 1 — найти проблемный прогон:** `/overview` → critical finding → `/findings/F-...` → evidence → linked `/runs/<id>` → `/runs/<id>/review` → dismissal/journal. После обновления статуса findings count обновляется во всех представлениях.

**Flow 2 — сравнить модели:** `/experiments` → experiment → выбрать 2–5 runs → `/runs/compare?ids=...` → baseline → deterministic compare → segment regressions → optional review. Некомпатибельные данные отмечены до запуска сравнения.

**Flow 3 — неполный contract:** `/runs/<id>` → `Data & contract` → `Check contract` → список missing columns/analyses → copy requirements → доработать training script → refresh → recheck.

**Flow 4 — выбрать метрики:** `/task` → `/metrics` → recommend → `/metrics/test` → открыть объяснения тестов → принять и сохранить `metrics:` в `task.yaml` после diff.

**Flow 5 — без MLflow и без LLM:** локальная директория с runs → `/runs` → `/runs/:id` → review → findings. Никаких network requests; нет искусственно заблокированных экранов.

**Flow 6 — агент + человек:** MCP/coding agent запускает review тем же ядром и возвращает URL вида `/findings/<id>` или `/runs/<id>/review`; инженер видит те же факты. Ссылки корректны только для доступного ему локального сервера и при существующем ID.

## 8. Предлагаемая backend архитектура

```text
Browser SPA (readable visual layer)
   | HTTP JSON + optional progress stream
   v
Local web app / API (НОВОЕ)
   |   authorization for loopback, validation, redaction
   |   source registry + view models + background task manager
   v
MLReview application services (общие для CLI / API / MCP / Web)
   ├─ task spec / run contract / source adapters (files, MLflow)
   ├─ deterministic analyzers / metrics / playbooks
   ├─ review / reporting / findings / journal
   ├─ optional grounded LLM
   └─ .mlreview/ config, SQLite, reports
```

- **Backend:** предлагается Python FastAPI + Uvicorn (в `web` extra) либо минимальный ASGI server — решение после spike; React + TypeScript + Vite для сложных таблиц/графиков, bundled static assets в wheel, без Node runtime у конечного пользователя. Это **вариант реализации**, не существующая зависимость.
- **Выделить application/service слой** для use cases, если CLI сейчас содержит orchestration logic: CLI и Web вызывают один сервис; endpoint **не** исполняет shell-команду `mlreview ...` как subprocess.
- SQLite — source of truth для findings/journal; MLflow/file adapters — source of truth для run metadata; task.yaml — source of truth спецификации. Не копировать новые сущности в browser state как долговременную базу.
- P0 long-running review: `POST` → job ID, polling `GET /jobs/:id`; worker ограничивает concurrency и ресурсы, не блокирует event loop. После restart потерянная job не может показываться как running; snapshots/results сохраняются явно или отмечаются interrupted.
- Кэш derived views keyed by `(run_id, source fingerprint, task spec hash, analyzer version, parameters)`; invalidate при изменении источника или spec. Не выдавать старую находку за проверку текущего датасета.
- API versioning: `/api/v1/...`, общие Pydantic DTO, OpenAPI для codegen frontend types; ошибки с машиночитаемыми кодами и human-readable details.
- Пример endpoint plan (**не текущий API**):

```http
GET    /api/v1/health
GET    /api/v1/workspace
GET    /api/v1/sources
POST   /api/v1/sources/test
GET    /api/v1/experiments
GET    /api/v1/runs?source=&experiment=&sort=&limit=&cursor=
GET    /api/v1/runs/{id}
POST   /api/v1/runs/{id}/contract-check
POST   /api/v1/reviews                 # {run_ids, baseline_run_ids, spec, options}
GET    /api/v1/reviews/{id}           # P1 persisted snapshots
GET    /api/v1/jobs/{id}
GET    /api/v1/findings?status=&severity=&run_id=
GET    /api/v1/findings/{id}
POST   /api/v1/findings/{id}/dismiss  # {reason}
GET    /api/v1/journal
POST   /api/v1/journal
GET    /api/v1/task
POST   /api/v1/task/validate
PUT    /api/v1/task                    # optimistic concurrency
GET    /api/v1/metrics/catalog
POST   /api/v1/metrics/test
GET    /api/v1/playbooks
GET    /api/v1/settings
```

**Специальные ограничения:** endpoint review валидирует run/source identity и task spec; сервер не получает произвольный Python/code execution из UI; операции файлов ограничены allowlisted paths/workspace, path traversal и symlink escape запрещены; открытие внешних артефактов строго контролируется. Адрес `127.0.0.1` не является достаточной защитой от CSRF/DNS rebinding: проверять Host/Origin, использовать CSRF protection для мутирующих запросов, не включать permissive CORS, при необходимости session token. Для `--host 0.0.0.0` потребовать отдельный явно документированный secure mode, не поддерживать без auth.

## 9. Требования к состояниям, данным и тестам

**Договоримся о терминах:** `not run`, `running`, `completed`, `failed`, `skipped` — статусы операции review, не run status источника; finding severity — отдельное поле от finding resolution status; `source unavailable` не равно `run deleted`.

**Контракты:**

- Run identity: stable namespaced ID → отображаемое native ID, source, experiment; никаких неоднозначных совпадений names.
- Review identity: либо `latest` по run и выбранной comparison spec, либо permanent snapshot ID; алгоритм выбора latest должен быть явным.
- Finding reference: стабильный ID из store + evidence/provenance, повторное ревью не должно терять dismissal semantics.
- Metrics: direction, units, split, sample size, confidence information (если вычислены); `NaN`/missing rendering, никакой "0" как fallback.
- Coverage: список applicable/ran/skipped/failed checks и причины; именно backend определяет applicability.

**Нефункциональные требования (предлагаемые):**

- Local-first; deterministic workflows без LLM и internet; telemetry **off** по умолчанию.
- Обслуживание 10k runs через server-side pagination; P0 table не грузит все данные разом. Target performance численно уточнить бенчмарком после первого прототипа.
- Результат для одного run UI vs CLI при одинаковых inputs: совпадают IDs (если уже persisted), severities, факты, рекомендации и skipped reasons.
- WCAG-aware keyboard navigation, aria-labels, contrast, color-independent severity; timestamps с часовым поясом; числовые форматы и precision не искажают факты.
- XSS protection для текста journal, findings, LLM responses, MLflow metadata; strict escaping и безопасная обработка markdown.
- Browser reload/direct deep link поддерживаются серверным SPA fallback **только для frontend routes**, `/api/*` возвращает корректные API errors.

## 10. Тестовые сценарии (черновик B-web)

| ID | Дано · когда · тогда |
|---|---|
| WEB-01 | Given initialized `.mlreview`, when `mlreview ui`, then server binds localhost, prints URL and `/overview` opens |
| WEB-02 | Given no workspace, when `mlreview ui`, then clear initialization message, no implicit writes |
| WEB-03 | Given files-only project, when no MLflow and LLM, then runs/review/findings work offline |
| WEB-04 | Given MLflow unavailable, then existing local runs and findings remain accessible |
| WEB-05 | Given incomplete contract, then unavailable checks and missing fields are shown, remaining checks run |
| WEB-06 | Given review run through CLI, then UI reads the same findings and statuses from store |
| WEB-07 | Given dismissed critical finding without reason, then backend rejects; valid reason persists and is visible in UI |
| WEB-08 | Given two different dataset versions, then comparison visibly warns and does not silently claim comparable effect |
| WEB-09 | Given direct URL `/runs/:id` after server restart, then same run resolves (or honest not-found if source removed) |
| WEB-10 | Given a review with no findings, then UI distinguishes clean result from review not performed/skipped |
| WEB-11 | Given unsafe local path / malicious journal text, then path traversal/XSS is blocked |
| WEB-12 | Given LLM disabled, then grounded deterministic review remains usable and hypothesis actions are disabled with explanation |
| WEB-13 | Given running review, then reload preserves a truthful job state and review result eventually reaches terminal state |
| WEB-14 | Given 10k runs, then server-side search/filter/pagination work without full client download |
| WEB-15 | Given YAML spec changed by another process, then Save detects conflict rather than silently overwriting |

В репозитории MLReview перед реализацией перенести утверждённые пункты в `docs/specs/web-ui.md` по проектному шаблону и сопоставить с FR/NFR; текущие `WEB-*` — **проектные идентификаторы**, не существующие FR.

## 11. Этапы разработки и граница MVP

**Phase 0 — подтверждение предметных DTO (обязательно):** изучить реальные модели `workspace.py`, `store.py`, `reporting.py`, `api.py`, CLI, config; зафиксировать stable run IDs, review persistence, filtering, links, security constraints. Написать snapshot/contract tests для общей логики. Не строить UI на предположениях о том, какие данные есть в store.

**Phase 1 — P0 Read-only foundation:** `mlreview ui`, HTTP/static, `/overview`, `/sources` (read-only), `/runs`, `/runs/:runId`, `/findings`, `/findings/:id`, `/task` (read-only); browse, deep links, states, health. Приоритет — реальная проверка чтения файлов и MLflow.

**Phase 2 — P0 actions:** `Check contract`, `Run review` и progress/jobs, findings dismissal, journal add/search, task validate/save и source setup. Сверить вывод CLI и Web end-to-end.

**Phase 3 — Analysis UX:** compare run(s), metrics tests, reports and coverage, experiment grouping, playbooks, settings; reviews history только при введённой модели snapshots.

**Phase 4 — Quality and optional intelligence:** LLM contextual ask (grounded facts only), export, proactive watch visualization, keyboard shortcuts, accessibility/performance/security gates.

Выпускать вертикальными slices: сначала "запустить → увидеть прогон → выполнить review → открыть finding", затем расширять. Не переносить UI-логику проверки в frontend. Обязательная проверка проекта: `ruff check`, `ruff format --check`, `pytest` плюс web integration/e2e tests.

## 12. Решения, которые нужно утвердить перед кодом

1. **Сервер:** FastAPI+Uvicorn и React/Vite как предложенный стек или simpler server/frontend; собирать frontend assets в Python wheel.
2. **Review persistence:** существующие `.mlreview/reports` vs новые immutable review snapshots; можно ли реализовать `/reviews/:id` без изменения store.
3. **Source index:** хранить ли отдельный run catalog в SQLite, как обновлять и инвалидировать MLflow/local index; не становиться вторым tracker.
4. **Project resolution:** cwd upward search vs only explicit root; как разрешать несколько workspace.
5. **Task spec path:** способ связывания `task.yaml` и `.mlreview/config.yaml`, optimistic locking и backup.
6. **UI watch control:** показывать только status или действительно запускать управляемый watch worker.
7. **Remote access:** MVP строго localhost; secure remote mode — отдельная спецификация.

## 13. Итоговое целевое поведение

После установки пользователь выполняет `mlreview init` → `mlreview ui`, открывается локальная веб-страница. Он видит подключённые MLflow/local прогоны, выбирает один или несколько runs, проверяет контракт, запускает детерминированное review, видит severity-ranked findings и evidence, узнаёт, какие проверки не были выполнены, сравнивает прогон с явным baseline, фиксирует решение в журнале и возвращается к результатам позже по постоянной ссылке. CLI, Python API, MCP и Web используют одну и ту же предметную логику и source of truth; Web добавляет **читаемость, интерактивную навигацию и визуальный анализ**, а не отдельный набор эвристик.

---

### Использованные источники на момент составления

- `https://github.com/Knohhon/mlreview` — README и текущее позиционирование.
- `https://github.com/Knohhon/mlreview/blob/main/CLAUDE.md` — карта модулей, инварианты, CLI families, порядок разработки.
- `https://github.com/Knohhon/mlreview/blob/main/docs/quickstart.md` — flow init → config → runs → review → findings → metrics → watch → MCP.

**Предел проверки:** чтение публичных страниц репозитория; исходники отдельных Python-модулей не были загружены, поэтому API DTO, реальные schema и сигнатуры нового Web слоя следует проверить в Phase 0. Это проектная спецификация, а не подтверждённый список поддерживаемых Web API.
