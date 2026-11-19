---
name: mlreview-analyzer
description: Добавление и изменение детерминированных анализаторов MLReview и проверок плейбуков моделей, включая факты, пропуски и бенчмарк. Не используется для анализа пользовательского датасета.
---

# Анализаторы и плейбуки MLReview

Работать по mlreview-develop: сначала профильная спецификация и тесты.
Перед реализацией выбрать существующую точку расширения:

- Общая проверка: src/mlreview/analyzers/registry.py (`Analyzer`, `ReviewContext`),
  builtin.py и docs/specs/analyzers.md. Группы: data, segments, metrics, model,
  curves, process. Возвращаемый тип — `CheckResults`.
- Проверка семейства модели: docs/specs/playbooks.md,
  src/mlreview/playbooks/context.py и соседний плейбук. Формат — каталог с
  playbook.yaml и необязательным checks.py с `run(ctx: PlaybookContext)`.
  Детект семейства/алиасы параметров — данные манифеста; не импортировать модель.
  Метаданные семейства и параметры нормализуются в resolve.py.

Находки создавать через `Finding`: устойчивый check, requirement, severity,
facts и при необходимости evidence/suggestion. Числа округлять до 6 знаков,
сообщения выводить из facts. Фиксировать RNG, не использовать скрытую сеть или LLM.
Без необходимых полей возвращать `Skipped` с причиной; отсутствие доказательств
не выдавать за отсутствие дефекта. При новом типе находки проверить fingerprint
и magnitude в core/findings.py и работу отклонений/дедупликации.

Проверить чистый пример, внедрённый дефект, граничный случай, отсутствующие данные
и воспроизводимость. Для плейбуков подменять loader.builtin_playbooks как в
tests/test_playbooks.py. При изменении общего отчёта проверить CLI/API/MCP и
LLM-контекст: исходные признаки и гиперпараметры не добавлять в исходящий запрос.

Если добавляется принимаемый дефект, синхронизировать benchmark/projects.py,
benchmark/run.py и docs/specs/benchmark.md. Прогнать профильные тесты,
tests/test_architecture.py, tests/test_offline.py и приёмочный бенчмарк:
учитывать и найденные дефекты, и ложные critical на чистых проектах.
Завершить проверками mlreview-check и сверкой документации.
