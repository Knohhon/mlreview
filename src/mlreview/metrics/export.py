"""Экспорт метрик задачи в исполняемый модуль на numpy (FR-M04).

Модуль содержит исходники `mlreview.metrics.kernels` без изменений и функцию `compute`;
сам `mlreview` ему не нужен. Поведение зафиксировано в `docs/specs/metrics.md`.
"""

from __future__ import annotations

import ast
import inspect

from mlreview.core.task_spec import TaskSpec
from mlreview.metrics import kernels
from mlreview.metrics.catalog import CATALOG, param_value, parse_name
from mlreview.metrics.library import spec_metrics

HEADER = '''"""Метрики задачи {task}: сгенерировано командой `mlreview metrics export` (FR-M04).

Основная метрика — {primary}. Зависимость — только numpy.

    from metrics import compute
    compute(labels, predictions)  # -> {{имя метрики: значение}}

labels — метки (бинарная задача — 0/1, multiclass — коды классов 0…C−1 в порядке колонок
prediction:<класс>); predictions — предсказания из контракта прогона (вектор или матрица).
"""

'''


def _kernels_source() -> str:
    """Исходник `kernels` без докстринга модуля."""
    source = inspect.getsource(kernels)
    tree = ast.parse(source)
    start = tree.body[1].lineno if ast.get_docstring(tree) else 1
    return "\n".join(source.splitlines()[start - 1 :]) + "\n"


def _classes_code(spec: TaskSpec) -> str:
    task = spec.task
    if task.type == "classification/binary" and task.output == "probabilities":
        return "(predictions >= THRESHOLD).astype(int)"
    if task.type == "classification/multiclass" and task.output != "labels":
        return "np.argmax(predictions, axis=1)"
    return "predictions.astype(int)"


def export_metrics(spec: TaskSpec) -> str:
    """Текст модуля `metrics.py` для метрик спецификации."""
    items = spec_metrics(spec)
    custom = {metric.name: metric for metric in spec.metrics.custom} if spec.metrics else {}
    threshold = spec.metrics.threshold if spec.metrics else 0.5
    lines = [
        HEADER.format(task=f"{spec.task.type} · {spec.task.output}", primary=items[0].name),
        _kernels_source(),
        "",
        f"TASK = {spec.task.type + ' · ' + spec.task.output!r}",
        f"THRESHOLD = {threshold!r}",
        f"PRIMARY = {items[0].name!r}",
        "",
    ]
    if custom:
        lines += [
            "",
            "def _load_custom(path, function):",
            '    """Пользовательская метрика из её файла."""',
            "    import importlib.util",
            "",
            '    spec = importlib.util.spec_from_file_location("custom_metric", path)',
            "    module = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(module)",
            "    return getattr(module, function)",
            "",
        ]
        for index, metric in enumerate(custom.values()):
            lines.append(f"_CUSTOM_{index} = _load_custom({metric.path!r}, {metric.function!r})")
        lines.append("")
    lines += [
        "",
        "def compute(labels, predictions):",
        '    """Значения метрик задачи: {имя: число}."""',
        "    labels = np.asarray(labels)",
        "    predictions = np.asarray(predictions, dtype=float)",
        f"    classes = {_classes_code(spec)}",
        "    return {",
    ]
    custom_index = {name: index for index, name in enumerate(custom)}
    for item in items:
        if item.name in custom_index:
            call = f"float(_CUSTOM_{custom_index[item.name]}(labels, predictions))"
        else:
            base, raw = parse_name(item.name)
            info = CATALOG[base]
            argument = "classes" if info.kind == "classes" else "predictions"
            param = param_value(info, raw)
            extra = "" if param is None else f", {param!r}"
            call = f"{base}(labels, {argument}{extra})"
        lines.append(f"        {item.name!r}: {call},")
    lines += ["    }", ""]
    return "\n".join(lines)
