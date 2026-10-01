"""Стражи архитектуры: ядро не зависит от фреймворков обучения и трекеров (NFR-01).

Спецификация: docs/specs/invariant-guards.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "mlreview"

FORBIDDEN = frozenset(
    {
        "catboost",
        "jax",
        "keras",
        "lightgbm",
        "mlflow",
        "sklearn",
        "tensorflow",
        "torch",
        "wandb",
        "xgboost",
    }
)
ALLOWED_DIRS = ("adapters",)


def forbidden_imports(package_root: Path) -> list[str]:
    """Возвращает нарушения вида ``<путь>: <модуль>`` для модулей вне адаптеров."""
    violations = []
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root)
        if relative.parts[0] in ALLOWED_DIRS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            violations.extend(
                f"{relative}: {name}" for name in names if name.split(".")[0] in FORBIDDEN
            )
    return violations


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_core_import_of_framework_is_reported(tmp_path):
    """B1: импорт фреймворка вне адаптеров — нарушение с файлом и модулем."""
    _write(tmp_path, "core/run.py", "import torch.nn\nfrom sklearn.metrics import f1_score\n")
    assert forbidden_imports(tmp_path) == ["core/run.py: torch.nn", "core/run.py: sklearn.metrics"]


def test_adapter_import_of_tracker_is_allowed(tmp_path):
    """B2: адаптерам можно импортировать трекеры."""
    _write(tmp_path, "adapters/mlflow_adapter.py", "import mlflow\n")
    assert forbidden_imports(tmp_path) == []


def test_package_has_no_forbidden_imports():
    """B1 на реальном пакете."""
    assert forbidden_imports(PACKAGE_ROOT) == []
