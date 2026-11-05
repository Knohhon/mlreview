"""Семейство модели по имени класса и нормализация гиперпараметров (FR-D11, FR-C07).

Распознавание — по шаблонам `detect` манифестов, гиперпараметры — по алиасам `params`:
так одни проверки работают для sklearn, LightGBM, XGBoost и CatBoost без их импорта (NFR-01).
Поведение зафиксировано в `docs/specs/playbooks.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mlreview.core.model_info import ModelInfo
from mlreview.playbooks import loader
from mlreview.playbooks.loader import Playbook

_NULLS = frozenset({"none", "null"})
_BOOLS = {"true": True, "false": False}


def resolve_family(
    model: ModelInfo | None, playbooks: Iterable[Playbook] | None = None
) -> str | None:
    """Явное семейство или первое распознанное по короткому имени класса (B3–B5)."""
    if model is None:
        return None
    if model.family is not None:
        return model.family
    if model.short_name is None:
        return None
    for playbook in loader.builtin_playbooks() if playbooks is None else playbooks:
        families = playbook.families_of(model.short_name)
        if families:
            return families[0]
    return None


def flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Плоский конфиг: вложенные словари — ключи через точку, значения как есть."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def parse_value(value: Any) -> Any:
    """Строки MLflow → значения: `None`/`null` → None, `True` → True, числа → числа (B7)."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLS:
        return None
    if text.lower() in _BOOLS:
        return _BOOLS[text.lower()]
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return value


def _matches(key: str, alias: str) -> bool:
    return key == alias or key.endswith((f"__{alias}", f".{alias}"))


def alias_keys(config: Mapping[str, Any], playbook: Playbook) -> set[str]:
    """Ключи плоского конфига, которые распознаются алиасами гиперпараметров плейбука."""
    aliases = [alias for spec in playbook.manifest.params.values() for alias in spec.aliases]
    return {key for key in flatten(config) if any(_matches(key, alias) for alias in aliases)}


def model_params(config: Mapping[str, Any], playbook: Playbook) -> dict[str, Any]:
    """Гиперпараметры под нормализованными именами плейбука (B6–B8).

    Точное имя алиаса важнее шага Pipeline (`<шаг>__<алиас>`) и вложенного ключа; среди
    них — первый по имени ключа. Значение из `unbounded` — `None` («без ограничения»).
    """
    flat = flatten(config)
    params = {}
    for name, spec in playbook.manifest.params.items():
        exact = [alias for alias in spec.aliases if alias in flat]
        if exact:
            key = exact[0]
        else:
            found = sorted(key for key in flat for alias in spec.aliases if _matches(key, alias))
            if not found:
                continue
            key = found[0]
        value = parse_value(flat[key])
        params[name] = None if value in spec.unbounded and value is not True else value
    return params
