"""Адаптер локальных файлов: каталог прогона → контракт прогона (FR-C03, FR-C02).

Таблицы — `predictions|examples|curves.parquet|csv`, конфиг — `config.yaml|json`, id прогона
и версия данных — `run.yaml|json`. Поведение зафиксировано в `docs/specs/files-adapter.md`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlreview.adapters.errors import AdapterError
from mlreview.core.run_contract import RunContract, build_run_contract
from mlreview.core.task_spec import TaskSpec

TABLE_FORMATS = ("parquet", "csv")
MAPPING_FORMATS = ("yaml", "json")
META_FILE = "run.yaml"


class FilesAdapterError(AdapterError):
    """Каталог прогона не прочитан или не записан."""


def _single(directory: Path, stem: str, formats: tuple[str, ...]) -> Path | None:
    """Файл `<stem>.<формат>`; ни одного — `None`, несколько форматов — ошибка (E3)."""
    found = [directory / f"{stem}.{fmt}" for fmt in formats]
    found = [path for path in found if path.is_file()]
    if len(found) > 1:
        names = ", ".join(str(path) for path in found)
        raise FilesAdapterError(f"{stem} записан в нескольких форматах ({names}); оставьте один")
    return found[0] if found else None


def _read_table(path: Path) -> pd.DataFrame:
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path, dtype={"example_id": str})
        return pd.read_parquet(path)
    except Exception as exc:  # читатели parquet и CSV бросают разные типы ошибок
        raise FilesAdapterError(f"не удалось прочитать {path}: {exc}") from exc


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        raise FilesAdapterError(f"не удалось прочитать {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise FilesAdapterError(f"{path}: ожидается словарь, а не {type(data).__name__}")
    return dict(data)


def load_files_contract(spec: TaskSpec, directory: str | Path) -> RunContract:
    """Читает каталог прогона и собирает контракт (FR-C02) против спецификации задачи.

    Ошибки чтения — `FilesAdapterError`; ошибки содержимого — `RunContractError` из ядра.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FilesAdapterError(f"каталог прогона не найден: {directory}")
    predictions_path = _single(directory, "predictions", TABLE_FORMATS)
    if predictions_path is None:
        expected = " или ".join(f"predictions.{fmt}" for fmt in TABLE_FORMATS)
        raise FilesAdapterError(f"{directory}: нет файла предсказаний {expected}")
    tables = {name: _single(directory, name, TABLE_FORMATS) for name in ("examples", "curves")}
    config_path = _single(directory, "config", MAPPING_FORMATS)
    meta_path = _single(directory, "run", MAPPING_FORMATS)
    meta = _read_mapping(meta_path) if meta_path else {}
    run_id = meta.get("run_id")
    return build_run_contract(
        spec,
        run_id=directory.resolve().name if run_id is None else str(run_id),
        predictions=_read_table(predictions_path),
        examples=_read_table(tables["examples"]) if tables["examples"] else None,
        config=_read_mapping(config_path) if config_path else None,
        curves=_read_table(tables["curves"]) if tables["curves"] else None,
        data_version=meta.get("data_version"),
    )


def save_run_files(  # noqa: PLR0913 — поля контракта прогона
    directory: str | Path,
    predictions: pd.DataFrame,
    *,
    examples: pd.DataFrame | None = None,
    curves: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    data_version: str | None = None,
    run_id: str | None = None,
) -> Path:
    """Сохраняет таблицы контракта в каталог прогона; `run.yaml` пишется последним (B4).

    Вызывается в конце скрипта обучения; появление `run.yaml` — признак «прогон завершён»
    для проактивного режима (FR-R04).
    """
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in (
            ("predictions", predictions),
            ("examples", examples),
            ("curves", curves),
        ):
            if frame is not None:
                frame.to_parquet(directory / f"{name}.parquet", index=False)
        if config is not None:
            text = yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=True)
            (directory / "config.yaml").write_text(text, encoding="utf-8")
        meta = {"run_id": run_id or directory.resolve().name}
        if data_version is not None:
            meta["data_version"] = str(data_version)
        text = yaml.safe_dump(meta, allow_unicode=True, sort_keys=True)
        (directory / META_FILE).write_text(text, encoding="utf-8")
    except OSError as exc:
        raise FilesAdapterError(f"не удалось записать прогон в {directory}: {exc}") from exc
    return directory


def is_finished(directory: Path) -> bool:
    """Каталог — завершённый прогон: есть `run.yaml` или `run.json` (FR-R04)."""
    return any((directory / f"run.{fmt}").is_file() for fmt in MAPPING_FORMATS)
