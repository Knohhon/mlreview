"""Временный workspace для браузерной приёмки; пользовательские данные не используются."""

from __future__ import annotations

import tempfile
from pathlib import Path

from mlreview.adapters.files_adapter import save_run_files
from mlreview.cli import main
from mlreview.core.store import Store
from mlreview.core.workspace import init_workspace
from tests._factories import BINARY_SPEC_TEXT, binary_tables


def serve() -> None:
    """Запускает локальный UI с синтетическим дефектным прогоном."""
    with tempfile.TemporaryDirectory(prefix="mlreview-e2e-") as directory:
        root = Path(directory)
        workspace, _ = init_workspace(root)
        Store(workspace.store_path).initialize()
        (root / "task.yaml").write_text(BINARY_SPEC_TEXT, encoding="utf-8")
        predictions, examples = binary_tables(n=30, duplicates=15)
        save_run_files(
            root / "runs" / "candidate",
            predictions,
            examples=examples,
            run_id="candidate",
            data_version="v1",
        )
        workspace.config_path.write_text(
            "llm: {provider: none}\nweb:\n  sources:\n    - id: local\n"
            f"      path: {root / 'runs'}\n      name: Local runs\n      kind: files\n",
            encoding="utf-8",
        )
        from mlreview.web.service import Service  # noqa: PLC0415
        from mlreview.web.state import State  # noqa: PLC0415

        service = Service(State(workspace))
        service.scan("local")
        service.close()
        main(["ui", "--project", str(root), "--port", "8769", "--no-browser"])


if __name__ == "__main__":
    serve()
