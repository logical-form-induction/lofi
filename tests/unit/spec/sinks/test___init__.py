"""Sink declaration import tests."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def test_sink_declarations_import_without_runtime_or_client_modules(
    tmp_path: Path,
) -> None:
    probe = """
import json
import sys

import lofi.spec.sinks.mlflow
import lofi.spec.sinks.trackio
import lofi.spec.sinks.wandb

names = (
    "lofi.run",
    "lofi.run._mlflow",
    "lofi.run._trackio",
    "lofi.run._wandb",
    "torch",
    "lairs",
    "panproto",
    "mlflow",
    "trackio",
    "wandb",
)
print(json.dumps({name: name in sys.modules for name in names}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    imported = cast("dict[str, JsonValue]", json.loads(completed.stdout))

    assert not any(imported.values()), imported
