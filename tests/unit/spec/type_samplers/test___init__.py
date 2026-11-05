"""Import tests for type-sampler declarations."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

_IMPORTS = """
import lofi.spec.type_samplers.recursive
import lofi.spec.type_samplers.exhaustive
import lofi.spec.type_samplers.distribution
"""


def test_type_sampler_declarations_import_without_runtime_layers(
    tmp_path: Path,
) -> None:
    probe = (
        _IMPORTS
        + """
import json
import sys

names = (
    "lofi.data",
    "lofi.nn",
    "lofi.run",
    "lofi.train",
    "torch",
    "panproto",
)
print(json.dumps({name: name in sys.modules for name in names}))
"""
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    imported = cast("dict[str, JsonValue]", json.loads(completed.stdout))

    assert not any(imported.values()), imported
