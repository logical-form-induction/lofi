"""Parser declaration import tests."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue

_IMPORTS = """
import lofi.spec.parsers.chart
import lofi.spec.parsers.flat
import lofi.spec.parsers.reduction
import lofi.spec.parsers.tagged
"""

_DISCOVERY = """
import json

from lofi.spec import ParserSpec
from lofi.spec._plugins import load_plugins

plugins = load_plugins()
declared = ParserSpec.model_validate({
    "kind": "flat",
    "width": 13,
    "features": ["left", "right", "bilinear"],
    "projection_initialization": "identity",
})
print(json.dumps({
    "plugins": plugins,
    "kind": declared.kind,
    "width": declared.width,
    "features": declared.features,
    "projection_initialization": declared.projection_initialization,
}))
"""


def test_parsers_declarations_import_without_the_neural_layer(tmp_path: Path) -> None:
    probe = (
        _IMPORTS
        + """
import json
import sys

names = (
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


def test_plugin_discovery_loads_the_flat_declaration_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _DISCOVERY],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    discovered = cast("dict[str, JsonValue]", json.loads(completed.stdout))
    plugins = discovered["plugins"]

    assert isinstance(plugins, list)
    assert all(isinstance(plugin, str) for plugin in plugins)
    assert "flat" in plugins
    assert set(discovered) == {
        "plugins",
        "kind",
        "width",
        "features",
        "projection_initialization",
    }
    assert discovered["kind"] == "flat"
    assert discovered["width"] == 13
    assert discovered["features"] == ["left", "right", "bilinear"]
    assert discovered["projection_initialization"] == "identity"
