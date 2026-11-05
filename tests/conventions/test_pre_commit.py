import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml

if TYPE_CHECKING:
    from lofi._types import JsonValue

TREE: Final = Path(__file__).resolve().parents[2]


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)

    return value


def _items(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)

    return value


def _text(value: JsonValue) -> str:
    assert isinstance(value, str)

    return value


def hooks_config() -> dict[str, JsonValue]:
    loaded = yaml.safe_load(
        (TREE / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )

    return _mapping(loaded)


def locked_version(package: str) -> str:
    lock = tomllib.loads((TREE / "uv.lock").read_text(encoding="utf-8"))
    versions = [
        _text(entry["version"]) for entry in lock["package"] if entry["name"] == package
    ]

    assert len(versions) == 1

    return versions[0]


def test_the_ruff_hooks_run_the_ruff_the_lock_pins() -> None:
    repos = [_mapping(repo) for repo in _items(hooks_config()["repos"])]
    ruff = [repo for repo in repos if _text(repo["repo"]).endswith("/ruff-pre-commit")]

    assert len(ruff) == 1
    assert _text(ruff[0]["rev"]) == f"v{locked_version('ruff')}"


def test_the_hooks_run_under_the_interpreter_the_tree_pins() -> None:
    pinned = (TREE / ".python-version").read_text(encoding="utf-8").strip()
    versions = _mapping(hooks_config()["default_language_version"])

    assert _text(versions["python"]) == f"python{pinned}"
