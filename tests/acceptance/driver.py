"""Drive LoFI from configuration documents through its public routes only.

A case names a document under ``documents/`` and reaches the package in one of
three ways: the composition API (``lofi.run.compose``), the planning API that
checks a composed run (``lofi.run.plan_training``), or the installed ``lofi``
console script. No case constructs a package value directly.
"""

import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, cast

from lofi.run import Selection, compose, plan_training, resolved_document
from lofi.spec import (
    RunContext,
    config_search_path,
    run_context,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi._types import JsonValue
    from lofi.data.synthetic import InterpretationFunction
    from lofi.data.synthetic.spec.corpus import CheckedSyntheticCorpus
    from lofi.spec import RunSpec

DOCUMENTS: Final = Path(__file__).resolve().parent / "documents"
"""The directory every case document and its fragments live in."""

EXPERIMENT_SEED: Final = 5
"""The root seed every tiny point derives its run seed from."""

RUN_SEED: Final = 0
"""The single run seed used by every unswept acceptance point."""

type Row = dict[str, JsonValue]


class PointFailedError(AssertionError):
    """A composed point that the console script did not complete."""


class TrainedPoint(NamedTuple):
    """What one completed point left in its run directory."""

    directory: Path
    point_id: str
    resolved: dict[str, JsonValue]
    rows: tuple[Row, ...]


def selection(document: str, *overrides: str, project: Path = DOCUMENTS) -> Selection:
    """Return the selection one document and its overrides describe."""
    return Selection(
        DOCUMENTS / document, None, overrides, (), config_search_path(project)
    )


def composed(document: str, *overrides: str, project: Path = DOCUMENTS) -> RunSpec:
    """Compose and validate a document through the composition API."""
    with run_context(RunContext(project, RUN_SEED)):
        return compose(selection(document, *overrides, project=project))


def resolved(
    document: str, *overrides: str, project: Path = DOCUMENTS
) -> dict[str, JsonValue]:
    """Return the resolved document a composed run dumps to."""
    return resolved_document(composed(document, *overrides, project=project))


def interpretation(document: str, *overrides: str) -> InterpretationFunction:
    """Return the interpretation a planned run's synthetic corpus is drawn under."""
    planned = plan_training(composed(document, *overrides))
    built = cast(
        "CheckedSyntheticCorpus", planned.checked_dataset("train")
    ).source_interpretation

    if built is None:
        msg = f"{document} declares no interpretation"
        raise AssertionError(msg)

    return built


def lofi() -> Callable[[Sequence[str]], int]:
    """Return the installed ``lofi`` console script."""
    (script,) = entry_points(group="console_scripts", name="lofi")

    return cast("Callable[[Sequence[str]], int]", script.load())


def train(document: str, root: Path, *overrides: str) -> TrainedPoint:
    """Compose a document, train it with the console script and read its outputs.

    Composition runs first so that a refused document raises the refusal
    itself; a point the console script does not complete raises
    ``PointFailedError`` carrying the run's own diagnostic.
    """
    composed(document, *overrides)
    argv = [
        "run",
        "--config",
        str(DOCUMENTS / document),
        "--experiment-seed",
        str(EXPERIMENT_SEED),
        "--root",
        str(root),
    ]

    for override in overrides:
        argv.extend(("--set", override))

    status = lofi()(argv)
    directories = sorted((root / "runs").glob("[0-9a-f]*"))

    if status != 0 or len(directories) != 1:
        diagnostics = [
            json.loads(failure.read_text(encoding="utf-8"))["diagnostic"]
            for directory in directories
            if (failure := directory / "failure.json").is_file()
        ]
        msg = f"{document} did not train (exit {status}): {diagnostics}"
        raise PointFailedError(msg)

    (directory,) = directories
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    if manifest["status"] != "completed":
        msg = f"{document} ended {manifest['status']}"
        raise PointFailedError(msg)

    rows = json.loads((directory / "rows.json").read_text(encoding="utf-8"))["rows"]

    return TrainedPoint(
        directory,
        directory.name,
        json.loads((directory / "config.resolved.json").read_text(encoding="utf-8")),
        tuple(rows),
    )


def leaf(document: JsonValue, path: str) -> JsonValue:
    """Return the value at a dotted path, a numeric segment indexing a list."""
    held = document

    for key in path.split("."):
        if isinstance(held, list):
            held = held[int(key)]
        elif isinstance(held, dict):
            held = held[key]
        else:
            msg = f"{path} passes through the scalar {held!r} at {key!r}"
            raise TypeError(msg)

    return held


def keys(document: JsonValue, path: str) -> frozenset[str]:
    """Return the keys of the mapping at a dotted path."""
    held = leaf(document, path)

    if not isinstance(held, dict):
        msg = f"{path} holds {held!r}, not a mapping"
        raise TypeError(msg)

    return frozenset(held)
