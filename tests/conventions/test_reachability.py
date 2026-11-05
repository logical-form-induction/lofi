"""Every module and spec root is reachable from a supported package boundary.

A module is reachable when the generated API reference exposes it, or when a
console script, a ``__main__`` module or a plugin entry point loads it,
directly or through the modules those import.
A spec root, a declaration class deriving directly from a tagged union, is
reachable when a document can hold one: a class some reachable module
composes a document against, or validates one from text, names it through
its fields, directly or through other declaration classes. An unreachable
component fails the suite and must be wired into the current package or removed.
"""

from typing import TYPE_CHECKING, Final

from tests.structure import (
    TREE,
    Package,
    reachable_declarations,
    reachable_modules,
    source_modules,
    spec_roots,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def unreachable(root: Path) -> dict[str, set[str]]:
    """Return the modules and spec roots of a tree nothing configurable reaches.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    dict[str, set[str]]
        The unreachable dotted module names under ``modules`` and the
        unreachable spec roots of the reachable modules, as ``module:class``,
        under ``spec-roots``.
    """
    modules = reachable_modules(root)
    package = Package(root)
    declarations = reachable_declarations(package)

    return {
        "modules": set(source_modules(root)) - modules,
        "spec-roots": {
            symbol.describe()
            for symbol in spec_roots(package)
            if symbol.module in modules and symbol not in declarations
        },
    }


FOUND: Final = unreachable(TREE)


def test_every_module_is_reachable() -> None:
    assert not FOUND["modules"], (
        "no supported package API, entry point or configuration reaches these "
        "modules; wire them, or delete them:\n" + "\n".join(sorted(FOUND["modules"]))
    )


def test_every_spec_root_is_reachable() -> None:
    assert not FOUND["spec-roots"], (
        "no document can hold these spec roots; give a document a field for "
        "them, or delete them:\n" + "\n".join(sorted(FOUND["spec-roots"]))
    )


def write_tree(root: Path, files: Mapping[str, str]) -> None:
    """Write a miniature repository from relative paths and their text."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


MINIATURE: Final = {
    "pyproject.toml": (
        '[project]\nname = "lofi"\n'
        '[project.scripts]\nlofi = "lofi.cli._main:main"\n'
        '[project.entry-points."lofi.metric"]\n'
        'scored = "lofi.metrics._scored:PLUGIN"\n'
    ),
    "lofi/__init__.py": "",
    "lofi/cli/__init__.py": "",
    "lofi/cli/__main__.py": "from lofi.cli._run import run\n",
    "lofi/cli/_run.py": "def run():\n    return 0\n",
    "lofi/cli/_main.py": (
        "import didactic.api as dx\n"
        "from lofi.cli._spec import RunSpec\n"
        "def main(loader):\n    return loader.compose('run.yaml', schema=RunSpec)\n"
    ),
    "lofi/cli/_spec.py": (
        "import didactic.api as dx\n"
        "class EncoderSpec(dx.TaggedUnion, discriminator='kind'):\n    pass\n"
        "class RnnSpec(EncoderSpec):\n    width: int = 1\n"
        "class SinkSpec(dx.TaggedUnion, discriminator='kind'):\n    pass\n"
        "class Section(dx.Model):\n    encoder: 'EncoderSpec | None' = None\n"
        "class RunSpec(dx.Model):\n    model: Section\n"
    ),
    "lofi/metrics/__init__.py": "from lofi.metrics._shared import ROWS\n",
    "lofi/metrics/_scored.py": "from ._helper import PLUGIN\n",
    "lofi/metrics/_helper.py": "PLUGIN = 1\n",
    "lofi/metrics/_shared.py": "ROWS = ()\n",
    "lofi/tools/__init__.py": "",
    "lofi/tools/_driven.py": "",
    "scripts/drive.py": "from lofi.tools import _driven\n",
}


def test_a_module_reached_only_by_a_script_or_nothing_is_unreachable(
    tmp_path: Path,
) -> None:
    write_tree(tmp_path, {**MINIATURE, "lofi/metrics/_orphan.py": "ROWS = ()\n"})

    found = unreachable(tmp_path)

    assert found["modules"] == {
        "lofi.metrics._orphan",
        "lofi.tools",
        "lofi.tools._driven",
    }
    assert {"lofi.cli._run", "lofi.cli._spec", "lofi.metrics._helper"} <= (
        reachable_modules(tmp_path)
    )


def test_a_module_loaded_by_a_literal_import_module_call_is_reached(
    tmp_path: Path,
) -> None:
    write_tree(
        tmp_path,
        {
            **MINIATURE,
            "lofi/metrics/_helper.py": (
                "from importlib import import_module\n"
                "PLUGIN = 1\n"
                "def read():\n"
                "    return import_module('lofi.metrics._reading')\n"
                "def named(name):\n"
                "    return import_module(name)\n"
            ),
            "lofi/metrics/_reading.py": "from lofi.metrics._deep import ROWS\n",
            "lofi/metrics/_deep.py": "ROWS = ()\n",
            "lofi/metrics/_unnamed.py": "ROWS = ()\n",
        },
    )

    reached = reachable_modules(tmp_path)

    assert {"lofi.metrics._reading", "lofi.metrics._deep"} <= reached
    assert "lofi.metrics._unnamed" not in reached


def test_a_documented_public_module_is_a_supported_package_api(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            **MINIATURE,
            "lofi/tools/public.py": "VALUE = 1\n",
            "lofi/tools/_private.py": "VALUE = 2\n",
            "docs/reference/tools.md": (
                "::: lofi.tools.public\n\n::: lofi.tools._private\n"
            ),
        },
    )

    reached = reachable_modules(tmp_path)

    assert "lofi.tools.public" in reached
    assert "lofi.tools._private" not in reached


def test_a_spec_root_no_document_holds_is_unreachable(tmp_path: Path) -> None:
    write_tree(tmp_path, MINIATURE)

    assert unreachable(tmp_path)["spec-roots"] == {"lofi.cli._spec:SinkSpec"}

    write_tree(
        tmp_path,
        {
            "lofi/cli/_spec.py": MINIATURE["lofi/cli/_spec.py"].replace(
                "    model: Section\n",
                "    model: Section\n    sinks: tuple[SinkSpec, ...] = ()\n",
            )
        },
    )

    assert unreachable(tmp_path)["spec-roots"] == set()


def test_a_document_type_alias_reaches_nested_union_families(tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            **MINIATURE,
            "lofi/cli/_spec.py": (
                "import didactic.api as dx\n"
                "class Family(dx.TaggedUnion, discriminator='family'):\n    pass\n"
                "class Ordinal(Family):\n    family: str = 'ordinal'\n"
                "class Predictor(dx.TaggedUnion, discriminator='kind'):\n    pass\n"
                "class Score(Predictor):\n    kind: str = 'score'\n"
                "class ResponseSpec(dx.Model):\n"
                "    family: Family\n"
                "    predictor: tuple[Predictor, ...]\n"
                "class BoundedResponseSpec(dx.Model):\n"
                "    predictor: tuple[Predictor, ...]\n"
                "type ResponseDeclaration = (\n"
                "    ResponseSpec | BoundedResponseSpec | ResponseDeclaration\n"
                ")\n"
                "class Section(dx.Model):\n    response: ResponseDeclaration\n"
                "class RunSpec(dx.Model):\n    model: Section\n"
            ),
        },
    )

    assert unreachable(tmp_path)["spec-roots"] == set()
