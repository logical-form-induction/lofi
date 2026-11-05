import ast
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

# use the repository checkout instead of an installed distribution
TREE: Final = Path(__file__).resolve().parents[2]

SOURCE_ROOT: Final = TREE / "lofi"
MIRROR_ROOT: Final = TREE / "tests" / "unit"

PREFIX: Final = "test_"


def declares_nothing(module: Path) -> bool:
    """Return whether a module contains only a docstring and imports.

    Parameters
    ----------
    module
        The source module to inspect.

    Returns
    -------
    bool
        Whether the module declares no executable members.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"))
    runnable = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))
    ]

    return not runnable


def listing(root: Path) -> Sequence[Path]:
    """Return Python files below a root in path order.

    Parameters
    ----------
    root
        The directory to search.

    Returns
    -------
    collections.abc.Sequence
        Sorted paths excluding bytecode caches.
    """
    modules = sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and path.name != "conftest.py"
        and path.is_file()
    )

    if not modules:
        pytest.fail(f"no Python module found under {root}, so nothing was compared")

    return modules


def compiled_stubs() -> Sequence[Path]:
    """Return the stubs below the source root in path order.

    A compiled extension module has no Python source, so its stub stands for it.

    Returns
    -------
    collections.abc.Sequence
        Sorted stub paths.
    """
    return sorted(SOURCE_ROOT.rglob("*.pyi"))


def mirror_of(module: Path) -> Path:
    """Return the test path corresponding to a source module or stub.

    Parameters
    ----------
    module
        A path below the source root.

    Returns
    -------
    pathlib.Path
        The corresponding mirror path, whether or not it exists.
    """
    relative = module.relative_to(SOURCE_ROOT)

    return MIRROR_ROOT / relative.parent / f"{PREFIX}{relative.stem}.py"


def source_of(mirror: Path) -> Path:
    """Return the source path corresponding to a mirror module.

    Parameters
    ----------
    mirror
        A path below the mirror root.

    Returns
    -------
    pathlib.Path
        The corresponding source path, whether or not it exists.
    """
    relative = mirror.relative_to(MIRROR_ROOT)

    return SOURCE_ROOT / relative.parent / relative.name.removeprefix(PREFIX)


def report(headline: str, paths: Sequence[Path]) -> str:
    """Return a failure report listing paths relative to the repository.

    Parameters
    ----------
    headline
        The description shared by the paths.
    paths
        The paths to list.

    Returns
    -------
    str
        The headline followed by one relative path per line.
    """
    named = "\n".join(f"  {path.relative_to(TREE)}" for path in paths)

    return f"{headline}:\n{named}"


def test_every_source_module_has_a_mirrored_test_module() -> None:
    unmirrored = [
        module
        for module in [*listing(SOURCE_ROOT), *compiled_stubs()]
        if not mirror_of(module).exists() and not declares_nothing(module)
    ]

    if unmirrored:
        pytest.fail(report("source modules with no mirrored test module", unmirrored))


def test_every_mirrored_test_module_still_has_its_source_module() -> None:
    orphaned = [
        mirror
        for mirror in listing(MIRROR_ROOT)
        if not source_of(mirror).exists()
        and not source_of(mirror).with_suffix(".pyi").exists()
    ]

    if orphaned:
        pytest.fail(report("test modules mirroring no source module", orphaned))


def test_no_mirrored_test_module_is_blank() -> None:
    blank = [
        mirror
        for mirror in listing(MIRROR_ROOT)
        if not mirror.read_text(encoding="utf-8").strip()
    ]

    if blank:
        pytest.fail(report("mirrored test modules holding nothing", blank))
