import subprocess
import sys
from pathlib import Path

import pytest

# packages that must import without numerical dependencies
SYMBOLIC_CORE = (
    "lofi.terms",
    "lofi.judge",
    "lofi.lexicon",
    "lofi.deduce",
    "lofi.interpret",
    "lofi.typecheck",
    "lofi.lf",
    "lofi.formalism",
    "lofi.grammars",
)

# dependencies prohibited from the symbolic core
POISONED = ("torch", "numpy", "quivers", "lairs", "bead")

# use the repository checkout instead of an installed distribution
TREE = Path(__file__).resolve().parents[2]

TIMEOUT_SECONDS = 120.0


def import_under_poison(package: str) -> subprocess.CompletedProcess[str]:
    """Import a package in a fresh interpreter with numerical libraries rejected.

    Parameters
    ----------
    package
        The package to import.

    Returns
    -------
    subprocess.CompletedProcess
        The completed interpreter process, including captured output.
    """
    return subprocess.run(
        [sys.executable, "-m", "tests.independence._import_under_poison"],
        input="\n".join((package, *POISONED)),
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=TREE,
        timeout=TIMEOUT_SECONDS,
    )


@pytest.mark.parametrize("package", SYMBOLIC_CORE)
def test_package_imports_with_the_poisoned_libraries_refused(package: str) -> None:
    result = import_under_poison(package)

    assert result.returncode == 0, (
        f"{package} failed to import with {', '.join(POISONED)} refused:\n"
        f"{result.stderr}"
    )


def test_a_package_that_does_import_torch_fails_under_the_same_poison() -> None:
    result = import_under_poison("lofi.nn")

    assert result.returncode != 0, (
        "lofi.nn imported with torch refused, so the poison reaches nothing and "
        "the packages above prove nothing"
    )

    named = [library for library in POISONED if library in result.stderr]

    assert named, (
        f"lofi.nn failed for a reason other than a refused library:\n{result.stderr}"
    )
