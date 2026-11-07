"""Source and runtime facts recorded for production point execution."""

import shutil
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.run._manifest import accelerator_runtime, installed_packages

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ExecutionEnvironment",
    "SourceRevisionError",
    "UnfrozenRevisionError",
    "execution_environment",
    "package_checkout",
    "require_frozen_revision",
    "source_revision",
]


class SourceRevisionError(RuntimeError):
    """The installed source tree has no readable Git revision."""


class UnfrozenRevisionError(RuntimeError):
    """An executing checkout that is not exactly the tree a freeze tags."""


def package_checkout() -> Path:
    """Return the Git checkout the executing run package is imported from.

    Raises
    ------
    SourceRevisionError
        If no ancestor of the installed package holds Git metadata.
    """
    source = Path(__file__).resolve()

    for parent in source.parents:
        if (parent / ".git").exists():
            return parent

    msg = f"no Git repository contains the installed run package at {source}"
    raise SourceRevisionError(msg)


def _git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Return the result of one fixed Git inspection of a checkout.

    Raises
    ------
    SourceRevisionError
        If Git is not installed.
    """
    executable = shutil.which("git")

    if executable is None:
        msg = "git is not installed, so no source revision can be read"
        raise SourceRevisionError(msg)

    return subprocess.run(
        (executable, "-C", str(checkout), *arguments),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )


def source_revision(checkout: Path | None = None, /) -> str:
    """Return the exact Git commit a checkout is at.

    Parameters
    ----------
    checkout : Path | None
        The checkout read; the one the executing package is imported from
        when None.

    Raises
    ------
    SourceRevisionError
        If Git is not installed or the checkout's ``HEAD`` cannot be read.
    """
    root = package_checkout() if checkout is None else checkout
    completed = _git(root, "rev-parse", "HEAD")
    revision = completed.stdout.strip()

    if completed.returncode or not revision:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        msg = f"the source revision under {root} cannot be read: {detail}"
        raise SourceRevisionError(msg)

    return revision


def require_frozen_revision(checkout: Path, freeze_id: str, /) -> str:
    """Return the commit a freeze tags once the executing checkout is that tree.

    Every number an experiment reports traces to the code that produced it, so
    the checkout the run package executes from must be the tree its freeze
    tagged: its ``HEAD`` must be the commit the tag ``freeze_id`` names and no
    tracked file may differ from that commit. Untracked files, such as a run
    store or a results directory inside the checkout, are not compared.

    Parameters
    ----------
    checkout : Path
        The checkout the executing run package is imported from.
    freeze_id : str
        The freeze the experiment runs against, ``freeze/<n>``.

    Returns
    -------
    str
        The tagged commit, which every manifest the experiment writes records.

    Raises
    ------
    UnfrozenRevisionError
        If the checkout has no such tag, its ``HEAD`` is another commit, or a
        tracked file differs from ``HEAD``.
    SourceRevisionError
        If Git is not installed.
    """
    tagged = _git(
        checkout,
        "rev-parse",
        "--verify",
        "--quiet",
        f"refs/tags/{freeze_id}^{{commit}}",
    )

    if tagged.returncode or not tagged.stdout.strip():
        msg = f"the executing checkout at {checkout} has no tag {freeze_id}"
        raise UnfrozenRevisionError(msg)

    commit = tagged.stdout.strip()
    head = _git(checkout, "rev-parse", "HEAD").stdout.strip()

    if head != commit:
        msg = (
            f"the executing checkout at {checkout} is at {head or 'no commit'}, "
            f"not the commit {commit} that {freeze_id} tags"
        )
        raise UnfrozenRevisionError(msg)

    status = _git(checkout, "status", "--porcelain", "--untracked-files=no")

    if status.returncode or status.stdout.strip():
        detail = status.stdout.strip() or status.stderr.strip()
        msg = (
            f"the executing checkout at {checkout} differs from the tree "
            f"{freeze_id} tags: {detail}"
        )
        raise UnfrozenRevisionError(msg)

    return commit


@final
class ExecutionEnvironment(FrozenValue):
    """Immutable package, source and accelerator facts for one executor."""

    _noun = "execution environment"

    __slots__ = ("_accelerator", "_git_revision", "_packages")

    def __init__(
        self,
        packages: Mapping[str, str],
        git_revision: str,
        accelerator: str,
    ) -> None:
        object.__setattr__(
            self,
            "_packages",
            MappingProxyType(
                {
                    ensure_name("a package name", name): ensure_name(
                        "a package version", value
                    )
                    for name, value in sorted(packages.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "_git_revision",
            ensure_name("a source revision", git_revision),
        )
        object.__setattr__(
            self,
            "_accelerator",
            ensure_name("an accelerator runtime", accelerator),
        )

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        del name, value
        msg = "an execution environment is immutable"
        raise AttributeError(msg)

    @property
    def packages(self) -> Mapping[str, str]:
        """Return installed distribution versions by name."""
        return self._packages

    @property
    def git_revision(self) -> str:
        """Return the exact source revision."""
        return self._git_revision

    @property
    def accelerator(self) -> str:
        """Return the installed accelerator runtime."""
        return self._accelerator


def execution_environment() -> ExecutionEnvironment:
    """Return current package, source and accelerator runtime facts."""
    return ExecutionEnvironment(
        installed_packages(),
        source_revision(),
        accelerator_runtime(),
    )
