"""Replacing a file's whole content while other processes may read it."""

from time import sleep
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["REPLACE_ATTEMPTS", "REPLACE_INTERVAL_SECONDS", "replace_path"]

REPLACE_ATTEMPTS: Final = 40
"""How many times a staged file is renamed onto its target before giving up.

Windows refuses to replace a file that another process holds open, and a
reader holds a run's record only for the length of one read, so a refused
replacement is tried again rather than failing the writer.
"""

REPLACE_INTERVAL_SECONDS: Final = 0.05
"""How long a refused replacement waits before it is tried again."""


def replace_path(
    staged: Path,
    target: Path,
    /,
    *,
    attempts: int = REPLACE_ATTEMPTS,
    interval: float = REPLACE_INTERVAL_SECONDS,
) -> None:
    """Rename a staged file onto its target so a reader sees either version.

    A rename the operating system refuses with ``PermissionError``, as
    Windows does while a reader holds the target open, is tried again after
    ``interval`` seconds, at most ``attempts`` times in all.

    Parameters
    ----------
    staged : Path
        The complete new content, beside the target.
    target : Path
        The file to replace.
    attempts : int
        The most renames tried. One rename is always tried.
    interval : float
        The seconds waited between refused renames.

    Raises
    ------
    PermissionError
        If every attempt is refused.
    """
    for _ in range(attempts - 1):
        try:
            staged.replace(target)

        except PermissionError:
            sleep(interval)

        else:
            return

    staged.replace(target)
