"""Executables that stand in for external programs, written as Python.

A prover is run as one executable path. The stand-in for one is a Python
program run by this interpreter through a launcher beside it: a POSIX shell
script that executes the interpreter, or on Windows a command script that
does the same. Each passes its arguments through unchanged.
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def stand_in(path: Path, program: str, /) -> Path:
    """Write a Python program and an executable that runs it, and return that.

    Parameters
    ----------
    path : Path
        Where the executable is written, without a suffix. The program is
        written beside it with the suffix ``.py``.
    program : str
        The program's source.
    """
    source = path.with_name(f"{path.name}.py")
    source.write_text(program, encoding="utf-8", newline="\n")

    if sys.platform == "win32":
        launcher = path.with_name(f"{path.name}.cmd")
        launcher.write_text(
            f'@"{sys.executable}" "{source}" %*\n', encoding="utf-8", newline="\r\n"
        )

        return launcher

    path.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{source}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    path.chmod(0o755)

    return path
