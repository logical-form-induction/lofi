"""The process environment of a sandboxed prover."""

import os

__all__ = ["sandbox_environment"]


def sandbox_environment(directory: str, /) -> dict[str, str]:
    """Return the environment of a prover process confined to a directory.

    The process sees the C locale, the platform's default search path, and the
    directory as its home and its temporary directory. POSIX names the
    temporary directory ``TMPDIR`` and Windows ``TEMP`` and ``TMP``, so all
    three are set. A Windows process cannot start without ``SYSTEMROOT``, so
    it is passed through wherever the parent environment defines it.

    Parameters
    ----------
    directory : str
        The sandbox directory the prover runs in.
    """
    environment = {
        "HOME": directory,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "TEMP": directory,
        "TMP": directory,
        "TMPDIR": directory,
    }
    system_root = os.environ.get("SYSTEMROOT")

    if system_root is not None:
        environment["SYSTEMROOT"] = system_root

    return environment
