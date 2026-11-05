"""Corpus-path resolution against a data root."""

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import final, override

__all__ = ["DataRoot", "DataRootError", "PathDeclarationError"]


class DataRootError(ValueError):
    """An error for an invalid data root.

    Parameters
    ----------
    directory : Path
        The directory offered as a root.
    reason : str
        What disqualifies it, as a noun phrase.
    """

    def __init__(self, directory: Path, reason: str) -> None:
        super().__init__(f"{directory} cannot serve as a data root: it is {reason}")

        #: The directory offered as a root.
        self.directory = directory

        #: The reason the directory is invalid.
        self.reason = reason


class PathDeclarationError(ValueError):
    """An error for an invalid declared path.

    Parameters
    ----------
    key : str
        The configuration key the text was declared under.
    declared : str
        The path text the configuration gave.
    reason : str
        What disqualifies the text, as a noun phrase.
    """

    def __init__(self, key: str, declared: str, reason: str) -> None:
        super().__init__(
            f"{key}: {declared!r} does not resolve against a data root: it is {reason}"
        )

        #: The configuration key the text was declared under.
        self.key = key

        #: The path text the configuration gave.
        self.declared = declared

        #: The reason the path is invalid.
        self.reason = reason


def _components(declared: str, key: str) -> tuple[str, ...]:
    """Return the components of a declared path.

    Parameters
    ----------
    declared : str
        The path text the configuration gave.
    key : str
        The configuration key it was declared under, which names the offender
        when the text is refused.

    Returns
    -------
    tuple[str, ...]
        The ordinary name components, in order, with any ``.`` component and
        any repeated or trailing separator dropped.

    Raises
    ------
    PathDeclarationError
        If the text is empty, spelled with a backslash, absolute,
        drive-qualified, home-relative, or spelled with a ``..`` component.
    """
    text = declared.strip()

    if not text:
        raise PathDeclarationError(key, declared, "empty")

    if "\\" in text:
        raise PathDeclarationError(key, declared, "spelled with a backslash")

    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).drive:
        raise PathDeclarationError(key, declared, "an absolute path")

    if text.startswith("~"):
        raise PathDeclarationError(key, declared, "a home-relative path")

    parts = PurePosixPath(text).parts

    if ".." in parts:
        raise PathDeclarationError(key, declared, "spelled with a '..' component")

    return parts


@final
class DataRoot:
    """Resolves corpus paths against one directory.

    Parameters
    ----------
    directory : Path
        An absolute directory naming no ``..`` component. It need not exist:
        a missing file is reported by the reader that opens it, which names the
        whole path.

    Raises
    ------
    DataRootError
        If the directory is relative or names a ``..`` component.
    """

    __slots__ = ("_directory",)

    def __init__(self, directory: Path) -> None:
        if not directory.is_absolute():
            raise DataRootError(directory, "relative to a working directory")

        if ".." in directory.parts:
            raise DataRootError(directory, "spelled with a '..' component")

        self._directory = directory

    @property
    def directory(self) -> Path:
        """Return the absolute directory declared paths resolve against."""
        return self._directory

    def resolve(self, declared: str, *, key: str) -> Path:
        """Return the absolute path a configuration key names.

        Parameters
        ----------
        declared : str
            The path text the configuration gave, in POSIX form and relative to
            the root.
        key : str
            The configuration key the text was declared under, which names the
            offender when the text is refused.

        Returns
        -------
        Path
            The resolved absolute path.

        Raises
        ------
        PathDeclarationError
            If the text is not a path relative to the root: empty, spelled with
            a backslash, absolute, drive-qualified, home-relative, or spelled
            with a ``..`` component.
        """
        return self._directory.joinpath(*_components(declared, key))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether both roots name the same directory."""
        if type(other) is not DataRoot:
            return NotImplemented

        return self._directory == other._directory

    @override
    def __hash__(self) -> int:
        """Return a hash agreeing with equality."""
        return hash((DataRoot, self._directory))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"{type(self).__name__}({self._directory!r})"
