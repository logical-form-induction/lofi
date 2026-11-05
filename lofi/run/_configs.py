"""The configuration groups packaged with LoFI.

A group is a directory under the packaged root whose relative path spells a
dotted slot of the specification it fills, and each document in it is one
option for that slot. Composition selects one option per slot, so a
combination of components is a list of selections rather than a document of
its own.
"""

from typing import TYPE_CHECKING, Final

from lofi.spec import PACKAGED_CONFIG_ROOT

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DOCUMENT_SUFFIXES",
    "UnknownConfigGroupError",
    "configuration_documents",
    "packaged_groups",
    "packaged_options",
    "section_root",
]

DOCUMENT_SUFFIXES: Final = frozenset({".json", ".toml", ".yaml", ".yml"})
"""The suffixes of files recognized as configuration documents."""


class UnknownConfigGroupError(LookupError):
    """The packaged library holds no group for a slot.

    Attributes
    ----------
    slot : str
        The dotted slot looked up.
    available : tuple[str, ...]
        The slots the packaged library holds, in slot order.
    """

    def __init__(self, *, slot: str, available: tuple[str, ...]) -> None:
        self.slot = slot
        self.available = available
        super().__init__(
            f"the packaged configuration library holds no group {slot!r}; "
            f"it holds {list(available)}"
        )


def configuration_documents(directory: Path, /) -> tuple[Path, ...]:
    """Return every configuration document below a directory in path order.

    Parameters
    ----------
    directory : Path
        The directory searched at every depth.

    Raises
    ------
    NotADirectoryError
        If the path is not a directory.
    """
    if not directory.is_dir():
        msg = f"not a configuration directory: {directory}"
        raise NotADirectoryError(msg)

    return tuple(
        sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in DOCUMENT_SUFFIXES
        )
    )


def section_root(slot: str, /) -> Path:
    """Return the directory the groups below one slot resolve against.

    Parameters
    ----------
    slot : str
        The dotted slot of the section whose own groups are wanted, empty for
        the library as a whole.
    """
    if not slot:
        return PACKAGED_CONFIG_ROOT

    return PACKAGED_CONFIG_ROOT.joinpath(*slot.split("."))


def packaged_groups() -> tuple[str, ...]:
    """Return the dotted slot of every group the packaged library holds.

    Returns
    -------
    tuple[str, ...]
        The slots in ascending order.
    """
    return tuple(
        sorted(
            {
                ".".join(document.parent.relative_to(PACKAGED_CONFIG_ROOT).parts)
                for document in configuration_documents(PACKAGED_CONFIG_ROOT)
            }
        )
    )


def packaged_options(slot: str, /) -> tuple[str, ...]:
    """Return the options the packaged library holds for one slot.

    Parameters
    ----------
    slot : str
        The dotted slot whose options are wanted.

    Returns
    -------
    tuple[str, ...]
        The option names in ascending order, each a fragment name a selection
        of the slot may take.

    Raises
    ------
    UnknownConfigGroupError
        If the packaged library holds no group for the slot.
    """
    directory = section_root(slot)
    options = tuple(
        sorted(
            document.stem
            for document in configuration_documents(PACKAGED_CONFIG_ROOT)
            if document.parent == directory
        )
    )

    if not options:
        raise UnknownConfigGroupError(slot=slot, available=packaged_groups())

    return options
