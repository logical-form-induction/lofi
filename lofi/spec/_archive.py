"""Indexed declarations retained with their checked theories."""

from typing import final, override

from lofi._frozen import FrozenValue
from lofi.spec._fingerprint import IndexedLanguage

__all__ = ["ArchivedLanguage"]


@final
class ArchivedLanguage(FrozenValue):
    """An indexed declaration and the theory it was checked to.

    Parameters
    ----------
    declaration : bytes
        The canonical bytes of the declaration the theory was checked from.
    theory : IndexedLanguage
        The theory specification and digest with the route it was lowered on.

    Raises
    ------
    TypeError
        If the declaration is not bytes or the theory is not an indexed
        language.
    ValueError
        If the declaration is empty.
    """

    _noun = "fingerprint value"

    __slots__ = ("_declaration", "_theory")

    def __init__(self, declaration: bytes, theory: IndexedLanguage) -> None:
        if not isinstance(declaration, bytes):
            msg = f"a declaration is not bytes: {declaration!r}"
            raise TypeError(msg)

        if not declaration:
            msg = "a declaration is empty"
            raise ValueError(msg)

        if type(theory) is not IndexedLanguage:
            msg = f"an archived theory is not an IndexedLanguage: {theory!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_theory", theory)

    @property
    def declaration(self) -> bytes:
        """Return the declaration's canonical bytes."""
        return self._declaration

    @property
    def theory(self) -> IndexedLanguage:
        """Return the theory the declaration was checked to."""
        return self._theory

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the declaration and the theory."""
        if type(other) is not ArchivedLanguage:
            return NotImplemented

        return self._declaration == other._declaration and self._theory == other._theory

    @override
    def __hash__(self) -> int:
        """Hash the declaration and the theory."""
        return hash((ArchivedLanguage, self._declaration, self._theory))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"ArchivedLanguage({self._declaration!r}, {self._theory!r})"
