"""Lexical type signatures."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_non_negative

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi.typecheck._values import Scheme

__all__ = ["LexicalSignature"]


@final
class LexicalSignature:
    """An immutable symbol-to-scheme environment."""

    __slots__ = ("_schemes",)

    def __init__(self, schemes: Mapping[int, Scheme]) -> None:
        """Initialize the environment from a copied mapping."""
        held = {
            ensure_non_negative(f"constant identifier {atom!r}", atom): scheme
            for atom, scheme in schemes.items()
        }
        object.__setattr__(self, "_schemes", MappingProxyType(held))

    @property
    def schemes(self) -> Mapping[int, Scheme]:
        """Return the read-only symbol declarations."""
        return self._schemes

    def __getitem__(self, symbol: int, /) -> Scheme:
        """Get the scheme declared for a constant identifier."""
        return self._schemes[symbol]

    def __iter__(self) -> Iterator[int]:
        """Iterate over declared constant identifiers."""
        return iter(self._schemes)

    def __len__(self) -> int:
        """Count declared constant identifiers."""
        return len(self._schemes)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a lexical signature is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a lexical signature is immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another signature has the same declarations."""
        if type(other) is not LexicalSignature:
            return NotImplemented

        return self._schemes == other._schemes

    @override
    def __hash__(self) -> int:
        """Return a hash over the declarations."""
        return hash((LexicalSignature, tuple(sorted(self._schemes.items()))))

    @override
    def __repr__(self) -> str:
        """Return the constructor call."""
        return f"LexicalSignature({dict(self._schemes)!r})"
