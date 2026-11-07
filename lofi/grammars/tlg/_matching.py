"""Exact matching for categories carrying declared type-logical modes."""

from hashlib import sha256
from json import dumps
from typing import TYPE_CHECKING, Self, final, override

from lofi._frozen import FrozenValue
from lofi.terms import RepresentativeKernel, Term

if TYPE_CHECKING:
    from lofi.terms import Signature

__all__ = ["ModalMatching"]


@final
class ModalMatching(FrozenValue):
    """Structural identity qualified by a finite modal order."""

    _noun = "modal matching"

    __slots__ = ("_modes", "_name", "_order")

    def __init__(
        self,
        modes: tuple[str, ...],
        order: tuple[tuple[str, str], ...],
    ) -> None:
        encoded = dumps(
            {"modes": modes, "order": order},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        object.__setattr__(self, "_modes", modes)
        object.__setattr__(self, "_order", order)
        digest = sha256(encoded.encode()).hexdigest()[:16]
        object.__setattr__(self, "_name", f"modal-exact:{digest}")

    @property
    def modes(self) -> tuple[str, ...]:
        """Return the declared modes in their canonical order."""
        return self._modes

    @property
    def order(self) -> tuple[tuple[str, str], ...]:
        """Return the reflexive-transitive modal order."""
        return self._order

    @property
    def name(self) -> str:
        """Return the relation name qualified by the modal declaration."""
        return self._name

    @property
    def has_key(self) -> bool:
        """Report that exact representatives are available."""
        return True

    @property
    def exact(self) -> bool:
        """Report that the modal order relates distinct modes."""
        return False

    @property
    def representatives(self) -> RepresentativeKernel:
        """Return structural identity as the native representative kernel."""
        return RepresentativeKernel.identity()

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str]:
        """Return the operator alone, since modal matching relates one term."""
        del signature
        return frozenset({operator})

    def key(self, term: Term, /) -> Term:
        """Return the exact representative of a category."""
        return term

    def equivalent(self, left: Term, right: Term, /) -> bool:
        """Decide structural identity between two interned categories."""
        return left is right

    @override
    def __copy__(self) -> Self:
        """Return this immutable relation."""
        return self

    @override
    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable relation."""
        del memo
        return self

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[ModalMatching],
        tuple[tuple[str, ...], tuple[tuple[str, str], ...]],
    ]:
        """Return the reconstruction call."""
        return ModalMatching, (self._modes, self._order)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the modal vocabulary and order."""
        if type(other) is not ModalMatching:
            return NotImplemented
        return self._modes == other._modes and self._order == other._order

    @override
    def __hash__(self) -> int:
        """Hash the modal vocabulary and order."""
        return hash((ModalMatching, self._modes, self._order))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"ModalMatching({self._modes!r}, {self._order!r})"
