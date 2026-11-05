"""Finite conjunction views of a grammar's coordination categories."""

from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Conjunctions"]


@final
class Conjunctions[T]:
    """Finite conjunction markers and category pairs."""

    __slots__ = ("_markers", "_marking", "_name", "_pairs")

    def __init__(
        self,
        name: str,
        markers: Sequence[T],
        pairs: Sequence[tuple[T, T]],
        *,
        marking: str | None = None,
    ) -> None:
        held_markers = tuple(markers)
        held_pairs = tuple(pairs)

        if not held_markers:
            msg = "a conjunction view requires at least one marker"
            raise ValueError(msg)

        if not held_pairs:
            msg = "a conjunction view requires at least one category pair"
            raise ValueError(msg)

        object.__setattr__(
            self, "_name", ensure_name("a conjunction view's name", name)
        )
        object.__setattr__(self, "_markers", held_markers)
        object.__setattr__(self, "_pairs", held_pairs)
        object.__setattr__(
            self,
            "_marking",
            None
            if marking is None
            else ensure_name("a conjunction marking constructor", marking),
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a conjunction view is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a conjunction view is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the conjunction view name."""
        return self._name

    @property
    def markers(self) -> tuple[T, ...]:
        """Return the copied conjunction markers."""
        return self._markers

    @property
    def pairs(self) -> tuple[tuple[T, T], ...]:
        """Return the copied unmarked and marked category pairs."""
        return self._pairs

    @property
    def marking(self) -> str | None:
        """Return the unary constructor that marks any category, if declared."""
        return self._marking
