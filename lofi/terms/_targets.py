"""Matching-aware category target sets."""

from typing import TYPE_CHECKING, Never, Self, final, override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi.terms._ports import Matching

__all__ = ["TargetSet"]


@final
class TargetSet[T]:
    """A frozen category set paired with its matching.

    Parameters
    ----------
    categories : Iterable[T]
        Target categories in declaration order.
    matching : Matching[T]
        Congruence used to compare categories.
    """

    __slots__ = ("_categories", "_matching")

    def __init__(self, categories: Iterable[T], matching: Matching[T]) -> None:
        object.__setattr__(self, "_categories", tuple(categories))
        object.__setattr__(self, "_matching", matching)

    @property
    def categories(self) -> tuple[T, ...]:
        """Return the target categories in declaration order."""
        return self._categories

    @property
    def matching(self) -> Matching[T]:
        """Return the matching used to compare categories."""
        return self._matching

    @property
    def matching_name(self) -> str:
        """Return the matching name."""
        return self._matching.name

    def __copy__(self) -> Self:
        """Return this immutable target set."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable target set."""
        del memo

        return self

    @override
    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[tuple[T, ...], Matching[T]]]:
        """Return the reconstruction call with the injected matching."""
        return (type(self), (self._categories, self._matching))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a target set is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a target set is immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the categories and matching identity agree."""
        if type(other) is not TargetSet:
            return NotImplemented

        return (
            self._categories == other._categories and self._matching is other._matching
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the categories and matching identity."""
        return hash((TargetSet, self._categories, id(self._matching)))

    @override
    def __repr__(self) -> str:
        """Return a rendering with the categories and matching name."""
        return (
            f"TargetSet(categories={self._categories!r}, "
            f"matching={self.matching_name!r})"
        )
