"""Training optimization values."""

from enum import Enum
from typing import TYPE_CHECKING, Never, final, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["OptInSet", "Optimization", "OptimizationResult", "Rewrite"]


@final
class OptimizationResult:
    """Reports one optimization-policy call."""

    __slots__ = ("_loss_finite", "_scale", "_stepped")

    def __init__(
        self,
        *,
        loss_finite: bool,
        stepped: bool,
        scale: float | None,
    ) -> None:
        object.__setattr__(self, "_loss_finite", loss_finite)
        object.__setattr__(self, "_stepped", stepped)
        object.__setattr__(self, "_scale", scale)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "an optimization result is immutable"

        raise AttributeError(msg)

    @property
    def loss_finite(self) -> bool:
        """Return whether the loss was finite."""
        return self._loss_finite

    @property
    def stepped(self) -> bool:
        """Return whether the optimizer stepped."""
        return self._stepped

    @property
    def scale(self) -> float | None:
        """Return the scale used for the step, if any."""
        return self._scale

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another result reports the same outcome."""
        if type(other) is not OptimizationResult:
            return NotImplemented

        return (
            self._loss_finite == other._loss_finite
            and self._stepped == other._stepped
            and self._scale == other._scale
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the reported outcome."""
        return hash((OptimizationResult, self._loss_finite, self._stepped, self._scale))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"OptimizationResult(loss_finite={self._loss_finite!r}, "
            f"stepped={self._stepped!r}, scale={self._scale!r})"
        )


class Rewrite(Enum):
    """Classifies an optimization rewrite."""

    #: Change scheduling, batching, or placement without changing values.
    SCHEDULE = "schedule"

    #: Permit changes to computed values.
    VALUE = "value"

    @property
    def requires_opt_in(self) -> bool:
        """Return whether the rewrite requires opt-in."""
        return self is Rewrite.VALUE


@final
class Optimization:
    """An optional computation rewrite.

    Parameters
    ----------
    name : str
        What the rewrite is called.
    rewrite : Rewrite
        What it changes: the schedule alone, or the values as well.

    Raises
    ------
    ValueError
        If the name is empty or has surrounding whitespace.
    """

    __slots__ = ("_name", "_rewrite")

    def __init__(self, name: str, rewrite: Rewrite) -> None:
        if not name or name != name.strip():
            msg = f"an optimization's name is non-empty and stripped: {name!r}"

            raise ValueError(msg)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_rewrite", rewrite)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "an optimization is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the rewrite name."""
        return self._name

    @property
    def rewrite(self) -> Rewrite:
        """Return the rewrite class."""
        return self._rewrite

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same rewrite under the same name."""
        if type(other) is not Optimization:
            return NotImplemented

        return self._name == other._name and self._rewrite == other._rewrite

    @override
    def __hash__(self) -> int:
        """Return a hash over the name and what the rewrite changes."""
        return hash((Optimization, self._name, self._rewrite))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this optimization."""
        return f"Optimization({self._name!r}, {self._rewrite})"


@final
class OptInSet:
    """Enabled value rewrites.

    Parameters
    ----------
    optimizations : Iterable[Optimization]
        The value rewrites to turn on.

    Raises
    ------
    ValueError
        If one of them is a schedule rewrite.
    """

    __slots__ = ("_members",)

    def __init__(self, optimizations: Iterable[Optimization] = ()) -> None:
        named = tuple(optimizations)

        for optimization in named:
            if not optimization.rewrite.requires_opt_in:
                msg = (
                    f"{optimization.name!r} rewrites the schedule and not the "
                    f"values, so it is on and cannot be opted into"
                )

                raise ValueError(msg)

        object.__setattr__(self, "_members", frozenset(named))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "an opt-in set is immutable"

        raise AttributeError(msg)

    def __contains__(self, optimization: Optimization) -> bool:
        """Return whether a rewrite is one this regime asked for.

        Parameters
        ----------
        optimization : Optimization
            The rewrite to look for.
        """
        return optimization in self._members

    def __iter__(self) -> Iterator[Optimization]:
        """Yield the rewrites this regime asked for, in name order."""
        return iter(sorted(self._members, key=lambda member: member.name))

    def __len__(self) -> int:
        """Return how many rewrites this regime asked for."""
        return len(self._members)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value contains exactly the same rewrites."""
        if type(other) is not OptInSet:
            return NotImplemented

        return self._members == other._members

    @override
    def __hash__(self) -> int:
        """Return a hash over the rewrites this regime asked for."""
        return hash((OptInSet, self._members))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this regime."""
        members = ", ".join(repr(member) for member in self)

        return f"OptInSet([{members}])"
