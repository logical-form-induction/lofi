"""Arity-generic factoring and combination of decoder states."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Never, Self, final, override

import torch
from torch import nn

from lofi._checks import ensure_non_negative
from lofi.nn._guards import checked_state, positive
from lofi.nn._values import Encoded
from lofi.nn.vsa import bind, bundle, bundle_scale, unbind

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor

    from lofi.judge import ClosureReport
    from lofi.terms import Operator, Term

__all__ = [
    "SHIPPED_RANKS",
    "ArityBound",
    "ArityHeadScheme",
    "ArityUnsupportedError",
    "RoleFillerScheme",
    "reached_operators",
]

SHIPPED_RANKS: Final[tuple[int, ...]] = (1, 2)
"""The operator arities the neural layer ships combination modules for."""


class ArityUnsupportedError(ValueError):
    """Reports an operator arity outside a scheme's shipped ranks."""


def reached_operators(closure: ClosureReport[Term], /) -> tuple[Operator, ...]:
    """Return the operators a closure's reached terms are built from.

    Parameters
    ----------
    closure : ClosureReport[Term]
        The structural closure a compiled grammar reports, whose reached terms are the
        categories the rules derive.

    Returns
    -------
    tuple[Operator, ...]
        One entry per distinct operator appearing anywhere in a reached term, first
        occurrence first.
    """
    found: dict[Operator, None] = {}

    for reached in closure.reached:
        pending = [reached]

        while pending:
            term = pending.pop()
            found.setdefault(term.op, None)
            pending.extend(reversed(term.args))

    return tuple(found)


@final
class ArityBound:
    """The operator arities a rank-indexed family serves.

    Parameters
    ----------
    ranks : Iterable[int]
        The ranks the family ships a combination module for.

    Raises
    ------
    TypeError
        If a rank is not an integer.
    ValueError
        If a rank is not positive, or no rank is given.
    """

    __slots__ = ("_ranks",)

    def __init__(self, ranks: Iterable[int] = SHIPPED_RANKS) -> None:
        held = tuple(
            sorted({positive("a shipped rank", rank) for rank in ranks}),
        )

        if not held:
            msg = "a rank-indexed family ships at least one rank"
            raise ValueError(msg)

        object.__setattr__(self, "_ranks", held)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an arity bound is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an arity bound is immutable"

        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable bound."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable bound."""
        del memo

        return self

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the shipped ranks in ascending order."""
        return self._ranks

    @property
    def ceiling(self) -> int:
        """Return the largest shipped rank."""
        return self._ranks[-1]

    @property
    def reason(self) -> str:
        """Return the declared refusal, naming the shipped ranks and the ceiling."""
        shipped = ", ".join(str(rank) for rank in self._ranks)

        return (
            f"a rank-indexed family ships ranks {shipped}, so it serves an "
            f"operator arity of at most {self.ceiling}"
        )

    def served(self, reached: Iterable[Operator], /) -> tuple[int, ...]:
        """Return the shipped ranks a closure actually reaches.

        Parameters
        ----------
        reached : Iterable[Operator]
            The operators the compiled grammar builds, which is what decides the
            branching a scheme pays for.

        Returns
        -------
        tuple[int, ...]
            The positive arities among them, in ascending order. A nullary operator
            takes no combination module and is absent.

        Raises
        ------
        ArityUnsupportedError
            If a reached operator has a positive arity outside the shipped ranks.
        """
        arities: dict[int, str] = {}

        for op in reached:
            if op.arity > 0:
                arities.setdefault(op.arity, op.name)

        for arity in sorted(arities):
            if arity not in self._ranks:
                msg = (
                    f"{self.reason}; the closure reaches arity {arity} at "
                    f"operator {arities[arity]!r}"
                )
                raise ArityUnsupportedError(msg)

        return tuple(sorted(arities))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value ships the same ranks."""
        if type(other) is not ArityBound:
            return NotImplemented

        return self._ranks == other._ranks

    @override
    def __hash__(self) -> int:
        """Return a hash over the shipped ranks."""
        return hash((ArityBound, self._ranks))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this bound."""
        return f"ArityBound({self._ranks!r})"


def _unserved(ranks: tuple[int, ...], asked: int, noun: str) -> str:
    """Return the refusal naming the ranks a scheme serves and the one asked of it.

    Parameters
    ----------
    ranks : tuple[int, ...]
        The ranks the scheme serves, ascending.
    asked : int
        The rank the caller asked for.
    noun : str
        What one served rank holds: a role, a head or a combination.
    """
    served = ", ".join(str(each) for each in ranks)

    if not served:
        return f"this scheme holds no {noun}, so none at rank {asked}"

    return f"this scheme holds a {noun} at ranks {served}, not at {asked}"


def _slot_offsets(ranks: Sequence[int]) -> Mapping[int, int]:
    """Return each rank's first argument-position slot.

    Parameters
    ----------
    ranks : Sequence[int]
        The served ranks, ascending.

    Returns
    -------
    Mapping[int, int]
        The slot at which a rank's argument positions start, keyed by rank.
    """
    offsets: dict[int, int] = {}
    at = 0

    for rank in ranks:
        offsets[rank] = at
        at += rank

    return MappingProxyType(offsets)


def _combined_mask(children: Sequence[Encoded], /) -> Tensor | None:
    """Return the mask covering a combined state.

    Parameters
    ----------
    children : Sequence[Encoded]
        The child states being combined.

    Returns
    -------
    Tensor | None
        True where every child holds data, or None where no child is masked.
    """
    held: Tensor | None = None

    for child in children:
        mask = child.mask

        if mask is None:
            continue

        held = mask if held is None else held & mask

    return held


def _checked_children(
    children: Sequence[Encoded], ranks: tuple[int, ...], width: int, /
) -> int:
    """Return the rank of a combination.

    Parameters
    ----------
    children : Sequence[Encoded]
        The child states to combine, in argument order.
    ranks : tuple[int, ...]
        The ranks the scheme serves.
    width : int
        The feature width the scheme was built at.

    Raises
    ------
    ArityUnsupportedError
        If the scheme serves no module at that rank.
    ValueError
        If a child is shaped wrong, or the children disagree about their axes or their
        shape.
    """
    rank = len(children)

    if rank not in ranks:
        raise ArityUnsupportedError(_unserved(ranks, rank, "combination"))

    first = checked_state(children[0], width, "scheme")

    for child in children[1:]:
        checked_state(child, width, "scheme")

        if child.axes != first.axes or child.tensor.shape != first.tensor.shape:
            msg = (
                "the children of one combination share their axes and their "
                f"shape: {first.axes} of {tuple(first.tensor.shape)} against "
                f"{child.axes} of {tuple(child.tensor.shape)}"
            )
            raise ValueError(msg)

    return rank


class RoleFillerScheme(nn.Module):
    r"""Holographic factoring and combination by role-filler binding.

    A combination of positive rank $r$ is the bundle
    $\mathbf{h}=r^{-1/2}\sum_{i=1}^{r}\mathbf{r}_i\odot\mathbf{h}_i$,
    where $\mathbf{r}_i$ and $\mathbf{h}_i$ are role and child vectors and
    $\odot$ denotes coordinatewise multiplication. The scaling normalizes the
    accumulation across roles; it does not guarantee a fixed norm for arbitrary
    learned vectors. Factoring divides by each nonzero role coordinate and undoes
    the scaling, so a rank-one round trip returns the original child.

    Parameters
    ----------
    width : int
        How wide a decoder state is.
    reached : Iterable[Operator]
        The operators the compiled grammar builds. Only the arities among them get a
        role, so a grammar that declares a ternary constructor and never reaches one
        builds nothing for it.
    bound : ArityBound | None
        The ranks this family ships, or None for the shipped ones.

    Raises
    ------
    TypeError
        If the width is not an integer.
    ValueError
        If the width is not positive.
    ArityUnsupportedError
        If a reached operator has a positive arity outside the shipped ranks.
    """

    _log_magnitudes: nn.Parameter
    _signs: Tensor

    def __init__(
        self,
        *,
        width: int,
        reached: Iterable[Operator],
        bound: ArityBound | None = None,
    ) -> None:
        super().__init__()
        held = ArityBound() if bound is None else bound
        served = held.served(reached)
        size = positive("a decoder state's width", width)
        slots = sum(served)

        # a role is a fixed sign pattern times a positive magnitude, so unbinding
        # never divides by a coordinate that is zero
        self.register_buffer(
            "_signs",
            torch.randint(0, 2, (slots, size)).mul(2).sub(1).float(),
        )
        self._log_magnitudes = nn.Parameter(torch.zeros(slots, size))
        self._offsets = _slot_offsets(served)
        self._bound = held
        self._served = served
        self._width = size

    @property
    def width(self) -> int:
        """Return the feature width of the states this scheme reads and returns."""
        return self._width

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the arities the closure reached, in ascending order."""
        return self._served

    @property
    def bound(self) -> ArityBound:
        """Return the shipped ranks this scheme was built against."""
        return self._bound

    def role(self, arity: int, position: int, /) -> Tensor:
        """Return the role vector of one argument position.

        Parameters
        ----------
        arity : int
            The arity of the operator the position belongs to.
        position : int
            Which argument position it is, counting from zero.

        Returns
        -------
        Tensor
            Rank 1 of the scheme's own width, at the parameters' precision, with no
            coordinate at zero.

        Raises
        ------
        ArityUnsupportedError
            If the scheme holds no role at that arity.
        ValueError
            If the position is outside the arity.
        """
        if arity not in self._offsets:
            raise ArityUnsupportedError(_unserved(self._served, arity, "role"))

        if not 0 <= position < arity:
            msg = f"an operator of arity {arity} has no argument at {position}"
            raise ValueError(msg)

        at = self._offsets[arity] + position

        return self._signs[at] * self._log_magnitudes[at].exp()

    def factor(self, parent: Encoded, /, *, arity: int) -> tuple[Encoded, ...]:
        """Return one child state per argument position of an operator.

        Parameters
        ----------
        parent : Encoded
            The state at the node being expanded, over the feature axis innermost.
        arity : int
            How many arguments the chosen operator takes.

        Returns
        -------
        tuple[Encoded, ...]
            One state per argument position, each of the parent's own axes, shape and
            mask. A nullary operator returns nothing.

        Raises
        ------
        ArityUnsupportedError
            If the scheme holds no role at that arity.
        ValueError
            If the parent is shaped wrong, or the arity is negative.
        """
        count = ensure_non_negative("an operator's arity", arity)
        held = checked_state(parent, self._width, "scheme")

        if not count:
            return ()

        correction = bundle_scale(count)

        return tuple(
            Encoded(
                unbind(held.tensor, self.role(count, position)) * correction,
                held.axes,
                held.mask,
            )
            for position in range(count)
        )

    def combine(self, children: Sequence[Encoded], /) -> Encoded:
        """Return the parent state one operator's children bundle into.

        Parameters
        ----------
        children : Sequence[Encoded]
            One state per argument position, in argument order.

        Returns
        -------
        Encoded
            The bundle, of the children's own axes and shape, masked where every child
            holds data.

        Raises
        ------
        ArityUnsupportedError
            If the scheme serves no module at that rank.
        ValueError
            If a child is shaped wrong, or the children disagree.
        """
        rank = _checked_children(children, self._served, self._width)

        return Encoded(
            bundle(
                tuple(
                    bind(child.tensor, self.role(rank, position))
                    for position, child in enumerate(children)
                )
            ),
            children[0].axes,
            _combined_mask(children),
        )


class ArityHeadScheme(nn.Module):
    """Recursive factoring and combination by an arity-indexed head family.

    One head per argument position of each reached arity reads a parent state down to a
    child, and one head per reached rank reads a run of children back up to a parent.

    Parameters
    ----------
    width : int
        How wide a decoder state is.
    reached : Iterable[Operator]
        The operators the compiled grammar builds. Only the arities among them get
        heads, so a grammar that declares a ternary constructor and never reaches one
        builds nothing for it.
    bound : ArityBound | None
        The ranks this family ships, or None for the shipped ones.

    Raises
    ------
    TypeError
        If the width is not an integer.
    ValueError
        If the width is not positive.
    ArityUnsupportedError
        If a reached operator has a positive arity outside the shipped ranks.
    """

    _down: nn.ModuleList
    _up: nn.ModuleList

    def __init__(
        self,
        *,
        width: int,
        reached: Iterable[Operator],
        bound: ArityBound | None = None,
    ) -> None:
        super().__init__()
        held = ArityBound() if bound is None else bound
        served = held.served(reached)
        size = positive("a decoder state's width", width)

        self._down = nn.ModuleList(
            nn.Linear(size, size) for rank in served for _position in range(rank)
        )
        self._up = nn.ModuleList(nn.Linear(rank * size, size) for rank in served)
        self._offsets = _slot_offsets(served)
        self._bound = held
        self._served = served
        self._width = size

    @property
    def width(self) -> int:
        """Return the feature width of the states this scheme reads and returns."""
        return self._width

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the arities the closure reached, in ascending order."""
        return self._served

    @property
    def bound(self) -> ArityBound:
        """Return the shipped ranks this scheme was built against."""
        return self._bound

    def _head(self, arity: int, position: int, /) -> nn.Module:
        """Return the head reading one argument position.

        Parameters
        ----------
        arity : int
            The arity of the operator the position belongs to.
        position : int
            Which argument position it is, counting from zero.

        Returns
        -------
        nn.Module
            The head at that slot.

        Raises
        ------
        ArityUnsupportedError
            If the scheme holds no head at that arity.
        """
        if arity not in self._offsets:
            raise ArityUnsupportedError(_unserved(self._served, arity, "head"))

        return self._down[self._offsets[arity] + position]

    def factor(self, parent: Encoded, /, *, arity: int) -> tuple[Encoded, ...]:
        """Return one child state per argument position of an operator.

        Parameters
        ----------
        parent : Encoded
            The state at the node being expanded, over the feature axis innermost.
        arity : int
            How many arguments the chosen operator takes.

        Returns
        -------
        tuple[Encoded, ...]
            One state per argument position, each of the parent's own axes, shape and
            mask. A nullary operator returns nothing.

        Raises
        ------
        ArityUnsupportedError
            If the scheme holds no head at that arity.
        ValueError
            If the parent is shaped wrong, or the arity is negative.
        """
        count = ensure_non_negative("an operator's arity", arity)
        held = checked_state(parent, self._width, "scheme")

        return tuple(
            Encoded(
                torch.tanh(self._head(count, position)(held.tensor)),
                held.axes,
                held.mask,
            )
            for position in range(count)
        )

    def combine(self, children: Sequence[Encoded], /) -> Encoded:
        """Return the parent state one operator's children read back up to.

        Parameters
        ----------
        children : Sequence[Encoded]
            One state per argument position, in argument order.

        Returns
        -------
        Encoded
            The parent, of the children's own axes and shape, masked where every child
            holds data.

        Raises
        ------
        ArityUnsupportedError
            If the scheme serves no module at that rank.
        ValueError
            If a child is shaped wrong, or the children disagree.
        """
        rank = _checked_children(children, self._served, self._width)
        run = torch.cat(tuple(child.tensor for child in children), dim=-1)
        head = self._up[self._served.index(rank)]

        return Encoded(
            torch.tanh(head(run)),
            children[0].axes,
            _combined_mask(children),
        )
