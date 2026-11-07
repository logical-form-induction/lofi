"""Partial terms paired with decoder states, and the context of their next hole."""

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Never, Self, final, override

from lofi._checks import ensure_non_negative
from lofi.terms._path import positions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._native import Term
    from lofi.terms._ports import Path, Sort

__all__ = ["Frontier", "HoleContext"]


class HoleContext(NamedTuple):
    """What a decoder knows about the hole it fills next.

    Attributes
    ----------
    path : Path
        The hole's argument path from the skeleton's root.
    sort : Sort
        The hole's sort.
    in_scope : Mapping[Sort, int]
        How many binding sites of each sort scope over the hole, so the bound
        indices of a sort the hole may take are zero up to that count.
    free_used : Mapping[Sort, int]
        How many distinct free variables of each sort the skeleton holds.
    remaining : int
        How many nodes the skeleton may still grow by under the size bound.
    """

    path: Path
    sort: Sort
    in_scope: Mapping[Sort, int]
    free_used: Mapping[Sort, int]
    remaining: int


def _is_hole(term: Term, /) -> bool:
    """Return whether a term is a kernel-derived hole."""
    return term.op.is_hole


def _hole_paths(skeleton: Term, /) -> tuple[Path, ...]:
    """Return hole paths in shallowest-leftmost order."""
    paths = positions(skeleton, _is_hole)

    return tuple(sorted(paths, key=lambda path: (len(path), path)))


def _free_counts(skeleton: Term, /) -> Mapping[Sort, int]:
    """Return how many distinct free variables of each sort a term holds."""
    seen: dict[Sort, set[int]] = {}
    pending = [skeleton]

    while pending:
        node = pending.pop()

        if node.op.is_free_variable and node.payload is not None:
            seen.setdefault(node.op.sort, set()).add(node.payload)

        pending.extend(node.args)

    return MappingProxyType({sort: len(held) for sort, held in seen.items()})


@final
class Frontier[S]:
    """A partial term with one decoder state per hole.

    Parameters
    ----------
    skeleton : Term
        The partial term, whose holes are the kernel's hole operators.
    states : Mapping[Path, S]
        Decoder states keyed by exactly the skeleton's hole paths.
    log_weight : float
        The accumulated log weight.
    """

    __slots__ = ("_holes", "_log_weight", "_skeleton", "_states")

    def __init__(
        self,
        skeleton: Term,
        states: Mapping[Path, S],
        log_weight: float,
    ) -> None:
        holes = _hole_paths(skeleton)
        held = dict(states)
        expected = frozenset(holes)
        received = frozenset(held)

        if received != expected:
            missing = tuple(path for path in holes if path not in received)
            extra = tuple(
                sorted(received - expected, key=lambda path: (len(path), path))
            )
            msg = (
                "frontier states do not match skeleton holes: "
                f"missing={missing!r}, extra={extra!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_skeleton", skeleton)
        object.__setattr__(self, "_states", MappingProxyType(held))
        object.__setattr__(self, "_holes", holes)
        object.__setattr__(self, "_log_weight", log_weight)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a frontier is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a frontier is immutable"

        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable frontier."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable frontier."""
        del memo

        return self

    @property
    def skeleton(self) -> Term:
        """Return the partial term."""
        return self._skeleton

    @property
    def states(self) -> Mapping[Path, S]:
        """Return the read-only state map."""
        return self._states

    @property
    def log_weight(self) -> float:
        """Return the accumulated log weight."""
        return self._log_weight

    @property
    def holes(self) -> tuple[Path, ...]:
        """Return holes in shallowest-leftmost order."""
        return self._holes

    @property
    def is_complete(self) -> bool:
        """Return whether the skeleton has no holes."""
        return not self._holes

    def context(self, max_size: int, /) -> HoleContext:
        """Return the context of the next hole, the first in hole order.

        Parameters
        ----------
        max_size : int
            The most nodes a completed term may hold.

        Raises
        ------
        ValueError
            If the frontier is complete or the bound is negative.
        """
        bound = ensure_non_negative("max_size", max_size)

        if not self._holes:
            msg = "a complete frontier has no next hole"
            raise ValueError(msg)

        path = self._holes[0]
        in_scope: dict[Sort, int] = {}
        node = self._skeleton

        for position in path:
            for site in node.op.scope(position):
                in_scope[site.sort] = in_scope.get(site.sort, 0) + 1

            node = node.args[position]

        return HoleContext(
            path,
            node.op.sort,
            MappingProxyType(in_scope),
            _free_counts(self._skeleton),
            bound - self._skeleton.size,
        )

    def with_weight(self, log_weight: float) -> Self:
        """Return this frontier with a new weight.

        Parameters
        ----------
        log_weight : float
            The replacement log weight.
        """
        return type(self)(self._skeleton, self._states, log_weight)

    @override
    def __repr__(self) -> str:
        """Return the hole count and weight."""
        return f"Frontier(holes={len(self._holes)}, log_weight={self._log_weight!r})"
