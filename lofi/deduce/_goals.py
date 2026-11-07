"""Spanning and covering parse goals."""

from enum import Enum
from typing import TYPE_CHECKING, Never, override

from lofi._checks import ensure_finite, ensure_non_negative

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.deduce._ports import Chart
    from lofi.deduce._spans import Cell
    from lofi.deduce._values import Edge, ItemKey
    from lofi.terms import Matching, TargetSet

__all__ = ["CoverGoal", "PenaltyForm", "RootGoal", "Tiling"]

type _RootGoalReduction[T] = tuple[
    Callable[[tuple[TargetSet[T], tuple[float, ...]]], RootGoal[T]],
    tuple[tuple[TargetSet[T], tuple[float, ...]]],
]
"""Root-goal pickle reduction."""

type _CoverGoalFields = tuple[int, float, PenaltyForm, Tiling]
"""Fields needed to reconstruct a cover goal."""

type _CoverGoalReduction[T] = tuple[
    Callable[[_CoverGoalFields], CoverGoal[T]],
    tuple[_CoverGoalFields],
]
"""Cover-goal pickle reduction."""


class PenaltyForm(Enum):
    """How a cover's fragment count incurs its declared log penalty."""

    AFTER_FIRST = "after-first"
    """Charge every fragment after the first."""

    EACH = "each-fragment"
    """Charge every fragment, including the first."""


class Tiling(Enum):
    """Which deterministic cover a fragment goal selects."""

    FEWEST = "fewest-fragments"
    """Use the fewest chart fragments, preferring the longest first on ties."""

    LEXICAL = "lexical"
    """Use exactly the one-token cells."""


class RootGoal[T]:
    """Requires one item spanning the input with a target category.

    Parameters
    ----------
    targets : TargetSet[T]
        Permitted categories and their matching.
    """

    __slots__ = ("_targets", "_weights")

    def __init__(
        self,
        targets: TargetSet[T],
        weights: tuple[float, ...] | None = None,
    ) -> None:
        held = (
            (0.0,) * len(targets.categories)
            if weights is None
            else tuple(
                ensure_finite("a root goal's category weight", weight)
                for weight in weights
            )
        )

        if len(held) != len(targets.categories):
            msg = (
                "a root goal needs one weight per target, not "
                f"{len(held)} weights for {len(targets.categories)} targets"
            )
            raise ValueError(msg)

        for position, category in enumerate(targets.categories):
            if any(
                targets.matching.equivalent(category, earlier)
                for earlier in targets.categories[:position]
            ):
                msg = "a root goal gives more than one weight to one matching class"
                raise ValueError(msg)

        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "_weights", held)

    @property
    def targets(self) -> TargetSet[T]:
        """Return the permitted root categories and matching."""
        return self._targets

    @property
    def categories(self) -> tuple[T, ...]:
        """Return the permitted root categories."""
        return self._targets.categories

    @property
    def matching(self) -> Matching[T]:
        """Return the matching used to compare root categories."""
        return self._targets.matching

    @property
    def weights(self) -> tuple[float, ...]:
        """Return target-category log weights in target order."""
        return self._weights

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the stable root-goal identity settings."""
        return {"kind": "root", "weights": list(self._weights)}

    def accepts[W](self, chart: Chart[T, W], key: ItemKey[T], /) -> bool:
        """Decide whether one item spans the input under a named category.

        Parameters
        ----------
        chart : Chart[T, W]
            Completed chart.
        key : ItemKey[T]
            Candidate item.
        """
        if key.span != (0, chart.index.n):
            return False

        return any(
            self._targets.matching.equivalent(key.reading, category)
            for category in self._targets.categories
        )

    def roots[W](self, chart: Chart[T, W], /) -> Iterator[ItemKey[T]]:
        """Enumerate the spanning items whose reading is a named category.

        Parameters
        ----------
        chart : Chart[T, W]
            The finished chart.

        Yields
        ------
        ItemKey
            The accepted items, none where the input has no spanning parse.
        """
        for key in chart.items_at((0, chart.index.n)):
            if self.accepts(chart, key):
                yield key

    def root_groups[W](
        self,
        chart: Chart[T, W],
        /,
    ) -> tuple[tuple[ItemKey[T], ...], ...]:
        """Return the spanning roots as one alternative group."""
        roots = tuple(self.roots(chart))

        return (roots,) if roots else ()

    def root_weight(self, root: ItemKey[T], /) -> float:
        """Return the log weight of one accepted root category."""
        for category, weight in zip(
            self._targets.categories,
            self._weights,
            strict=True,
        ):
            if self._targets.matching.equivalent(root.reading, category):
                return weight

        msg = f"root goal accepts no item {root!r}"
        raise ValueError(msg)

    def combination_weight(self, group_count: int, /) -> float:
        """Return the neutral weight for one spanning root group."""
        del group_count

        return 0.0

    def admits_edge[W](
        self,
        chart: Chart[T, W],
        parent: ItemKey[T],
        edge: Edge[T, W],
        /,
    ) -> bool:
        """Admit every edge in the forest."""
        del chart, parent, edge

        return True

    def __copy__(self) -> RootGoal[T]:
        """Reconstruct the goal with shared dependencies."""
        return _restore_root_goal((self._targets, self._weights))

    def __deepcopy__(self, memo: dict[int, RootGoal[T]]) -> RootGoal[T]:
        """Reconstruct the goal with shared dependencies."""
        del memo

        return _restore_root_goal((self._targets, self._weights))

    @override
    def __reduce__(self) -> _RootGoalReduction[T]:
        """Return the root-goal reconstruction call."""
        return (_restore_root_goal, ((self._targets, self._weights),))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a root goal is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a root goal is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the target set."""
        return f"RootGoal(targets={self._targets!r}, weights={self._weights!r})"


class CoverGoal[T]:
    """Requires a sequence of items tiling the input.

    Parameters
    ----------
    max_fragments : int
        Positive maximum fragment count.
    penalty : float
        Nonnegative log cost per fragment after the first.

    Raises
    ------
    TypeError
        If the fragment bound is not an integer, or the penalty is not a number.
    ValueError
        If the fragment bound is below one, or the penalty is negative or not finite.
    """

    __slots__ = ("_max_fragments", "_penalty", "_penalty_form", "_tiling")

    def __init__(
        self,
        max_fragments: int,
        penalty: float = 0.0,
        *,
        penalty_form: PenaltyForm = PenaltyForm.AFTER_FIRST,
        tiling: Tiling = Tiling.FEWEST,
    ) -> None:
        object.__setattr__(
            self,
            "_max_fragments",
            ensure_non_negative("a cover goal's fragment bound", max_fragments),
        )

        if self._max_fragments < 1:
            msg = "a cover goal's fragment bound is at least one, not 0"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "_penalty",
            ensure_finite("a cover goal's penalty", penalty),
        )

        if self._penalty < 0.0:
            msg = f"a cover goal's penalty is non-negative, not {penalty!r}"
            raise ValueError(msg)

        if not isinstance(penalty_form, PenaltyForm):
            msg = f"a cover goal's penalty form is not a PenaltyForm: {penalty_form!r}"
            raise TypeError(msg)
        if not isinstance(tiling, Tiling):
            msg = f"a cover goal's tiling is not a Tiling: {tiling!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_penalty_form", penalty_form)
        object.__setattr__(self, "_tiling", tiling)

    @property
    def max_fragments(self) -> int:
        """Return the most fragments a cover may use."""
        return self._max_fragments

    @property
    def penalty(self) -> float:
        """Return the log cost of each additional fragment."""
        return self._penalty

    @property
    def penalty_form(self) -> PenaltyForm:
        """Return which fragments incur the penalty."""
        return self._penalty_form

    @property
    def tiling(self) -> Tiling:
        """Return how the goal chooses a cover."""
        return self._tiling

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the stable cover-goal identity settings."""
        return {
            "kind": "cover",
            "max_fragments": self._max_fragments,
            "penalty": self._penalty,
            "penalty_form": self._penalty_form.value,
            "tiling": self._tiling.value,
        }

    def penalty_for(self, fragments: int, /) -> float:
        """Return the log penalty for a fragment count.

        Parameters
        ----------
        fragments : int
            Fragment count.

        Returns
        -------
        float
            Log penalty, with zero for at most one fragment.
        """
        charged = (
            max(fragments - 1, 0)
            if self._penalty_form is PenaltyForm.AFTER_FIRST
            else max(fragments, 0)
        )

        return -self._penalty * charged

    def fragments[W](self, chart: Chart[T, W], /) -> tuple[Cell, ...]:
        """Return the spans of the cover, left to right.

        Parameters
        ----------
        chart : Chart[T, W]
            The finished chart.

        Returns
        -------
        tuple[Cell, ...]
            Half-open cover spans from left to right, or an empty tuple if unavailable.
        """
        n = chart.index.n

        if n == 0:
            return ()

        if self._tiling is Tiling.LEXICAL:
            if n > self._max_fragments:
                return ()

            spans = tuple(chart.index.leaves())
            return spans if all(chart.items_at(span) for span in spans) else ()

        # compute the fewest fragments covering [i, n) from right to left.
        none = n + 1
        best = [none] * (n + 1)
        reach = [0] * (n + 1)
        best[n] = 0

        for i in range(n - 1, -1, -1):
            # descending j prefers the longest fragment on ties.
            for j in range(n, i, -1):
                if best[j] == none or not chart.items_at((i, j)):
                    continue

                if best[j] + 1 < best[i]:
                    best[i] = best[j] + 1
                    reach[i] = j

        if best[0] > min(self._max_fragments, n):
            return ()

        spans: list[Cell] = []
        at = 0

        while at < n:
            spans.append((at, reach[at]))
            at = reach[at]

        return tuple(spans)

    def accepts[W](self, chart: Chart[T, W], key: ItemKey[T], /) -> bool:
        """Decide whether one item is a fragment of the cover.

        Parameters
        ----------
        chart : Chart[T, W]
            The finished chart.
        key : ItemKey[T]
            The item to judge.
        """
        return key.span in self.fragments(chart) and key in chart.items_at(key.span)

    def roots[W](self, chart: Chart[T, W], /) -> Iterator[ItemKey[T]]:
        """Enumerate the items the cover's fragments are read from.

        Parameters
        ----------
        chart : Chart[T, W]
            The finished chart.

        Yields
        ------
        ItemKey
            Items from cover spans in left-to-right order.
        """
        for span in self.fragments(chart):
            yield from chart.items_at(span)

    def root_groups[W](
        self,
        chart: Chart[T, W],
        /,
    ) -> tuple[tuple[ItemKey[T], ...], ...]:
        """Return one alternative root group per required fragment."""
        return tuple(chart.items_at(span) for span in self.fragments(chart))

    def root_weight(self, root: ItemKey[T], /) -> float:
        """Return the neutral log weight of one fragment alternative."""
        del root

        return 0.0

    def combination_weight(self, group_count: int, /) -> float:
        """Return the cover's fragment penalty in log space."""
        return self.penalty_for(group_count)

    def admits_edge[W](
        self,
        chart: Chart[T, W],
        parent: ItemKey[T],
        edge: Edge[T, W],
        /,
    ) -> bool:
        """Admit every edge in the forest."""
        del chart, parent, edge

        return True

    def __copy__(self) -> CoverGoal[T]:
        """Reconstruct the goal from its frozen fields."""
        return _restore_cover_goal(
            (self._max_fragments, self._penalty, self._penalty_form, self._tiling)
        )

    def __deepcopy__(self, memo: dict[int, CoverGoal[T]]) -> CoverGoal[T]:
        """Reconstruct the goal from its frozen fields."""
        del memo

        return _restore_cover_goal(
            (self._max_fragments, self._penalty, self._penalty_form, self._tiling)
        )

    @override
    def __reduce__(self) -> _CoverGoalReduction[T]:
        """Return the cover-goal reconstruction call."""
        return (
            _restore_cover_goal,
            (
                (
                    self._max_fragments,
                    self._penalty,
                    self._penalty_form,
                    self._tiling,
                ),
            ),
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a cover goal is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a cover goal is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the fragment bound and the penalty."""
        return (
            f"CoverGoal(max_fragments={self._max_fragments!r}, "
            f"penalty={self._penalty!r}, penalty_form={self._penalty_form!r}, "
            f"tiling={self._tiling!r})"
        )


def _restore_root_goal[T](
    fields: tuple[TargetSet[T], tuple[float, ...]],
) -> RootGoal[T]:
    """Reconstruct a root goal from its target set."""
    targets, weights = fields

    return RootGoal(targets, weights)


def _restore_cover_goal[T](fields: _CoverGoalFields) -> CoverGoal[T]:
    """Reconstruct a cover goal from frozen fields."""
    max_fragments, penalty, penalty_form, tiling = fields

    return CoverGoal(
        max_fragments,
        penalty,
        penalty_form=penalty_form,
        tiling=tiling,
    )
