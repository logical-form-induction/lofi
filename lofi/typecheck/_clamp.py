"""Clamp enforcement over packed-forest goals."""

from typing import TYPE_CHECKING, Never, Self, final, override

from lofi._checks import ensure_non_negative
from lofi.lexicon import ClampMode, ClampScope

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.deduce import Cell, Chart, Edge, Goal, ItemKey
    from lofi.lexicon import Clamp
    from lofi.terms import Matching
    from lofi.typecheck._ports import ClampProjection

__all__ = ["CategoryProjection", "ClampedGoal", "PairProjection"]


@final
class CategoryProjection[T]:
    """Reads a chart item's reading as its category, with no meaning."""

    __slots__ = ()

    @property
    def has_meaning(self) -> bool:
        """Return whether readings hold a meaning a clamp may pin."""
        return False

    def category(self, reading: T, /) -> T:
        """Return the reading itself, which is the category.

        Parameters
        ----------
        reading : T
            One chart item's reading.
        """
        return reading

    def meaning(self, reading: T, /) -> T | None:
        """Return None, since a bare category holds no meaning.

        Parameters
        ----------
        reading : T
            One chart item's reading.
        """
        del reading

        return None

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this projection."""
        return "CategoryProjection()"


@final
class PairProjection[T]:
    """Reads a category and a meaning off a reading that pairs them."""

    __slots__ = ()

    @property
    def has_meaning(self) -> bool:
        """Return whether readings hold a meaning a clamp may pin."""
        return True

    def category(self, reading: tuple[T, T], /) -> T:
        """Return the first component of the pair.

        Parameters
        ----------
        reading : tuple[T, T]
            One chart item's reading, as a category and a meaning.
        """
        return reading[0]

    def meaning(self, reading: tuple[T, T], /) -> T | None:
        """Return the second component of the pair.

        Parameters
        ----------
        reading : tuple[T, T]
            One chart item's reading, as a category and a meaning.
        """
        return reading[1]

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this projection."""
        return "PairProjection()"


def _checked_clamp[T, R](
    clamp: Clamp[T],
    projection: ClampProjection[T, R],
) -> Clamp[T]:
    """Validate and return a clamp against the readings it will test.

    Parameters
    ----------
    clamp : Clamp[T]
        Clamp to validate.
    projection : ClampProjection[T, R]
        How the readings the clamp tests are read.

    Returns
    -------
    Clamp[T]
        The validated clamp.

    Raises
    ------
    TypeError
        If a span endpoint is not an integer.
    ValueError
        If the span is empty, the clamp names neither a category nor a meaning,
        or the clamp pins a meaning over readings that hold none.
    """
    start, end = clamp.span
    ensure_non_negative("a clamped span's start", start)
    ensure_non_negative("a clamped span's end", end)

    if end <= start:
        msg = f"a clamp covers no position over the span [{start}, {end})"
        raise ValueError(msg)

    if not clamp.require_constituent and clamp.scope is ClampScope.OCCURRENCE:
        msg = "a token-scoped clamp cannot exclude a constituent"
        raise ValueError(msg)

    if not clamp.require_constituent and (clamp.category or clamp.meaning is not None):
        msg = "a negative-bracket clamp names no category or meaning"
        raise ValueError(msg)

    if clamp.require_constituent and not clamp.category and clamp.meaning is None:
        msg = f"a clamp names the category or the meaning it pins: {clamp!r}"
        raise ValueError(msg)

    if clamp.meaning is not None and not projection.has_meaning:
        msg = f"a clamp pins a meaning, but the readings hold none: {clamp!r}"
        raise ValueError(msg)

    return clamp


def _within(inner: Cell, outer: Cell) -> bool:
    """Return whether one span lies inside another, equality included.

    Parameters
    ----------
    inner : Cell
        The candidate inner span.
    outer : Cell
        The candidate outer span.
    """
    return outer[0] <= inner[0] and inner[1] <= outer[1]


@final
class ClampedGoal[T, R]:
    """A goal narrowed by clamps on the readings of chart items.

    A clamp requires its span to be a constituent with an admitted category,
    meaning, or both; negation reverses that test. Root clamps test the root,
    while inner clamps filter crossing edges and test the exact child once.
    Clamps outside every root, or clamps that remove every root, admit no parse.

    Parameters
    ----------
    base : Goal[R]
        The goal that decides roots and edges before clamping.
    clamps : Iterable[Clamp[T]]
        The clamps to enforce, each over a non-empty span and naming a category
        set, a meaning, or both.
    projection : ClampProjection[T, R]
        How a category and a meaning are read off a chart item's reading.
    matching : Matching[T]
        The congruence the chart's categories are packed under, which decides
        whether a reading's category is one a clamp names.

    Raises
    ------
    TypeError
        If a span endpoint is not an integer.
    ValueError
        If no clamp is supplied, a span is empty, a clamp names neither a
        category nor a meaning, or a clamp pins a meaning over readings that
        hold none.
    """

    __slots__ = (
        "_base",
        "_clamps",
        "_matching",
        "_meaning_matching",
        "_projection",
    )

    def __init__(
        self,
        base: Goal[R],
        clamps: Iterable[Clamp[T]],
        projection: ClampProjection[T, R],
        matching: Matching[T],
        *,
        meaning_matching: Matching[T] | None = None,
    ) -> None:
        enforced = tuple(_checked_clamp(clamp, projection) for clamp in clamps)

        if not enforced:
            msg = "a clamped goal with no clamp narrows nothing"
            raise ValueError(msg)

        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_clamps", enforced)
        object.__setattr__(self, "_projection", projection)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(
            self,
            "_meaning_matching",
            matching if meaning_matching is None else meaning_matching,
        )

    @property
    def base(self) -> Goal[R]:
        """Return the goal narrowed by the clamps."""
        return self._base

    @property
    def clamps(self) -> tuple[Clamp[T], ...]:
        """Return the enforced clamps in declaration order."""
        return self._clamps

    @property
    def projection(self) -> ClampProjection[T, R]:
        """Return how a category and a meaning are read off a reading."""
        return self._projection

    @property
    def matching(self) -> Matching[T]:
        """Return the congruence clamp categories are compared under."""
        return self._matching

    @property
    def meaning_matching(self) -> Matching[T]:
        """Return the congruence clamp meanings are compared under."""
        return self._meaning_matching

    def accepts[W](self, chart: Chart[R, W], key: ItemKey[R], /) -> bool:
        """Decide whether an item is one of the roots the clamps leave.

        Parameters
        ----------
        chart : Chart[R, W]
            The finished chart.
        key : ItemKey[R]
            The item to judge.

        Returns
        -------
        bool
            Whether the base accepts the item, the clamps at its span pass on
            it, and the chart keeps a parse at all, so that an item is accepted
            exactly when it is enumerated.

        Raises
        ------
        ValueError
            If a clamped span reaches past the input.
        """
        return (
            self._base.accepts(chart, key)
            and self._passes_at_span(key)
            and bool(self.root_groups(chart))
        )

    def roots[W](self, chart: Chart[R, W], /) -> Iterator[ItemKey[R]]:
        """Enumerate the base's roots that survive the clamps.

        Parameters
        ----------
        chart : Chart[R, W]
            The finished chart.

        Yields
        ------
        ItemKey[R]
            The surviving roots, none where any goal component loses every root.
        """
        for group in self.root_groups(chart):
            yield from group

    def root_groups[W](
        self,
        chart: Chart[R, W],
        /,
    ) -> tuple[tuple[ItemKey[R], ...], ...]:
        """Return the base's root groups with the clamps applied at their spans.

        Parameters
        ----------
        chart : Chart[R, W]
            The finished chart.

        Returns
        -------
        tuple[tuple[ItemKey[R], ...], ...]
            The base's groups, each narrowed to the roots passing the clamps at
            its span, or an empty tuple when a group loses every root or a
            clamped span lies inside no root.

        Raises
        ------
        ValueError
            If a clamped span reaches past the input.
        """
        n = chart.index.n

        for clamp in self._constituent_clamps():
            if clamp.span[1] > n:
                msg = (
                    f"the clamp on the span {clamp.span!r} reaches past the "
                    f"{n}-token input"
                )
                raise ValueError(msg)

        groups: list[tuple[ItemKey[R], ...]] = []

        for group in self._base.root_groups(chart):
            kept = tuple(root for root in group if self._passes_at_span(root))

            if not kept:
                return ()

            groups.append(kept)

        spans = {root.span for group in groups for root in group}

        for clamp in self._constituent_clamps():
            if not any(_within(clamp.span, span) for span in spans):
                return ()

        return tuple(groups)

    def root_weight(self, root: ItemKey[R], /) -> float:
        """Return the base's local log weight of one root alternative.

        Parameters
        ----------
        root : ItemKey[R]
            Root alternative being weighed.
        """
        return self._base.root_weight(root)

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the base goal's stable settings."""
        return self._base.settings

    def combination_weight(self, group_count: int, /) -> float:
        """Return the base's shared log weight of a root combination."""
        return self._base.combination_weight(group_count)

    def admits_edge[W](
        self,
        chart: Chart[R, W],
        parent: ItemKey[R],
        edge: Edge[R, W],
        /,
    ) -> bool:
        """Return whether the base admits an edge that keeps every clamped span.

        Parameters
        ----------
        chart : Chart[R, W]
            The finished chart.
        parent : ItemKey[R]
            The item the edge concludes.
        edge : Edge[R, W]
            The step to judge.

        Returns
        -------
        bool
            Whether the base admits the edge and, for every clamped span the
            parent strictly contains, a child contains the span and a child
            covering it exactly passes the clamp.
        """
        if not self._base.admits_edge(chart, parent, edge):
            return False

        start, end = parent.span

        for clamp in self._constituent_clamps():
            s, e = clamp.span

            if start > s or e > end or (start == s and end == e):
                continue

            if not clamp.require_constituent:
                if any(child.span == clamp.span for child in edge.children):
                    return False

                continue

            for child in edge.children:
                a, b = child.span

                if a <= s and e <= b:
                    if a == s and b == e and not self._passes(clamp, child.reading):
                        return False

                    break

            else:
                return False

        return True

    def _passes_at_span(self, key: ItemKey[R]) -> bool:
        """Return whether every clamp at an item's own span passes on it.

        Parameters
        ----------
        key : ItemKey[R]
            The item to test.
        """
        return all(
            self._passes(clamp, key.reading)
            for clamp in self._constituent_clamps()
            if clamp.span == key.span
        )

    def _constituent_clamps(self) -> tuple[Clamp[T], ...]:
        """Return hard constraints whose constituency the goal decides."""
        return tuple(
            clamp
            for clamp in self._clamps
            if clamp.mode is ClampMode.HARD and clamp.scope is ClampScope.CONSTITUENT
        )

    def _passes(self, clamp: Clamp[T], reading: R) -> bool:
        """Return whether a clamp allows a reading at its span.

        Parameters
        ----------
        clamp : Clamp[T]
            The clamp to apply.
        reading : R
            The reading to test.
        """
        if not clamp.require_constituent:
            return False

        named = True

        if clamp.category:
            category = self._projection.category(reading)
            named = any(
                self._matching.equivalent(category, held) for held in clamp.category
            )

        if named and clamp.meaning is not None:
            meaning = self._projection.meaning(reading)
            named = meaning is not None and self._meaning_matching.equivalent(
                meaning, clamp.meaning
            )

        return named is not clamp.negate

    def __copy__(self) -> Self:
        """Return this immutable goal."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable goal."""
        del memo

        return self

    @override
    def __reduce__(self) -> Never:
        """Refuse serialization of clamps over live term identities."""
        msg = "a clamped goal holds live term identities and cannot be pickled"
        raise TypeError(msg)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a clamped goal is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a clamped goal is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the base and how many clamps narrow it."""
        return (
            f"ClampedGoal({self._base!r}, clamps={len(self._clamps)}, "
            f"matching={self._matching.name!r})"
        )
