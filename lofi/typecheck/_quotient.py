"""The logical-form chart, keyed by span and normal-form meaning."""

from math import fsum, isclose, isfinite
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Never,
    Protocol,
    final,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.registry import Registry
from lofi.terms import Normalization, normalize

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.deduce import Cell, SplitProbabilities
    from lofi.terms import Calculus, Term, TermStore

__all__ = [
    "DISTINCT_MEANING_WEIGHTING",
    "MULTIPLICITY_MEANING_WEIGHTING",
    "LogicalFormChart",
    "MeaningItem",
    "MeaningSource",
    "MeaningWeighting",
    "MeaningWeightingPlugin",
    "meaning_shares",
    "meaning_weighting",
]


@runtime_checkable
class MeaningWeighting(Protocol):
    """Assign relative mass to distinct meanings reached at one split."""

    @property
    def name(self) -> str:
        """Return the installed policy name."""

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        """Return normalized shares keyed by distinct meaning."""


def meaning_shares(
    weighting: MeaningWeighting,
    reached: Sequence[Term],
    /,
) -> Mapping[Term, float]:
    """Validate one policy result and return its positive shares.

    A policy accounts for every distinct reached meaning exactly once. Zero
    shares are valid, but are absent from the result so logarithmic consumers
    never construct an impossible route.
    """
    held = tuple(reached)

    if not held:
        msg = "meaning weighting needs at least one reached meaning"
        raise ValueError(msg)

    distinct = tuple(dict.fromkeys(held))
    supplied = dict(weighting.shares(held))

    if set(supplied) != set(distinct):
        missing = tuple(meaning for meaning in distinct if meaning not in supplied)
        extra = tuple(meaning for meaning in supplied if meaning not in distinct)
        msg = (
            "a meaning-weighting result must account for exactly the reached "
            f"meanings; missing={missing!r}, extra={extra!r}"
        )
        raise ValueError(msg)

    for meaning, share in supplied.items():
        if not isinstance(share, float) or not isfinite(share) or share < 0.0:
            msg = (
                "a meaning-weighting share is a finite nonnegative float: "
                f"{meaning!r} has {share!r}"
            )
            raise ValueError(msg)

    total = fsum(supplied.values())

    if not isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        msg = f"meaning-weighting shares sum to one, not {total!r}"
        raise ValueError(msg)

    return MappingProxyType(
        {meaning: supplied[meaning] for meaning in distinct if supplied[meaning] > 0.0}
    )


@final
class _DistinctMeaningWeighting(FrozenValue):
    """Divide a split's mass uniformly among distinct meanings."""

    _noun = "meaning-weighting policy"
    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the installed policy name."""
        return "distinct"

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        """Return one equal share per distinct meaning."""
        distinct = tuple(dict.fromkeys(reached))
        return dict.fromkeys(distinct, 1.0 / len(distinct))


@final
class _MultiplicityMeaningWeighting(FrozenValue):
    """Divide a split's mass in proportion to derivational multiplicity."""

    _noun = "meaning-weighting policy"
    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the installed policy name."""
        return "multiplicity"

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        """Return each meaning's share of derivational contributions."""
        distinct = tuple(dict.fromkeys(reached))
        return {meaning: reached.count(meaning) / len(reached) for meaning in distinct}


DISTINCT_MEANING_WEIGHTING: Final[MeaningWeighting] = _DistinctMeaningWeighting()
"""The built-in uniform-over-distinct-meanings policy."""

MULTIPLICITY_MEANING_WEIGHTING: Final[MeaningWeighting] = (
    _MultiplicityMeaningWeighting()
)
"""The built-in derivational-multiplicity policy."""


class MeaningWeightingPlugin(FrozenValue):
    """Build one immutable meaning-weighting policy."""

    _noun = "meaning-weighting plugin"
    __slots__ = ("_build",)

    def __init__(self, build: Callable[[], MeaningWeighting], /) -> None:
        object.__setattr__(self, "_build", build)

    def construct(self) -> MeaningWeighting:
        """Build and validate the plugin's weighting policy."""
        weighting = self._build()

        if not isinstance(weighting, MeaningWeighting):
            msg = "a meaning-weighting plugin built no weighting policy"
            raise TypeError(msg)

        ensure_name("a meaning-weighting policy", weighting.name)
        return weighting


def meaning_weighting(name: str, /) -> MeaningWeighting:
    """Build a named weighting through its unique installed plugin."""
    held = ensure_name("a meaning weighting", name)
    member = Registry[object]("meaning_weighting").get(held)

    if not isinstance(member, MeaningWeightingPlugin):
        msg = f"the meaning-weighting entry {held!r} names no plugin"
        raise TypeError(msg)

    weighting = member.construct()

    if weighting.name != held:
        msg = f"the meaning-weighting entry {held!r} built policy {weighting.name!r}"
        raise ValueError(msg)

    return weighting


DISTINCT_PLUGIN: Final = MeaningWeightingPlugin(_DistinctMeaningWeighting)
MULTIPLICITY_PLUGIN: Final = MeaningWeightingPlugin(_MultiplicityMeaningWeighting)
"""Installed builders for the package's meaning-weighting policies."""


class MeaningSource(NamedTuple):
    """One derivation's logical form at a span, before normalization.

    Attributes
    ----------
    split : int
        The binary split point the derivation used.
    form : Term
        The logical form the derivation produced, as the rule left it.
    """

    split: int
    form: Term


@final
class MeaningItem:
    """One distinct meaning at one span, with the weight the quotient gives it.

    Two items are equal when they hold the same span and the same normal form.
    The weight, the splits and the multiplicity are read off the contributions
    and take no part in equality.

    Parameters
    ----------
    span : Cell
        The half-open token interval the meaning covers.
    meaning : Term
        The normal form the contributions share.
    weight : float
        The posterior mass the declared weighting gives the meaning.
    splits : tuple[int, ...]
        The distinct split points that produced the meaning, ascending.
    multiplicity : int
        How many contributions produced the meaning.
    """

    __slots__ = ("_meaning", "_multiplicity", "_span", "_splits", "_weight")

    def __init__(
        self,
        span: Cell,
        meaning: Term,
        weight: float,
        splits: tuple[int, ...],
        multiplicity: int,
        /,
    ) -> None:
        object.__setattr__(self, "_span", span)
        object.__setattr__(self, "_meaning", meaning)
        object.__setattr__(self, "_weight", weight)
        object.__setattr__(self, "_splits", splits)
        object.__setattr__(self, "_multiplicity", multiplicity)

    @property
    def span(self) -> Cell:
        """Return the half-open token interval the meaning covers."""
        return self._span

    @property
    def meaning(self) -> Term:
        """Return the normal form the contributions share."""
        return self._meaning

    @property
    def weight(self) -> float:
        """Return the posterior mass the declared weighting gives the meaning."""
        return self._weight

    @property
    def splits(self) -> tuple[int, ...]:
        """Return the distinct split points that produced the meaning, ascending."""
        return self._splits

    @property
    def multiplicity(self) -> int:
        """Return how many contributions produced the meaning."""
        return self._multiplicity

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another item holds the same span and normal form."""
        return (
            type(other) is MeaningItem
            and self._span == other._span
            and self._meaning == other._meaning
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the span and the normal form."""
        return hash((MeaningItem, self._span, self._meaning))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a meaning item is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a meaning item is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this item."""
        return (
            f"MeaningItem({self._span!r}, {self._meaning!r}, {self._weight!r}, "
            f"{self._splits!r}, {self._multiplicity!r})"
        )


def _row(alpha: SplitProbabilities, span: Cell) -> Mapping[int, float]:
    """Return one span's split posterior, or refuse a span it does not declare.

    Parameters
    ----------
    alpha : SplitProbabilities
        The split posterior to read.
    span : Cell
        The span whose row to read.
    """
    try:
        return alpha.at(span)

    except KeyError:
        msg = f"the split posterior declares no row for span {span!r}"

        raise ValueError(msg) from None


@final
class LogicalFormChart:
    """A logical-form chart quotiented by declared normal meaning.

    Alpha variants and forms with one declared normal form share a meaning. Each split's
    posterior mass is divided either among distinct meanings or among derivational
    contributions, then summed by meaning across splits.

    Parameters
    ----------
    store : TermStore
        The store the contributions and their normal forms are interned in.
    contributions : Mapping[Cell, Sequence[MeaningSource]]
        The logical forms each span received, in derivation order.
    alpha : SplitProbabilities
        The split posterior the weights are read from.
    calculus : Calculus
        The abstraction and application operators reduction runs under.
    normalization : Normalization
        The beta order, eta treatment, AC quotient and fuel.
    weighting : MeaningWeighting
        Which weight a meaning several derivations reach receives.

    Raises
    ------
    ValueError
        If the weighting is undeclared, if a span received nothing, if the
        posterior declares no row for a span, or if a contribution names a split
        the posterior does not score.
    """

    __slots__ = ("_alpha_mass", "_items", "_sources", "_weighting")

    def __init__(
        self,
        store: TermStore,
        contributions: Mapping[Cell, Sequence[MeaningSource]],
        alpha: SplitProbabilities,
        /,
        *,
        calculus: Calculus,
        normalization: Normalization,
        weighting: MeaningWeighting,
    ) -> None:
        if not isinstance(weighting, MeaningWeighting):
            msg = "a logical-form chart needs a meaning-weighting policy"
            raise TypeError(msg)

        held = weighting
        items: dict[Cell, dict[Term, MeaningItem]] = {}
        sources: dict[Cell, int] = {}
        mass: dict[Cell, float] = {}

        for span in sorted(contributions):
            supplied = tuple(contributions[span])

            if not supplied:
                msg = f"span {span!r} received no logical form"

                raise ValueError(msg)

            row = _row(alpha, span)
            reached = _normalized(
                store,
                supplied,
                row,
                span,
                calculus=calculus,
                normalization=normalization,
            )
            items[span] = _quotient(span, reached, row, weighting=held)
            sources[span] = len(supplied)
            mass[span] = fsum(
                row[split] for split in sorted({source.split for source in supplied})
            )

        object.__setattr__(self, "_weighting", held)
        object.__setattr__(self, "_items", items)
        object.__setattr__(self, "_sources", sources)
        object.__setattr__(self, "_alpha_mass", mass)

    @property
    def weighting(self) -> MeaningWeighting:
        """Return the declared weighting the items were built under."""
        return self._weighting

    @property
    def spans(self) -> tuple[Cell, ...]:
        """Return the quotiented spans, ascending."""
        return tuple(self._items)

    @property
    def distinct_total(self) -> int:
        """Return how many distinct meanings the whole chart holds."""
        return sum(len(row) for row in self._items.values())

    @property
    def contribution_total(self) -> int:
        """Return how many contributions the whole chart received."""
        return sum(self._sources.values())

    def meanings(self, span: Cell, /) -> tuple[MeaningItem, ...]:
        """Return one span's distinct meanings, in first-contribution order.

        Parameters
        ----------
        span : Cell
            The half-open token interval to read.
        """
        return tuple(self._items[span].values())

    def weight(self, span: Cell, meaning: Term, /) -> float:
        """Return the weight one meaning has at one span.

        Parameters
        ----------
        span : Cell
            The half-open token interval to read.
        meaning : Term
            The normal form to look the weight up by.

        Raises
        ------
        KeyError
            If no contribution at that span normalizes to that meaning.
        """
        return self._items[span][meaning].weight

    def contribution_count(self, span: Cell, /) -> int:
        """Return how many contributions one span received.

        Parameters
        ----------
        span : Cell
            The half-open token interval to read.
        """
        return self._sources[span]

    def alpha_mass(self, span: Cell, /) -> float:
        """Return the posterior mass of the distinct split points one span used.

        Parameters
        ----------
        span : Cell
            The half-open token interval to read.
        """
        return self._alpha_mass[span]

    def weight_mass(self, span: Cell, /) -> float:
        """Return the summed weight of one span's distinct meanings.

        Parameters
        ----------
        span : Cell
            The half-open token interval to read.
        """
        return fsum(item.weight for item in self._items[span].values())

    def multiplicity_ratio(self) -> float:
        """Return contributions per distinct meaning over the whole chart.

        Raises
        ------
        ValueError
            If the chart holds no meaning, so no ratio is defined.
        """
        distinct = self.distinct_total

        if not distinct:
            msg = "a chart holding no meaning has no multiplicity ratio"

            raise ValueError(msg)

        return self.contribution_total / distinct

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a logical-form chart is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a logical-form chart is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return the weighting, the span count and the two totals."""
        return (
            f"LogicalFormChart(weighting={self._weighting.name!r}, "
            f"spans={len(self._items)}, distinct={self.distinct_total}, "
            f"contributions={self.contribution_total})"
        )


def _normalized(
    store: TermStore,
    supplied: Sequence[MeaningSource],
    row: Mapping[int, float],
    span: Cell,
    *,
    calculus: Calculus,
    normalization: Normalization,
) -> tuple[tuple[int, Term], ...]:
    """Normalize a span's contributions, refusing a split the posterior omits.

    Parameters
    ----------
    store : TermStore
        The store that interns the terms.
    supplied : Sequence[MeaningSource]
        The contributions to normalize.
    row : Mapping[int, float]
        The split posterior for the span.
    span : Cell
        The span receiving the contributions.
    calculus : Calculus
        The operators under which to normalize.
    normalization : Normalization
        The beta order, eta treatment, AC quotient and fuel.
    """
    reached: list[tuple[int, Term]] = []

    for source in supplied:
        if source.split not in row:
            msg = (
                f"the split posterior for span {span!r} does not score split "
                f"{source.split}"
            )

            raise ValueError(msg)

        reached.append(
            (
                source.split,
                normalize(
                    store,
                    source.form,
                    calculus,
                    normalization=normalization,
                ),
            )
        )

    return tuple(reached)


def _shares(
    reached: Sequence[Term],
    mass: float,
    *,
    weighting: MeaningWeighting,
) -> dict[Term, float]:
    """Divide one split's posterior mass among the meanings it reached.

    Parameters
    ----------
    reached : Sequence[Term]
        The meanings reached at the split.
    mass : float
        The split's posterior mass.
    weighting : MeaningWeighting
        The rule for dividing the mass.
    """
    return {
        meaning: mass * share
        for meaning, share in meaning_shares(weighting, reached).items()
    }


def _quotient(
    span: Cell,
    reached: Sequence[tuple[int, Term]],
    row: Mapping[int, float],
    *,
    weighting: MeaningWeighting,
) -> dict[Term, MeaningItem]:
    """Build one span's items, keyed by normal form in first-contribution order.

    Parameters
    ----------
    span : Cell
        The span receiving the items.
    reached : Sequence[tuple[int, Term]]
        The split and normal form of each contribution.
    row : Mapping[int, float]
        The split posterior for the span.
    weighting : MeaningWeighting
        The rule for weighting repeated meanings.
    """
    by_split: dict[int, list[Term]] = {}
    shares: dict[Term, list[float]] = {}
    splits: dict[Term, set[int]] = {}
    counts: dict[Term, int] = {}

    for split, meaning in reached:
        by_split.setdefault(split, []).append(meaning)
        shares.setdefault(meaning, [])
        splits.setdefault(meaning, set()).add(split)
        counts[meaning] = counts.get(meaning, 0) + 1

    for split, meanings in by_split.items():
        for meaning, share in _shares(
            meanings, row[split], weighting=weighting
        ).items():
            shares[meaning].append(share)

    return {
        meaning: MeaningItem(
            span,
            meaning,
            fsum(shares[meaning]),
            tuple(sorted(splits[meaning])),
            counts[meaning],
        )
        for meaning in shares
    }
