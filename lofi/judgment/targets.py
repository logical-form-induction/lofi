"""Judgment targets stratified by item-level agreement.

Each row receives its item's response distribution, its recorded response,
and a low, medium, or high agreement stratum.
"""

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

import torch

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.judgment._spec import AgreementBin

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor

    from lofi.judgment._design import BoundedDesign, JudgmentDesign
    from lofi.judgment._ports import StratifiedTarget, StratifiedTargets

__all__ = [
    "AgreementStrata",
    "ItemTarget",
    "JudgmentTargets",
    "ProbabilityTargets",
    "StratifiedMeans",
    "StratumMean",
    "TargetRow",
    "agreement_of",
    "stratified_mean",
]


def agreement_of(distribution: Sequence[float]) -> float:
    """Return how far a distribution over levels is from uniform.

    The agreement is one minus the distribution's Shannon entropy divided by
    the entropy of the uniform distribution over the same levels, so it is
    one where every response is the same level and zero where every level
    is equally likely.

    Parameters
    ----------
    distribution : Sequence[float]
        Probability of each level, over at least two levels.

    Raises
    ------
    ValueError
        If fewer than two levels are given, a probability is negative, or
        the probabilities do not sum to one.
    """
    if len(distribution) < 2:
        msg = f"agreement is over at least two levels, and {len(distribution)} given"

        raise ValueError(msg)

    if any(probability < 0.0 for probability in distribution):
        msg = f"a distribution has a negative probability: {tuple(distribution)!r}"

        raise ValueError(msg)

    total = math.fsum(distribution)

    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        msg = f"a distribution's probabilities sum to {total!r}, not one"

        raise ValueError(msg)

    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in distribution
        if probability > 0.0
    )

    return 1.0 - entropy / math.log(len(distribution))


def _tertile(ordered: Sequence[float], which: int) -> float:
    """Return a tertile of sorted values by linear interpolation.

    The cut lies at the position ``which`` thirds of the way along the
    values, interpolated between the two values around it, so equal
    neighbors give the cut exactly.

    Parameters
    ----------
    ordered : Sequence[float]
        Values in ascending order, at least one.
    which : int
        The tertile wanted, one or two.
    """
    position = (len(ordered) - 1) * which / 3
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    fraction = position - below

    return ordered[below] + fraction * (ordered[above] - ordered[below])


@final
class AgreementStrata:
    """Cuts of the item agreements into the low, medium and high strata.

    The cuts are the lower and upper tertiles. Values below the lower cut are
    low, values at or above the upper cut are high, and the rest are medium.
    If both cuts coincide, every equal value is high.

    Parameters
    ----------
    agreements : Iterable[float]
        Agreement of each item.

    Raises
    ------
    ValueError
        If no agreement is given, or one is not finite.
    """

    __slots__ = ("_lower", "_upper")

    def __init__(self, agreements: Iterable[float]) -> None:
        values = [ensure_finite("an item agreement", value) for value in agreements]

        if not values:
            msg = "agreement strata are cut over at least one item"

            raise ValueError(msg)

        ordered = sorted(values)
        self._lower = _tertile(ordered, 1)
        self._upper = _tertile(ordered, 2)

    @property
    def lower(self) -> float:
        """Return the agreement below which an item is in the low stratum."""
        return self._lower

    @property
    def upper(self) -> float:
        """Return the agreement from which an item is in the high stratum."""
        return self._upper

    def bin_of(self, agreement: float, /) -> AgreementBin:
        """Return the stratum of an item.

        Parameters
        ----------
        agreement : float
            The item's agreement.
        """
        if agreement < self._lower:
            return AgreementBin.LOW

        if agreement >= self._upper:
            return AgreementBin.HIGH

        return AgreementBin.MEDIUM

    @override
    def __repr__(self) -> str:
        """Return the two cuts."""
        return f"AgreementStrata(lower={self._lower!r}, upper={self._upper!r})"


@final
class ItemTarget:
    """The population distribution of one item's responses and its stratum.

    Parameters
    ----------
    item : str
        Item identifier.
    counts : tuple[int, ...]
        Number of responses at each level, at least one in all.
    agreement : float
        Agreement of the distribution.
    stratum : AgreementBin
        Stratum the item falls in.

    Raises
    ------
    ValueError
        If no response is counted.
    """

    __slots__ = ("_agreement", "_counts", "_distribution", "_item", "_stratum")

    def __init__(
        self,
        item: str,
        counts: tuple[int, ...],
        agreement: float,
        stratum: AgreementBin,
    ) -> None:
        total = sum(ensure_non_negative(f"a count of {item!r}", n) for n in counts)

        if total == 0:
            msg = f"the item {item!r} has no response to take a distribution over"

            raise ValueError(msg)

        self._item = item
        self._counts = counts
        self._distribution = tuple(count / total for count in counts)
        self._agreement = agreement
        self._stratum = stratum

    @property
    def item(self) -> str:
        """Return the item identifier."""
        return self._item

    @property
    def counts(self) -> tuple[int, ...]:
        """Return the number of responses at each level."""
        return self._counts

    @property
    def responses(self) -> int:
        """Return the number of responses."""
        return sum(self._counts)

    @property
    def distribution(self) -> tuple[float, ...]:
        """Return the proportion of responses at each level."""
        return self._distribution

    @property
    def agreement(self) -> float:
        """Return the agreement of the distribution."""
        return self._agreement

    @property
    def stratum(self) -> AgreementBin:
        """Return the stratum the item falls in."""
        return self._stratum

    @override
    def __repr__(self) -> str:
        """Return the item, its counts and its stratum."""
        return (
            f"ItemTarget({self._item!r}, {self._counts!r}, {self._agreement!r}, "
            f"{self._stratum!r})"
        )


@final
class TargetRow:
    """The two prediction targets of one judgment and its item's stratum.

    Parameters
    ----------
    participant : str
        Participant who gave the response.
    item : ItemTarget
        The item's population distribution and stratum.
    response : int
        Zero-based level of the participant's own response.

    Raises
    ------
    ValueError
        If the response is outside the item's levels.
    """

    __slots__ = ("_item", "_participant", "_response")

    def __init__(self, participant: str, item: ItemTarget, response: int) -> None:
        if not 0 <= response < len(item.counts):
            msg = (
                f"a response of {response} lies outside the {len(item.counts)} "
                f"levels of {item.item!r}"
            )

            raise ValueError(msg)

        self._participant = participant
        self._item = item
        self._response = response

    @property
    def participant(self) -> str:
        """Return the participant identifier."""
        return self._participant

    @property
    def item(self) -> str:
        """Return the item identifier."""
        return self._item.item

    @property
    def response(self) -> int:
        """Return the participant's own response, the perspectivist target."""
        return self._response

    @property
    def distribution(self) -> tuple[float, ...]:
        """Return the item's population distribution, the soft-label target."""
        return self._item.distribution

    @property
    def agreement(self) -> float:
        """Return the item's agreement."""
        return self._item.agreement

    @property
    def stratum(self) -> AgreementBin:
        """Return the item's stratum."""
        return self._item.stratum

    @override
    def __repr__(self) -> str:
        """Return the participant, the item, the response and the stratum."""
        return (
            f"TargetRow({self._participant!r}, {self.item!r}, {self._response!r}, "
            f"{self.stratum!r})"
        )


@final
class JudgmentTargets:
    """The prediction targets of every row of a design.

    Each item's distribution is the proportion of the design's responses to
    it at each level, its agreement is that distribution's, and its stratum
    is cut over the items' agreements. The rows are in the design's order,
    one per judgment.

    Parameters
    ----------
    design : JudgmentDesign
        Design with at least one row.

    Raises
    ------
    ValueError
        If the design has no rows.
    """

    __slots__ = ("_items", "_rows", "_strata")

    def __init__(self, design: JudgmentDesign) -> None:
        if design.n_rows == 0:
            msg = (
                "targets are taken over at least one judgment, and the design has none"
            )

            raise ValueError(msg)

        levels = design.response_levels
        cells = design.items * levels + design.responses
        counts = torch.bincount(cells, minlength=len(design.item_levels) * levels)
        counted = counts.reshape(len(design.item_levels), levels).tolist()
        agreements = [
            agreement_of([count / sum(row) for count in row]) for row in counted
        ]
        strata = AgreementStrata(agreements)
        items = tuple(
            ItemTarget(name, tuple(row), agreement, strata.bin_of(agreement))
            for name, row, agreement in zip(
                design.item_levels, counted, agreements, strict=True
            )
        )
        self._strata = strata
        self._items: Mapping[str, ItemTarget] = MappingProxyType(
            {target.item: target for target in items}
        )
        self._rows = tuple(
            TargetRow(design.participant_levels[participant], items[item], response)
            for participant, item, response in zip(
                design.participants.tolist(),
                design.items.tolist(),
                design.responses.tolist(),
                strict=True,
            )
        )

    @property
    def strata(self) -> AgreementStrata:
        """Return the cuts the items are stratified by."""
        return self._strata

    @property
    def items(self) -> Mapping[str, ItemTarget]:
        """Return each item's target, by identifier."""
        return self._items

    @property
    def rows(self) -> tuple[TargetRow, ...]:
        """Return one row per judgment, in the design's order."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the row and item counts and the cuts."""
        return (
            f"JudgmentTargets(rows={len(self._rows)}, items={len(self._items)}, "
            f"strata={self._strata!r})"
        )


@final
class _ProbabilityItemTarget(FrozenValue):
    """One probability item's mean response and uncertainty stratum."""

    _noun = "probability item target"

    __slots__ = ("_agreement", "_item", "_response", "_stratum")

    def __init__(
        self,
        item: str,
        response: float,
        agreement: float,
        stratum: AgreementBin,
    ) -> None:
        response = ensure_finite(f"the mean response of {item!r}", response)

        if not 0.0 <= response <= 1.0:
            msg = f"the mean response of {item!r} is not a probability: {response!r}"

            raise ValueError(msg)

        object.__setattr__(self, "_item", item)
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "_agreement", agreement)
        object.__setattr__(self, "_stratum", stratum)

    @property
    def item(self) -> str:
        """Return the item identifier."""
        return self._item

    @property
    def response(self) -> float:
        """Return the item's mean unit interval response."""
        return self._response

    @property
    def agreement(self) -> float:
        """Return the item's distance from maximum binary uncertainty."""
        return self._agreement

    @property
    def stratum(self) -> AgreementBin:
        """Return the item's uncertainty stratum."""
        return self._stratum

    @override
    def __repr__(self) -> str:
        """Return the item, mean response, agreement and stratum."""
        return (
            f"_ProbabilityItemTarget({self._item!r}, {self._response!r}, "
            f"{self._agreement!r}, {self._stratum!r})"
        )


@final
class _ProbabilityTargetRow(FrozenValue):
    """One probability response beside its item's uncertainty stratum."""

    _noun = "probability target row"

    __slots__ = ("_item", "_participant", "_response")

    def __init__(
        self,
        participant: str,
        item: _ProbabilityItemTarget,
        response: float,
    ) -> None:
        response = ensure_finite(f"a response to {item.item!r}", response)

        if not 0.0 <= response <= 1.0:
            msg = f"a response to {item.item!r} is not a probability: {response!r}"

            raise ValueError(msg)

        object.__setattr__(self, "_participant", participant)
        object.__setattr__(self, "_item", item)
        object.__setattr__(self, "_response", response)

    @property
    def participant(self) -> str:
        """Return the participant identifier."""
        return self._participant

    @property
    def item(self) -> str:
        """Return the item identifier."""
        return self._item.item

    @property
    def response(self) -> float:
        """Return the response on the closed unit interval."""
        return self._response

    @property
    def agreement(self) -> float:
        """Return the item's distance from maximum binary uncertainty."""
        return self._item.agreement

    @property
    def stratum(self) -> AgreementBin:
        """Return the item's uncertainty stratum."""
        return self._item.stratum

    @override
    def __repr__(self) -> str:
        """Return the participant, item, response and stratum."""
        return (
            f"_ProbabilityTargetRow({self._participant!r}, {self.item!r}, "
            f"{self._response!r}, {self.stratum!r})"
        )


@final
class ProbabilityTargets(FrozenValue):
    """Bounded probabilities stratified by item-level binary uncertainty.

    An item's mean unit interval response defines a binary distribution over
    the relation and its complement. Its agreement is one minus that
    distribution's normalized entropy, on the same scale used for ordinal
    item distributions.

    Parameters
    ----------
    design : BoundedDesign
        Bounded design with at least one row.

    Raises
    ------
    ValueError
        If the design has no rows or a response was not collected on the
        closed unit interval.
    """

    _noun = "probability targets"

    __slots__ = ("_items", "_rows", "_strata")

    def __init__(self, design: BoundedDesign) -> None:
        if design.n_rows == 0:
            msg = (
                "targets are taken over at least one judgment, and the design has none"
            )

            raise ValueError(msg)

        if any(scale.lower != 0.0 or scale.upper != 1.0 for scale in design.scales):
            msg = "probability targets require every response scale to be [0, 1]"

            raise ValueError(msg)

        grouped: list[list[float]] = [[] for _ in design.item_levels]

        for item, response in zip(
            design.items.tolist(), design.unit_responses.tolist(), strict=True
        ):
            grouped[item].append(response)

        means = [math.fsum(responses) / len(responses) for responses in grouped]
        agreements = [agreement_of((1.0 - mean, mean)) for mean in means]
        strata = AgreementStrata(agreements)
        items = tuple(
            _ProbabilityItemTarget(name, mean, agreement, strata.bin_of(agreement))
            for name, mean, agreement in zip(
                design.item_levels, means, agreements, strict=True
            )
        )
        object.__setattr__(self, "_strata", strata)
        object.__setattr__(
            self,
            "_items",
            MappingProxyType({target.item: target for target in items}),
        )
        object.__setattr__(
            self,
            "_rows",
            tuple(
                _ProbabilityTargetRow(
                    design.participant_levels[participant], items[item], response
                )
                for participant, item, response in zip(
                    design.participants.tolist(),
                    design.items.tolist(),
                    design.unit_responses.tolist(),
                    strict=True,
                )
            ),
        )

    @property
    def strata(self) -> AgreementStrata:
        """Return the cuts over item-level bounded agreement."""
        return self._strata

    @property
    def items(self) -> Mapping[str, StratifiedTarget]:
        """Return each item's bounded target, by identifier."""
        return self._items

    @property
    def rows(self) -> Sequence[StratifiedTarget]:
        """Return one target row per bounded response."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the row and item counts and the cuts."""
        return (
            f"ProbabilityTargets(rows={len(self._rows)}, items={len(self._items)}, "
            f"strata={self._strata!r})"
        )


@final
class StratumMean:
    """The mean of a per-row score over the rows of one stratum.

    A stratum with no rows has no value and no interval.

    Parameters
    ----------
    value : float | None
        Mean score, or None where the stratum has no rows.
    lower : float | None
        Lower bound of the bootstrap interval, if the stratum has rows.
    upper : float | None
        Upper bound of the bootstrap interval, if the stratum has rows.
    items : int
        Number of items in the stratum.
    rows : int
        Number of rows in the stratum.

    Raises
    ------
    ValueError
        If a value is reported over no rows, or rows without a value or an
        interval.
    """

    __slots__ = ("_items", "_lower", "_rows", "_upper", "_value")

    def __init__(
        self,
        value: float | None,
        lower: float | None,
        upper: float | None,
        *,
        items: int,
        rows: int,
    ) -> None:
        present = (value is not None, lower is not None, upper is not None)

        if rows == 0 and any(present):
            msg = f"a stratum with no rows reports a mean: {value!r}"

            raise ValueError(msg)

        if rows > 0 and not all(present):
            msg = f"a stratum with {rows} rows reports no mean or no interval"

            raise ValueError(msg)

        self._value = value
        self._lower = lower
        self._upper = upper
        self._items = ensure_non_negative("a stratum's item count", items)
        self._rows = ensure_non_negative("a stratum's row count", rows)

    @property
    def value(self) -> float | None:
        """Return the mean score, or None where the stratum has no rows."""
        return self._value

    @property
    def interval(self) -> tuple[float, float] | None:
        """Return the bootstrap interval, or None where the stratum has no rows."""
        if self._lower is None or self._upper is None:
            return None

        return self._lower, self._upper

    @property
    def half_width(self) -> float | None:
        """Return half the interval's width, or None where there is none."""
        if self._lower is None or self._upper is None:
            return None

        return (self._upper - self._lower) / 2.0

    @property
    def items(self) -> int:
        """Return the number of items."""
        return self._items

    @property
    def rows(self) -> int:
        """Return the number of rows."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the mean, the interval and the counts."""
        return (
            f"StratumMean({self._value!r}, {self._lower!r}, {self._upper!r}, "
            f"items={self._items}, rows={self._rows})"
        )


@final
class StratifiedMeans:
    """The mean of a per-row score over every row and within each stratum.

    Parameters
    ----------
    aggregate : StratumMean
        Mean over every row.
    strata : Mapping[AgreementBin, StratumMean]
        Mean within each stratum, in stratum order.
    """

    __slots__ = ("_aggregate", "_strata")

    def __init__(
        self, aggregate: StratumMean, strata: Mapping[AgreementBin, StratumMean]
    ) -> None:
        self._aggregate = aggregate
        self._strata: Mapping[AgreementBin, StratumMean] = MappingProxyType(
            {stratum: strata[stratum] for stratum in AgreementBin}
        )

    @property
    def aggregate(self) -> StratumMean:
        """Return the mean over every row."""
        return self._aggregate

    @property
    def strata(self) -> Mapping[AgreementBin, StratumMean]:
        """Return the mean within each stratum, in stratum order."""
        return self._strata

    @property
    def lowest(self) -> StratumMean:
        """Return the mean within the lowest-agreement stratum that has rows."""
        return next(mean for mean in self._strata.values() if mean.rows > 0)

    @override
    def __repr__(self) -> str:
        """Return the aggregate and the strata."""
        return f"StratifiedMeans({self._aggregate!r}, {dict(self._strata)!r})"


def _cluster_bootstrap(
    scores: Tensor,
    items: Tensor,
    *,
    replicates: int,
    level: float,
    generator: torch.Generator,
) -> StratumMean:
    """Return the mean of scores with an interval from resampling items.

    Each replicate draws items with replacement and takes the mean of the
    scores of every row of the drawn items, counted as often as their item
    was drawn. The interval is the percentile interval of the replicate
    means.

    Parameters
    ----------
    scores : Tensor
        One score per row.
    items : Tensor
        Dense item index of each row.
    replicates : int
        Number of resamples.
    level : float
        Nominal coverage of the interval.
    generator : torch.Generator
        Source of the resampling.
    """
    distinct = int(items.max()) + 1
    drawn = torch.randint(
        distinct, (replicates, distinct), generator=generator, dtype=torch.int64
    )
    weights = torch.zeros((replicates, distinct), dtype=scores.dtype).scatter_add_(
        1, drawn, torch.ones_like(drawn, dtype=scores.dtype)
    )
    per_row = weights[:, items]
    means = (per_row * scores).sum(dim=1) / per_row.sum(dim=1)
    tail = (1.0 - level) / 2.0
    lower, upper = torch.quantile(
        means, torch.tensor([tail, 1.0 - tail], dtype=scores.dtype)
    ).tolist()

    return StratumMean(
        float(scores.mean()),
        lower,
        upper,
        items=distinct,
        rows=scores.numel(),
    )


def stratified_mean(
    targets: StratifiedTargets,
    scores: Sequence[float],
    *,
    replicates: int = 999,
    level: float = 0.95,
    seed: int = 0,
) -> StratifiedMeans:
    """Return the mean of a per-row score over every row and within each stratum.

    Each mean has a bootstrap percentile interval from resampling items,
    since the rows of one item are not independent of each other.

    Parameters
    ----------
    targets : StratifiedTargets
        Targets whose rows the scores are over, in row order.
    scores : Sequence[float]
        One finite score per row.
    replicates : int
        Number of resamples, at least one.
    level : float
        Nominal coverage of each interval, strictly between zero and one.
    seed : int
        Seed of the resampling.

    Raises
    ------
    ValueError
        If the scores are not one per row or not finite, no replicate is
        requested, or the level is not a proportion.
    """
    rows = targets.rows

    if len(scores) != len(rows):
        msg = f"{len(scores)} scores were given for {len(rows)} rows"

        raise ValueError(msg)

    if replicates < 1:
        msg = f"a bootstrap draws at least one replicate: {replicates}"

        raise ValueError(msg)

    if not 0.0 < level < 1.0:
        msg = f"a coverage level is not a proportion: {level!r}"

        raise ValueError(msg)

    values = torch.tensor(
        [ensure_finite("a row's score", score) for score in scores],
        dtype=torch.float64,
    )
    item_index = {name: index for index, name in enumerate(targets.items)}
    items = torch.tensor([item_index[row.item] for row in rows], dtype=torch.int64)
    strata = torch.tensor(
        [list(AgreementBin).index(row.stratum) for row in rows], dtype=torch.int64
    )
    generator = torch.Generator().manual_seed(seed)
    aggregate = _cluster_bootstrap(
        values, items, replicates=replicates, level=level, generator=generator
    )
    means: dict[AgreementBin, StratumMean] = {}

    for position, stratum in enumerate(AgreementBin):
        within = strata == position

        if not bool(within.any()):
            means[stratum] = StratumMean(None, None, None, items=0, rows=0)

            continue

        chosen = items[within]
        relabeled = torch.unique(chosen, return_inverse=True)[1]
        means[stratum] = _cluster_bootstrap(
            values[within],
            relabeled,
            replicates=replicates,
            level=level,
            generator=generator,
        )

    return StratifiedMeans(aggregate, means)
