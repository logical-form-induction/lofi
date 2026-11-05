"""Interannotator correlation ceilings."""

from itertools import combinations
from math import fsum, isfinite, sqrt
from statistics import fmean
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

from lofi.eval._values import MetricResult, Resampling
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi._types import JsonValue

__all__ = [
    "INTERANNOTATOR_COEFFICIENTS",
    "InterannotatorCeilingRow",
    "InterannotatorCoefficient",
    "ParticipantResponse",
    "interannotator_ceiling",
]

_PAIR_UNIT: Final = "participant-pair"
_MINIMUM_CORRELATION_ITEMS: Final = 2
_MINIMUM_BOOTSTRAP_PAIRS: Final = 2
_MINIMUM_GROUP_PARTICIPANTS: Final = 2

type InterannotatorCoefficient = Literal["spearman", "pearson"]
"""A correlation coefficient an interannotator ceiling reports."""

INTERANNOTATOR_COEFFICIENTS: Final[tuple[InterannotatorCoefficient, ...]] = (
    "spearman",
    "pearson",
)
"""Every coefficient an interannotator ceiling can report."""

type _ParticipantRow = tuple[str, tuple[tuple[str, float], ...]]


class ParticipantResponse(NamedTuple):
    """One participant's numeric response to one item."""

    participant: str
    item: str
    value: float


class InterannotatorCeilingRow(NamedTuple):
    """One interannotator correlation estimate."""

    coefficient: InterannotatorCoefficient
    result: MetricResult
    participant_count: int
    pair_count: int

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as serializable report columns."""
        interval = self.result.ci

        return {
            "coefficient": self.coefficient,
            "value": self.result.value,
            "participant_count": self.participant_count,
            "pair_count": self.pair_count,
            "resampling_unit": self.result.resampling.unit,
            "bootstrap_replicates": int(self.result.components["bootstrap_replicates"]),
            "bootstrap_seed": self.result.resampling.seed,
            "ci_low": None if interval is None else interval.low,
            "ci_high": None if interval is None else interval.high,
            "ci_level": None if interval is None else interval.level,
        }


def _ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Assign average ranks to values, including ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0

    while start < len(order):
        stop = start + 1

        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1

        rank = (start + stop + 1) / 2.0

        for position in range(start, stop):
            ranks[order[position]] = rank

        start = stop

    return tuple(ranks)


def _centered(values: Sequence[float]) -> tuple[float, ...]:
    """Center values after scaling their magnitude."""
    scale = max(abs(value) for value in values)

    if scale == 0.0:
        return (0.0,) * len(values)

    scaled = tuple(value / scale for value in values)
    mean = fsum(scaled) / len(scaled)

    return tuple(value - mean for value in scaled)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Pearson correlation for two nonconstant vectors."""
    size = len(left)

    if size != len(right):
        msg = f"correlation vectors have lengths {size} and {len(right)}"
        raise ValueError(msg)

    if size < _MINIMUM_CORRELATION_ITEMS:
        msg = f"a correlation needs at least {_MINIMUM_CORRELATION_ITEMS} items"
        raise ValueError(msg)

    left_centered = _centered(left)
    right_centered = _centered(right)
    left_spread = fsum(value * value for value in left_centered)
    right_spread = fsum(value * value for value in right_centered)

    if left_spread == 0.0 or right_spread == 0.0:
        msg = "a correlation vector is constant"
        raise ValueError(msg)

    covariance = fsum(
        value * other
        for value, other in zip(left_centered, right_centered, strict=True)
    )
    correlation = covariance / sqrt(left_spread * right_spread)

    return min(1.0, max(-1.0, correlation))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute tied-rank Spearman correlation."""
    return _pearson(_ranks(left), _ranks(right))


def _participants(
    responses: Iterable[ParticipantResponse],
) -> tuple[_ParticipantRow, ...]:
    """Collect complete responses by participant and item."""
    collected: dict[str, dict[str, float]] = {}

    for position, response in enumerate(responses):
        if not isinstance(response, ParticipantResponse):
            msg = f"interannotator response {position} is not a ParticipantResponse"
            raise TypeError(msg)

        participant, item, value = response

        if not participant:
            msg = f"interannotator response {position} has no participant"
            raise ValueError(msg)

        if not isfinite(value):
            msg = f"participant {participant!r} has no finite response to {item!r}"
            raise ValueError(msg)

        held = collected.setdefault(participant, {})

        if item in held:
            msg = f"participant {participant!r} responds twice to {item!r}"
            raise ValueError(msg)

        held[item] = value

    return tuple(
        (participant, tuple(sorted(held.items())))
        for participant, held in sorted(collected.items())
    )


def _pair_correlations(
    participants: Sequence[_ParticipantRow],
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Compute correlations for participants with identical item sets."""
    by_items: dict[tuple[str, ...], list[_ParticipantRow]] = {}

    for participant in participants:
        items = tuple(item for item, _ in participant[1])
        by_items.setdefault(items, []).append(participant)

    spearman: list[float] = []
    pearson: list[float] = []
    paired_participants: set[str] = set()

    for item_set, group in by_items.items():
        if len(group) < _MINIMUM_GROUP_PARTICIPANTS:
            continue

        for left, right in combinations(group, 2):
            left_values = tuple(value for _, value in left[1])
            right_values = tuple(value for _, value in right[1])

            try:
                spearman.append(_spearman(left_values, right_values))
                pearson.append(_pearson(left_values, right_values))
            except ValueError as error:
                msg = (
                    f"participants {left[0]!r} and {right[0]!r} on list "
                    f"{item_set!r} have undefined correlation"
                )
                raise ValueError(msg) from error

            paired_participants.update((left[0], right[0]))

    return tuple(spearman), tuple(pearson), len(paired_participants)


def _bootstrap(bootstrap: Bootstrap) -> None:
    """Validate the participant-pair bootstrap declaration."""
    if not isinstance(bootstrap, Bootstrap):
        msg = f"an interannotator bootstrap is not one: {bootstrap!r}"
        raise TypeError(msg)

    if bootstrap.unit != _PAIR_UNIT:
        msg = (
            f"an interannotator bootstrap resamples {bootstrap.unit!r}, "
            f"not {_PAIR_UNIT!r}"
        )
        raise ValueError(msg)

    if bootstrap.groups is not None:
        msg = "participant-pair correlations are already the resampling units"
        raise ValueError(msg)


def _row(
    coefficient: InterannotatorCoefficient,
    correlations: Sequence[float],
    participant_count: int,
    bootstrap: Bootstrap,
) -> InterannotatorCeilingRow:
    """Build one bootstrapped correlation row."""
    drawn = bootstrap.interval(correlations, fmean)
    pair_count = len(correlations)
    result = MetricResult(
        drawn.observed,
        pair_count,
        Resampling(bootstrap.unit, bootstrap.seed),
        components={
            "participants": float(participant_count),
            "pairs": float(pair_count),
            "bootstrap_replicates": float(bootstrap.replicates),
        },
        ci=Interval(drawn.bca.low, drawn.bca.high, drawn.bca.level),
    )

    return InterannotatorCeilingRow(
        coefficient,
        result,
        participant_count,
        pair_count,
    )


def interannotator_ceiling(
    responses: Iterable[ParticipantResponse],
    *,
    coefficients: Sequence[InterannotatorCoefficient],
    bootstrap: Bootstrap,
) -> tuple[InterannotatorCeilingRow, ...]:
    """Compute the declared correlation ceilings over participant pairs.

    Participants who responded to exactly the same items form the pairs; each
    pair's correlation is one resampling unit of the bootstrap.

    Parameters
    ----------
    responses : Iterable[ParticipantResponse]
        Every participant's response to each item they saw.
    coefficients : Sequence[InterannotatorCoefficient]
        The coefficients to report, one row each in this order.
    bootstrap : Bootstrap
        The participant-pair bootstrap the intervals are drawn under.

    Raises
    ------
    ValueError
        If no coefficient or an unknown one is declared, or fewer than two
        same-list participant pairs are available.
    """
    _bootstrap(bootstrap)
    declared = tuple(coefficients)

    if not declared or len(set(declared)) != len(declared):
        msg = f"an interannotator ceiling declares distinct coefficients: {declared!r}"
        raise ValueError(msg)

    unknown = tuple(
        name for name in declared if name not in INTERANNOTATOR_COEFFICIENTS
    )

    if unknown:
        msg = f"unknown interannotator coefficients {unknown!r}"
        raise ValueError(msg)

    participants = _participants(responses)
    spearman, pearson, participant_count = _pair_correlations(participants)

    if len(spearman) < _MINIMUM_BOOTSTRAP_PAIRS:
        msg = (
            f"an interannotator ceiling needs at least {_MINIMUM_BOOTSTRAP_PAIRS} "
            f"same-list participant pairs and found {len(spearman)}"
        )
        raise ValueError(msg)

    correlations = {"spearman": spearman, "pearson": pearson}

    return tuple(
        _row(name, correlations[name], participant_count, bootstrap)
        for name in declared
    )
