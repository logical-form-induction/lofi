"""Agreement metrics for annotator responses."""

from itertools import combinations, islice
from math import atanh, sqrt, tanh
from statistics import NormalDist, fmean
from typing import TYPE_CHECKING, Final

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.judgment._values import AgreementCeiling

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "fisher_interval",
    "rank_correlation",
    "spearman_brown",
    "split_half_ceiling",
]

_STANDARD_NORMAL: Final = NormalDist()
"""The standard normal used for Fisher quantiles."""

_MINIMUM_PAIRED = 3
"""The minimum pair count for rank correlation."""

_MINIMUM_FISHER = 4
"""The minimum observation count for a Fisher interval."""

_MINIMUM_RESPONSES = 2
"""The minimum response count for a split."""


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return ranks with ties assigned their mean rank.

    Parameters
    ----------
    values : Sequence[float]
        Values to rank.

    Returns
    -------
    tuple[float, ...]
        One rank per input value, in input order and counting from one.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0

    while start < len(order):
        stop = start

        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1

        shared = (start + stop) / 2.0 + 1.0

        for position in range(start, stop + 1):
            ranks[order[position]] = shared

        start = stop + 1

    return tuple(ranks)


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the Spearman correlation between two paired sequences.

    Parameters
    ----------
    first : Sequence[float]
        First sequence of paired values.
    second : Sequence[float]
        Second sequence of paired values.

    Returns
    -------
    float
        The correlation in the closed interval [-1, 1].

    Raises
    ------
    ValueError
        If lengths differ, fewer than three pairs are provided, or either
        sequence is constant.
    """
    if len(first) != len(second):
        msg = (
            f"a rank correlation needs two sides of one length: "
            f"{len(first)}, {len(second)}"
        )

        raise ValueError(msg)

    if len(first) < _MINIMUM_PAIRED:
        msg = f"a rank correlation needs at least {_MINIMUM_PAIRED} pairs: {len(first)}"

        raise ValueError(msg)

    left = _average_ranks(first)
    right = _average_ranks(second)
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_spread = fmean([(value - left_mean) ** 2 for value in left])
    right_spread = fmean([(value - right_mean) ** 2 for value in right])

    if left_spread == 0.0 or right_spread == 0.0:
        msg = "a rank correlation is undefined where one side never varies"

        raise ValueError(msg)

    covariance = fmean(
        [
            (value - left_mean) * (other - right_mean)
            for value, other in zip(left, right, strict=True)
        ]
    )

    return covariance / sqrt(left_spread * right_spread)


def spearman_brown(correlation: float, factor: float) -> float:
    """Apply the Spearman-Brown correction.

    Parameters
    ----------
    correlation : float
        Correlation at the measured length.
    factor : float
        Ratio of target length to measured length.

    Returns
    -------
    float
        Corrected correlation at the target length.

    Raises
    ------
    TypeError
        If either argument is not numeric or is a boolean.
    ValueError
        If an argument is not finite, ``factor`` is not positive, or the
        correction denominator is not positive.
    """
    ensure_finite("a correlation", correlation)
    ensure_finite("a lengthening factor", factor)

    if factor <= 0.0:
        msg = f"a lengthening factor is not positive: {factor!r}"

        raise ValueError(msg)

    denominator = 1.0 + (factor - 1.0) * correlation

    if denominator <= 0.0:
        msg = (
            f"lengthening by {factor!r} is undefined at a correlation of "
            f"{correlation!r}"
        )

        raise ValueError(msg)

    return factor * correlation / denominator


def fisher_interval(
    correlation: float, observations: int, *, level: float = 0.95
) -> tuple[float, float]:
    """Return a Fisher confidence interval for a correlation.

    Parameters
    ----------
    correlation : float
        Correlation strictly between -1 and 1.
    observations : int
        Number of paired observations.
    level : float
        Confidence level strictly between 0 and 1.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds on the correlation scale.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the correlation or level is out of range, or fewer than four
        observations are provided.
    """
    ensure_finite("a correlation", correlation)
    ensure_non_negative("an observation count", observations)
    ensure_finite("a coverage level", level)

    if not -1.0 < correlation < 1.0:
        msg = f"a Fisher interval is undefined at a correlation of {correlation!r}"

        raise ValueError(msg)

    if observations < _MINIMUM_FISHER:
        msg = (
            f"a Fisher interval needs at least {_MINIMUM_FISHER} observations: "
            f"{observations}"
        )

        raise ValueError(msg)

    if not 0.0 < level < 1.0:
        msg = f"a coverage level is not a proportion: {level!r}"

        raise ValueError(msg)

    center = atanh(correlation)
    spread = _STANDARD_NORMAL.inv_cdf(1.0 - (1.0 - level) / 2.0) / sqrt(
        observations - 3
    )

    return tanh(center - spread), tanh(center + spread)


def _balanced_splits(size: int, wanted: int) -> tuple[tuple[int, ...], ...]:
    """Return the first balanced half-splits.

    Parameters
    ----------
    size : int
        Number of response positions.
    wanted : int
        Maximum number of splits to return.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Lexicographically ordered positions in the smaller half of each split.
    """
    return tuple(islice(combinations(range(size), size // 2), wanted))


def split_half_ceiling(
    items: Iterable[Sequence[float]], *, splits: int = 10, level: float = 0.95
) -> AgreementCeiling:
    """Return a split-half agreement ceiling corrected to the full sample.

    Parameters
    ----------
    items : Iterable[Sequence[float]]
        Response sequence for each item.
    splits : int
        Number of balanced splits to average.
    level : float
        Confidence level for the Fisher interval.

    Returns
    -------
    AgreementCeiling
        Corrected agreement, interval, and supporting counts.

    Raises
    ------
    TypeError
        If a numeric value or count has the wrong type.
    ValueError
        If no splits are requested, an item has fewer than two responses, or
        the resulting correlation is undefined.
    """
    ensure_non_negative("a split count", splits)

    if splits == 0:
        msg = "a split-half ceiling was asked to average over no splits"

        raise ValueError(msg)

    held = [tuple(responses) for responses in items]

    for responses in held:
        if len(responses) < _MINIMUM_RESPONSES:
            msg = (
                f"an item has fewer than {_MINIMUM_RESPONSES} responses and "
                f"cannot be split: {len(responses)}"
            )

            raise ValueError(msg)

    arrangements = {
        size: _balanced_splits(size, splits) for size in {len(r) for r in held}
    }
    halved: list[float] = []

    for split in range(splits):
        left: list[float] = []
        right: list[float] = []

        for responses in held:
            taken = arrangements[len(responses)][
                split % len(arrangements[len(responses)])
            ]
            aside = set(taken)

            left.append(fmean([responses[position] for position in taken]))
            right.append(
                fmean(
                    [
                        response
                        for position, response in enumerate(responses)
                        if position not in aside
                    ]
                )
            )

        halved.append(rank_correlation(left, right))

    average = fmean(halved)
    lower, upper = fisher_interval(average, len(held), level=level)

    return AgreementCeiling(
        spearman_brown(average, 2.0),
        spearman_brown(lower, 2.0),
        spearman_brown(upper, 2.0),
        items=len(held),
        responses=sum(len(responses) for responses in held),
    )
