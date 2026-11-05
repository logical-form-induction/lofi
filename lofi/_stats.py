"""Rank statistics over paired measurements."""

from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["average_ranks", "spearman"]


def average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return the one-based rank of each value, tied values sharing their mean rank.

    Parameters
    ----------
    values : Sequence[float]
        The values to rank, in any order.
    """
    order = sorted(range(len(values)), key=lambda at: values[at])
    ranks = [0.0] * len(values)
    start = 0

    while start < len(order):
        stop = start

        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1

        shared = (start + stop) / 2 + 1

        for at in order[start : stop + 1]:
            ranks[at] = shared

        start = stop + 1

    return tuple(ranks)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return the rank correlation of two equally long sequences.

    Parameters
    ----------
    xs : Sequence[float]
        The first measurement, one value per unit.
    ys : Sequence[float]
        The second measurement, one value per unit, in the same order.

    Returns
    -------
    float | None
        The Pearson correlation of the average ranks, or nothing when either
        measurement has no variation to correlate.

    Raises
    ------
    ValueError
        If the sequences differ in length.
    """
    if len(xs) != len(ys):
        msg = f"{len(xs)} values cannot be correlated with {len(ys)}"
        raise ValueError(msg)

    return _pearson(average_ranks(xs), average_ranks(ys))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return the product-moment correlation, or nothing without variation."""
    if not xs:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    spread_x = sum((x - mean_x) ** 2 for x in xs)
    spread_y = sum((y - mean_y) ** 2 for y in ys)

    if spread_x == 0.0 or spread_y == 0.0:
        return None

    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))

    return covariance / sqrt(spread_x * spread_y)
