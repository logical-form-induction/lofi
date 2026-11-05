"""Rank statistics over the frequencies a sample gives its forms."""

from math import log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from lofi._stats import average_ranks, spearman

__all__ = ["average_ranks", "log_log_slope", "spearman"]


def log_log_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return the least-squares slope of one log measurement on another.

    Parameters
    ----------
    xs : Sequence[float]
        The positive regressor, one value per unit.
    ys : Sequence[float]
        The positive response, one value per unit, in the same order.

    Returns
    -------
    float | None
        The slope, or nothing when fewer than two units are given or the
        regressor does not vary.

    Raises
    ------
    ValueError
        If the sequences differ in length or a value is not positive.
    """
    if len(xs) != len(ys):
        msg = f"{len(xs)} values cannot be regressed on {len(ys)}"
        raise ValueError(msg)

    for label, values in (("regressor", xs), ("response", ys)):
        if any(value <= 0 for value in values):
            msg = f"a {label} is not positive: {values!r}"
            raise ValueError(msg)

    if len(xs) < 2:
        return None

    log_xs = [log(x) for x in xs]
    log_ys = [log(y) for y in ys]
    mean_x = sum(log_xs) / len(log_xs)
    mean_y = sum(log_ys) / len(log_ys)
    spread = sum((x - mean_x) ** 2 for x in log_xs)

    if spread == 0.0:
        return None

    return (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(log_xs, log_ys, strict=True))
        / spread
    )
