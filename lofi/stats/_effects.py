"""Paired effect-size estimates."""

from math import fsum
from statistics import stdev
from typing import TYPE_CHECKING, Final, Literal, override

from lofi.stats._bootstrap import Bootstrap
from lofi.stats._permutation import _paired
from lofi.stats._values import BootstrapResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "ESTIMATORS",
    "EffectSize",
    "Estimator",
    "paired_cliffs_delta",
    "paired_cohens_d",
    "paired_effect_size",
]

_MINIMUM_ITEMS: Final = 2
"""The minimum item count for a standardized difference."""


type Estimator = Literal["paired-cohens-d", "paired-cliffs-delta"]
"""Specifies which effect size an estimate reports."""

ESTIMATORS: Final[tuple[Estimator, ...]] = ("paired-cohens-d", "paired-cliffs-delta")
"""Declared estimators in report order."""


def paired_cohens_d(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the mean per-item difference in units of its own spread.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item, in item order.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.

    Returns
    -------
    float
        The mean difference divided by the sample standard deviation of the differences,
        positive where the left condition scores above the right.

    Raises
    ------
    TypeError
        If a score is not a number.
    ValueError
        If a score is not finite, if the two sequences differ in length, if there are
        fewer than two items, or if every item shows the same difference, which leaves
        the standardized difference undefined.
    """
    one, other = _paired(left, right)

    if len(one) < _MINIMUM_ITEMS:
        msg = (
            f"a standardized difference needs at least {_MINIMUM_ITEMS} items "
            f"and was given {len(one)}"
        )
        raise ValueError(msg)

    differences = tuple(
        first - second for first, second in zip(one, other, strict=True)
    )
    spread = stdev(differences)

    if spread <= 0.0:
        msg = (
            f"every item shows the same difference of {differences[0]!r}, so a "
            f"standardized difference divides by nothing"
        )
        raise ValueError(msg)

    return fsum(differences) / len(differences) / spread


def paired_cliffs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    """Return how often one condition wins an item less how often it loses.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item, in item order.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.

    Returns
    -------
    float
        The proportion of items the left condition wins less the proportion it loses,
        between minus one and one.

    Raises
    ------
    TypeError
        If a score is not a number.
    ValueError
        If a score is not finite, if the two sequences differ in length, or if there are
        no items.
    """
    one, other = _paired(left, right)
    won = sum(1 for first, second in zip(one, other, strict=True) if first > second)
    lost = sum(1 for first, second in zip(one, other, strict=True) if first < second)

    return (won - lost) / len(one)


class EffectSize:
    """Stores a paired effect estimate and interval.

    Parameters
    ----------
    estimator : Estimator
        Which effect size the interval is around.
    interval : BootstrapResult
        What the bootstrap returned: the estimate, the unit resampled, the replicate
        budget and seed, and the two intervals.

    Raises
    ------
    TypeError
        If the interval is not a bootstrap result.
    ValueError
        If the estimator is not one this package declares.
    """

    __slots__ = ("_estimator", "_interval")

    def __init__(self, estimator: Estimator, interval: BootstrapResult) -> None:
        if estimator not in ESTIMATORS:
            declared = ", ".join(ESTIMATORS)
            msg = (
                f"an effect size names the estimator {estimator!r}, not one "
                f"of {declared}"
            )
            raise ValueError(msg)

        if not isinstance(interval, BootstrapResult):
            msg = f"an effect size's interval is not a bootstrap result: {interval!r}"
            raise TypeError(msg)

        self._estimator: Estimator = estimator
        self._interval = interval

    @property
    def estimator(self) -> Estimator:
        """Return which effect size this is."""
        return self._estimator

    @property
    def interval(self) -> BootstrapResult:
        """Return the bootstrap result."""
        return self._interval

    @property
    def value(self) -> float:
        """Return the observed effect estimate."""
        return self._interval.observed

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same estimator and interval."""
        if type(other) is not EffectSize:
            return NotImplemented

        return self._estimator == other._estimator and self._interval == other._interval

    @override
    def __hash__(self) -> int:
        """Return a hash over the estimator and the interval."""
        return hash((EffectSize, self._estimator, self._interval))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this effect size."""
        return f"EffectSize({self._estimator!r}, {self._interval!r})"


def _estimate(
    estimator: Estimator,
) -> Callable[[Sequence[float], Sequence[float]], float]:
    """Return the function one estimator names.

    Parameters
    ----------
    estimator : Estimator
        Which effect size to compute.

    Returns
    -------
    Callable[[Sequence[float], Sequence[float]], float]
        The estimator, taking the two conditions' aligned scores.

    Raises
    ------
    ValueError
        If the estimator is not one this package declares.
    """
    if estimator not in ESTIMATORS:
        declared = ", ".join(ESTIMATORS)
        msg = f"an effect size was asked for {estimator!r}, not one of {declared}"
        raise ValueError(msg)

    if estimator == "paired-cohens-d":
        return paired_cohens_d

    return paired_cliffs_delta


def paired_effect_size(
    left: Sequence[float],
    right: Sequence[float],
    bootstrap: Bootstrap,
    /,
    *,
    estimator: Estimator,
) -> EffectSize:
    """Return a paired effect size with an interval over the declared unit.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item, in item order.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.
    bootstrap : Bootstrap
        The resampling to draw the interval with, which declares the unit, the replicate
        budget, the seed and the coverage.
    estimator : Estimator
        Which effect size to report.

    Returns
    -------
    EffectSize
        The named estimator and the bootstrap's estimate and intervals.

    Raises
    ------
    TypeError
        If a score is not a number, or the bootstrap is not one.
    ValueError
        If a score is not finite, if the two sequences differ in length, if there are no
        items, if the estimator is not one this package declares, or if a resample
        leaves the estimator undefined.
    """
    if not isinstance(bootstrap, Bootstrap):
        msg = f"an effect size's bootstrap is not one: {bootstrap!r}"
        raise TypeError(msg)

    one, other = _paired(left, right)
    point = _estimate(estimator)
    pairs = tuple(zip(one, other, strict=True))

    def over_pairs(drawn: Sequence[tuple[float, float]]) -> float:
        return point([first for first, _ in drawn], [second for _, second in drawn])

    return EffectSize(estimator, bootstrap.interval(pairs, over_pairs))
