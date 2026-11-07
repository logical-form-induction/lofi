"""Paired permutation tests."""

from math import fsum
from random import Random
from typing import TYPE_CHECKING, Final

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.stats._values import PValue, p_value_against_replicates

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from lofi.stats._values import Tail

__all__ = ["paired_permutation", "paired_rank_inversion"]

_EXACT_LIMIT: Final = 1 << 16
"""The maximum reassignment space for exact tests."""


def _paired(
    left: Sequence[float], right: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate and return paired score sequences.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item, in item order.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.

    Returns
    -------
    tuple[tuple[float, ...], tuple[float, ...]]
        The two sequences, as finite floats.

    Raises
    ------
    TypeError
        If a score is not a number.
    ValueError
        If a score is not finite, if the two sequences differ in length, or if there are
        no items at all.
    """
    if len(left) != len(right):
        msg = (
            f"a paired test was given {len(left)} scores on one side and "
            f"{len(right)} on the other"
        )
        raise ValueError(msg)

    if not left:
        msg = "a paired test was given no items"
        raise ValueError(msg)

    return (
        tuple(ensure_finite("a score on the left", score) for score in left),
        tuple(ensure_finite("a score on the right", score) for score in right),
    )


def _differences(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    """Return paired score differences.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item, in item order.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.

    Returns
    -------
    tuple[float, ...]
        One difference per item, left minus right.

    Raises
    ------
    TypeError
        If a score is not a number.
    ValueError
        If a score is not finite, if the two sequences differ in length, or if there are
        no items at all.
    """
    one, other = _paired(left, right)

    return tuple(first - second for first, second in zip(one, other, strict=True))


def _sign_units(
    groups: Sequence[str] | None, items: int
) -> tuple[tuple[int, ...], int]:
    """Return one sign-unit index per item and the number of units.

    Parameters
    ----------
    groups : Sequence[str] | None
        The group each item belongs to, in item order, or None where every item
        receives its own sign.
    items : int
        How many paired items the labels must cover.

    Returns
    -------
    tuple[tuple[int, ...], int]
        The unit index aligned with each item and the number of distinct units.

    Raises
    ------
    TypeError
        If a group label is not a string.
    ValueError
        If a group label is empty or the labels do not cover every item.
    """
    if groups is None:
        return tuple(range(items)), items

    if len(groups) != items:
        msg = (
            f"a paired permutation test was given {len(groups)} group labels "
            f"for {items} items"
        )
        raise ValueError(msg)

    indices: dict[str, int] = {}
    units: list[int] = []

    for group in groups:
        label = ensure_name("a permutation group label", group)
        units.append(indices.setdefault(label, len(indices)))

    return tuple(units), len(indices)


def _mean_under(
    differences: Sequence[float],
    total: float,
    kept: int,
    units: Sequence[int],
) -> float:
    """Return the mean difference for a sign assignment.

    Parameters
    ----------
    differences : Sequence[float]
        The per-item differences.
    total : float
        Their sum.
    kept : int
        A bit per sign unit, set where that unit keeps its observed sign.
    units : Sequence[int]
        The sign-unit index aligned with each item.

    Returns
    -------
    float
        The mean difference under that assignment.
    """
    inside = fsum(
        difference
        for difference, unit in zip(differences, units, strict=True)
        if kept >> unit & 1
    )

    return (2.0 * inside - total) / len(differences)


def _enumerated_means(
    differences: Sequence[float], units: Sequence[int], unit_count: int
) -> Iterator[float]:
    """Yield the mean difference under every assignment of signs to the pairs.

    Parameters
    ----------
    differences : Sequence[float]
        The per-item differences.
    units : Sequence[int]
        The sign-unit index aligned with each item.
    unit_count : int
        How many independent signs are assigned.

    Yields
    ------
    float
        The mean difference under one assignment.
    """
    total = fsum(differences)

    for kept in range(1 << unit_count):
        yield _mean_under(differences, total, kept, units)


def _sampled_means(
    differences: Sequence[float],
    units: Sequence[int],
    unit_count: int,
    replicates: int,
    rng: Random,
) -> Iterator[float]:
    """Yield the mean difference under a fixed number of drawn sign assignments.

    Parameters
    ----------
    differences : Sequence[float]
        The per-item differences.
    units : Sequence[int]
        The sign-unit index aligned with each item.
    unit_count : int
        How many independent signs are assigned.
    replicates : int
        How many assignments to draw.
    rng : Random
        The generator the draw runs on.

    Yields
    ------
    float
        The mean difference under one drawn assignment.
    """
    total = fsum(differences)
    for _ in range(replicates):
        yield _mean_under(differences, total, rng.getrandbits(unit_count), units)


def paired_permutation(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    seed: int,
    tail: Tail,
    groups: Sequence[str] | None = None,
) -> PValue:
    """Return the p-value of a paired difference in per-item scores.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item, in item order.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.
    replicates : int
        How many reassignments to draw.
    seed : int
        The seed a draw runs under.
    tail : Tail
        Which direction counts as extreme.
    groups : Sequence[str] | None
        The group each item belongs to, in item order, or None where every item
        receives its own sign.

    Returns
    -------
    PValue
        The tail probability, recording the provenance of this test.

    Raises
    ------
    TypeError
        If a score, group label, replicate count or seed is of the wrong type.
    ValueError
        If a score is not finite, if the two sequences differ in length, if there are no
        items, if the grouping does not label every item, if a group label is empty, if
        the replicate budget is zero, or if the seed is negative.
    """
    differences = _differences(left, right)
    units, unit_count = _sign_units(groups, len(differences))
    budget = ensure_non_negative("a replicate budget", replicates)
    drawn_under = ensure_non_negative("a permutation seed", seed)

    if budget == 0:
        msg = "a paired permutation test was given no replicate budget"
        raise ValueError(msg)

    observed = fsum(differences) / len(differences)
    exact = (1 << unit_count) <= min(budget, _EXACT_LIMIT)
    reference = (
        tuple(_enumerated_means(differences, units, unit_count))
        if exact
        else tuple(
            _sampled_means(differences, units, unit_count, budget, Random(drawn_under))
        )
    )

    return p_value_against_replicates(
        observed,
        reference,
        tail=tail,
        # exchange the conditions within each pair
        null="paired sign flips",
        family="permutation",
        exact=exact,
        seed=None if exact else drawn_under,
    )


def _inversions(left: Sequence[float], right: Sequence[float]) -> float:
    """Count pairwise ranking disagreements.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item.
    right : Sequence[float]
        The other condition's score for the same items, in the same order.

    Returns
    -------
    float
        The number of discordant pairs.
    """
    items = len(left)

    return float(
        sum(
            1
            for one in range(items)
            for other in range(one + 1, items)
            if (left[one] - left[other]) * (right[one] - right[other]) < 0
        )
    )


def _inversions_under(
    left: Sequence[float], right: Sequence[float], swapped: int
) -> float:
    """Return the inversion count for one score assignment.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item.
    right : Sequence[float]
        The other condition's score for the same items.
    swapped : int
        A bit per item, set where the two conditions trade that item's scores.

    Returns
    -------
    float
        The number of discordant pairs under that assignment.
    """
    one = tuple(
        right[index] if swapped >> index & 1 else score
        for index, score in enumerate(left)
    )
    other = tuple(
        left[index] if swapped >> index & 1 else score
        for index, score in enumerate(right)
    )

    return _inversions(one, other)


def _enumerated_inversions(
    left: Sequence[float], right: Sequence[float]
) -> Iterator[float]:
    """Yield the inversion count under every assignment of scores to conditions.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item.
    right : Sequence[float]
        The other condition's score for the same items.

    Yields
    ------
    float
        The count under one assignment.
    """
    for swapped in range(1 << len(left)):
        yield _inversions_under(left, right, swapped)


def _sampled_inversions(
    left: Sequence[float], right: Sequence[float], replicates: int, rng: Random
) -> Iterator[float]:
    """Yield the inversion count under a fixed number of drawn assignments.

    Parameters
    ----------
    left : Sequence[float]
        One condition's score for each item.
    right : Sequence[float]
        The other condition's score for the same items.
    replicates : int
        How many assignments to draw.
    rng : Random
        The generator the draw runs on.

    Yields
    ------
    float
        The count under one drawn assignment.
    """
    items = len(left)

    for _ in range(replicates):
        yield _inversions_under(left, right, rng.getrandbits(items))


def paired_rank_inversion(
    left: Sequence[float],
    right: Sequence[float],
    *,
    replicates: int,
    seed: int,
    tail: Tail,
) -> PValue:
    """Return the p-value of the inversions between two conventions' rankings.

    Parameters
    ----------
    left : Sequence[float]
        One convention's score for each item, in item order.
    right : Sequence[float]
        The other convention's score for the same items, in the same order.
    replicates : int
        How many reassignments to draw.
    seed : int
        The seed a draw runs under, recorded only where a draw happened.
    tail : Tail
        Which direction counts as extreme.

    Returns
    -------
    PValue
        The tail probability, recording the provenance of this test.

    Raises
    ------
    TypeError
        If a score, the replicate count or the seed is of the wrong type.
    ValueError
        If a score is not finite, if the two sequences differ in length, if there are no
        items, if the replicate budget is zero, or if the seed is negative.
    """
    one, other = _paired(left, right)
    budget = ensure_non_negative("a replicate budget", replicates)
    drawn_under = ensure_non_negative("a permutation seed", seed)

    if budget == 0:
        msg = "a paired rank test was given no replicate budget"
        raise ValueError(msg)

    observed = _inversions(one, other)
    exact = (1 << len(one)) <= min(budget, _EXACT_LIMIT)
    reference = (
        tuple(_enumerated_inversions(one, other))
        if exact
        else tuple(_sampled_inversions(one, other, budget, Random(drawn_under)))
    )

    return p_value_against_replicates(
        observed,
        reference,
        tail=tail,
        # exchange the convention ranks within each item
        null="paired rank swaps",
        family="permutation",
        exact=exact,
        seed=None if exact else drawn_under,
    )
