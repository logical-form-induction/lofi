"""Declared allocation of synthetic occurrences to named retained splits."""

from math import floor
from random import Random
from typing import TYPE_CHECKING

from lofi._checks import ensure_finite, ensure_name

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "split_assignments",
    "training_occurrences",
    "validate_retained",
    "validate_training_split",
]


def validate_retained(values: Mapping[str, float], /) -> dict[str, float]:
    """Return name-ordered positive allocations whose sum leaves training data."""
    if not values:
        msg = "retained declares at least one named split"
        raise ValueError(msg)
    held: dict[str, float] = {}
    for raw_name in sorted(values):
        name = ensure_name("a retained split", raw_name)
        fraction = ensure_finite(f"retained.{name}", values[raw_name])
        if not 0.0 < fraction < 1.0:
            msg = f"retained.{name} is not strictly between zero and one: {fraction!r}"
            raise ValueError(msg)
        held[name] = fraction
    total = sum(held.values())
    if total >= 1.0:
        msg = f"retained allocations sum to {total!r}, leaving no training occurrence"
        raise ValueError(msg)
    return held


def validate_training_split(name: str, retained: Mapping[str, float], /) -> str:
    """Return a named training split that no retained split shares.

    Raises
    ------
    ValueError
        If the name is empty or names a retained split.
    """
    held = ensure_name("a training split", name)

    if held in retained:
        msg = f"retained split {held!r} is the training split"
        raise ValueError(msg)

    return held


def split_assignments(
    count: int,
    retained: Mapping[str, float],
    seed: int,
    /,
    *,
    training_split: str,
) -> tuple[str, ...]:
    """Assign every occurrence to the training split or one named retained split."""
    allocations = validate_retained(retained)
    validate_training_split(training_split, allocations)
    positions = list(range(count))
    Random(seed).shuffle(positions)
    assignments = [training_split] * count
    at = 0
    for name, fraction in allocations.items():
        amount = floor(count * fraction)
        for position in positions[at : at + amount]:
            assignments[position] = name
        at += amount
    return tuple(assignments)


def training_occurrences(count: int, retained: Mapping[str, float], /) -> int:
    """Return how many declared occurrences remain assigned to training."""
    return count - sum(floor(count * fraction) for fraction in retained.values())
