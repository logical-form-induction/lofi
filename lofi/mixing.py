"""Declared mixing policies for multi-step training."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction
from math import floor, fsum
from typing import TYPE_CHECKING, Literal, final, override

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Alternation",
    "Interleaving",
    "MixingSpec",
    "ProportionalMixingSpec",
    "RatioMixingSpec",
    "RoundRobinMixingSpec",
    "TemperatureMixingSpec",
]


class Alternation(Enum):
    """How an interleaving fixes the weight of each task."""

    ROUND_ROBIN = "round_robin"
    PROPORTIONAL = "proportional"
    RATIO = "ratio"
    TEMPERATURE = "temperature"


@final
class Interleaving(FrozenValue):
    """Fix how many steps each task takes and their interleaved order."""

    _noun = "interleaving"

    __slots__ = ("_alternation", "_temperature", "_weights")

    def __init__(
        self,
        alternation: Alternation,
        weights: Sequence[int] = (),
        *,
        temperature: float | None = None,
    ) -> None:
        declared = tuple(
            ensure_positive(f"the weight of task {position}", weight)
            for position, weight in enumerate(weights)
        )

        if alternation is Alternation.RATIO and not declared:
            msg = "a ratio declares one weight per task and declared none"
            raise ValueError(msg)

        if alternation is not Alternation.RATIO and declared:
            msg = (
                f"{alternation.value} alternation reads its weights off the tasks "
                f"and was given {list(declared)}"
            )
            raise ValueError(msg)

        if alternation is Alternation.TEMPERATURE:
            if isinstance(temperature, bool) or not isinstance(
                temperature, (int, float)
            ):
                msg = f"temperature is not a number: {temperature!r}"
                raise TypeError(msg)

            held_temperature = float(temperature)

            if not 0.0 < held_temperature < float("inf"):
                msg = f"temperature is not positive and finite: {temperature!r}"
                raise ValueError(msg)

        elif temperature is not None:
            msg = (
                f"{alternation.value} alternation declares no temperature, "
                f"not {temperature!r}"
            )
            raise ValueError(msg)

        else:
            held_temperature = None

        object.__setattr__(self, "_alternation", alternation)
        object.__setattr__(self, "_weights", declared)
        object.__setattr__(self, "_temperature", held_temperature)

    @classmethod
    def round_robin(cls) -> Interleaving:
        """Return equal task alternation."""
        return cls(Alternation.ROUND_ROBIN)

    @classmethod
    def proportional(cls) -> Interleaving:
        """Return one-pass-per-task alternation."""
        return cls(Alternation.PROPORTIONAL)

    @classmethod
    def ratio(cls, weights: Sequence[int]) -> Interleaving:
        """Return alternation under declared integer task weights."""
        return cls(Alternation.RATIO, weights)

    @classmethod
    def temperature(cls, value: float) -> Interleaving:
        """Return size-proportional alternation softened by ``value``."""
        return cls(Alternation.TEMPERATURE, temperature=value)

    @property
    def alternation(self) -> Alternation:
        """Return how task weights are fixed."""
        return self._alternation

    @property
    def declared(self) -> tuple[int, ...]:
        """Return declared weights, empty where task sizes fix them."""
        return self._weights

    @property
    def temperature_value(self) -> float | None:
        """Return the declared temperature, when temperature-scaled."""
        return self._temperature

    def _weights_over(self, sizes: Sequence[int], /) -> tuple[int, ...]:
        """Return each task's validated weight."""
        for position, size in enumerate(sizes):
            if ensure_non_negative(f"the size of task {position}", size) == 0:
                msg = f"task {position} holds no batch to fit on"
                raise ValueError(msg)

        if self._alternation is Alternation.PROPORTIONAL:
            return tuple(sizes)

        if self._alternation is Alternation.ROUND_ROBIN:
            return (1,) * len(sizes)

        if self._alternation is Alternation.TEMPERATURE:
            return tuple(sizes)

        if len(self._weights) != len(sizes):
            msg = (
                f"a ratio over {len(self._weights)} tasks does not weight "
                f"{len(sizes)} of them"
            )
            raise ValueError(msg)

        return self._weights

    def _temperature_counts(self, sizes: Sequence[int], /) -> tuple[int, ...]:
        """Allocate one fixed-size epoch under temperature-scaled masses."""
        self._weights_over(sizes)
        temperature = self._temperature

        if temperature is None:
            msg = "temperature counts require a declared temperature"
            raise RuntimeError(msg)

        total = sum(sizes)
        masses = tuple(size ** (1.0 / temperature) for size in sizes)
        normalizer = fsum(masses)
        quotas = tuple(total * mass / normalizer for mass in masses)
        counts = [max(1, floor(quota)) for quota in quotas]

        while sum(counts) < total:
            position = max(
                range(len(counts)),
                key=lambda index: (quotas[index] - counts[index], -index),
            )
            counts[position] += 1

        while sum(counts) > total:
            position = max(
                (index for index, count in enumerate(counts) if count > 1),
                key=lambda index: (counts[index] - quotas[index], -index),
            )
            counts[position] -= 1

        return tuple(counts)

    def counts(self, sizes: Sequence[int], /) -> tuple[int, ...]:
        """Return the number of steps each task takes over one epoch."""
        if not sizes:
            msg = "an interleaving runs over at least one task"
            raise ValueError(msg)

        if self._alternation is Alternation.TEMPERATURE:
            return self._temperature_counts(sizes)

        weights = self._weights_over(sizes)
        rounds = max(
            -(-size // weight) for size, weight in zip(sizes, weights, strict=True)
        )
        return tuple(weight * rounds for weight in weights)

    def schedule(self, sizes: Sequence[int], /) -> tuple[int, ...]:
        """Return task positions in their evenly interleaved epoch order."""
        counts = self.counts(sizes)
        places = sorted(
            (Fraction(2 * position + 1, 2 * count), task)
            for task, count in enumerate(counts)
            for position in range(count)
        )
        return tuple(task for _, task in places)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another interleaving fixes the same policy."""
        if type(other) is not Interleaving:
            return NotImplemented

        return (
            self._alternation is other._alternation
            and self._weights == other._weights
            and self._temperature == other._temperature
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the complete policy."""
        return hash((Interleaving, self._alternation, self._weights, self._temperature))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        if self._temperature is not None:
            return (
                f"Interleaving({self._alternation!r}, "
                f"temperature={self._temperature!r})"
            )

        if not self._weights:
            return f"Interleaving({self._alternation!r})"

        return f"Interleaving({self._alternation!r}, {list(self._weights)!r})"


class MixingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how one epoch distributes work among training tasks."""


class RoundRobinMixingSpec(MixingSpec, extra="forbid"):
    """Give every task as many steps as the largest task needs."""

    kind: Literal["round-robin"] = "round-robin"


class ProportionalMixingSpec(MixingSpec, extra="forbid"):
    """Visit every task batch exactly once."""

    kind: Literal["proportional"] = "proportional"


class RatioMixingSpec(MixingSpec, extra="forbid"):
    """Repeat positive integer task weights until every task is covered."""

    kind: Literal["ratio"] = "ratio"
    weights: tuple[int, ...]

    @dx.validates("weights")
    @classmethod
    def _weights_are_positive(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        """Validate a nonempty positive weight for every eventual task."""
        if not values:
            msg = "ratio mixing declares at least one task weight"
            raise ValueError(msg)

        return tuple(
            ensure_positive(f"mixing weight {position}", value)
            for position, value in enumerate(values)
        )


class TemperatureMixingSpec(MixingSpec, extra="forbid"):
    """Allocate a fixed-size epoch by temperature-scaled dataset size."""

    kind: Literal["temperature"] = "temperature"
    temperature: float

    @dx.validates("temperature")
    @classmethod
    def _temperature_is_positive(cls, value: float) -> float:
        """Validate a positive finite sampling temperature."""
        held = ensure_finite("mixing temperature", value)

        if held <= 0.0:
            msg = f"mixing temperature is not positive: {value!r}"
            raise ValueError(msg)

        return held
