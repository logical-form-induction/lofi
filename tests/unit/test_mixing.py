"""Tests for declared and runtime task-mixing policies."""

import math
from copy import copy, deepcopy
from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.mixing import (
    Alternation,
    Interleaving,
    MixingSpec,
    ProportionalMixingSpec,
    RatioMixingSpec,
    RoundRobinMixingSpec,
    TemperatureMixingSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


@pytest.mark.parametrize(
    ("document", "variant"),
    [
        ({"kind": "round-robin"}, RoundRobinMixingSpec),
        ({"kind": "proportional"}, ProportionalMixingSpec),
        ({"kind": "ratio", "weights": [2, 1]}, RatioMixingSpec),
        ({"kind": "temperature", "temperature": 2.0}, TemperatureMixingSpec),
    ],
)
def test_each_mixing_document_reaches_its_one_variant(
    document: dict[str, JsonValue], variant: type[MixingSpec]
) -> None:
    assert type(MixingSpec.model_validate(document)) is variant


@pytest.mark.parametrize("weights", [[], [1, 0], [-1, 2], [True, 1]])
def test_ratio_mixing_requires_positive_integer_weights(weights: list[int]) -> None:
    with pytest.raises(dx.ValidationError, match=r"weights|mixing weight"):
        MixingSpec.model_validate(
            {"kind": "ratio", "weights": cast("list[JsonValue]", weights)}
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan, True])
def test_temperature_mixing_requires_a_positive_finite_number(
    temperature: float,
) -> None:
    with pytest.raises(dx.ValidationError, match="temperature"):
        MixingSpec.model_validate({"kind": "temperature", "temperature": temperature})


def test_round_robin_gives_every_task_as_many_steps_as_the_largest_dataset() -> None:
    policy = Interleaving.round_robin()

    assert policy.counts([3, 5]) == (5, 5)
    assert policy.schedule([3, 5]) == (0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    assert policy.alternation is Alternation.ROUND_ROBIN
    assert policy.declared == ()


def test_proportional_alternation_gives_each_task_exactly_its_own_batches() -> None:
    policy = Interleaving.proportional()

    assert policy.counts([3, 5]) == (3, 5)
    assert policy.counts([4, 4]) == (4, 4)
    assert policy.alternation is Alternation.PROPORTIONAL


def test_temperature_one_is_proportional_and_other_temperatures_reallocate() -> None:
    proportional = Interleaving.temperature(1.0)
    softened = Interleaving.temperature(2.0)
    sharpened = Interleaving.temperature(0.5)

    assert proportional.counts([2, 8]) == (2, 8)
    assert softened.counts([2, 8]) == (3, 7)
    assert sharpened.counts([2, 8]) == (1, 9)
    assert sum(softened.counts([2, 8])) == 10
    assert proportional.alternation is Alternation.TEMPERATURE
    assert proportional.temperature_value == pytest.approx(1.0)


def test_a_ratio_scales_until_every_dataset_is_covered_at_least_once() -> None:
    assert Interleaving.ratio([2, 1]).counts([3, 5]) == (10, 5)
    assert Interleaving.ratio([1, 2]).counts([3, 5]) == (3, 6)
    assert Interleaving.ratio([1, 1]).counts([3, 5]) == (5, 5)
    assert Interleaving.ratio([3, 1]).counts([6, 2]) == (6, 2)


def test_a_ratio_runs_enough_rounds_to_reach_the_end_of_every_dataset() -> None:
    assert Interleaving.ratio([2, 1]).counts([7, 2]) == (8, 4)
    assert Interleaving.ratio([3, 2]).counts([10, 3]) == (12, 8)
    assert Interleaving.ratio([4, 1]).counts([9, 1]) == (12, 3)


def test_a_ratio_spreads_the_heavier_task_evenly_through_the_epoch() -> None:
    assert Interleaving.ratio([1, 2]).schedule([2, 2]) == (1, 0, 1, 1, 0, 1)


def test_an_alternation_that_reads_its_own_weights_is_given_none() -> None:
    with pytest.raises(ValueError, match="round_robin alternation reads its weights"):
        Interleaving(Alternation.ROUND_ROBIN, [1, 1])

    with pytest.raises(ValueError, match="proportional alternation reads its weights"):
        Interleaving(Alternation.PROPORTIONAL, [1, 1])


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_a_temperature_is_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="temperature is not positive and finite"):
        Interleaving.temperature(value)


def test_a_ratio_declares_at_least_one_positive_weight() -> None:
    with pytest.raises(ValueError, match="a ratio declares one weight per task"):
        Interleaving(Alternation.RATIO)

    with pytest.raises(ValueError, match="the weight of task 1 is not positive"):
        Interleaving.ratio([1, 0])

    with pytest.raises(ValueError, match="the weight of task 0 is negative"):
        Interleaving.ratio([-1])


def test_an_interleaving_runs_over_at_least_one_task() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        Interleaving.round_robin().counts([])


def test_an_interleaving_is_a_frozen_value() -> None:
    policy = Interleaving.ratio([2, 3])

    assert policy == Interleaving(Alternation.RATIO, [2, 3])
    assert policy != Interleaving.ratio([3, 2])
    assert policy != Interleaving.round_robin()
    assert len({policy, Interleaving(Alternation.RATIO, (2, 3))}) == 1
    assert repr(policy) == f"Interleaving({Alternation.RATIO!r}, [2, 3])"
    assert repr(Interleaving.round_robin()) == (
        f"Interleaving({Alternation.ROUND_ROBIN!r})"
    )

    with pytest.raises(AttributeError, match="immutable"):
        policy.__setattr__("_weights", ())

    with pytest.raises(AttributeError, match="immutable"):
        policy.__delattr__("_weights")

    assert copy(policy) is policy
    assert deepcopy(policy) is policy
