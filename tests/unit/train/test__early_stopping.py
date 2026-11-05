"""Tests for absolute-delta early stopping."""

import math
from typing import Literal, cast

import pytest

from lofi.train._early_stopping import (
    AccuracyStopping,
    EarlyStopping,
    EarlyStoppingState,
)


@pytest.mark.parametrize(
    ("mode", "values"),
    [
        ("min", (4.0, 3.0, 2.0)),
        ("max", (2.0, 3.0, 4.0)),
    ],
)
def test_first_and_strictly_better_observations_establish_the_best(
    mode: str,
    values: tuple[float, ...],
) -> None:
    rule = EarlyStopping(
        "loss",
        cast("Literal['min', 'max']", mode),
        2,
        0.5,
    )
    state = EarlyStoppingState()

    for value in values:
        decision = rule.observe(state, value)
        state = decision.state

        assert decision.improved
        assert not decision.stop
        assert state.best == value
        assert state.failures == 0


@pytest.mark.parametrize(
    ("mode", "initial", "boundary", "beyond"),
    [
        ("min", 3.0, 2.5, 2.49),
        ("max", 3.0, 3.5, 3.51),
    ],
)
def test_equality_at_the_delta_boundary_does_not_improve(
    mode: str,
    initial: float,
    boundary: float,
    beyond: float,
) -> None:
    rule = EarlyStopping(
        "score",
        cast("Literal['min', 'max']", mode),
        2,
        0.5,
    )
    first = rule.observe(EarlyStoppingState(), initial)
    equal = rule.observe(first.state, boundary)
    improved = rule.observe(equal.state, beyond)

    assert not equal.improved
    assert equal.state.best == initial
    assert equal.state.failures == 1
    assert improved.improved
    assert improved.state.best == beyond
    assert improved.state.failures == 0


def test_stopping_occurs_at_exactly_patience_consecutive_failures() -> None:
    rule = EarlyStopping("loss", "min", 3, 0.0)
    state = rule.observe(EarlyStoppingState(), 1.0).state

    for failures in range(1, 4):
        decision = rule.observe(state, 1.0)
        state = decision.state

        assert decision.stop is (failures == 3)
        assert state.failures == failures


@pytest.mark.parametrize(
    ("metric", "mode", "patience", "delta", "error"),
    [
        ("", "min", 1, 0.0, ValueError),
        ("loss", "other", 1, 0.0, ValueError),
        ("loss", "min", 0, 0.0, ValueError),
        ("loss", "min", 1, -1.0, ValueError),
        ("loss", "min", 1, math.inf, ValueError),
    ],
)
def test_invalid_runtime_controls_are_refused(
    metric: str,
    mode: str,
    patience: int,
    delta: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        EarlyStopping(
            metric,
            cast("Literal['min', 'max']", mode),
            patience,
            delta,
        )


def test_nonfinite_observations_are_refused() -> None:
    rule = EarlyStopping("loss", "min", 2, 0.0)

    with pytest.raises(ValueError, match="loss"):
        rule.observe(EarlyStoppingState(), math.nan)


def test_accuracy_stopping_names_the_target_and_ceiling_outcomes() -> None:
    rule = AccuracyStopping("held-out/exact-match", 0.95, 3)

    continuing = rule.observe(0.8, 1)
    met = rule.observe(0.95, 2)
    exhausted = rule.observe(0.9, 3)

    assert not continuing.stop
    assert continuing.reason is None
    assert met.stop
    assert met.met
    assert met.reason == "target-met"
    assert exhausted.stop
    assert not exhausted.met
    assert exhausted.reason == "target-not-met"


@pytest.mark.parametrize(
    ("metric", "target", "steps"),
    [
        ("", 0.95, 2),
        ("accuracy", -0.1, 2),
        ("accuracy", 1.1, 2),
        ("accuracy", math.inf, 2),
        ("accuracy", 0.95, 0),
    ],
)
def test_invalid_accuracy_stopping_controls_are_refused(
    metric: str, target: float, steps: int
) -> None:
    with pytest.raises(ValueError, match=r"metric|target|max_steps"):
        AccuracyStopping(metric, target, steps)


def test_accuracy_stopping_refuses_nonaccuracies_and_steps_past_its_ceiling() -> None:
    rule = AccuracyStopping("held-out/exact-match", 0.95, 2)

    with pytest.raises(ValueError, match="not an accuracy"):
        rule.observe(1.1, 1)

    with pytest.raises(ValueError, match="exceeds"):
        rule.observe(0.5, 3)
