import importlib
import math
from typing import TYPE_CHECKING, cast

import pytest

from lofi import _checks
from lofi._checks import (
    NumericalFailureError,
    ensure_commit,
    ensure_finite,
    ensure_fraction,
    ensure_name,
    ensure_non_negative,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_checks_is_on_only_for_one(
    reload_with_checks: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for on in (True, False):
        reload_with_checks(on=on)
        assert _checks.CHECKS is on

    monkeypatch.setenv("LOFI_CHECK", "true")
    assert importlib.reload(_checks).CHECKS is False

    monkeypatch.delenv("LOFI_CHECK")
    assert importlib.reload(_checks).CHECKS is False


def test_a_finite_number_comes_back_as_a_float() -> None:
    assert ensure_finite("a bound", 0.25) == 0.25
    assert ensure_finite("a bound", -3.5) == -3.5
    assert ensure_finite("a bound", 0.0) == 0.0


def test_a_whole_number_is_widened_rather_than_refused() -> None:
    widened = ensure_finite("a bound", 2)

    assert isinstance(widened, float)
    assert widened == 2.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_number_that_is_not_finite_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="a bound is not finite"):
        ensure_finite("a bound", value)


@pytest.mark.parametrize("value", ["0.5", None, (0.5,), complex(1, 1), True])
def test_a_value_that_is_not_a_number_is_refused(
    value: str | tuple[float] | complex | None,
) -> None:
    with pytest.raises(TypeError, match="a bound is not a number"):
        ensure_finite("a bound", cast("float", value))


def test_a_fraction_comes_back_as_a_float_at_and_between_the_bounds() -> None:
    assert ensure_fraction("a share", 0) == 0.0
    assert ensure_fraction("a share", 0.25) == 0.25
    assert isinstance(ensure_fraction("a share", 1), float)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
def test_a_value_outside_the_unit_interval_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="a share"):
        ensure_fraction("a share", value)


def test_a_fraction_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(TypeError, match="a share is not a number"):
        ensure_fraction("a share", cast("float", "0.5"))


def test_a_non_negative_integer_comes_back_unchanged() -> None:
    assert ensure_non_negative("a seed", 0) == 0
    assert ensure_non_negative("a seed", 17) == 17


def test_a_negative_integer_is_refused() -> None:
    with pytest.raises(ValueError, match="a seed is negative: -1"):
        ensure_non_negative("a seed", -1)


@pytest.mark.parametrize("value", [1.0, "1", None, True])
def test_a_count_that_is_not_an_integer_is_refused(
    value: float | str | None,
) -> None:
    with pytest.raises(TypeError, match="a seed is not an integer"):
        ensure_non_negative("a seed", cast("int", value))


def test_a_non_empty_string_comes_back_unchanged() -> None:
    assert ensure_name("a unit", "sentence") == "sentence"
    assert ensure_name("a unit", " ") == " "


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="a unit is empty"):
        ensure_name("a unit", "")


@pytest.mark.parametrize("value", [None, 3, b"sentence", ("sentence",)])
def test_a_name_that_is_not_a_string_is_refused(
    value: int | bytes | tuple[str] | None,
) -> None:
    with pytest.raises(TypeError, match="a unit is not a string"):
        ensure_name("a unit", cast("str", value))


def test_the_field_a_caller_names_is_what_the_message_reports() -> None:
    with pytest.raises(ValueError, match="an interval's lower bound is not finite"):
        ensure_finite("an interval's lower bound", math.nan)

    with pytest.raises(ValueError, match="a null's replicate count is negative"):
        ensure_non_negative("a null's replicate count", -4)

    with pytest.raises(ValueError, match="a bound's consumer is empty"):
        ensure_name("a bound's consumer", "")


def test_a_full_commit_hash_comes_back_unchanged() -> None:
    assert ensure_commit("a revision", "0a" * 20) == "0a" * 20


@pytest.mark.parametrize("value", ["main", "0A" * 20, "0" * 39, "0" * 41, ""])
def test_a_revision_that_is_not_a_full_commit_hash_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="a revision"):
        ensure_commit("a revision", value)


def test_a_numerical_failure_names_the_quantity_and_any_value_reported() -> None:
    reported = NumericalFailureError("metrics['loss']", float("inf"))
    unreported = NumericalFailureError("the parser loss in epoch 2", None)

    assert (reported.quantity, reported.value) == ("metrics['loss']", float("inf"))
    assert str(reported) == "metrics['loss'] is not finite: inf"
    assert (unreported.quantity, unreported.value) == (
        "the parser loss in epoch 2",
        None,
    )
    assert str(unreported) == "the parser loss in epoch 2 is not finite"
    assert isinstance(reported, FloatingPointError)
