from typing import Final, cast

import pytest
import torch

from lofi.nn._guards import WIDTH_AXIS, checked_state, positive
from lofi.nn._values import Encoded

WIDTH: Final = 6


def _state(width: int = WIDTH, axes: tuple[str, ...] = (WIDTH_AXIS,)) -> Encoded:
    return Encoded(torch.zeros(width), axes)


def test_a_positive_number_passes_through_unchanged() -> None:
    assert positive("a width", 1) == 1
    assert positive("a width", 12) == 12


def test_a_number_that_is_not_positive_is_refused_by_its_field_name() -> None:
    with pytest.raises(ValueError, match="a width is zero"):
        positive("a width", 0)

    with pytest.raises(ValueError, match="negative"):
        positive("a width", -3)

    with pytest.raises(TypeError):
        positive("a width", cast("int", 1.5))


def test_a_state_of_the_readers_own_width_passes_through_unchanged() -> None:
    state = _state()

    assert checked_state(state, WIDTH, "scheme") is state
    assert checked_state(
        Encoded(torch.zeros(3, WIDTH), ("cell", WIDTH_AXIS)), WIDTH, "head"
    ).axes == ("cell", WIDTH_AXIS)


def test_a_state_that_does_not_end_in_the_feature_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="innermost"):
        checked_state(_state(axes=("symbol",)), WIDTH, "scheme")

    with pytest.raises(ValueError, match="innermost"):
        checked_state(
            Encoded(torch.zeros(WIDTH, 3), (WIDTH_AXIS, "cell")), WIDTH, "scheme"
        )


def test_a_state_of_another_width_is_refused_by_the_readers_noun() -> None:
    with pytest.raises(ValueError, match="this scheme reads states 6 wide, not 7"):
        checked_state(_state(WIDTH + 1), WIDTH, "scheme")

    with pytest.raises(ValueError, match="this head reads states 6 wide, not 7"):
        checked_state(_state(WIDTH + 1), WIDTH, "head")

    with pytest.raises(ValueError, match="this head reads states 6 wide, not 5"):
        checked_state(_state(WIDTH - 1), WIDTH, "head")
