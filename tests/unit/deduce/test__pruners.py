from math import inf, log

import pytest

from lofi.deduce._ports import CellPruner
from lofi.deduce._pruners import TopKPruner
from lofi.deduce._values import ItemKey
from lofi.deduce.laws import PrunerFixture, check_pruner_laws


def cell(*weights: float) -> tuple[tuple[ItemKey[str], float], ...]:
    return tuple(
        (ItemKey((0, 2), f"X{at}", "step"), weight) for at, weight in enumerate(weights)
    )


def readings(kept: tuple[ItemKey[str], ...]) -> tuple[str, ...]:
    return tuple(key.reading for key in kept)


def test_a_pruner_is_a_port_the_chart_can_hold() -> None:
    assert isinstance(TopKPruner[str](3), CellPruner)
    assert not isinstance(object(), CellPruner)


def test_a_cell_no_bigger_than_the_beam_keeps_every_item() -> None:
    pruner = TopKPruner[str](3)
    items = cell(log(0.1), log(0.9), log(0.5))

    assert readings(pruner.keep((0, 2), items)) == ("X0", "X1", "X2")
    assert pruner.keep((0, 2), ()) == ()


def test_the_beam_keeps_the_heaviest_items_and_returns_them_in_arrival_order() -> None:
    pruner = TopKPruner[str](2)
    items = cell(log(0.1), log(0.9), log(0.5))

    assert readings(pruner.keep((0, 2), items)) == ("X1", "X2")


def test_items_of_equal_weight_are_kept_in_the_order_they_arrived() -> None:
    pruner = TopKPruner[str](2)
    items = cell(log(0.5), log(0.5), log(0.5), log(0.5))

    assert readings(pruner.keep((0, 2), items)) == ("X0", "X1")


def test_a_log_zero_item_is_dropped_before_any_item_with_weight() -> None:
    pruner = TopKPruner[str](2)
    items = cell(-inf, log(0.01), -inf, log(0.02))

    assert readings(pruner.keep((0, 2), items)) == ("X1", "X3")


def test_the_span_does_not_change_which_items_the_beam_keeps() -> None:
    pruner = TopKPruner[str](1)
    items = cell(log(0.2), log(0.8))

    assert pruner.keep((0, 2), items) == pruner.keep((4, 9), items)


@pytest.mark.parametrize("k", [0, -1])
def test_a_beam_that_would_empty_every_cell_is_refused(k: int) -> None:
    with pytest.raises(ValueError, match="at least one item"):
        TopKPruner[str](k)


def test_the_beam_is_read_back_and_rendered() -> None:
    pruner = TopKPruner[str](7)

    assert pruner.k == 7
    assert repr(pruner) == "TopKPruner(7)"


def test_the_shipped_pruner_satisfies_the_cell_pruner_laws() -> None:
    check_pruner_laws(
        PrunerFixture(
            TopKPruner[str](2),
            (
                ((0, 2), cell(log(0.1), log(0.9), log(0.5), -inf)),
                ((0, 1), cell(log(0.5))),
                ((1, 4), cell(log(0.5), log(0.5), log(0.5))),
                ((2, 3), ()),
            ),
        )
    )
