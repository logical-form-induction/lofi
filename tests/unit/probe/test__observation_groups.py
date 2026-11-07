"""Tests for caller-declared observation row groups."""

from copy import copy, deepcopy
from pickle import dumps, loads
from typing import cast

import pytest

from lofi.probe._observation_groups import ObservationGrouping


def test_a_grouping_preserves_group_order_and_unordered_membership() -> None:
    grouping = ObservationGrouping(6, ([4, 1], [5], [0, 3]))

    assert grouping.rows == 6
    assert grouping.groups == (
        frozenset({1, 4}),
        frozenset({5}),
        frozenset({0, 3}),
    )


def test_rows_may_remain_ungrouped_and_every_row_may_be_ungrouped() -> None:
    assert ObservationGrouping(4, ((1, 3),)).groups == (frozenset({1, 3}),)
    assert ObservationGrouping(4).groups == ()


def test_a_grouping_snapshots_mutable_memberships_and_copies_by_identity() -> None:
    first = [0, 2]
    groups = [first, [3]]
    grouping = ObservationGrouping(4, groups)
    first.reverse()
    groups.clear()

    assert grouping.groups == (frozenset({0, 2}), frozenset({3}))
    assert copy(grouping) is grouping
    assert deepcopy(grouping) is grouping


def test_a_grouping_refuses_assignment_and_deletion() -> None:
    grouping = ObservationGrouping(3, ((0, 2),))

    with pytest.raises(AttributeError, match="observation grouping is immutable"):
        grouping.__setattr__("_rows", 4)

    with pytest.raises(AttributeError, match="observation grouping is immutable"):
        grouping.__delattr__("_groups")


def test_groupings_are_equal_and_hash_on_their_declaration() -> None:
    one = ObservationGrouping(5, ((3, 1), (4,)))
    same = ObservationGrouping(5, ((1, 3), (4,)))
    reordered = ObservationGrouping(5, ((4,), (1, 3)))

    assert one == same
    assert hash(one) == hash(same)
    assert one != reordered
    assert one != ObservationGrouping(4, ((1, 3),))
    assert one != "grouping"
    assert repr(one) == "ObservationGrouping(5, ((1, 3), (4,)))"


def test_a_grouping_round_trips_through_a_pickle() -> None:
    grouping = ObservationGrouping(5, ((3, 1), (4,)))
    restored = loads(dumps(grouping))

    assert type(restored) is ObservationGrouping
    assert restored == grouping
    assert hash(restored) == hash(grouping)
    assert repr(restored) == repr(grouping)


@pytest.mark.parametrize("rows", [True, 0, -1, 2.5])
def test_a_grouping_requires_a_positive_non_boolean_integer_extent(
    rows: float,
) -> None:
    with pytest.raises((TypeError, ValueError), match="row extent"):
        ObservationGrouping(cast("int", rows))


@pytest.mark.parametrize("row", [True, -1, 3, 1.5])
def test_group_rows_must_be_non_boolean_integers_inside_the_extent(
    row: float,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"row 1 of observation group 0"):
        ObservationGrouping(3, ((0, cast("int", row)),))


def test_each_observation_group_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="observation group 1 is empty"):
        ObservationGrouping(3, ((0,), ()))


def test_a_group_names_the_row_it_repeats() -> None:
    with pytest.raises(ValueError, match=r"group 0 repeats row 1"):
        ObservationGrouping(3, ((1, 0, 1),))


def test_groups_must_be_pairwise_disjoint() -> None:
    with pytest.raises(ValueError, match=r"group 1 overlaps.*row.*2"):
        ObservationGrouping(4, ((0, 2), (2, 3)))


def test_groups_must_not_duplicate_an_earlier_group() -> None:
    with pytest.raises(ValueError, match=r"group 1 duplicates.*rows.*0.*2"):
        ObservationGrouping(4, ((0, 2), (2, 0)))
