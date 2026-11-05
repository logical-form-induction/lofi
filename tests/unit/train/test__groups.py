from typing import Final

import pytest

from lofi.train._groups import (
    ActiveGroups,
    GroupError,
    ParameterGroup,
    ParameterGroups,
    qualified,
)

EMBEDDINGS: Final = ParameterGroup("embeddings", "embeddings")
TOP: Final = ParameterGroup("top", "layers", range(2, 4))
WHOLE: Final = ParameterGroup("whole", "")


def test_a_name_under_a_prefix_drops_whichever_side_is_empty() -> None:
    assert qualified("encoder", "layers") == "encoder.layers"
    assert qualified("", "layers") == "layers"
    assert qualified("encoder", "") == "encoder"
    assert qualified("", "") == ""


def test_a_group_refuses_an_empty_name() -> None:
    with pytest.raises(ValueError, match="a group name is empty"):
        ParameterGroup("", "layers")


@pytest.mark.parametrize("layers", [range(4, 0, -1), range(-1, 3)])
def test_a_group_refuses_a_range_that_does_not_run_forward_from_zero(
    layers: range,
) -> None:
    with pytest.raises(ValueError, match="runs forward"):
        ParameterGroup("top", "layers", layers)


def test_a_group_under_a_prefix_names_and_points_from_the_enclosing_model() -> None:
    assert TOP.under("encoder") == ParameterGroup(
        "encoder.top", "encoder.layers", range(2, 4)
    )
    assert WHOLE.under("encoder") == ParameterGroup("encoder.whole", "encoder")
    assert TOP.under("") == TOP


def test_two_groups_are_equal_when_they_select_the_same_way_under_one_name() -> None:
    same = ParameterGroup("top", "layers", range(2, 4))

    assert same == TOP
    assert hash(TOP) == hash(same)
    assert ParameterGroup("top", "layers", range(1, 4)) != TOP
    assert ParameterGroup("other", "layers", range(2, 4)) != TOP
    assert TOP != EMBEDDINGS
    assert TOP.__eq__(range(2, 4)) is NotImplemented


def test_a_group_reports_what_it_selects_and_rebuilds_from_its_repr() -> None:
    assert TOP.name == "top"
    assert TOP.module == "layers"
    assert TOP.layers == range(2, 4)
    assert EMBEDDINGS.layers is None
    assert repr(TOP) == "ParameterGroup('top', 'layers', range(2, 4))"
    assert repr(EMBEDDINGS) == "ParameterGroup('embeddings', 'embeddings')"


def test_a_group_refuses_reassignment() -> None:
    with pytest.raises(AttributeError, match="immutable"):
        TOP.__setattr__("_name", "other")


def test_a_declaration_refuses_two_groups_under_one_name() -> None:
    with pytest.raises(GroupError, match="two groups are called 'top'"):
        ParameterGroups([TOP, ParameterGroup("top", "embeddings")])


def test_a_declaration_holds_its_groups_in_name_order() -> None:
    declared = ParameterGroups([TOP, WHOLE, EMBEDDINGS])

    assert list(declared) == [EMBEDDINGS, TOP, WHOLE]
    assert declared.names == {"embeddings", "top", "whole"}
    assert len(declared) == 3
    assert len(ParameterGroups()) == 0


def test_a_declaration_under_a_prefix_qualifies_every_group() -> None:
    declared = ParameterGroups([EMBEDDINGS, TOP]).under("encoder")

    assert declared.names == {"encoder.embeddings", "encoder.top"}
    assert list(declared) == [EMBEDDINGS.under("encoder"), TOP.under("encoder")]


def test_two_declarations_are_equal_when_they_hold_the_same_groups() -> None:
    one = ParameterGroups([TOP, EMBEDDINGS])
    other = ParameterGroups([EMBEDDINGS, TOP])

    assert one == other
    assert hash(one) == hash(other)
    assert one != ParameterGroups([TOP])
    assert one.__eq__([TOP, EMBEDDINGS]) is NotImplemented
    assert repr(ParameterGroups([EMBEDDINGS])) == (
        "ParameterGroups([ParameterGroup('embeddings', 'embeddings')])"
    )


def test_a_declaration_refuses_reassignment() -> None:
    declared = ParameterGroups([TOP])

    with pytest.raises(AttributeError, match="immutable"):
        declared.__setattr__("_groups", ())


def test_a_configuration_refuses_an_empty_group_name() -> None:
    with pytest.raises(ValueError, match="an active group name is empty"):
        ActiveGroups(["top", ""])


def test_a_configuration_names_the_groups_a_run_trains() -> None:
    active = ActiveGroups(["top", "embeddings"])

    assert "top" in active
    assert "whole" not in active
    assert list(active) == ["embeddings", "top"]
    assert len(active) == 2
    assert active.names == {"embeddings", "top"}


def test_two_configurations_are_equal_when_they_train_the_same_groups() -> None:
    one = ActiveGroups(["top", "embeddings"])

    assert one == ActiveGroups(["embeddings", "top"])
    assert hash(one) == hash(ActiveGroups(["embeddings", "top"]))
    assert one != ActiveGroups(["top"])
    assert one.__eq__({"top", "embeddings"}) is NotImplemented
    assert repr(ActiveGroups(["top"])) == "ActiveGroups(['top'])"


def test_a_configuration_refuses_reassignment() -> None:
    active = ActiveGroups(["top"])

    with pytest.raises(AttributeError, match="immutable"):
        active.__setattr__("_names", frozenset())
