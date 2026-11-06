"""Tests for immutable learner-signature assignments."""

from copy import copy, deepcopy

import pytest

from lofi.spec._signature import SignatureAssignments


def test_assignments_preserve_each_slot_order_and_count_scrambles() -> None:
    assignments = SignatureAssignments(
        ("entity", "event"),
        ("arrow",),
        ("apply", "compose", "raise"),
    )

    assert assignments.primitives == ("entity", "event")
    assert assignments.constructors == ("arrow",)
    assert assignments.combinators == ("apply", "compose", "raise")
    assert assignments.scramble_space == 11


def test_assignments_are_value_equal_hashable_and_frozen() -> None:
    assignments = SignatureAssignments(("entity",), ("arrow",), ("apply",))
    equal = SignatureAssignments(("entity",), ("arrow",), ("apply",))

    assert assignments == equal
    assert hash(assignments) == hash(equal)
    assert repr(assignments) == (
        "SignatureAssignments(('entity',), ('arrow',), ('apply',))"
    )
    assert copy(assignments) is assignments
    assert deepcopy(assignments) is assignments

    with pytest.raises(AttributeError, match=r"signature assignments.*immutable"):
        assignments.__setattr__("_primitives", ("event",))


@pytest.mark.parametrize("field", ["primitives", "constructors", "combinators"])
def test_each_inventory_requires_distinct_nonempty_names(field: str) -> None:
    inventories = {
        "primitives": ("entity",),
        "constructors": ("arrow",),
        "combinators": ("apply",),
    }
    inventories[field] = ("",)

    with pytest.raises(ValueError, match="name"):
        SignatureAssignments(**inventories)

    inventories[field] = ("same", "same")

    with pytest.raises(ValueError, match="repeats"):
        SignatureAssignments(**inventories)


@pytest.mark.parametrize("field", ["primitives", "constructors", "combinators"])
def test_text_is_not_an_inventory_of_names(field: str) -> None:
    inventories: dict[str, tuple[str, ...] | str] = {
        "primitives": ("entity",),
        "constructors": ("arrow",),
        "combinators": ("apply",),
    }
    inventories[field] = "one-name"

    with pytest.raises(TypeError, match="text, not a sequence"):
        SignatureAssignments(**inventories)
