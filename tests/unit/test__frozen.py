"""Tests for the shared frozen-value base."""

from copy import copy, deepcopy
from pickle import dumps

import pytest

from lofi._frozen import FrozenValue, Unpicklable


class Point(FrozenValue):
    """A frozen pair of coordinates."""

    _noun = "point"

    __slots__ = ("_x", "_y")

    def __init__(self, x: int, y: int) -> None:
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_y", y)

    @property
    def x(self) -> int:
        """Return the first coordinate."""
        return self._x

    @property
    def y(self) -> int:
        """Return the second coordinate."""
        return self._y


class Origin(Point):
    """A point that takes its noun from the class it refines."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(0, 0)


class Handle(Unpicklable, FrozenValue):
    """A frozen value standing for something serialization would lose."""

    _noun = "handle"
    _because = "names a live object"

    __slots__ = ()


def test_a_frozen_value_fills_its_fields_through_the_object_setter() -> None:
    point = Point(2, 3)

    assert (point.x, point.y) == (2, 3)


def test_a_frozen_value_refuses_field_reassignment() -> None:
    point = Point(2, 3)

    with pytest.raises(AttributeError, match="a point is immutable"):
        point.__setattr__("_x", 5)


def test_a_frozen_value_refuses_field_deletion() -> None:
    point = Point(2, 3)

    with pytest.raises(AttributeError, match="a point is immutable"):
        point.__delattr__("_x")


def test_a_frozen_value_refuses_a_field_it_never_declared() -> None:
    point = Point(2, 3)

    with pytest.raises(AttributeError, match="a point is immutable"):
        point.__setattr__("z", 1)


def test_a_frozen_value_keeps_no_instance_dictionary() -> None:
    point = Point(2, 3)

    with pytest.raises(TypeError):
        vars(point)


def test_both_copies_of_a_frozen_value_are_the_value() -> None:
    point = Point(2, 3)

    assert copy(point) is point
    assert deepcopy(point) is point


def test_a_deep_copy_ignores_the_memo_it_is_handed() -> None:
    point = Point(2, 3)
    memo: dict[int, Point] = {id(point): Point(9, 9)}

    assert point.__deepcopy__(memo) is point


def test_a_refinement_inherits_the_noun_of_the_value_it_refines() -> None:
    origin = Origin()

    with pytest.raises(AttributeError, match="a point is immutable"):
        origin.__setattr__("_x", 5)


def test_a_value_declaring_no_noun_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="Nameless declares no noun"):

        class Nameless(FrozenValue):
            """A value that names no kind."""

            __slots__ = ()


def test_a_value_declaring_an_empty_noun_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="Blank declares no noun"):

        class Blank(FrozenValue):
            """A value whose kind is the empty string."""

            _noun = ""

            __slots__ = ()


def test_a_value_declaring_a_cause_refuses_serialization() -> None:
    handle = Handle()

    with pytest.raises(
        TypeError, match="a handle names a live object and cannot be pickled"
    ):
        dumps(handle)


def test_a_value_refusing_serialization_still_refuses_reassignment() -> None:
    handle = Handle()

    with pytest.raises(AttributeError, match="a handle is immutable"):
        handle.__setattr__("_probe", 1)


def test_a_value_refusing_serialization_declares_no_cause_is_refused() -> None:
    with pytest.raises(TypeError, match="Causeless declares no cause"):

        class Causeless(Unpicklable, FrozenValue):
            """A value that refuses serialization without saying why."""

            _noun = "causeless value"

            __slots__ = ()
