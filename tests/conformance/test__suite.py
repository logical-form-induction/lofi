from itertools import count
from typing import TYPE_CHECKING, Final, Self, override

import pytest
from tests.conformance._suite import (
    CONFORMANCE_LAWS,
    Conformance,
    ConformanceSuite,
)

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class Value:
    __slots__ = ("_tag",)

    def __init__(self, tag: int) -> None:
        self._tag = tag

    @property
    def tag(self) -> int:
        return self._tag

    def neighbors(self) -> tuple[int, ...]:
        return (self._tag - 1, self._tag + 1)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Value) and other.tag == self._tag

    @override
    def __hash__(self) -> int:
        return hash(self._tag)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tag})"


class Tolerant(Value):
    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Value) and abs(other.tag - self.tag) <= 1

    @override
    def __hash__(self) -> int:
        return hash(Tolerant)


class Sloppy(Value):
    __slots__ = ()

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)


class Everything(Value):
    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        return True

    @override
    def __hash__(self) -> int:
        return hash(Everything)


class Loose(Value):
    pass


class Leaky(Value):
    __slots__ = ("_marks",)

    def __init__(self, tag: int, marks: Sequence[int]) -> None:
        super().__init__(tag)
        self._marks = list(marks)

    def marks(self) -> list[int]:
        return self._marks


class Growing:
    __slots__ = ("_marks",)

    def __init__(self, marks: Sequence[int]) -> None:
        self._marks = list(marks)

    @property
    def marks(self) -> tuple[int, ...]:
        return tuple(self._marks)

    def counted(self) -> int:
        self._marks.append(len(self._marks))

        return len(self._marks)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Growing) and other.marks == self.marks

    @override
    def __hash__(self) -> int:
        return hash(self.marks)

    @override
    def __repr__(self) -> str:
        return f"Growing({self.marks})"


class Moving:
    __slots__ = ("_at", "_data")

    def __init__(self, data: tuple[int, ...], at: str = "") -> None:
        self._data = data
        self._at = at

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    def moved(self, placement: str) -> Self:
        return type(self)(self._data, placement)

    def total(self) -> int:
        return sum(self._data)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Moving) and other.data == self._data

    @override
    def __hash__(self) -> int:
        return hash(self._data)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data}, {self._at!r})"


class Cached:
    __slots__ = ("_at", "_data", "_home")

    def __init__(self, data: tuple[int, ...], at: str = "", home: str = "") -> None:
        self._data = data
        self._at = at
        self._home = home

    @property
    def data(self) -> tuple[int, ...]:
        return self._data

    def moved(self, placement: str) -> Self:
        return type(self)(self._data, placement, self._home or placement)

    def total(self) -> int:
        return sum(self._data) if self._at == self._home else 0

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cached) and other.data == self._data

    @override
    def __hash__(self) -> int:
        return hash(self._data)

    @override
    def __repr__(self) -> str:
        return f"Cached({self._data}, {self._at!r}, {self._home!r})"


def _rising() -> Callable[[], Tolerant]:
    """Return a builder handing out a tag one further along each time.

    Returns
    -------
    The builder, which the tolerant value needs: two of its values are close
    enough to be equal and three far enough apart not to chain.
    """
    tags = count()

    return lambda: Tolerant(next(tags))


type Subject = (
    Conformance[Value]
    | Conformance[Leaky]
    | Conformance[Growing]
    | Conformance[Moving]
    | Conformance[Cached]
)

_NEIGHBORS: Final = (("neighbors", Value.neighbors),)

_CONFORMANT: Final[tuple[Subject, ...]] = (
    Conformance[Value](
        build=lambda: Value(1),
        unequal=Value(9),
        operations=_NEIGHBORS,
    ),
    Conformance[Moving](
        build=lambda: Moving((1, 2)),
        unequal=Moving((3,)),
        operations=(("total", Moving.total),),
        placements=("cpu", "meta"),
        place=Moving.moved,
        read=Moving.total,
    ),
)

_BROKEN: Final[tuple[tuple[str, Subject], ...]] = (
    (
        "test_equality_is_transitive",
        Conformance[Value](
            build=_rising(),
            unequal=Tolerant(99),
            operations=_NEIGHBORS,
        ),
    ),
    (
        "test_equal_values_hash_alike",
        Conformance[Value](
            build=lambda: Sloppy(1),
            unequal=Sloppy(9),
            operations=_NEIGHBORS,
        ),
    ),
    (
        "test_a_different_value_is_not_equal",
        Conformance[Value](
            build=lambda: Everything(1),
            unequal=Everything(9),
            operations=_NEIGHBORS,
        ),
    ),
    (
        "test_no_attribute_can_be_attached",
        Conformance[Value](
            build=lambda: Loose(1),
            unequal=Loose(9),
            operations=_NEIGHBORS,
        ),
    ),
    (
        "test_an_operation_leaves_the_value_unchanged",
        Conformance[Growing](
            build=lambda: Growing(()),
            unequal=Growing((7,)),
            operations=(("counted", Growing.counted),),
        ),
    ),
    (
        "test_no_operation_hands_out_a_live_collection",
        Conformance[Leaky](
            build=lambda: Leaky(1, ()),
            unequal=Leaky(9, ()),
            operations=(("marks", Leaky.marks),),
        ),
    ),
    (
        "test_placement_does_not_change_the_reading",
        Conformance[Cached](
            build=lambda: Cached((1, 2)),
            unequal=Cached((3,)),
            operations=(("total", Cached.total),),
            placements=("cpu", "meta"),
            place=Cached.moved,
            read=Cached.total,
        ),
    ),
)


def _red(conformance: Subject) -> frozenset[str]:
    """Return the name of every law that fails against a subject.

    Parameters
    ----------
    conformance
        The subject and what the laws need to run against it.

    Returns
    -------
    The names of the failing laws. A law raising anything other than a law
    violation is a fault in the subject rather than a verdict, and is left to
    reach the caller.
    """
    suite = ConformanceSuite()
    red: list[str] = []

    for law in CONFORMANCE_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


@pytest.mark.parametrize("conformance", _CONFORMANT, ids=("value", "moving"))
def test_every_law_holds_of_a_conformant_subject(conformance: Subject) -> None:
    assert _red(conformance) == frozenset()


@pytest.mark.parametrize(
    ("law", "conformance"),
    _BROKEN,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, conformance: Subject) -> None:
    assert _red(conformance) == frozenset({law})
