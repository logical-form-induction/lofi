from operator import eq
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from collections.abc import Set as AbstractSet

    from torch import Tensor

__all__ = [
    "CONFORMANCE_LAWS",
    "Conformance",
    "ConformanceSuite",
]

_PROBE: Final = "_conformance_probe"

_LIVE: Final = (list, dict, set, bytearray)


def _stay[T](subject: T, _placement: str) -> T:
    """Return the subject where it is, which is all a pure value can do.

    Parameters
    ----------
    subject : T
        The subject to place.
    _placement : str
        Where it was asked to go, which a pure value has no notion of.

    Returns
    -------
    The subject, unchanged.
    """
    return subject


type _Observation = (
    bool
    | int
    | float
    | str
    | Tensor
    | Sequence[_Observation]
    | AbstractSet[_Observation]
    | None
)


def _itself[T](subject: T) -> T:
    """Return the subject as its own reading.

    Parameters
    ----------
    subject : T
        The placed subject.

    Returns
    -------
    The subject, which the placement law then compares with equality.
    """
    return subject


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


class Conformance[T](NamedTuple):
    """Inputs shared by the generic conformance laws.

    Parameters
    ----------
    build : Callable[[], T]
        Return a value equal to every other value this returns, and a fresh object each
        time.
    unequal : T
        A value of the same kind that must not compare equal to a built one.
    operations : tuple[tuple[str, Callable[[T], _Observation]], ...]
        The public operations a caller may run on the value, each under the name a
        message reports it by.
    placements : tuple[str, ...]
        Where the subject runs, the first being where it is built.
    place : Callable[[T, str], T]
        Put the subject on a placement and return what runs there.
    read : Callable[[T], T | _Observation]
        What the placement law compares, given a placed subject.
    """

    build: Callable[[], T]
    unequal: T
    operations: tuple[tuple[str, Callable[[T], _Observation]], ...] = ()
    placements: tuple[str, ...] = ("cpu",)
    place: Callable[[T, str], T] = _stay
    read: Callable[[T], T | _Observation] = _itself


class ConformanceSuite:
    pytestmark = pytest.mark.conformance

    def test_equality_is_reflexive[T](self, conformance: Conformance[T]) -> None:
        """Check that a value equals itself.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a built value does not equal itself.
        """
        value = conformance.build()

        # use operator equality because self-comparison is the tested property
        if not eq(value, value):
            _fail(f"{value!r} does not equal itself", law="equality_is_reflexive")

    def test_equality_is_symmetric[T](self, conformance: Conformance[T]) -> None:
        """Check that two built values agree in both directions.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If two values the fixture built do not compare equal both ways.
        """
        one = conformance.build()
        other = conformance.build()

        if not (one == other and other == one):
            _fail(
                f"{one!r} and {other!r} do not compare equal in both directions",
                law="equality_is_symmetric",
            )

    def test_equality_is_transitive[T](self, conformance: Conformance[T]) -> None:
        """Check that equality holds across three built values.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the first two and the last two agree while the ends do not.
        """
        first = conformance.build()
        middle = conformance.build()
        last = conformance.build()

        if not (first == middle and middle == last):
            _fail(
                f"the fixture built {first!r}, {middle!r} and {last!r}, "
                "which are not all equal",
                law="equality_is_transitive",
            )

        if first != last:
            _fail(
                f"{first!r} equals {middle!r} and {middle!r} equals {last!r}, "
                f"but {first!r} does not equal {last!r}",
                law="equality_is_transitive",
            )

    def test_equal_values_hash_alike[T](self, conformance: Conformance[T]) -> None:
        """Check that two equal values hash to the same number.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If two equal values hash differently, which loses one of them in any set or
            mapping that holds the other.
        """
        one = conformance.build()
        other = conformance.build()

        if one == other and hash(one) != hash(other):
            _fail(
                f"{one!r} equals {other!r} and hashes to {hash(one)} "
                f"against {hash(other)}",
                law="equal_values_hash_alike",
            )

    def test_a_different_value_is_not_equal[T](
        self,
        conformance: Conformance[T],
    ) -> None:
        """Check that the value the fixture names as different is not equal.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a built value compares equal to it in either direction.
        """
        value = conformance.build()

        if value == conformance.unequal or conformance.unequal == value:
            _fail(
                f"{value!r} compares equal to {conformance.unequal!r}",
                law="a_different_value_is_not_equal",
            )

    def test_no_attribute_can_be_attached[T](
        self,
        conformance: Conformance[T],
    ) -> None:
        """Check that a built value takes on nothing after it is built.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If a name the value does not declare can be set on it, which an instance
            dictionary on the value allows.
        """
        value = conformance.build()

        try:
            setattr(value, _PROBE, 0)
        except AttributeError, TypeError:
            return

        _fail(
            f"{value!r} accepted an attribute named {_PROBE}",
            law="no_attribute_can_be_attached",
        )

    def test_an_operation_leaves_the_value_unchanged[T](
        self,
        conformance: Conformance[T],
    ) -> None:
        """Check that no public operation moves the value it is run on.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If an operation changes the value's hash or moves it out of its equality
            class, either of which loses it in a set or a mapping it was already in.
        """
        value = conformance.build()
        peer = conformance.build()
        before = hash(value)

        for name, operation in conformance.operations:
            operation(value)

            if hash(value) != before:
                _fail(
                    f"{name} moved the hash of {value!r} from {before} "
                    f"to {hash(value)}",
                    law="an_operation_leaves_the_value_unchanged",
                )

            if value != peer:
                _fail(
                    f"{name} moved {value!r} out of the class of {peer!r}",
                    law="an_operation_leaves_the_value_unchanged",
                )

    def test_no_operation_hands_out_a_live_collection[T](
        self,
        conformance: Conformance[T],
    ) -> None:
        """Check that an accessor returns nothing the caller can mutate.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If an operation returns a mutable container.
        """
        value = conformance.build()

        for name, operation in conformance.operations:
            result = operation(value)

            if isinstance(result, _LIVE):
                _fail(
                    f"{name} handed out a {type(result).__name__}, "
                    "which the caller can mutate",
                    law="no_operation_hands_out_a_live_collection",
                )

    def test_placement_does_not_change_the_reading[T](
        self,
        conformance: Conformance[T],
    ) -> None:
        """Check that moving the subject leaves what it computes alone.

        Parameters
        ----------
        conformance : Conformance[T]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the fixture names no placement, or if a reading taken at one placement
            differs from the reading taken at the first.
        """
        if not conformance.placements:
            _fail(
                "the fixture names no placement to run on",
                law="placement_does_not_change_the_reading",
            )

        home = conformance.placements[0]
        subject = conformance.place(conformance.build(), home)
        baseline = conformance.read(subject)

        for placement in (*conformance.placements[1:], home):
            subject = conformance.place(subject, placement)
            reading = conformance.read(subject)

            if reading != baseline:
                _fail(
                    f"the reading on {home} is {baseline!r} and the reading "
                    f"on {placement} is {reading!r}",
                    law="placement_does_not_change_the_reading",
                )


CONFORMANCE_LAWS: Final = (
    ConformanceSuite.test_equality_is_reflexive,
    ConformanceSuite.test_equality_is_symmetric,
    ConformanceSuite.test_equality_is_transitive,
    ConformanceSuite.test_equal_values_hash_alike,
    ConformanceSuite.test_a_different_value_is_not_equal,
    ConformanceSuite.test_no_attribute_can_be_attached,
    ConformanceSuite.test_an_operation_leaves_the_value_unchanged,
    ConformanceSuite.test_no_operation_hands_out_a_live_collection,
    ConformanceSuite.test_placement_does_not_change_the_reading,
)
