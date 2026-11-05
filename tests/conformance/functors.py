from typing import ClassVar, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.terms import Functors, Term, at_path

__all__ = [
    "FUNCTORS_CONFORMANCE_LAWS",
    "FunctorFixture",
    "FunctorsConformance",
    "functor_construction_round_trips",
    "functor_paths_are_distinct_children",
    "opposites_are_total_and_involutive",
    "slash_inventory_is_nonempty_and_distinguished",
]


class FunctorFixture(NamedTuple):
    """Values for functor-view checks."""

    functors: Functors[Term]
    result: Term
    argument: Term


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation with its name."""
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def slash_inventory_is_nonempty_and_distinguished(
    fixture: FunctorFixture,
) -> None:
    """Check that the distinguished slashes belong to a nonempty inventory."""
    functors = fixture.functors

    if not functors.slashes:
        _fail("the slash inventory is empty", law="slash inventory")

    missing = tuple(
        slash
        for slash in (functors.forward_slash, functors.backward_slash)
        if slash not in functors.slashes
    )

    if missing:
        _fail(
            f"the distinguished slashes {missing!r} are absent from "
            f"{functors.slashes!r}",
            law="slash inventory",
        )


def functor_paths_are_distinct_children(
    fixture: FunctorFixture,
) -> None:
    """Check that result and argument use distinct one-child paths."""
    functors = fixture.functors
    paths = (functors.result_path, functors.argument_path)

    if any(len(path) != 1 for path in paths):
        _fail(f"the result and argument paths are {paths!r}", law="functor paths")

    if functors.result_path == functors.argument_path:
        _fail(f"both roles use {functors.result_path!r}", law="functor paths")


def opposites_are_total_and_involutive(
    fixture: FunctorFixture,
) -> None:
    """Check that every slash has an opposite that maps back to it."""
    functors = fixture.functors

    for slash in functors.slashes:
        try:
            opposite = functors.opposite(slash)
            back = functors.opposite(opposite)

        except Exception as refusal:
            detail = f"opposite refused {slash!r} with {type(refusal).__name__}"
            message = f"opposite: {detail}"

            raise LawViolationError(message) from refusal

        if opposite not in functors.slashes:
            _fail(
                f"the opposite of {slash!r} is {opposite!r}, outside "
                f"{functors.slashes!r}",
                law="opposite",
            )

        if back != slash:
            _fail(
                f"{slash!r} maps to {opposite!r} and then to {back!r}",
                law="opposite",
            )


def functor_construction_round_trips(fixture: FunctorFixture) -> None:
    """Check that construction preserves slash, sort, result, and argument."""
    functors = fixture.functors

    for slash in functors.slashes:
        built = functors.make(slash, fixture.result, fixture.argument)

        if not functors.is_functor(built):
            _fail(f"make returned nonfunctor {built!r}", law="functor construction")

        if functors.slash(built) != slash:
            _fail(
                f"make used {slash!r} and slash returned {functors.slash(built)!r}",
                law="functor construction",
            )

        if built.op.sort != functors.category_sort:
            _fail(
                f"make built sort {built.op.sort!r}, not the declared category "
                f"sort {functors.category_sort!r}",
                law="functor construction",
            )

        result = at_path(built, functors.result_path)
        argument = at_path(built, functors.argument_path)

        if result is not fixture.result:
            _fail(
                f"the result path reads {result!r}, not {fixture.result!r}",
                law="functor construction",
            )

        if argument is not fixture.argument:
            _fail(
                f"the argument path reads {argument!r}, not {fixture.argument!r}",
                law="functor construction",
            )


class FunctorsConformance:
    """Reusable functor-view conformance checks."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_slash_inventory_is_nonempty_and_distinguished(
        self, functors_fixture: FunctorFixture
    ) -> None:
        """Run the slash-inventory law."""
        slash_inventory_is_nonempty_and_distinguished(functors_fixture)

    def test_functor_paths_are_distinct_children(
        self, functors_fixture: FunctorFixture
    ) -> None:
        """Run the functor-path law."""
        functor_paths_are_distinct_children(functors_fixture)

    def test_opposites_are_total_and_involutive(
        self, functors_fixture: FunctorFixture
    ) -> None:
        """Run the opposite law."""
        opposites_are_total_and_involutive(functors_fixture)

    def test_functor_construction_round_trips(
        self, functors_fixture: FunctorFixture
    ) -> None:
        """Run the construction law."""
        functor_construction_round_trips(functors_fixture)


FUNCTORS_CONFORMANCE_LAWS: Final = (
    FunctorsConformance.test_slash_inventory_is_nonempty_and_distinguished,
    FunctorsConformance.test_functor_paths_are_distinct_children,
    FunctorsConformance.test_opposites_are_total_and_involutive,
    FunctorsConformance.test_functor_construction_round_trips,
)
