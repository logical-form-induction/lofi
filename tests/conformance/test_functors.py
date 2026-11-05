from typing import Final, override

import pytest
from tests.conformance.functors import (
    FUNCTORS_CONFORMANCE_LAWS,
    FunctorFixture,
    FunctorsConformance,
    functor_construction_round_trips,
    functor_paths_are_distinct_children,
    opposites_are_total_and_involutive,
    slash_inventory_is_nonempty_and_distinguished,
)

from lofi._checks import LawViolationError
from lofi.terms import Operator, Path, Signature, Term, TermStore

SIGNATURE: Final = Signature(
    "functor-conformance",
    ("Cat",),
    (
        Operator("np", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)
STORE: Final = TermStore(SIGNATURE)
NP: Final = STORE.op(SIGNATURE.by_name("np"))
S: Final = STORE.op(SIGNATURE.by_name("s"))


class View:
    """A binary functor view backed by one store."""

    __slots__ = ("_backward", "_forward", "_store")

    def __init__(
        self,
        store: TermStore,
        forward: str,
        backward: str,
    ) -> None:
        self._store = store
        self._forward = forward
        self._backward = backward

    @property
    def slashes(self) -> frozenset[str]:
        """Return the declared slash inventory."""
        return frozenset({self._forward, self._backward})

    @property
    def forward_slash(self) -> str:
        """Return the forward slash."""
        return self._forward

    @property
    def backward_slash(self) -> str:
        """Return the backward slash."""
        return self._backward

    @property
    def result_path(self) -> Path:
        """Return the result path."""
        return (0,)

    @property
    def argument_path(self) -> Path:
        """Return the argument path."""
        return (1,)

    @property
    def category_sort(self) -> str:
        """Return the sort the functor constructors build."""
        return "Cat"

    def is_functor(self, term: Term, /) -> bool:
        """Return whether a term uses a declared slash."""
        return term.op.name in self.slashes

    def slash(self, term: Term, /) -> str:
        """Return a functor's slash."""
        return term.op.name

    def make(
        self,
        slash: str,
        result: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return the interned functor."""
        return self._store.op(self._store.signature.by_name(slash), result, argument)

    def opposite(self, slash: str, /) -> str:
        """Return the oppositely oriented slash."""
        if slash == self._forward:
            return self._backward

        if slash == self._backward:
            return self._forward

        raise KeyError(slash)


class SamePaths(View):
    """A view with one path for both functor roles."""

    __slots__ = ()

    @property
    @override
    def argument_path(self) -> Path:
        """Return the result path again."""
        return self.result_path


class Vacant(View):
    """A view with no slash inventory."""

    __slots__ = ()

    @property
    @override
    def slashes(self) -> frozenset[str]:
        """Return no slashes."""
        return frozenset()


class Cycling(View):
    """A view with a noninvolutive opposite operation."""

    __slots__ = ()

    @override
    def opposite(self, slash: str, /) -> str:
        """Return the forward slash for every input."""
        del slash

        return self.forward_slash


class Reversing(View):
    """A view that reverses result and argument roles."""

    __slots__ = ()

    @override
    def make(
        self,
        slash: str,
        result: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return a functor with its argument first."""
        return super().make(slash, argument, result)


def _fixture(view: View) -> FunctorFixture:
    """Return the shared terms under one view."""
    return FunctorFixture(view, S, NP)


class TestDirectedFunctors(FunctorsConformance):
    """The conformance suite over two directed slashes."""

    @pytest.fixture
    def functors_fixture(self) -> FunctorFixture:
        """Return a directed functor fixture."""
        return _fixture(View(STORE, "fwd", "bwd"))


class TestUndirectedFunctors(FunctorsConformance):
    """The conformance suite over one self-opposite slash."""

    @pytest.fixture
    def functors_fixture(self) -> FunctorFixture:
        """Return an undirected functor fixture."""
        return _fixture(View(STORE, "arrow", "arrow"))


def test_the_inventory_law_catches_an_empty_set() -> None:
    with pytest.raises(LawViolationError, match="slash inventory is empty"):
        slash_inventory_is_nonempty_and_distinguished(
            _fixture(Vacant(STORE, "fwd", "bwd"))
        )


def test_the_path_law_catches_two_roles_at_one_child() -> None:
    with pytest.raises(LawViolationError, match="both roles use"):
        functor_paths_are_distinct_children(_fixture(SamePaths(STORE, "fwd", "bwd")))


def test_the_opposite_law_catches_a_noninvolution() -> None:
    with pytest.raises(LawViolationError, match="maps to"):
        opposites_are_total_and_involutive(_fixture(Cycling(STORE, "fwd", "bwd")))


def test_the_construction_law_catches_reversed_roles() -> None:
    with pytest.raises(LawViolationError, match="the result path reads"):
        functor_construction_round_trips(_fixture(Reversing(STORE, "fwd", "bwd")))


class Misdeclared(View):
    """A view declaring a category sort its constructors do not build."""

    __slots__ = ()

    @property
    @override
    def category_sort(self) -> str:
        """Return a sort the constructors do not build."""
        return "Other"


def test_the_construction_law_catches_a_misdeclared_category_sort() -> None:
    with pytest.raises(LawViolationError, match="declared category sort 'Other'"):
        functor_construction_round_trips(_fixture(Misdeclared(STORE, "fwd", "bwd")))


def test_the_suite_binds_every_functor_law_once() -> None:
    assert tuple(
        method.__name__.removeprefix("test_") for method in FUNCTORS_CONFORMANCE_LAWS
    ) == (
        "slash_inventory_is_nonempty_and_distinguished",
        "functor_paths_are_distinct_children",
        "opposites_are_total_and_involutive",
        "functor_construction_round_trips",
    )
