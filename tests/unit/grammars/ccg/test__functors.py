import pickle
from copy import copy, deepcopy
from typing import Final

import pytest
from tests.conformance.functors import FunctorFixture, FunctorsConformance

from lofi.grammars.ccg import FunctorView
from lofi.terms import Operator, Signature, TermStore

SIGNATURE: Final = Signature(
    "functors",
    ("Cat", "Meaning"),
    (
        Operator("S", "Cat"),
        Operator("NP", "Cat"),
        Operator("m", "Meaning"),
        Operator("/", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("\\", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("->", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("pair", "Meaning", arg_sorts=("Meaning", "Meaning")),
        Operator("un", "Cat", arg_sorts=("Cat",)),
        Operator("tag", "Cat", arg_sorts=("Cat", "Meaning")),
    ),
)
STORE: Final = TermStore(SIGNATURE)
S: Final = STORE.op(SIGNATURE.by_name("S"))
NP: Final = STORE.op(SIGNATURE.by_name("NP"))
SLASHES: Final = FunctorView(STORE, "/", "\\", result=0, argument=1)
ARROW: Final = FunctorView(STORE, "->", "->", result=1, argument=0)


def test_a_directed_view_reads_its_two_constructors_and_slots() -> None:
    backward = SLASHES.make("\\", S, NP)

    assert SLASHES.slashes == frozenset({"/", "\\"})
    assert (SLASHES.forward_slash, SLASHES.backward_slash) == ("/", "\\")
    assert (SLASHES.result_path, SLASHES.argument_path) == ((0,), (1,))
    assert SLASHES.category_sort == "Cat"
    assert SLASHES.opposite("/") == "\\"
    assert SLASHES.opposite("\\") == "/"
    assert backward.args == (S, NP)
    assert SLASHES.is_functor(backward)
    assert not SLASHES.is_functor(NP)
    assert SLASHES.slash(backward) == "\\"
    assert not SLASHES.is_functor(ARROW.make("->", S, NP))

    with pytest.raises(KeyError):
        SLASHES.opposite("->")

    with pytest.raises(KeyError):
        SLASHES.make("->", S, NP)


def test_a_self_opposite_view_places_its_result_at_the_declared_slot() -> None:
    built = ARROW.make("->", S, NP)

    assert ARROW.slashes == frozenset({"->"})
    assert ARROW.forward_slash == ARROW.backward_slash == "->"
    assert ARROW.opposite("->") == "->"
    assert (ARROW.result_path, ARROW.argument_path) == ((1,), (0,))
    assert built.args == (NP, S)

    with pytest.raises(KeyError):
        ARROW.opposite("/")


@pytest.mark.parametrize(
    ("forward", "backward", "slots", "message"),
    [
        ("un", "->", (0, 1), "not a binary endofunctor"),
        ("tag", "->", (0, 1), "not a binary endofunctor"),
        ("->", "pair", (0, 1), "build different sorts"),
        ("->", "->", (0, 0), "two child positions"),
        ("->", "->", (1, 2), "two child positions"),
    ],
)
def test_a_functor_view_refuses_constructors_that_build_no_functor(
    forward: str, backward: str, slots: tuple[int, int], message: str
) -> None:
    result, argument = slots

    with pytest.raises(ValueError, match=message):
        FunctorView(STORE, forward, backward, result=result, argument=argument)


def test_a_functor_view_is_frozen_and_copies_by_identity() -> None:
    field = "_forward"

    assert copy(SLASHES) is SLASHES
    assert deepcopy(SLASHES) is SLASHES

    with pytest.raises(AttributeError, match="immutable"):
        setattr(SLASHES, field, "->")

    with pytest.raises(TypeError):
        pickle.dumps(SLASHES)


class TestDirectedFunctorView(FunctorsConformance):
    """Check the functor view of a directed slash pair."""

    @pytest.fixture
    def functors_fixture(self) -> FunctorFixture:
        """Build a directed functor fixture."""
        return FunctorFixture(SLASHES, S, NP)


class TestSelfOppositeFunctorView(FunctorsConformance):
    """Check the functor view of one self-opposite constructor at swapped slots."""

    @pytest.fixture
    def functors_fixture(self) -> FunctorFixture:
        """Build a self-opposite functor fixture."""
        return FunctorFixture(ARROW, S, NP)
