from typing import Final

import pytest
from hypothesis import given, settings
from tests.support.term_strategies import terms

from lofi.judge import SpineLimit, SpineMatch, SpineMatches, match_spine
from lofi.terms import MVar, Node, Operator, Signature, Spine, Term, TermStore, build

SIGNATURE: Final = Signature(
    "spine-matching",
    ("Ty",),
    (
        Operator("n", "Ty"),
        Operator("np", "Ty"),
        Operator("s", "Ty"),
        Operator("fwd", "Ty", arg_sorts=("Ty", "Ty")),
        Operator("bwd", "Ty", arg_sorts=("Ty", "Ty")),
    ),
)
STORE: Final = TermStore(SIGNATURE)
N: Final = STORE.op(SIGNATURE.by_name("n"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
FWD: Final = SIGNATURE.by_name("fwd")
BWD: Final = SIGNATURE.by_name("bwd")


def _spine(order: int) -> Term:
    """Build a left-result spine."""
    built = N

    for at in range(order):
        built = STORE.op(FWD if at % 2 == 0 else BWD, built, NP)

    return built


def _orders(matches: SpineMatches) -> tuple[int, ...]:
    """Return the orders an enumeration found."""
    return tuple(found.order for found in matches.matches)


@pytest.mark.parametrize("order", range(4))
def test_one_schema_matches_each_order_from_zero_through_three(order: int) -> None:
    pattern = Spine(
        "context",
        Node("n"),
        ("fwd", "bwd"),
        min_order=0,
        max_order=3,
    )
    result = match_spine(pattern, _spine(order))

    assert _orders(result) == (order,)
    assert result.exhausted == frozenset()
    assert build(pattern, STORE, result.matches[0].bindings) is _spine(order)


def test_the_same_schema_declines_order_four_and_reports_the_maximum() -> None:
    pattern = Spine(
        "context",
        Node("n"),
        ("fwd", "bwd"),
        min_order=0,
        max_order=3,
    )
    result = match_spine(pattern, _spine(4))

    assert result.matches == ()
    assert result.exhausted == frozenset({"max_order"})


def test_an_unbounded_schema_matches_the_finite_spine_above_order_three() -> None:
    pattern = Spine(
        "context",
        Node("n"),
        ("fwd", "bwd"),
        min_order=0,
        max_order=None,
    )
    result = match_spine(pattern, _spine(6))

    assert _orders(result) == (6,)
    assert result.exhausted == frozenset()


def test_an_empty_result_distinguishes_no_match_from_exhausted_fuel() -> None:
    pattern = Spine("context", Node("n"), ("fwd", "bwd"), max_order=3)

    no_match = match_spine(pattern, NP)
    no_fuel = match_spine(pattern, N, fuel=0)

    assert no_match.matches == no_fuel.matches == ()
    assert no_match.exhausted == frozenset()
    assert no_fuel.exhausted == frozenset({"fuel"})


def test_every_bound_hit_is_named_in_the_exhausted_set() -> None:
    pattern = Spine("context", Node("n"), ("fwd", "bwd"), max_order=3)
    result = match_spine(pattern, _spine(4), fuel=2)

    assert result.matches == ()
    assert result.exhausted == frozenset[SpineLimit]({"max_order", "fuel"})


def test_fuel_that_covers_the_declared_band_does_not_report_exhaustion() -> None:
    pattern = Spine("context", Node("n"), ("fwd", "bwd"), max_order=3)

    assert _orders(match_spine(pattern, _spine(3), fuel=4)) == (3,)
    assert match_spine(pattern, _spine(3), fuel=4).exhausted == frozenset()


@settings(max_examples=2000, deadline=None)
@given(terms(STORE, sort="Ty", max_size=8, closed=True))
def test_every_enumerated_context_recomposes_to_the_original_term(
    term: Term,
) -> None:
    pattern = Spine(
        "context",
        MVar("core"),
        ("fwd", "bwd"),
        min_order=0,
        max_order=3,
    )
    result = match_spine(pattern, term)

    for found in result.matches:
        assert build(pattern, STORE, found.bindings) is term


def test_match_values_are_frozen_slotted_values() -> None:
    bindings = (
        match_spine(Spine("context", Node("n"), (), max_order=3), N).matches[0].bindings
    )
    match = SpineMatch(0, bindings)
    result = SpineMatches((match,), ("fuel",))

    assert match == SpineMatch(0, bindings)
    assert result == SpineMatches((match,), ("fuel",))
    assert hash(match) == hash(SpineMatch(0, bindings))
    assert hash(result) == hash(SpineMatches((match,), ("fuel",)))

    with pytest.raises(TypeError):
        vars(match)

    with pytest.raises(TypeError):
        vars(result)

    with pytest.raises(AttributeError, match="immutable"):
        match.__setattr__("_order", 1)

    with pytest.raises(AttributeError, match="immutable"):
        result.__setattr__("_matches", ())


def test_negative_fuel_is_refused() -> None:
    with pytest.raises(ValueError, match="fuel is negative"):
        match_spine(Spine("context", Node("n"), (), max_order=3), N, fuel=-1)
