"""Neural-facing logical-form properties."""

from math import isfinite
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.lf._fixtures import (
    PendingLogicalFormFixture,
    PendingValue,
    logical_form_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["logical_form_fixture"]

pytestmark = pytest.mark.conformance


def structure_partitions_nodes(fixture: PendingLogicalFormFixture) -> None:
    """Require current signature classes to partition every encoded node."""
    forms = fixture.term.forms(fixture.store)
    assert forms
    structures = {"abstraction", "application", "variable", "constant"}
    all_nodes = tuple(node for form in forms for node in fixture.neural.nodes(form))
    assert all_nodes
    assert {fixture.neural.structure(node) for node in all_nodes} == structures

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        nodes = fixture.neural.nodes(form)
        assert nodes
        for node in nodes:
            structure = fixture.neural.structure(node)
            children = fixture.neural.children(node)
            assert structure in structures

            if structure in {"variable", "constant"}:
                assert not children
            elif structure == "application":
                assert len(children) == 2
            else:
                assert len(children) == 1

    check()


def linearization_round_trips(fixture: PendingLogicalFormFixture) -> None:
    """Require linearization to preserve forms and parent-child indices."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        encoding = fixture.neural.linearize(form)
        symbols, parents, children = encoding
        assert symbols
        assert len(symbols) == len(parents) == len(children)
        assert all(-1 <= index < len(symbols) for index in parents)
        assert all(-1 <= index < len(symbols) for pair in children for index in pair)
        assert fixture.alpha_equal(fixture.neural.delinearize(encoding), form)

    check()


def application_is_binary(fixture: PendingLogicalFormFixture) -> None:
    """Require curried application to expose explicit binary nodes."""
    application = fixture.neural.application_example()
    children = fixture.neural.application_children(application)
    nodes = fixture.neural.nodes(application)

    assert len(children) == 2
    assert fixture.neural.children(application) == children
    assert nodes
    assert all(child in nodes for child in children)


def symbol_ids_are_stable(fixture: PendingLogicalFormFixture) -> None:
    """Require the symbol registry to assign stable integer identifiers."""
    first = fixture.neural.symbol_ids()
    second = fixture.neural.symbol_ids()
    assert first
    assert first == second
    assert len({name for name, _ in first}) == len(first)
    assert len({identifier for _, identifier in first}) == len(first)
    assert all(identifier >= 0 for _, identifier in first)


def clamps_are_respected(fixture: PendingLogicalFormFixture) -> None:
    """Require enumeration and scoring to respect whole-form and span clamps."""
    cases = fixture.neural.clamp_cases()
    assert cases

    assert {span is None for _, span, _, _ in cases} == {False, True}

    for pin, span, matching, conflicting in cases:
        candidates = fixture.neural.enumerate_with_pin(pin, span)
        assert candidates
        assert any(fixture.alpha_equal(candidate, matching) for candidate in candidates)
        assert not any(
            fixture.alpha_equal(candidate, conflicting) for candidate in candidates
        )
        assert isfinite(fixture.neural.score_with_pin(matching, pin, span))
        assert fixture.neural.score_with_pin(conflicting, pin, span) == float("-inf")


def combinators_compute_meanings(fixture: PendingLogicalFormFixture) -> None:
    """Require four combinator families to produce reduced meanings."""
    cases = fixture.neural.combinator_cases()
    assert cases
    assert {name for name, *_ in cases} == {
        "backward",
        "composition",
        "forward",
        "type_raising",
    }

    for name, left, right, expected in cases:
        result = fixture.neural.apply_combinator(name, left, right)
        assert fixture.alpha_equal(result, expected)
        assert fixture.alpha_equal(fixture.normalize(result), result)


_PROPERTIES = (
    pytest.param(structure_partitions_nodes, id="N1"),
    pytest.param(linearization_round_trips, id="N2"),
    pytest.param(application_is_binary, id="N3"),
    pytest.param(symbol_ids_are_stable, id="N4"),
    pytest.param(clamps_are_respected, id="N5"),
    pytest.param(combinators_compute_meanings, id="N6"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_neural_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one neural-facing property."""
    property_test(lf_fixture)
