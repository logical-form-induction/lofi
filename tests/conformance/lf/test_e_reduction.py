"""Logical-form substitution and reduction properties."""

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


def reduction_is_pure(fixture: PendingLogicalFormFixture) -> None:
    """Require reduction to produce a new result without changing its inputs."""
    redexes = fixture.reduction.redexes()
    assert redexes

    @given(form=st.sampled_from(redexes))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        rendered = fixture.print(form)
        fingerprint = hash(form)
        result = fixture.reduction.reduce_once(form)

        assert result is not form
        assert fixture.print(form) == rendered
        assert hash(form) == fingerprint

    check()


def substitution_avoids_capture(fixture: PendingLogicalFormFixture) -> None:
    """Require substitution to preserve every free argument variable."""
    cases = fixture.reduction.substitution_cases()
    assert cases

    @given(case=st.sampled_from(cases))
    @settings(max_examples=100, deadline=None)
    def check(case: tuple[PendingValue, PendingValue, int]) -> None:
        body, argument, variable = case
        result = fixture.reduction.substitute(body, argument)
        expected = (
            fixture.reduction.free_variables(body) - {variable}
        ) | fixture.reduction.free_variables(argument)

        assert fixture.reduction.free_variables(result) == expected

    check()


def reduction_preserves_types(fixture: PendingLogicalFormFixture) -> None:
    """Require reduction to preserve the term type."""
    redexes = fixture.reduction.redexes()
    assert redexes

    @given(form=st.sampled_from(redexes))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        reduced = fixture.reduction.reduce_once(form)

        assert fixture.term.type_of(reduced) == fixture.term.type_of(form)

    check()


def typed_terms_normalize(fixture: PendingLogicalFormFixture) -> None:
    """Require every well-typed term to normalize within its step budget."""
    forms = tuple(form for form, _ in fixture.term.typed_forms(fixture.store))
    assert forms
    assert fixture.reduction.fuel > 0

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        bounded = fixture.reduction.normalize_with_fuel(
            form,
            fixture.reduction.fuel,
        )
        assert fixture.alpha_equal(bounded, fixture.normalize(form))

    check()


def reduction_is_confluent(fixture: PendingLogicalFormFixture) -> None:
    """Require either reduction order to reach alpha-equivalent normal forms."""
    redexes = fixture.reduction.redexes()
    assert redexes

    @given(form=st.sampled_from(redexes))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        leftmost = fixture.normalize(form)
        rightmost = fixture.reduction.normalize_rightmost(form)

        assert fixture.alpha_equal(leftmost, rightmost)

    check()


def normalization_orders_are_explicit(fixture: PendingLogicalFormFixture) -> None:
    """Require separate declared weak-head and full normalization operations."""
    redexes = fixture.reduction.redexes()
    assert redexes
    form = redexes[0]
    weak = fixture.reduction.weak_head(form)
    full = fixture.normalize(form)

    assert fixture.alpha_equal(fixture.normalize(weak), full)


def beta_examples_reduce(fixture: PendingLogicalFormFixture) -> None:
    """Require the standard beta and capture-avoidance examples to reduce."""
    examples = fixture.reduction.beta_examples()
    assert examples

    for form, expected in examples:
        assert fixture.alpha_equal(fixture.normalize(form), expected)


def eta_is_opt_in(fixture: PendingLogicalFormFixture) -> None:
    """Keep eta out of the core and make explicit contraction opt in."""
    expanded, contracted = fixture.reduction.eta_pair()

    assert not fixture.alpha_equal(expanded, contracted)
    assert fixture.alpha_equal(
        fixture.reduction.eta_contract(expanded),
        contracted,
    )
    assert not fixture.alpha_equal(fixture.normalize(expanded), contracted)


def omega_is_bounded(fixture: PendingLogicalFormFixture) -> None:
    """Construct Omega, reject its typing, and raise NonTerminating after fuel."""
    omega = fixture.reduction.omega()
    judgment = fixture.check(omega)

    assert not fixture.typing.is_well_typed(judgment)
    assert fixture.typing.failure_path(judgment) is not None

    with pytest.raises(fixture.reduction.non_terminating):
        fixture.normalize(omega)


def reduction_commutes_with_printing(fixture: PendingLogicalFormFixture) -> None:
    """Require reduction to commute with printing modulo normalization."""
    redexes = fixture.reduction.redexes()
    assert redexes

    @given(form=st.sampled_from(redexes))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        restored = fixture.desugar(fixture.parse(fixture.print(form)))

        assert fixture.alpha_equal(
            fixture.normalize(restored),
            fixture.normalize(form),
        )

    check()


_PROPERTIES = (
    pytest.param(reduction_is_pure, id="E1"),
    pytest.param(substitution_avoids_capture, id="E2"),
    pytest.param(reduction_preserves_types, id="E3"),
    pytest.param(typed_terms_normalize, id="E4"),
    pytest.param(reduction_is_confluent, id="E5"),
    pytest.param(normalization_orders_are_explicit, id="E6"),
    pytest.param(beta_examples_reduce, id="E7"),
    pytest.param(eta_is_opt_in, id="E8"),
    pytest.param(omega_is_bounded, id="E9"),
    pytest.param(reduction_commutes_with_printing, id="E10"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_reduction_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one substitution or reduction property."""
    property_test(lf_fixture)
