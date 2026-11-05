"""Logical-form alpha-equivalence and binding properties."""

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


def alpha_is_an_equivalence(fixture: PendingLogicalFormFixture) -> None:
    """Require alpha-equivalence to be reflexive, symmetric, and transitive."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(
        first=st.sampled_from(forms),
        second=st.sampled_from(forms),
        third=st.sampled_from(forms),
    )
    @settings(max_examples=100, deadline=None)
    def check(
        first: PendingValue,
        second: PendingValue,
        third: PendingValue,
    ) -> None:
        assert fixture.alpha_equal(first, first)
        forward = fixture.alpha_equal(first, second)
        backward = fixture.alpha_equal(second, first)

        assert forward is backward

        if forward and fixture.alpha_equal(second, third):
            assert fixture.alpha_equal(first, third)

    check()


def alpha_preserves_observables(fixture: PendingLogicalFormFixture) -> None:
    """Require alpha-equivalence to preserve types and free variables."""
    forms = fixture.term.forms(fixture.store)
    pairs = fixture.binding.annotation_variants() + tuple(
        (form, form) for form in forms
    )
    assert pairs

    @given(pair=st.sampled_from(pairs))
    @settings(max_examples=100, deadline=None)
    def check(pair: tuple[PendingValue, PendingValue]) -> None:
        left, right = pair
        assert fixture.alpha_equal(left, right)
        assert fixture.term.type_of(left) == fixture.term.type_of(right)
        assert fixture.reduction.free_variables(
            left
        ) == fixture.reduction.free_variables(right)

    check()


def annotation_position_is_irrelevant(fixture: PendingLogicalFormFixture) -> None:
    """Require equal binder annotations to be alpha-equivalent across positions."""
    pairs = fixture.binding.annotation_variants()
    assert pairs

    @given(pair=st.sampled_from(pairs))
    @settings(max_examples=100, deadline=None)
    def check(pair: tuple[PendingValue, PendingValue]) -> None:
        left, right = pair
        left_types = tuple(type_ for _, type_ in fixture.printing.annotations(left))
        right_types = tuple(type_ for _, type_ in fixture.printing.annotations(right))
        assert left_types
        assert left_types == right_types
        assert fixture.alpha_equal(left, right)

    check()


def canonical_forms_decide_alpha(fixture: PendingLogicalFormFixture) -> None:
    """Require canonical forms and their hashes to decide alpha-equivalence."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(left=st.sampled_from(forms), right=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(left: PendingValue, right: PendingValue) -> None:
        canonical_left = fixture.binding.canonical(left)
        canonical_right = fixture.binding.canonical(right)

        assert fixture.alpha_equal(left, canonical_left)
        assert fixture.alpha_equal(right, canonical_right)
        assert (canonical_left == canonical_right) is fixture.alpha_equal(left, right)
        if canonical_left == canonical_right:
            assert hash(canonical_left) == hash(canonical_right)

    check()


def binding_has_unique_owners(fixture: PendingLogicalFormFixture) -> None:
    """Require each bound occurrence to have one enclosing binder owner."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        owners = fixture.binding.binding_owners(form)
        occurrences = tuple(occurrence for occurrence, _ in owners)
        assert len(occurrences) == len(set(occurrences))
        assert all(occurrence >= 0 and binder >= 0 for occurrence, binder in owners)

    check()


def binder_renaming_preserves_terms(fixture: PendingLogicalFormFixture) -> None:
    """Require named binder renaming to preserve the lowered term."""
    cases = fixture.binding.renamed_pairs()
    assert cases

    @given(case=st.sampled_from(cases))
    @settings(max_examples=100, deadline=None)
    def check(case: tuple[PendingValue, PendingValue]) -> None:
        left, right = case
        assert fixture.alpha_equal(left, right)
        assert fixture.binding.canonical(left) == fixture.binding.canonical(right)

    check()


_PROPERTIES = (
    pytest.param(alpha_is_an_equivalence, id="F1"),
    pytest.param(alpha_preserves_observables, id="F2"),
    pytest.param(annotation_position_is_irrelevant, id="F3"),
    pytest.param(canonical_forms_decide_alpha, id="F4"),
    pytest.param(binding_has_unique_owners, id="F5"),
    pytest.param(binder_renaming_preserves_terms, id="F6"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_binding_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one alpha-equivalence or binding property."""
    property_test(lf_fixture)
