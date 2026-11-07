"""Term-algebra and well-formedness properties."""

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


def terms_are_well_scoped(fixture: PendingLogicalFormFixture) -> None:
    """Require every term to be well scoped by construction."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        for index, depth in fixture.term.scope_depths(form):
            assert 0 <= index < depth

    check()

    with pytest.raises(fixture.term.scope_error):
        fixture.term.ill_scoped()


def terms_carry_derivable_types(fixture: PendingLogicalFormFixture) -> None:
    """Require one-pass typing to be total on terms."""
    cases = fixture.term.typed_forms(fixture.store)
    assert cases

    @given(case=st.sampled_from(cases))
    @settings(max_examples=100, deadline=None)
    def check(case: tuple[PendingValue, PendingValue]) -> None:
        form, expected = case
        judgment = fixture.check(form)

        assert fixture.term.type_of(form) == expected
        assert fixture.typing.is_well_typed(judgment)
        assert fixture.typing.judgment_type(judgment) == expected

    check()


def signatures_are_data(fixture: PendingLogicalFormFixture) -> None:
    """Require distinct language declarations to lower without global state."""
    first, second = fixture.term.alternate_languages()

    assert first != second
    assert first.store is not second.store
    assert first.type_of(fixture.term.term_for_language(first)) == first.read_type("t")
    assert second.type_of(fixture.term.term_for_language(second)) == second.read_type(
        "t"
    )


def binders_are_extensible(fixture: PendingLogicalFormFixture) -> None:
    """Require new binders to need no syntax or term-algebra edit."""
    extended, text = fixture.term.extended_binder()

    assert extended != fixture.language
    form = extended.read(text)
    assert extended.print(form)


def terms_are_immutable_values(fixture: PendingLogicalFormFixture) -> None:
    """Require immutable terms with structural equality and hashing."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        rebuilt = fixture.desugar(fixture.parse(fixture.print(form)))
        probe = "_conformance_probe"

        assert rebuilt == form
        assert hash(rebuilt) == hash(form)

        with pytest.raises((AttributeError, TypeError)):
            setattr(form, probe, 0)

    check()


def vacuous_binding_follows_policy(fixture: PendingLogicalFormFixture) -> None:
    """Keep vacuous abstraction in the language's one declared binder treatment."""
    form = fixture.term.vacuous()

    assert fixture.print(form)
    assert fixture.term.type_of(form)


_PROPERTIES = (
    pytest.param(terms_are_well_scoped, id="A1"),
    pytest.param(terms_carry_derivable_types, id="A2"),
    pytest.param(signatures_are_data, id="A3"),
    pytest.param(binders_are_extensible, id="A4"),
    pytest.param(terms_are_immutable_values, id="A5"),
    pytest.param(vacuous_binding_follows_policy, id="A6"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_term_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one term-algebra property."""
    property_test(lf_fixture)
