"""Logical-form printing properties."""

import re
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.lf._fixtures import (
    ACCEPTED,
    PendingLogicalFormFixture,
    PendingValue,
    logical_form_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["logical_form_fixture"]

pytestmark = pytest.mark.conformance


def print_parse_round_trips(fixture: PendingLogicalFormFixture) -> None:
    """Require printing and parsing to preserve every well-formed term."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        printed = fixture.print(form)
        reparsed = fixture.desugar(fixture.parse(printed))

        assert fixture.alpha_equal(reparsed, form)

    check()


def parse_print_is_idempotent(fixture: PendingLogicalFormFixture) -> None:
    """Require parse-print-parse to preserve alpha-equivalence."""
    assert ACCEPTED

    @given(text=st.sampled_from(ACCEPTED))
    @settings(max_examples=100, deadline=None)
    def check(text: str) -> None:
        first = fixture.desugar(fixture.parse(text))
        second = fixture.desugar(fixture.parse(fixture.print(first)))

        assert fixture.alpha_equal(first, second)

    check()


def printing_avoids_host_artifacts(fixture: PendingLogicalFormFixture) -> None:
    """Require printed forms to contain no host-language artifacts."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        printed = fixture.print(form)

        assert not re.search(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", printed)

    check()


def printing_is_total(fixture: PendingLogicalFormFixture) -> None:
    """Require printing to retain every valid free-variable name."""

    @given(name=st.from_regex(r"[A-Za-z][A-Za-z0-9_]*", fullmatch=True))
    @settings(max_examples=100, deadline=None)
    def check(name: str) -> None:
        form = fixture.term.free(name)
        assert fixture.print(form)

    check()


def annotations_round_trip(fixture: PendingLogicalFormFixture) -> None:
    """Require requested annotations to reach the one-store checker unchanged."""
    forms = fixture.printing.annotated_forms()
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        before = fixture.printing.annotations(form)
        assert before
        judgment = fixture.check(form)

        assert fixture.typing.is_well_typed(judgment)
        assert (
            fixture.printing.annotations(fixture.typing.checked_form(judgment))
            == before
        )

    check()


def printing_is_alpha_injective(fixture: PendingLogicalFormFixture) -> None:
    """Require identical printing exactly for alpha-equivalent terms."""
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(left=st.sampled_from(forms), right=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(left: PendingValue, right: PendingValue) -> None:
        same_printing = fixture.print(left) == fixture.print(right)

        assert same_printing is fixture.alpha_equal(left, right)

    check()


def pipeline_stages_round_trip(fixture: PendingLogicalFormFixture) -> None:
    """Require all four pipeline stages to be separate and round-trippable."""
    source = "sleep(john)"
    parsed = fixture.parse(source)
    form = fixture.desugar(parsed)
    judgment = fixture.check(form)
    assert fixture.typing.is_well_typed(judgment)
    restored = fixture.desugar(fixture.parse(fixture.print(form)))
    assert fixture.alpha_equal(restored, form)
    assert fixture.check(fixture.typing.checked_form(judgment)) == judgment
    normal = fixture.normalize(form)
    assert fixture.alpha_equal(fixture.normalize(normal), normal)
    assert fixture.alpha_equal(fixture.printing.pipeline(source), normal)


_PROPERTIES = (
    pytest.param(print_parse_round_trips, id="C1"),
    pytest.param(parse_print_is_idempotent, id="C2"),
    pytest.param(printing_avoids_host_artifacts, id="C3"),
    pytest.param(printing_is_total, id="C4"),
    pytest.param(annotations_round_trip, id="C5"),
    pytest.param(printing_is_alpha_injective, id="C6"),
    pytest.param(pipeline_stages_round_trip, id="C7"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_printing_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one printing property."""
    property_test(lf_fixture)
