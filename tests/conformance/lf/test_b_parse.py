"""Logical-form parsing properties."""

from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.lf._fixtures import (
    ACCEPTED,
    ALPHABET_PAIRS,
    PRECEDENCE_PAIRS,
    REJECTED,
    PendingLogicalFormFixture,
    PendingValue,
    logical_form_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["logical_form_fixture"]

pytestmark = pytest.mark.conformance


def accepted_strings_parse(fixture: PendingLogicalFormFixture) -> None:
    """Require every accepted corpus string to parse."""
    assert ACCEPTED

    for text in ACCEPTED:
        fixture.parse(text)


def rejected_strings_raise_parse_errors(fixture: PendingLogicalFormFixture) -> None:
    """Require rejected corpus strings to raise only parse errors."""
    assert REJECTED

    for text in REJECTED:
        with pytest.raises(fixture.parse_error):
            fixture.parse(text)


def whitespace_preserves_trees(fixture: PendingLogicalFormFixture) -> None:
    """Require token-boundary whitespace to preserve the parsed tree."""
    tokenizations = (
        ("λ", "x", ".", "x"),
        ("∀", "x", ".", "p", "(", "x", ")"),
        ("p", "(", "x", ")", "∧", "q", "(", "x", ")"),
    )
    assert tokenizations

    @given(
        tokens=st.sampled_from(tokenizations),
        spaces=st.lists(
            st.sampled_from(("", " ", "\t", "\n", " \n ")),
            min_size=10,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def check(tokens: tuple[str, ...], spaces: list[str]) -> None:
        compact = "".join(tokens)
        spaced = spaces[0] + "".join(
            token + spaces[index + 1] for index, token in enumerate(tokens)
        )
        assert fixture.parse(compact) == fixture.parse(spaced)

    check()


def alphabets_are_synonymous(fixture: PendingLogicalFormFixture) -> None:
    """Require matching ASCII and Unicode forms to parse identically."""
    assert ALPHABET_PAIRS
    forms = fixture.term.forms(fixture.store)
    assert forms

    @given(pair=st.sampled_from(ALPHABET_PAIRS))
    @settings(max_examples=100, deadline=None)
    def check_pair(pair: tuple[str, str]) -> None:
        unicode_text, ascii_text = pair
        unicode_tree = fixture.parse(unicode_text)
        assert unicode_tree == fixture.parse(ascii_text)
        assert fixture.alpha_equal(
            fixture.desugar(
                fixture.parse(fixture.print(fixture.desugar(unicode_tree)))
            ),
            fixture.desugar(unicode_tree),
        )

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check_form(form: PendingValue) -> None:
        assert fixture.alpha_equal(
            fixture.desugar(fixture.parse(fixture.print(form))), form
        )

    check_pair()
    check_form()


def precedence_is_structural(fixture: PendingLogicalFormFixture) -> None:
    """Require the five precedence examples to produce their declared trees."""
    assert PRECEDENCE_PAIRS

    for implicit, explicit in PRECEDENCE_PAIRS:
        assert fixture.parse(implicit) == fixture.parse(explicit)


def application_is_curried(fixture: PendingLogicalFormFixture) -> None:
    """Require application to be curried and left associative."""
    atoms = fixture.parsing.application_atoms()
    assert atoms

    @given(left=st.sampled_from(atoms), right=st.sampled_from(atoms))
    @settings(max_examples=100, deadline=None)
    def check(left: str, right: str) -> None:
        assert fixture.parse(f"f({left})({right})") == fixture.parse(
            f"(f({left}))({right})"
        )

    check()


def parsing_is_total(fixture: PendingLogicalFormFixture) -> None:
    """Require parsing to be stateless across nested inputs."""

    @given(depth=st.integers(min_value=0, max_value=40))
    @settings(max_examples=100, deadline=None)
    def check(depth: int) -> None:
        source = "(" * depth + "p(x)" + ")" * depth
        before = fixture.parsing.state_snapshot()
        assert fixture.parse(source) == fixture.parse(source)
        assert fixture.parsing.state_snapshot() == before

    check()


def binder_names_are_lexically_scoped(fixture: PendingLogicalFormFixture) -> None:
    """Require a bound name to differ from the same name outside its binder."""
    bound = fixture.desugar(fixture.parse("\\x. x"))
    free = fixture.desugar(fixture.parse("x"))

    assert not fixture.alpha_equal(bound, free)
    assert fixture.desugar(fixture.parse("\\x. x")) == fixture.desugar(
        fixture.parse("\\y. y")
    )


_PROPERTIES = (
    pytest.param(accepted_strings_parse, id="B1"),
    pytest.param(rejected_strings_raise_parse_errors, id="B2"),
    pytest.param(whitespace_preserves_trees, id="B3"),
    pytest.param(alphabets_are_synonymous, id="B4"),
    pytest.param(precedence_is_structural, id="B5"),
    pytest.param(application_is_curried, id="B6"),
    pytest.param(parsing_is_total, id="B7"),
    pytest.param(binder_names_are_lexically_scoped, id="B8"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_parsing_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one parsing property."""
    property_test(lf_fixture)
