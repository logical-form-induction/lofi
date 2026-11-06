"""Logical-form typing and inference properties."""

from typing import TYPE_CHECKING, cast

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

    from lofi.terms import Term

__all__ = ["logical_form_fixture"]

pytestmark = pytest.mark.conformance


def inference_is_sound(fixture: PendingLogicalFormFixture) -> None:
    """Require successful inference to agree with the independent checker."""
    cases = fixture.term.typed_forms(fixture.store)
    assert cases

    @given(case=st.sampled_from(cases))
    @settings(max_examples=100, deadline=None)
    def check(case: tuple[PendingValue, PendingValue]) -> None:
        form, expected = case
        judgment = fixture.check(form)

        assert fixture.typing.is_well_typed(judgment)
        assert fixture.typing.judgment_type(judgment) == expected
        assert fixture.typing.independent_type(form) == expected

    check()


def inference_is_idempotent(fixture: PendingLogicalFormFixture) -> None:
    """Require repeated inference to equal one inference."""
    forms = tuple(form for form, _ in fixture.term.typed_forms(fixture.store))
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        first = fixture.check(form)
        second = fixture.check(fixture.typing.checked_form(first))

        assert second == first

    check()


def inference_preserves_annotations(fixture: PendingLogicalFormFixture) -> None:
    """Require inference to preserve every user annotation verbatim."""
    forms = fixture.printing.annotated_forms()
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        expected = fixture.printing.annotations(form)
        assert expected
        inferred = fixture.typing.checked_form(fixture.check(form))

        assert fixture.printing.annotations(inferred) == expected

    check()


def annotation_conflicts_are_located(fixture: PendingLogicalFormFixture) -> None:
    """Require annotation conflicts to produce located residuals."""
    judgment = fixture.check(fixture.typing.annotation_conflict())
    failures = fixture.typing.failures(judgment)

    assert not fixture.typing.is_well_typed(judgment)
    assert len(failures) == 1
    kind, left, right, path = failures[0]
    assert kind == "annotation"
    assert left != right
    assert path == fixture.typing.failure_path(judgment)


def inference_is_principal(fixture: PendingLogicalFormFixture) -> None:
    """Require inference to produce the most general type."""
    cases = fixture.typing.principal_cases()
    assert cases

    @given(case=st.sampled_from(cases))
    @settings(max_examples=100, deadline=None)
    def check(case: tuple[PendingValue, PendingValue, PendingValue]) -> None:
        form, general, specific = case
        general_term = cast("Term", general)
        specific_term = cast("Term", specific)
        inferred = fixture.typing.judgment_type(fixture.check(form))

        assert inferred is not None
        assert fixture.typing.subsumes(inferred, general_term)
        assert fixture.typing.subsumes(general_term, inferred)
        assert fixture.typing.subsumes(general_term, specific_term)
        assert not fixture.typing.subsumes(specific_term, general_term)

    check()


def self_application_fails_finitely(fixture: PendingLogicalFormFixture) -> None:
    """Require self-application to produce a located residual."""
    judgment = fixture.check(fixture.typing.self_application())
    failures = fixture.typing.failures(judgment)

    assert not fixture.typing.is_well_typed(judgment)
    assert len(failures) == 1
    kind, left, right, path = failures[0]
    assert kind == "infinite_type"
    assert left != right
    assert path == fixture.typing.failure_path(judgment)


def inference_is_deterministic(fixture: PendingLogicalFormFixture) -> None:
    """Require inference to hide fresh-variable numbering."""
    forms = tuple(form for form, _ in fixture.term.typed_forms(fixture.store))
    assert forms

    @given(form=st.sampled_from(forms))
    @settings(max_examples=100, deadline=None)
    def check(form: PendingValue) -> None:
        first = fixture.check(form)
        second = fixture.check(form)

        assert fixture.typing.renumber(first) == fixture.typing.renumber(second)

    check()


def unification_has_mgu_laws(fixture: PendingLogicalFormFixture) -> None:
    """Require symmetric and idempotent most-general unification."""
    cases = fixture.typing.unification_cases()
    assert cases

    @given(case=st.sampled_from(cases))
    @settings(max_examples=100, deadline=None)
    def check(
        case: tuple[PendingValue, PendingValue, PendingValue, PendingValue],
    ) -> None:
        left, right, expected, candidate = case
        left_term = cast("Term", left)
        right_term = cast("Term", right)
        expected_term = cast("Term", expected)
        forward = fixture.typing.unify(left_term, right_term)
        backward = fixture.typing.unify(right_term, left_term)

        assert fixture.typing.apply_substitution(left_term, forward) == expected_term
        assert fixture.typing.apply_substitution(right_term, forward) == expected_term
        assert fixture.typing.apply_substitution(left_term, backward) == expected_term
        assert fixture.typing.apply_substitution(right_term, backward) == expected_term
        candidate_left = fixture.typing.apply_substitution(left_term, candidate)
        candidate_right = fixture.typing.apply_substitution(right_term, candidate)
        assert candidate_left == candidate_right
        assert fixture.typing.is_substitution_instance(forward, candidate)
        fixed = fixture.typing.unify(expected_term, expected_term)
        assert fixture.typing.apply_substitution(expected_term, fixed) == expected_term

    check()


def substitution_has_algebraic_laws(fixture: PendingLogicalFormFixture) -> None:
    """Require structural, transitive, ground-safe, idempotent substitution."""
    cases = fixture.typing.substitution_cases()
    assert cases

    ground = fixture.typing.ground_types()
    assert ground

    @given(case=st.sampled_from(cases), ground_type=st.sampled_from(ground))
    @settings(max_examples=100, deadline=None)
    def check(
        case: tuple[PendingValue, PendingValue, PendingValue, PendingValue],
        ground_type: PendingValue,
    ) -> None:
        source, first, second, expected = case
        source_term = cast("Term", source)
        expected_term = cast("Term", expected)
        ground_term = cast("Term", ground_type)
        composed = fixture.typing.compose_substitutions(first, second)
        stepwise = fixture.typing.apply_substitution(
            fixture.typing.apply_substitution(source_term, first),
            second,
        )

        assert stepwise == expected_term
        assert fixture.typing.apply_substitution(source_term, composed) == expected_term
        assert (
            fixture.typing.apply_substitution(expected_term, composed) == expected_term
        )
        assert fixture.typing.apply_substitution(ground_term, composed) == ground_term
        assert (
            fixture.typing.apply_substitution(
                source_term,
                fixture.typing.identity_substitution(),
            )
            == source
        )

    check()


def parser_and_typer_agree(fixture: PendingLogicalFormFixture) -> None:
    """Require every parsed corpus form to produce a total judgment."""
    assert ACCEPTED

    for source in ACCEPTED:
        form = fixture.desugar(fixture.parse(source))
        judgment = fixture.check(form)
        failures = fixture.typing.failures(judgment)
        paths = tuple(path for _, _, _, path in failures)

        assert paths == fixture.typing.failure_paths(judgment)
        assert all(
            kind in ("unification", "annotation", "infinite_type") and left != right
            for kind, left, right, _ in failures
        )

        if paths:
            assert not fixture.typing.is_well_typed(judgment)
        else:
            assert fixture.typing.is_well_typed(judgment)

        repeated = fixture.check(form)
        assert fixture.typing.failures(repeated) == failures


def predicate_arity_is_declared(fixture: PendingLogicalFormFixture) -> None:
    """Require predicate arity to come only from the signature."""
    judgment = fixture.check(fixture.typing.arity_form())
    assert fixture.typing.is_well_typed(judgment)


def quantifier_types_are_inferred(fixture: PendingLogicalFormFixture) -> None:
    """Require inference to determine quantifier binder types."""
    cases = fixture.typing.quantifier_forms()
    assert len(cases) >= 2
    assert len({expected for _, expected in cases}) >= 2

    for form, expected in cases:
        judgment = fixture.check(form)
        assert fixture.typing.is_well_typed(judgment)
        assert fixture.typing.judgment_type(judgment) == expected


def concrete_typing_rules_hold(fixture: PendingLogicalFormFixture) -> None:
    """Require worked examples to obey the declared typing rules."""
    forms = fixture.typing.concrete_rule_forms()
    assert forms

    for form in forms:
        assert fixture.typing.is_well_typed(fixture.check(form))


def negation_has_truth_type(fixture: PendingLogicalFormFixture) -> None:
    """Require proposition negation and a negated conjunct to have type truth."""
    form = fixture.typing.negated_conjunct()
    judgment = fixture.check(form)

    assert fixture.typing.is_well_typed(judgment)
    truth = fixture.term.truth_type()
    assert fixture.typing.judgment_type(judgment) == truth
    assert fixture.term.type_of(form) == truth


def typing_failures_and_boundaries_are_distinct(
    fixture: PendingLogicalFormFixture,
) -> None:
    """Require residual mismatches and exceptional boundaries to stay distinct."""
    for form, expected_kind in (
        (fixture.typing.unification_conflict(), "unification"),
        (fixture.typing.annotation_conflict(), "annotation"),
        (fixture.typing.self_application(), "infinite_type"),
    ):
        judgment = fixture.check(form)
        failures = fixture.typing.failures(judgment)
        assert not fixture.typing.is_well_typed(judgment)
        assert failures
        assert all(kind == expected_kind for kind, _, _, _ in failures)

    left, right = fixture.typing.infinite_unification_case()

    with pytest.raises(fixture.typing.infinite_type_error) as caught_infinite:
        fixture.typing.unify(left, right)

    assert fixture.typing.infinite_error_fields(caught_infinite.value) == (left, right)

    unknown, symbol, term, path = fixture.typing.unknown_symbol_case()

    with pytest.raises(fixture.typing.unknown_symbol_error) as caught_unknown:
        unknown()

    assert fixture.typing.unknown_symbol_fields(caught_unknown.value) == (
        symbol,
        term,
        path,
    )

    cases = fixture.typing.boundary_error_cases()
    assert cases

    for operation, error_type in cases:
        with pytest.raises(error_type):
            operation()


_PROPERTIES = (
    pytest.param(inference_is_sound, id="D1"),
    pytest.param(inference_is_idempotent, id="D2"),
    pytest.param(inference_preserves_annotations, id="D3"),
    pytest.param(annotation_conflicts_are_located, id="D4"),
    pytest.param(inference_is_principal, id="D5"),
    pytest.param(self_application_fails_finitely, id="D6"),
    pytest.param(inference_is_deterministic, id="D7"),
    pytest.param(unification_has_mgu_laws, id="D8"),
    pytest.param(substitution_has_algebraic_laws, id="D9"),
    pytest.param(parser_and_typer_agree, id="D10"),
    pytest.param(predicate_arity_is_declared, id="D11"),
    pytest.param(quantifier_types_are_inferred, id="D12"),
    pytest.param(concrete_typing_rules_hold, id="D13"),
    pytest.param(negation_has_truth_type, id="D14"),
    pytest.param(typing_failures_and_boundaries_are_distinct, id="D15"),
)


@pytest.mark.parametrize(
    "property_test",
    _PROPERTIES,
)
def test_typing_property(
    lf_fixture: PendingLogicalFormFixture,
    property_test: Callable[[PendingLogicalFormFixture], None],
) -> None:
    """Check one typing property."""
    property_test(lf_fixture)
