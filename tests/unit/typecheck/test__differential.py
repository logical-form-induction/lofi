from copy import copy, deepcopy
from typing import Final

import pytest
import torch
from tests.language import language, lexicon

from lofi._types import ItemId
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language
from lofi.terms import Term, substrate_reaches
from lofi.typecheck import (
    DifferentialReport,
    Scheme,
    TypeAgreement,
    differential_report,
)

LANGUAGE: Final = language(("bo",))
STORE: Final = LANGUAGE.store
CHECKER: Final = LANGUAGE.checker
RULES: Final = LANGUAGE.rules
ENTITY: Final = LANGUAGE.read_type("e")
TRUTH: Final = LANGUAGE.read_type("t")
BO: Final = LANGUAGE.syntax.desugar(LANGUAGE.syntax.parse("bo")).term
ILL_TYPED: Final = LANGUAGE.syntax.desugar(LANGUAGE.syntax.parse("bo(bo)")).term
BO_ATOM: Final = BO.payload

if BO_ATOM is None:
    msg = "the constant fixture has no atom"
    raise ValueError(msg)

SIGNATURE: Final = lexicon(LANGUAGE, {BO_ATOM: Scheme(ENTITY)})
VERSION: Final = "typing-1"
DIGEST: Final = "0" * 64


def fixture(
    count: int = 18,
) -> tuple[
    dict[tuple[ItemId, tuple[int, int]], Term],
    dict[tuple[ItemId, tuple[int, int]], Term],
]:
    """Build a one-span chart and decoded output per item.

    Parameters
    ----------
    count : int
        Number of items.
    """
    chart = {
        (ItemId(f"sentence-{position:02d}"), (0, 1)): BO for position in range(count)
    }
    decoded = dict.fromkeys(chart, ENTITY)

    return chart, decoded


def report(
    chart: dict[tuple[ItemId, tuple[int, int]], Term],
    decoded: dict[tuple[ItemId, tuple[int, int]], Term],
    *,
    agreement: TypeAgreement = TypeAgreement.EXACT,
    digest: str = DIGEST,
) -> DifferentialReport:
    """Run the differential comparison for test inputs.

    Parameters
    ----------
    chart : dict[tuple[ItemId, tuple[int, int]], Term]
        Logical forms keyed by item and span.
    decoded : dict[tuple[ItemId, tuple[int, int]], Term]
        Decoded types under the same keys.
    digest : str
        Theory digest attached to the report.
    """
    return differential_report(
        chart,
        decoded,
        CHECKER,
        SIGNATURE,
        RULES,
        agreement=agreement,
        theory_version=VERSION,
        theory_digest=digest,
    )


def test_the_eighteen_item_fixture_agrees_at_every_span() -> None:
    chart, decoded = fixture()

    found = report(chart, decoded)

    assert found.agrees
    assert len(found.items) == 18
    assert found.disagreements == ()
    assert found.residuals == ()
    assert found.theory_version == VERSION
    assert found.theory_digest == DIGEST


def test_one_injected_error_produces_one_located_disagreement() -> None:
    chart, decoded = fixture()
    address = next(iter(decoded))
    decoded[address] = TRUTH

    found = report(chart, decoded)

    assert not found.agrees
    assert len(found.disagreements) == 1
    assert found.disagreements[0].address == address
    assert found.disagreements[0].predicted is TRUTH
    assert found.disagreements[0].inferred is ENTITY
    assert found.residuals == ()


def test_unification_accepts_a_free_decoded_type() -> None:
    chart, decoded = fixture(1)
    address = next(iter(decoded))
    decoded[address] = STORE.fvar("Ty", "A")

    exact = report(chart, decoded)
    unifies = report(chart, decoded, agreement=TypeAgreement.UNIFIES)

    assert not exact.agrees
    assert unifies.agrees
    assert unifies.agreement is TypeAgreement.UNIFIES


def test_an_ill_typed_form_retains_its_native_residual() -> None:
    address = (ItemId("ill-typed"), (2, 4))

    found = report({address: ILL_TYPED}, {address: ENTITY})

    assert not found.agrees
    assert found.disagreements == ()
    assert len(found.residuals) == 1
    assert found.residuals[0].address == address
    assert found.residuals[0].residual


def test_reports_from_different_theories_refuse_comparison() -> None:
    chart, decoded = fixture(1)
    first = report(chart, decoded)
    second = report(chart, decoded, digest="1" * 64)

    with pytest.raises(ValueError, match="different theories"):
        first.compare(second)


def test_reports_under_the_same_theory_compare_by_outcome() -> None:
    chart, decoded = fixture(2)
    same = report(chart, decoded)
    changed = dict(decoded)
    changed[next(iter(changed))] = TRUTH

    assert same.compare(report(chart, decoded))
    assert not same.compare(report(chart, changed))


def test_a_report_is_frozen_and_contains_no_tensor_or_substrate_value() -> None:
    chart, decoded = fixture(1)
    found = report(chart, decoded)

    assert copy(found) is found
    assert deepcopy(found) is found
    assert substrate_reaches(found, refused=(torch.Tensor,)) == ()

    with pytest.raises(AttributeError, match="immutable"):
        found.__setattr__("_theory_digest", "changed")


def test_a_checked_language_builds_a_substrate_free_report() -> None:
    standard = registered_language("neo-davidsonian")
    checked = LanguageIso().forward(
        standard.with_(
            constants=(
                *standard.constants,
                ConstantSpec(name="bo", type="e"),
            )
        )
    )
    formula = checked.read("bo")
    entity = checked.read_type("e")
    atom = formula.payload

    assert atom is not None
    signature = lexicon(checked, {atom: Scheme(entity)})
    address = (ItemId("checked-language"), (0, 1))
    found = differential_report(
        {address: formula},
        {address: entity},
        checked.checker,
        signature,
        checked.rules,
        agreement=TypeAgreement.EXACT,
        theory_version=checked.theory.version,
        theory_digest=checked.theory.digest,
    )

    assert found.agrees
    assert found.theory_digest == checked.theory.digest
    assert substrate_reaches(found, refused=(torch.Tensor,)) == ()


def test_the_chart_and_decoder_must_cover_the_same_spans() -> None:
    chart, decoded = fixture(2)
    decoded.pop(next(iter(decoded)))

    with pytest.raises(ValueError, match="decoded spans differ"):
        report(chart, decoded)


@pytest.mark.parametrize("span", [(-1, 1), (1, 1), (2, 1)])
def test_an_invalid_span_is_refused(span: tuple[int, int]) -> None:
    address = (ItemId("item"), span)

    with pytest.raises(ValueError, match="span"):
        report({address: BO}, {address: ENTITY})


def test_an_empty_chart_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        report({}, {})
