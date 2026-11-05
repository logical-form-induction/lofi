"""Finite choice semantics and its THF encoding."""

import pytest

from lofi.lf._constants import CHOICE, ChoiceFunction
from lofi.lf._semantics import SMT_SYNTAX, THF_SYNTAX, TPTP_SYNTAX, ProverSyntax
from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Element,
    Frame,
    FunctionValue,
    Structure,
    TruthAlgebra,
)
from lofi.spec.language import LanguageIso, registered_language


def _algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def _predicate(outputs: tuple[bool, ...], /) -> FunctionValue:
    domain = tuple(Element("e", index) for index in range(len(outputs)))
    return FunctionValue(domain, outputs, (True,) * len(outputs))


def test_choice_selects_the_structure_witness_for_a_true_predicate() -> None:
    predicate = _predicate((False, True))
    witness = predicate.domain[1]
    choice = ChoiceFunction({predicate: witness})

    assert CHOICE.denote((predicate,), _algebra(), choice) == witness


def test_choice_refuses_a_selection_that_violates_the_choice_axiom() -> None:
    predicate = _predicate((False, True))

    language = LanguageIso().forward(registered_language("neo-davidsonian"))
    frame = Frame(
        language.store,
        function_type=language.function_type,
        truth_type="t",
        carriers={"e": 2, "v": 1},
        full_function_space_cap=64,
        henkin_pool=8,
        seed=0,
    )

    with pytest.raises(ValueError, match="choice axiom"):
        Structure(
            "violating-choice",
            frame,
            {},
            choice={predicate: predicate.domain[0]},
        )


def test_choice_uses_a_total_canonical_completion_for_an_unlisted_predicate() -> None:
    predicate = _predicate((False, True, True))

    assert (
        CHOICE.denote((predicate,), _algebra(), ChoiceFunction({}))
        == (predicate.domain[1])
    )


def test_choice_emits_the_thf_choice_binder() -> None:
    assert CHOICE.model_resource == "choice-function"
    assert (
        CHOICE.encode(
            "unused",
            ("(^ [X0: lofi_sort_e]: (lofi_dog @ X0))",),
            dialect=THF_SYNTAX,
        )
        == "(@+ [X0: lofi_sort_e]: (lofi_dog @ X0))"
    )


@pytest.mark.parametrize("dialect", [TPTP_SYNTAX, SMT_SYNTAX])
def test_choice_refuses_a_dialect_without_a_choice_binder(
    dialect: ProverSyntax,
) -> None:
    with pytest.raises(ValueError, match="THF"):
        CHOICE.encode("unused", ("predicate",), dialect=dialect)
