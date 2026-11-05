import pytest

from lofi.lf._constants import ChoiceFunction
from lofi.lf._semantics import MOST, SMT_SYNTAX, THF_SYNTAX, TPTP_SYNTAX
from lofi.lf.models import (
    CRISP,
    EXTREME,
    MATERIAL,
    PRODUCT,
    Element,
    FunctionValue,
    TruthAlgebra,
)


def _algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=MATERIAL,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def test_most_denotes_strict_majority_over_one_finite_domain() -> None:
    domain = tuple(Element("e", at) for at in range(3))
    known = (True,) * len(domain)
    restrictor = FunctionValue(domain, (True, True, True), known)
    majority = FunctionValue(domain, (True, True, False), known)
    minority = FunctionValue(domain, (True, False, False), known)

    model = ChoiceFunction({})

    assert MOST.denote((restrictor, majority), _algebra(), model) == 1.0
    assert MOST.denote((restrictor, minority), _algebra(), model) == 0.0


def test_most_emits_one_declared_symbol_application() -> None:
    assert (
        MOST.encode("lofi_most", ("(^ [X: e]: p)", "(^ [X: e]: q)"), dialect=THF_SYNTAX)
        == "(lofi_most @ (^ [X: e]: p) @ (^ [X: e]: q))"
    )
    assert MOST.rules == ("strict-majority",)

    with pytest.raises(ValueError, match="restrictor and scope"):
        MOST.encode("lofi_most", ("p",), dialect=THF_SYNTAX)


def test_most_is_written_in_each_dialect_s_application_syntax() -> None:
    assert MOST.encode("most", ("p", "q"), dialect=SMT_SYNTAX) == "(most p q)"
    assert MOST.encode("most", ("p", "q"), dialect=TPTP_SYNTAX) == "most(p, q)"
    assert THF_SYNTAX.higher_order
    assert not SMT_SYNTAX.higher_order
    assert not TPTP_SYNTAX.higher_order


def test_most_refuses_unknown_finite_facts() -> None:
    domain = tuple(Element("e", at) for at in range(2))
    unknown = FunctionValue(domain, (True, False), (True, False))

    with pytest.raises(ValueError, match="known restrictor and scope"):
        MOST.denote((unknown, unknown), _algebra(), ChoiceFunction({}))
