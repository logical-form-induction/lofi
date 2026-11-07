"""Finite join-semilattices and their logical-constant semantics."""

import pytest

from lofi.lf._plural import ATOM, PART_OF, SUM
from lofi.lf._semantics import SMT_SYNTAX, THF_SYNTAX, TPTP_SYNTAX, ProverSyntax
from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Element,
    JoinSemilattice,
    TruthAlgebra,
)


def _algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def test_plural_constants_use_the_model_resource_and_emit_declared_symbols() -> None:
    john, mary, group = tuple(Element("e", at) for at in range(3))
    lattice = JoinSemilattice("e", (john, mary, group))

    assert SUM.denote((john, mary), _algebra(), lattice) == group
    assert PART_OF.denote((john, group), _algebra(), lattice) == 1.0
    assert ATOM.denote((group,), _algebra(), lattice) == 0.0
    assert SUM.model_resource == "join-semilattice"
    assert SUM.encode("lofi_sum", ("john", "mary"), dialect=THF_SYNTAX) == (
        "(lofi_sum @ john @ mary)"
    )


@pytest.mark.parametrize("dialect", [TPTP_SYNTAX, SMT_SYNTAX])
def test_plural_constants_refuse_non_thf_emission(
    dialect: ProverSyntax,
) -> None:
    with pytest.raises(ValueError, match="THF"):
        SUM.encode("lofi_sum", ("john", "mary"), dialect=dialect)
