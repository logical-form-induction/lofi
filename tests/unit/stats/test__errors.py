import pytest

from lofi.stats import Dependence
from lofi.stats._errors import DependenceError


def test_the_refusal_names_the_procedure_the_regime_and_what_is_covered() -> None:
    error = DependenceError(
        correction="fdr_bh",
        dependence=Dependence.ARBITRARY,
        valid_under=frozenset({Dependence.INDEPENDENT, Dependence.PRDS}),
    )

    assert error.correction == "fdr_bh"
    assert error.dependence is Dependence.ARBITRARY
    assert error.valid_under == {Dependence.INDEPENDENT, Dependence.PRDS}
    assert (
        str(error)
        == "fdr_bh is valid under INDEPENDENT, PRDS and was applied under ARBITRARY"
    )


def test_a_procedure_covering_nothing_says_so() -> None:
    error = DependenceError(
        correction="nothing",
        dependence=Dependence.INDEPENDENT,
        valid_under=frozenset(),
    )

    assert "valid under nothing" in str(error)


def test_the_refusal_is_not_caught_by_a_handler_for_argument_errors() -> None:
    error = DependenceError(
        correction="fdr_bh",
        dependence=Dependence.ARBITRARY,
        valid_under=frozenset({Dependence.PRDS}),
    )

    assert not isinstance(error, ValueError | TypeError)

    with pytest.raises(DependenceError):
        raise error
