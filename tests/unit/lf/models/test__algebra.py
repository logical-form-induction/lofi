import pytest

from lofi.lf.models import (
    CRISP,
    EXTREME,
    FLOAT_ARITHMETIC,
    GOEDEL,
    LUKASIEWICZ,
    MARGIN,
    MATERIAL,
    PRODUCT,
    RESIDUUM,
    TNORM_QUANTIFIER,
    Arithmetic,
    FuzzyAlgebra,
    FuzzyLogicFactory,
    fuzzy_algebra,
)
from lofi.registry import RegistryError


@pytest.mark.parametrize(
    ("tnorm", "expected"),
    [(PRODUCT, 0.24), (GOEDEL, 0.4), (LUKASIEWICZ, 0.0)],
)
def test_each_registered_tnorm_interprets_conjunction(
    tnorm: FuzzyLogicFactory, expected: float
) -> None:
    algebra = tnorm.build(
        implication=RESIDUUM,
        quantifier=TNORM_QUANTIFIER,
        equality=CRISP,
        margin=0.0,
    )

    assert algebra.conjunction(0.4, 0.6) == pytest.approx(expected)


def test_quantifiers_and_truth_tables_share_the_declared_algebra() -> None:
    algebra = PRODUCT.build(
        implication=MATERIAL,
        quantifier=EXTREME,
        equality=MARGIN,
        margin=0.1,
    )

    assert algebra.universal((0.8, 0.3)) == 0.3
    assert algebra.existential((0.8, 0.3)) == 0.8
    assert algebra.connective((False, False, False, True), (0.5, 0.4)) == 0.2
    assert algebra.equal(0.2, 0.25) == 1.0


def test_an_algebra_refuses_truth_degrees_outside_the_unit_interval() -> None:
    algebra = PRODUCT.build(
        implication=RESIDUUM,
        quantifier=TNORM_QUANTIFIER,
        equality=CRISP,
        margin=0.0,
    )

    with pytest.raises(ValueError, match="unit interval"):
        algebra.negate(1.1)


class _Drastic:
    """A t-norm outside the three registered ones."""

    @property
    def name(self) -> str:
        return "drastic"

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        zero = arithmetic.constant(0.0, left)
        return arithmetic.where_at_most(
            one,
            left,
            right,
            arithmetic.where_at_most(one, right, left, zero),
        )

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        zero = arithmetic.constant(0.0, left)
        return arithmetic.where_at_most(
            left,
            zero,
            right,
            arithmetic.where_at_most(right, zero, left, one),
        )

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        one = arithmetic.constant(1.0, left)
        return arithmetic.where_at_most(left, right, one, right)


def test_an_algebra_interprets_any_declared_tnorm() -> None:
    algebra = FuzzyAlgebra(
        _Drastic(),
        implication=RESIDUUM,
        quantifier=TNORM_QUANTIFIER,
        equality=CRISP,
        margin=0.0,
    )

    assert algebra.name == "drastic"
    assert algebra.conjunction(0.4, 0.6) == 0.0
    assert algebra.conjunction(1.0, 0.6) == 0.6
    assert algebra.disjunction(0.0, 0.6) == 0.6
    assert algebra.disjunction(0.4, 0.6) == 1.0
    assert algebra.universal((1.0, 0.3)) == 0.3


def test_lifted_operations_follow_the_declared_implication() -> None:
    residuated = PRODUCT.build(
        implication=RESIDUUM, quantifier=TNORM_QUANTIFIER, equality=CRISP, margin=0.0
    )
    material = PRODUCT.build(
        implication=MATERIAL, quantifier=TNORM_QUANTIFIER, equality=CRISP, margin=0.0
    )

    assert residuated.conjoin(FLOAT_ARITHMETIC, 0.4, 0.5) == pytest.approx(0.2)
    assert residuated.disjoin(FLOAT_ARITHMETIC, 0.4, 0.5) == pytest.approx(0.7)
    assert residuated.imply(FLOAT_ARITHMETIC, 0.5, 0.25) == pytest.approx(0.5)
    assert material.imply(FLOAT_ARITHMETIC, 0.5, 0.25) == pytest.approx(0.625)


def test_the_installed_families_compose_one_declared_algebra() -> None:
    algebra = fuzzy_algebra("goedel", "material", "tnorm", "margin", 0.1)

    assert algebra.name == "goedel"
    assert algebra.conditional(0.7, 0.2) == pytest.approx(0.3)
    assert algebra.universal((0.8, 0.3)) == 0.3
    assert algebra.existential((0.8, 0.3)) == 0.8
    assert algebra.equal(0.2, 0.25) == 1.0
    assert algebra.equal(0.2, 0.5) == 0.0

    with pytest.raises(RegistryError, match="fuzzy_quantifier"):
        fuzzy_algebra("goedel", "material", "majority", "margin", 0.1)


def test_each_quantifier_aggregates_a_nonempty_sequence_in_any_arithmetic() -> None:
    values = (0.5, 0.4, 0.9)

    folded = fuzzy_algebra("product", "residuum", "tnorm", "crisp", 0.0)
    extreme = fuzzy_algebra("product", "residuum", "extreme", "crisp", 0.0)

    assert folded.every(FLOAT_ARITHMETIC, values) == pytest.approx(0.18)
    assert folded.some(FLOAT_ARITHMETIC, values) == pytest.approx(0.97)
    assert extreme.every(FLOAT_ARITHMETIC, values) == 0.4
    assert extreme.some(FLOAT_ARITHMETIC, values) == 0.9
    assert CRISP.equal(0.2, 0.25, 1.0) == 0.0
    assert MARGIN.equal("a", "a", 0.0) == 1.0
