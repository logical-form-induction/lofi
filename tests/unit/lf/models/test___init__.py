"""Tests for the public finite-model capability surface."""

from lofi.lf import models


def test_the_public_product_factory_builds_a_truth_algebra() -> None:
    algebra = models.PRODUCT.build(
        implication=models.RESIDUUM,
        quantifier=models.EXTREME,
        equality=models.CRISP,
        margin=0.0,
    )

    assert isinstance(algebra, models.TruthAlgebra)
    assert algebra.conjunction(0.5, 0.25) == 0.125
