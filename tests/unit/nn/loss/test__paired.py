import math
from typing import Final

import pytest
import torch

from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._divergence import CrossEntropy, KullbackLeibler
from lofi.nn.loss._paired import PairedTermDivergence
from lofi.terms import Operator, Signature, Term, TermStore

SIGNATURE: Final = Signature(
    "paired-loss", ("T",), (Operator("a", "T"), Operator("b", "T"))
)
STORE: Final = TermStore(SIGNATURE)
A: Final = STORE.op(SIGNATURE.by_name("a"))
B: Final = STORE.op(SIGNATURE.by_name("b"))


def weighted(
    terms: tuple[Term, ...],
    probabilities: tuple[float, ...],
    *,
    residual: float = 0.0,
    estimate: Estimate = Estimate.EXACT,
) -> WeightedTerms:
    return WeightedTerms(
        terms,
        torch.tensor(probabilities, dtype=torch.float64).log(),
        torch.tensor(residual, dtype=torch.float64).log(),
        estimate,
    )


def test_exact_paired_cross_entropy_compares_the_complete_term_support() -> None:
    reference = weighted((A, B), (0.75, 0.25))
    candidate = weighted((A, B), (0.5, 0.5))

    result = PairedTermDivergence(CrossEntropy())(reference, candidate)

    assert result.value.item() == pytest.approx(math.log(2.0))
    assert result.residual is not None
    assert result.residual.item() == 0.0


def test_truncated_paired_cross_entropy_persists_the_union_residual() -> None:
    reference = weighted((A,), (0.7,), residual=0.3, estimate=Estimate.TRUNCATED)
    candidate = weighted((A,), (0.8,), residual=0.2, estimate=Estimate.TRUNCATED)

    result = PairedTermDivergence(CrossEntropy())(reference, candidate)

    assert torch.isfinite(result.value)
    assert result.residual is not None
    assert result.residual.item() == pytest.approx(0.44)


def test_paired_cross_entropy_pushes_reference_mass_through_one_morphism() -> None:
    reference = weighted((A, B), (0.25, 0.75))
    candidate = weighted((B,), (1.0,))

    result = PairedTermDivergence(CrossEntropy(), reference=lambda _term: B)(
        reference, candidate
    )

    assert result.value.item() == pytest.approx(0.0)
    assert result.residual is not None
    assert result.residual.item() == 0.0


def test_paired_cross_entropy_refuses_sampled_weights() -> None:
    sampled = weighted((A,), (1.0,), estimate=Estimate.SAMPLED)

    with pytest.raises(ValueError, match="sampled"):
        PairedTermDivergence(CrossEntropy())(sampled, sampled)


def test_truncated_paired_divergence_conditions_on_the_terms_both_sides_list() -> None:
    reference = weighted((A, B), (0.5, 0.2), residual=0.3, estimate=Estimate.TRUNCATED)
    candidate = weighted((A,), (0.8,), residual=0.2, estimate=Estimate.TRUNCATED)

    result = PairedTermDivergence(KullbackLeibler())(reference, candidate)

    assert result.value.item() == pytest.approx(0.0)
    assert result.residual is not None
    assert result.residual.item() == pytest.approx(1.0 - 0.5 * 0.8)


def test_truncated_paired_divergence_over_disjoint_listings_omits_all_mass() -> None:
    reference = weighted((A,), (0.7,), residual=0.3, estimate=Estimate.TRUNCATED)
    candidate = weighted((B,), (0.8,), residual=0.2, estimate=Estimate.TRUNCATED)

    result = PairedTermDivergence(KullbackLeibler())(reference, candidate)

    assert result.value.item() == 0.0
    assert result.residual is not None
    assert result.residual.item() == pytest.approx(1.0)


def test_exact_paired_divergence_keeps_mass_the_candidate_gives_no_term() -> None:
    reference = weighted((A, B), (0.5, 0.5))
    candidate = weighted((A,), (1.0,))

    result = PairedTermDivergence(KullbackLeibler())(reference, candidate)

    assert result.value.item() == math.inf
    assert result.residual is not None
    assert result.residual.item() == 0.0
