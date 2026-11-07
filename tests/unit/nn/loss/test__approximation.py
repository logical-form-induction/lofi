import pytest
import torch

from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._approximation import ApproximationBound
from lofi.terms import Operator, Signature, TermStore

SIGNATURE = Signature("approximation-bound", ("T",), (Operator("a", "T"),))
STORE = TermStore(SIGNATURE)
TERM = STORE.op(SIGNATURE.by_name("a"))


def terms(residual: float, estimate: Estimate = Estimate.TRUNCATED) -> WeightedTerms:
    return WeightedTerms(
        (TERM,),
        torch.tensor([1.0 - residual], dtype=torch.float64).log(),
        torch.tensor(residual, dtype=torch.float64).log(),
        estimate,
    )


def test_a_truncating_term_set_at_or_below_the_bound_is_returned_unchanged() -> None:
    bound = ApproximationBound(0.1)
    held = terms(0.1)

    assert bound.check_terms(held, component="forms") is held


def test_a_truncating_term_set_above_the_bound_is_refused_with_its_component() -> None:
    with pytest.raises(ValueError, match=r"forms.*0\.2.*0\.1"):
        ApproximationBound(0.1).check_terms(terms(0.2), component="forms")


def test_an_exact_term_set_is_not_a_measured_truncation() -> None:
    held = terms(0.2, Estimate.EXACT)

    assert ApproximationBound(0.1).check_terms(held, component="forms") is held


def test_a_truncating_objective_residual_is_checked_against_the_same_bound() -> None:
    bound = ApproximationBound(0.1)
    residual = torch.tensor(0.05)

    assert bound.check_residual(residual, component="coherence") is residual

    with pytest.raises(ValueError, match=r"coherence.*0\.2.*0\.1"):
        bound.check_residual(torch.tensor(0.2), component="coherence")
