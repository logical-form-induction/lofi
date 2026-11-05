import math
from copy import copy, deepcopy
from typing import Final, cast

import pytest
import torch

from lofi.nn.forms import Estimate, WeightedTerms
from lofi.terms import Operator, Signature, TermStore

SIGNATURE: Final = Signature(
    "forms-test", ("T",), (Operator("a", "T"), Operator("b", "T"))
)
STORE: Final = TermStore(SIGNATURE)
A: Final = STORE.op(SIGNATURE.by_name("a"))
B: Final = STORE.op(SIGNATURE.by_name("b"))


def test_weighted_terms_keep_graph_connected_weights_and_an_explicit_estimate() -> None:
    logits = torch.tensor([1.0, 2.0], requires_grad=True)
    weights = logits.log_softmax(dim=0) + math.log(0.75)
    value = WeightedTerms(
        (A, B), weights, torch.tensor(math.log(0.25)), Estimate.TRUNCATED
    )

    assert value.terms == (A, B)
    assert value.log_weights is weights
    assert value.log_residual.exp() == pytest.approx(0.25)
    assert value.estimate is Estimate.TRUNCATED
    assert len(value) == 2

    value.log_weights.exp().sum().backward()
    assert logits.grad is not None


def test_weighted_terms_are_frozen_and_validate_their_tensor_shapes() -> None:
    value = WeightedTerms(
        (A,), torch.tensor([0.0]), torch.tensor(-torch.inf), Estimate.EXACT
    )

    assert copy(value) is value
    assert deepcopy(value) is value

    with pytest.raises(AttributeError, match="immutable"):
        value.__setattr__("_terms", ())

    with pytest.raises(AttributeError, match="immutable"):
        value.__delattr__("_terms")

    assert not hasattr(value, "__dict__")

    with pytest.raises(ValueError, match="one log weight"):
        WeightedTerms((A,), torch.tensor([]), torch.tensor(0.0), Estimate.EXACT)

    with pytest.raises(ValueError, match="scalar"):
        WeightedTerms((A,), torch.tensor([0.0]), torch.tensor([0.0]), Estimate.EXACT)

    with pytest.raises(TypeError, match="floating point"):
        WeightedTerms((A,), torch.tensor([0]), torch.tensor(0.0), Estimate.EXACT)

    with pytest.raises(TypeError, match="Estimate"):
        WeightedTerms(
            (A,),
            torch.tensor([0.0]),
            torch.tensor(-torch.inf),
            cast("Estimate", "exact"),
        )


@pytest.mark.parametrize(
    "residual",
    [torch.tensor([0.0]), torch.tensor(float("nan")), torch.tensor(0.1)],
)
def test_invalid_residual_log_mass_is_refused(residual: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="residual"):
        WeightedTerms((A,), torch.tensor([-1.0]), residual, Estimate.TRUNCATED)
