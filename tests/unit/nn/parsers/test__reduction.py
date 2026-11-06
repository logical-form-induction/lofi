import math
from typing import TYPE_CHECKING

import pytest
import torch

from lofi.nn.parsers._reduction import LogSumExpReduction, MaximumReduction

if TYPE_CHECKING:
    from lofi.nn.parsers._ports import LogWeightReduction

NEG_INF = float("-inf")


def test_neither_reduction_holds_mutable_state() -> None:
    for reduction in (LogSumExpReduction(), MaximumReduction()):
        assert not hasattr(reduction, "__dict__")

        with pytest.raises(AttributeError, match="is immutable"):
            reduction.__setattr__("label", "changed")


@pytest.mark.parametrize("reduction", [LogSumExpReduction(), MaximumReduction()])
def test_a_reduction_preserves_device_dtype_and_gradient(
    reduction: LogWeightReduction,
) -> None:
    weights = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    combined = reduction.reduce(
        weights, torch.ones_like(weights, dtype=torch.bool), dim=1
    )

    combined.sum().backward()

    assert combined.device == weights.device
    assert combined.dtype is weights.dtype
    assert weights.grad is not None
    assert bool(torch.isfinite(weights.grad).all())


def test_log_sum_exp_sums_the_probabilities_of_the_alternatives() -> None:
    weights = torch.tensor([[math.log(2.0), math.log(3.0)]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    combined = LogSumExpReduction().reduce(weights, mask, dim=-1)

    assert combined.item() == pytest.approx(math.log(5.0))


def test_log_sum_exp_agrees_with_torch_where_nothing_is_masked() -> None:
    weights = torch.randn(4, 7, dtype=torch.float64)
    mask = torch.ones(4, 7, dtype=torch.bool)

    combined = LogSumExpReduction().reduce(weights, mask, dim=-1)

    assert torch.allclose(combined, torch.logsumexp(weights, dim=-1))


def test_log_sum_exp_leaves_a_masked_alternative_out_of_the_sum() -> None:
    weights = torch.tensor([[0.0, 5.0]])
    mask = torch.tensor([[True, False]])

    combined = LogSumExpReduction().reduce(weights, mask, dim=-1)

    assert combined.item() == 0.0


def test_log_sum_exp_mixes_alternatives_by_their_normalized_probability() -> None:
    weights = torch.tensor([[math.log(1.0), math.log(3.0)]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    mixing = LogSumExpReduction().mixing_weights(weights, mask, dim=-1).exp()

    assert mixing.tolist()[0] == pytest.approx([0.25, 0.75])


def test_a_mask_of_lower_extent_broadcasts_over_the_items() -> None:
    weights = torch.tensor([[0.0, 5.0], [1.0, 6.0]])
    mask = torch.tensor([[True, False]])

    combined = LogSumExpReduction().reduce(weights, mask, dim=-1)

    assert combined.tolist() == [0.0, 1.0]


def test_log_sum_exp_reduces_a_fully_masked_row_without_a_nan_gradient() -> None:
    weights = torch.zeros(1, 3, requires_grad=True)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    combined = LogSumExpReduction().reduce(weights, mask, dim=-1)
    combined.sum().backward()

    assert combined.item() == NEG_INF
    assert weights.grad is not None
    assert bool(torch.isfinite(weights.grad).all())


def test_log_sum_exp_reduces_real_negative_infinities_without_a_nan_gradient() -> None:
    weights = torch.full((1, 3), NEG_INF, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)

    combined = LogSumExpReduction().reduce(weights, mask, dim=-1)
    combined.sum().backward()

    assert combined.item() == NEG_INF
    assert weights.grad is not None
    assert bool(torch.isfinite(weights.grad).all())


def test_a_row_that_reduces_to_zero_leaves_the_other_rows_untouched() -> None:
    rows = torch.tensor([[NEG_INF, NEG_INF], [0.0, 0.0]], requires_grad=True)
    mask = torch.ones(2, 2, dtype=torch.bool)

    combined = LogSumExpReduction().reduce(rows, mask, dim=-1)
    combined[1].backward()

    assert combined[0].item() == NEG_INF
    assert combined[1].item() == pytest.approx(math.log(2.0))
    assert rows.grad is not None
    assert bool(torch.isfinite(rows.grad).all())
    assert rows.grad[1].tolist() == pytest.approx([0.5, 0.5])


def test_maximum_keeps_the_best_of_the_alternatives() -> None:
    weights = torch.tensor([[1.0, 3.0, 2.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    assert MaximumReduction().reduce(weights, mask, dim=-1).item() == 3.0


def test_maximum_leaves_a_masked_alternative_out() -> None:
    weights = torch.tensor([[1.0, 3.0, 2.0]])
    mask = torch.tensor([[True, False, True]])

    assert MaximumReduction().reduce(weights, mask, dim=-1).item() == 2.0


def test_maximum_mixes_only_the_first_largest_real_alternative() -> None:
    weights = torch.tensor([[3.0, 3.0, 9.0]])
    mask = torch.tensor([[True, True, False]])

    mixing = MaximumReduction().mixing_weights(weights, mask, dim=-1).exp()

    assert mixing.tolist() == [[1.0, 0.0, 0.0]]


def test_maximum_reduces_a_fully_masked_row_without_a_nan_gradient() -> None:
    weights = torch.zeros(1, 3, requires_grad=True)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    combined = MaximumReduction().reduce(weights, mask, dim=-1)
    combined.sum().backward()

    assert combined.item() == NEG_INF
    assert weights.grad is not None
    assert bool(torch.isfinite(weights.grad).all())


def test_maximum_reduces_real_negative_infinities_without_a_nan_gradient() -> None:
    weights = torch.full((1, 3), NEG_INF, requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)

    combined = MaximumReduction().reduce(weights, mask, dim=-1)
    combined.sum().backward()

    assert combined.item() == NEG_INF
    assert weights.grad is not None
    assert bool(torch.isfinite(weights.grad).all())


@pytest.mark.parametrize("reduction", [LogSumExpReduction(), MaximumReduction()])
def test_the_reduction_removes_the_axis_it_was_given(
    reduction: LogWeightReduction,
) -> None:
    weights = torch.zeros(2, 3, 4)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)

    assert reduction.reduce(weights, mask, dim=1).shape == (2, 4)
    assert reduction.reduce(weights, mask, dim=-1).shape == (2, 3)


def test_the_two_reductions_disagree_where_the_alternatives_differ() -> None:
    weights = torch.tensor([[0.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    assert LogSumExpReduction().reduce(weights, mask, dim=-1).item() == pytest.approx(
        math.log(2)
    )
    assert MaximumReduction().reduce(weights, mask, dim=-1).item() == 0.0


def test_only_log_sum_exp_admits_expected_counts() -> None:
    assert LogSumExpReduction().probabilistic
    assert not MaximumReduction().probabilistic

    LogSumExpReduction().validate_expected_counts()

    with pytest.raises(ValueError, match="requires log-sum-exp reduction"):
        MaximumReduction().validate_expected_counts()
