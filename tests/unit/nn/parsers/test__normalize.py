import pytest
import torch

from lofi.nn.parsers._normalize import masked_log_softmax

NEG_INF = float("-inf")

#: Absolute tolerance for probabilities.
TOLERANCE = 1e-9


def test_a_row_of_real_alternatives_sums_to_one() -> None:
    weights = torch.tensor([[1.0, -2.0, 0.5], [-4.0, 0.0, 3.0]], dtype=torch.float64)
    mask = torch.ones_like(weights, dtype=torch.bool)

    share = masked_log_softmax(weights, mask).exp()

    assert share.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0], abs=TOLERANCE)
    assert share.reshape(-1).tolist() == pytest.approx(
        weights.softmax(dim=-1).reshape(-1).tolist(), abs=TOLERANCE
    )


def test_a_masked_position_weighs_negative_infinity() -> None:
    weights = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])

    share = masked_log_softmax(weights, mask)

    assert share[0, 1].item() == NEG_INF
    assert share.exp().sum().item() == pytest.approx(1.0, abs=TOLERANCE)
    assert share[0, ::2].tolist() == pytest.approx(
        weights[0, ::2].log_softmax(dim=-1).tolist()
    )


def test_a_row_whose_real_positions_are_all_zero_sums_to_zero() -> None:
    weights = torch.tensor([[NEG_INF, 4.0, NEG_INF]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])

    share = masked_log_softmax(weights, mask)

    assert share.tolist() == [[NEG_INF, NEG_INF, NEG_INF]]
    assert share.exp().sum().item() == 0.0


def test_a_row_with_no_real_position_sums_to_zero() -> None:
    weights = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)

    share = masked_log_softmax(weights, torch.zeros_like(weights, dtype=torch.bool))

    assert share.tolist() == [[NEG_INF, NEG_INF, NEG_INF]]


def test_a_row_that_reduces_to_zero_differentiates_without_a_nan() -> None:
    weights = torch.full((2, 3), NEG_INF, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])

    masked_log_softmax(weights, mask).exp().sum().backward()

    assert weights.grad is not None
    assert bool(torch.isfinite(weights.grad).all())


def test_the_mask_broadcasts_against_the_weights() -> None:
    weights = torch.tensor([[1.0, -2.0, 0.5], [-4.0, 0.0, 3.0]], dtype=torch.float64)

    share = masked_log_softmax(weights, torch.tensor([[True, True, False]]))

    assert share[:, 2].tolist() == [NEG_INF, NEG_INF]
    assert share.exp().sum(dim=-1).tolist() == pytest.approx([1.0, 1.0], abs=TOLERANCE)


def test_a_shift_of_a_whole_row_leaves_the_normalized_weights_alone() -> None:
    weights = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    mask = torch.ones_like(weights, dtype=torch.bool)

    assert masked_log_softmax(weights + 800.0, mask)[0].tolist() == pytest.approx(
        masked_log_softmax(weights, mask)[0].tolist(), abs=TOLERANCE
    )
