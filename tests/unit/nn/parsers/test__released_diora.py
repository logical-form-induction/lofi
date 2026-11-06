from __future__ import annotations

import pytest
import torch

from lofi.nn.parsers._released_diora import ReleasedDioraMLP, unit_vectors


def test_a_released_chart_refuses_an_incomplete_tensor_state() -> None:
    with pytest.raises(ValueError, match=r"module\.embed\.mat"):
        ReleasedDioraMLP({})


def test_normalization_keeps_a_zero_vector_at_zero_and_scales_the_rest() -> None:
    vectors = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1e-12, 0.0]])

    normalized = unit_vectors(vectors)

    assert torch.equal(normalized[0], torch.zeros(2))
    torch.testing.assert_close(normalized[1], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(normalized[2], torch.tensor([1.0, 0.0]))
