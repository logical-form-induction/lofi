"""Tests for shared controller mechanics."""

import torch

from lofi.nn.controllers._base import PairFeatures


def test_pair_features_follow_the_declared_order() -> None:
    left = torch.tensor([[2.0, 3.0]])
    right = torch.tensor([[5.0, 7.0]])
    features = PairFeatures(("right", "difference", "product", "left"), width=2)

    assert torch.equal(
        features(left, right),
        torch.tensor([[5.0, 7.0, -3.0, -4.0, 10.0, 21.0, 2.0, 3.0]]),
    )
    assert features.output_width == 8
