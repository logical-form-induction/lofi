import pytest
import torch

from lofi.nn.controllers._features import (
    DIFFERENCE,
    LEFT,
    PRODUCT,
    RIGHT,
    PairFeature,
    pair_feature,
)
from lofi.registry import RegistryError


def test_each_installed_feature_derives_its_pair_value() -> None:
    left = torch.tensor([2.0, 3.0])
    right = torch.tensor([5.0, 7.0])

    assert torch.equal(LEFT(left, right), left)
    assert torch.equal(RIGHT(left, right), right)
    assert torch.equal(DIFFERENCE(left, right), torch.tensor([-3.0, -4.0]))
    assert torch.equal(PRODUCT(left, right), torch.tensor([10.0, 21.0]))


@pytest.mark.parametrize(
    ("name", "feature"),
    [
        ("left", LEFT),
        ("right", RIGHT),
        ("difference", DIFFERENCE),
        ("product", PRODUCT),
    ],
)
def test_a_feature_resolves_through_its_family(name: str, feature: PairFeature) -> None:
    assert pair_feature(name) is feature


def test_an_uninstalled_feature_is_refused() -> None:
    with pytest.raises(RegistryError, match="sum"):
        pair_feature("sum")
