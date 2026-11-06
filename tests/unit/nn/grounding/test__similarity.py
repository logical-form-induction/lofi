import math

import pytest
import torch

from lofi.nn.grounding._similarity import COSINE, RBF, similarity_kernel
from lofi.registry import RegistryError


def test_each_packaged_kernel_is_installed_under_its_name() -> None:
    assert similarity_kernel("cosine") is COSINE
    assert similarity_kernel("rbf") is RBF


def test_an_uninstalled_kernel_is_refused_by_the_family() -> None:
    with pytest.raises(RegistryError, match="similarity_kernel"):
        similarity_kernel("laplacian")


def test_the_cosine_kernel_maps_agreement_onto_the_unit_interval() -> None:
    left = torch.tensor([1.0, 0.0])

    assert float(COSINE.similarity(left, left, temperature=1.0)) == 1.0
    assert float(COSINE.similarity(left, -left, temperature=1.0)) == 0.0
    assert float(
        COSINE.similarity(left, torch.tensor([0.0, 1.0]), temperature=0.5)
    ) == pytest.approx(0.25)


def test_the_rbf_kernel_decays_with_mean_squared_distance() -> None:
    left = torch.tensor([0.0, 0.0])
    right = torch.tensor([1.0, 1.0])

    assert float(RBF.similarity(left, left, temperature=1.0)) == 1.0
    assert float(RBF.similarity(left, right, temperature=2.0)) == pytest.approx(
        math.exp(-0.5)
    )
