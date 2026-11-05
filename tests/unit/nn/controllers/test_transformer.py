"""Tests for the transformer controller."""

from typing import Literal

import pytest
import torch

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.controllers import TransformerController
from lofi.nn.sites import ComponentId
from lofi.spec.controllers import TransformerControllerSpec


def _actions() -> ActionSet:
    identity = Action("identity")
    return ActionSet((Action("combine", "rule"), identity), identity)


def test_a_transformer_controller_scores_each_pair() -> None:
    controller = TransformerController(
        _actions(),
        width=4,
        component=ComponentId("controller"),
        settings=TransformerControllerSpec(
            layers=2, heads=2, feedforward_width=9
        ).settings(),
        raising=None,
    )
    cells = Encoded(torch.randn(3, 4), ("cell", "width"))

    assert controller.weigh(cells, right=cells).tensor.shape == (3, 2)


def test_a_transformer_controller_refuses_indivisible_attention_heads() -> None:
    with pytest.raises(ValueError, match="cannot divide"):
        TransformerController(
            _actions(),
            width=5,
            component=ComponentId("controller"),
            settings=TransformerControllerSpec(heads=2).settings(),
            raising=None,
        )


def test_a_transformer_controller_reads_its_declared_norm_placement() -> None:
    cells = Encoded(torch.randn(3, 4), ("cell", "width"))

    def scored(placement: Literal["pre", "post"]) -> torch.Tensor:
        torch.manual_seed(0)
        controller = TransformerController(
            _actions(),
            width=4,
            component=ComponentId("controller"),
            settings=TransformerControllerSpec(
                heads=2, feedforward_width=9, norm_placement=placement
            ).settings(),
            raising=None,
        )
        return controller.weigh(cells, right=cells).tensor

    assert not torch.allclose(scored("pre"), scored("post"))
