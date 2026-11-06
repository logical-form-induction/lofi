"""Tests for the bilinear controller."""

import torch

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.controllers import BilinearController
from lofi.nn.sites import ComponentId
from lofi.spec.controllers import BilinearControllerSpec


def test_a_bilinear_controller_sends_gradients_to_the_interaction() -> None:
    identity = Action("identity")
    actions = ActionSet((Action("combine", "rule"), identity), identity)
    controller = BilinearController(
        actions,
        width=3,
        component=ComponentId("controller"),
        settings=BilinearControllerSpec().settings(),
        raising=None,
    )
    cells = Encoded(torch.randn(4, 3), ("cell", "width"))

    loss = -controller.weigh(cells, right=cells).tensor[:, 0].mean()
    loss.backward()

    bilinear = dict(controller.named_parameters())["_head._bilinear.weight"]
    assert bilinear.grad is not None
