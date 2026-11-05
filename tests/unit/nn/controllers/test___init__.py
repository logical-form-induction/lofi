"""Tests for the controller implementation exports."""

from typing import Literal

import pytest
import torch

from lofi.judge import Action, ActionSet
from lofi.nn import controllers
from lofi.nn._values import Encoded
from lofi.nn.controllers import (
    BilinearController,
    MlpController,
    TransformerController,
)
from lofi.nn.sites import ComponentId
from lofi.spec.controllers import (
    BilinearControllerSpec,
    ControllerInput,
    MlpControllerSpec,
    TransformerControllerSpec,
)


def test_the_controller_package_exports_only_current_variants() -> None:
    assert controllers.__all__ == [
        "BilinearController",
        "MlpController",
        "TransformerController",
    ]


@pytest.mark.parametrize("input_source", ["inside", "span"])
@pytest.mark.parametrize("kind", ["mlp", "bilinear", "transformer"])
def test_every_controller_variant_trains_at_each_input_source(
    kind: Literal["mlp", "bilinear", "transformer"],
    input_source: ControllerInput,
) -> None:
    identity = Action("identity")
    actions = ActionSet((Action("combine", "rule"), identity), identity)
    component = ComponentId("controller")

    match kind:
        case "mlp":
            controller = MlpController(
                actions,
                width=4,
                component=component,
                settings=MlpControllerSpec(input=input_source).settings(),
                raising=None,
            )
        case "bilinear":
            controller = BilinearController(
                actions,
                width=4,
                component=component,
                settings=BilinearControllerSpec(input=input_source).settings(),
                raising=None,
            )
        case "transformer":
            controller = TransformerController(
                actions,
                width=4,
                component=component,
                settings=TransformerControllerSpec(
                    input=input_source,
                    heads=2,
                    dropout=0.0,
                ).settings(),
                raising=None,
            )

    cells = Encoded(torch.randn(5, 4), ("cell", "width"))
    weights = controller.weigh(cells, right=cells)
    (-weights.tensor[:, 0].mean()).backward()

    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0.0).any())
        for parameter in controller.parameters()
    )
