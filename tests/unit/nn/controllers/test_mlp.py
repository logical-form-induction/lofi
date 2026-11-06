"""Tests for the multilayer controller."""

from typing import TYPE_CHECKING, Final

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.controller import (
    CONTROLLER_LAWS,
    ControllerConformance,
    ControllerSuite,
)

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.controllers import MlpController
from lofi.nn.sites import ComponentId, Role, Site
from lofi.spec.controllers import (
    EntmaxControllerNormalizerSpec,
    MlpControllerSpec,
)
from lofi.spec.layers import FeedForwardLayerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

WIDTH: Final = 4
ACTIONS: Final = ActionSet(
    (Action("combine", "application"), Action("identity")),
    Action("identity"),
)
COMPONENT: Final = ComponentId("controller")
WEIGHING: Final = Site(COMPONENT, Role.COMBINATOR_LOGITS)


def _cells(rows: int) -> Encoded:
    values = torch.linspace(-1.0, 1.0, rows * WIDTH).reshape(rows, WIDTH)
    return Encoded(values, ("cell", "width"))


def _controller(actions: ActionSet = ACTIONS) -> MlpController:
    return MlpController(
        actions,
        width=WIDTH,
        component=COMPONENT,
        settings=MlpControllerSpec(
            hidden=(FeedForwardLayerSpec(width=6, activation="relu"),)
        ).settings(),
        raising=None,
    )


def _loaded(controller: MlpController, digest: str) -> None:
    controller.load_state_dict({**controller.state_dict(), "_extra_state": digest})


def _conformance() -> ControllerConformance[MlpController]:
    return ControllerConformance(
        build=_controller,
        actions=ACTIONS,
        cells=_cells,
        load=_loaded,
        sites=SiteObservation(
            build=_controller,
            observe=lambda controller: {
                WEIGHING: controller.weigh(_cells(3), right=_cells(3))
            },
            read=lambda controller: (
                controller.weigh(_cells(3), right=_cells(3)).tensor,
            ),
        ),
    )


@pytest.mark.conformance
@pytest.mark.parametrize(
    "law",
    CONTROLLER_LAWS,
    ids=[law.__name__.removeprefix("test_") for law in CONTROLLER_LAWS],
)
def test_the_mlp_controller_satisfies_every_controller_law(
    law: Callable[[ControllerSuite, ControllerConformance[MlpController]], None],
) -> None:
    law(ControllerSuite(), _conformance())


def test_an_mlp_controller_normalizes_one_column_per_action() -> None:
    controller = _controller()

    weights = controller.weigh(_cells(3), right=_cells(3))

    assert weights.axes == ("cell", "action")
    assert weights.normalized_over == "action"
    assert torch.allclose(weights.tensor.logsumexp(-1), torch.zeros(3), atol=1e-5)


def test_an_entmax_controller_trains_through_sparse_action_weights() -> None:
    controller = MlpController(
        ACTIONS,
        width=WIDTH,
        component=ComponentId("controller"),
        settings=MlpControllerSpec(
            normalizer=EntmaxControllerNormalizerSpec(alpha=1.5),
            hidden=(FeedForwardLayerSpec(width=6, activation="relu"),),
        ).settings(),
        raising=None,
    )
    weights = controller.weigh(_cells(5), right=_cells(5)).tensor
    loss = -weights[:, 0][weights[:, 0].isfinite()].mean()
    loss.backward()

    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in controller.parameters()
    )
