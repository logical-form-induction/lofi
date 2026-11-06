"""Tests for optimizers built from their declarations through their plugins."""

from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from torch import nn
from torch.optim import SGD, Adam, AdamW

from lofi.nn.optimizers import ADAM_PLUGIN, ADAMW_PLUGIN, SGD_PLUGIN
from lofi.registry import Registry, plugin_for
from lofi.spec.optimizers.adam import AdamOptimizerSpec
from lofi.spec.optimizers.adamw import AdamWOptimizerSpec
from lofi.spec.optimizers.sgd import SGDOptimizerSpec

if TYPE_CHECKING:
    from torch.nn import Parameter

    from lofi.spec import ConfiguredSpec

LEARNING_RATE: Final = 0.02
EPSILON: Final = 1e-6
WEIGHT_DECAY: Final = 0.04
BETAS: Final = (0.8, 0.95)
DECLARED: Final = AdamWOptimizerSpec(
    learning_rate=LEARNING_RATE,
    betas=BETAS,
    epsilon=EPSILON,
    weight_decay=WEIGHT_DECAY,
)


def held_parameters(optimizer: AdamW | Adam | SGD) -> tuple[Parameter, ...]:
    """Return parameters from the optimizer's groups in their held order."""
    return tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )


def built(declaration: AdamWOptimizerSpec, parameters: tuple[Parameter, ...]) -> AdamW:
    return ADAMW_PLUGIN.construct(declaration, parameters)


def test_installed_metadata_lists_every_optimizer_plugin() -> None:
    registry = Registry[object]("optimizer")

    assert registry.get("adamw") is ADAMW_PLUGIN
    assert registry.get("adam") is ADAM_PLUGIN
    assert registry.get("sgd") is SGD_PLUGIN
    assert registry.available() == ("adam", "adamw", "sgd")


def test_adamw_holds_the_supplied_parameters_in_order_without_changing_identity() -> (
    None
):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    selected = tuple(model.parameters())[1:]
    optimizer = built(DECLARED, selected)

    assert held_parameters(optimizer) == selected
    assert [id(parameter) for parameter in held_parameters(optimizer)] == [
        id(parameter) for parameter in selected
    ]


def test_adamw_applies_the_declared_hyperparameters() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    group = built(DECLARED, (parameter,)).param_groups[0]

    assert group["lr"] == LEARNING_RATE
    assert group["betas"] == BETAS
    assert group["eps"] == EPSILON
    assert group["weight_decay"] == WEIGHT_DECAY


def test_the_declared_defaults_reach_the_built_optimizer() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    group = built(AdamWOptimizerSpec(), (parameter,)).param_groups[0]

    assert group["lr"] == 1e-3
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.01


class TestAdamWPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        parameters = (nn.Parameter(torch.tensor([1.0, 2.0])),)
        before = parameters[0].detach().clone()

        def observe(component: object) -> dict[str, bool]:
            optimizer = cast("AdamW", component)

            return {
                "holds the supplied parameters": held_parameters(optimizer)
                == parameters,
                "steps at the declared rate": optimizer.param_groups[0]["lr"]
                == LEARNING_RATE,
            }

        declaration = cast("ConfiguredSpec[object]", DECLARED)

        return ConfiguredComponentFixture(
            declaration=declaration,
            context=parameters,
            expected_type=AdamW,
            observe=observe,
            unchanged=lambda: torch.equal(parameters[0], before),
        )


def test_the_plugin_is_resolved_from_the_declaration() -> None:
    assert plugin_for(cast("ConfiguredSpec[object]", DECLARED)) is ADAMW_PLUGIN


def test_adam_applies_the_declared_hyperparameters_over_the_supplied_order() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    selected = tuple(model.parameters())[::-1]
    optimizer = ADAM_PLUGIN.construct(
        AdamOptimizerSpec(
            learning_rate=LEARNING_RATE,
            betas=BETAS,
            epsilon=EPSILON,
            weight_decay=WEIGHT_DECAY,
        ),
        selected,
    )
    group = optimizer.param_groups[0]

    assert type(optimizer) is Adam
    assert held_parameters(optimizer) == selected
    assert (group["lr"], group["betas"], group["eps"], group["weight_decay"]) == (
        LEARNING_RATE,
        BETAS,
        EPSILON,
        WEIGHT_DECAY,
    )


def test_sgd_applies_the_declared_hyperparameters_over_the_supplied_order() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    selected = tuple(model.parameters())[1:]
    optimizer = SGD_PLUGIN.construct(
        SGDOptimizerSpec(
            learning_rate=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
            nesterov=True,
        ),
        selected,
    )
    group = optimizer.param_groups[0]

    assert type(optimizer) is SGD
    assert held_parameters(optimizer) == selected
    assert (
        group["lr"],
        group["momentum"],
        group["dampening"],
        group["weight_decay"],
        group["nesterov"],
    ) == (LEARNING_RATE, 0.9, 0.0, WEIGHT_DECAY, True)


def test_sgd_applies_a_declared_dampening_to_its_momentum() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = SGD_PLUGIN.construct(
        SGDOptimizerSpec(learning_rate=1.0, momentum=1.0, dampening=0.5),
        (parameter,),
    )

    for _ in range(2):
        optimizer.zero_grad()
        parameter.sum().backward()
        optimizer.step()

    assert optimizer.param_groups[0]["dampening"] == 0.5
    assert torch.equal(parameter.detach(), torch.tensor([-1.5]))


def test_one_sgd_step_moves_a_parameter_against_its_gradient() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = SGD_PLUGIN.construct(SGDOptimizerSpec(learning_rate=0.5), (parameter,))
    (parameter * 2.0).sum().backward()
    optimizer.step()

    assert torch.equal(parameter.detach(), torch.tensor([0.0]))


@pytest.mark.parametrize(
    ("declaration", "plugin"),
    [(AdamOptimizerSpec(), ADAM_PLUGIN), (SGDOptimizerSpec(), SGD_PLUGIN)],
)
def test_each_declaration_resolves_its_own_plugin(
    declaration: AdamOptimizerSpec | SGDOptimizerSpec, plugin: object
) -> None:
    assert plugin_for(cast("ConfiguredSpec[object]", declaration)) is plugin


class TestAdamPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        parameters = (nn.Parameter(torch.tensor([1.0, 2.0])),)
        before = parameters[0].detach().clone()

        def observe(component: object) -> dict[str, bool]:
            optimizer = cast("Adam", component)

            return {
                "holds the supplied parameters": held_parameters(optimizer)
                == parameters,
                "steps at the declared rate": optimizer.param_groups[0]["lr"]
                == LEARNING_RATE,
            }

        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                AdamOptimizerSpec(learning_rate=LEARNING_RATE),
            ),
            context=parameters,
            expected_type=Adam,
            observe=observe,
            unchanged=lambda: torch.equal(parameters[0], before),
        )


class TestSGDPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        parameters = (nn.Parameter(torch.tensor([1.0, 2.0])),)
        before = parameters[0].detach().clone()

        def observe(component: object) -> dict[str, bool]:
            optimizer = cast("SGD", component)

            return {
                "holds the supplied parameters": held_parameters(optimizer)
                == parameters,
                "steps at the declared rate": optimizer.param_groups[0]["lr"]
                == LEARNING_RATE,
            }

        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                SGDOptimizerSpec(learning_rate=LEARNING_RATE),
            ),
            context=parameters,
            expected_type=SGD,
            observe=observe,
            unchanged=lambda: torch.equal(parameters[0], before),
        )
