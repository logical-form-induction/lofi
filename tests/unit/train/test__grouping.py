import re
from typing import Final, override

import pytest
import torch
from torch import nn

from lofi.train._grouping import GroupResolution, declare, resolve_groups
from lofi.train._groups import (
    ActiveGroups,
    GroupError,
    ParameterGroup,
    ParameterGroups,
)

WIDTH: Final = 4
LAYERS: Final = 4
DEPTH: Final = 2
TUNED: Final = 2


class Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Linear(WIDTH, WIDTH)
        self.up = nn.Linear(WIDTH, WIDTH)

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return vectors + self.up(torch.tanh(self.down(vectors)))


class AdapterStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(Adapter() for _ in range(DEPTH))

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        for adapter in self.adapters:
            vectors = adapter(vectors)

        return vectors


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(WIDTH, WIDTH)
        self.adapters = AdapterStack()

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.adapters(torch.tanh(self.dense(vectors)))


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(5, WIDTH)
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))

    @property
    def parameter_groups(self) -> ParameterGroups:
        positions = range(len(self.layers))

        return ParameterGroups(
            [
                ParameterGroup("embeddings", "embeddings"),
                *(
                    ParameterGroup(
                        f"layer.{position}", "layers", range(position, position + 1)
                    )
                    for position in positions
                ),
                *(
                    ParameterGroup(
                        f"adapters.{position}", f"layers.{position}.adapters"
                    )
                    for position in positions
                ),
            ]
        )

    @override
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        vectors = self.embeddings(ids)

        for layer in self.layers:
            vectors = layer(vectors)

        return vectors


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(WIDTH, 1)

    @property
    def parameter_groups(self) -> ParameterGroups:
        return ParameterGroups([ParameterGroup("weights", "")])

    @override
    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.score(vectors)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = Head()

    @override
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(ids))


def wired(seed: int = 0) -> Model:
    torch.manual_seed(seed)

    return Model()


def declared(model: Model) -> ParameterGroups:
    return declare({"encoder": model.encoder, "head": model.head})


def configuration() -> ActiveGroups:
    return ActiveGroups(
        [
            *(f"encoder.layer.{position}" for position in range(TUNED, LAYERS)),
            *(f"encoder.adapters.{position}" for position in range(LAYERS)),
            "head.weights",
        ]
    )


def unfrozen(model: Model) -> frozenset[str]:
    tuned = tuple(f"encoder.layers.{position}." for position in range(TUNED, LAYERS))

    return frozenset(
        name
        for name, _ in model.named_parameters()
        if ".adapters." in name or name.startswith("head.") or name.startswith(tuned)
    )


def stepped(model: Model, resolution: GroupResolution) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(resolution.parameters, lr=0.5)
    model.zero_grad(set_to_none=True)
    model(torch.tensor([[0, 1, 2]])).sum().backward()
    optimizer.step()

    return optimizer


def test_declaring_over_components_names_each_group_under_its_component() -> None:
    groups = declared(wired())

    assert groups.names == {
        "encoder.embeddings",
        "head.weights",
        *(f"encoder.layer.{position}" for position in range(LAYERS)),
        *(f"encoder.adapters.{position}" for position in range(LAYERS)),
    }


def test_a_configuration_trains_the_named_layers_and_every_nested_adapter() -> None:
    model = wired()
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    trainable = unfrozen(model)

    resolution = resolve_groups(declared(model), configuration(), model)
    optimizer = stepped(model, resolution)

    held = dict(model.named_parameters())
    stepping = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    assert resolution.trainable == trainable
    assert resolution.frozen == frozenset(held) - trainable
    assert stepping == [
        id(parameter) for name, parameter in held.items() if name in trainable
    ]
    assert {name for name, held_at in held.items() if held_at.grad is None} == (
        frozenset(held) - trainable
    )
    assert {
        name for name, held_at in held.items() if not torch.equal(held_at, before[name])
    } == trainable


def test_a_group_holds_only_what_no_group_nested_inside_it_holds() -> None:
    model = wired()

    resolution = resolve_groups(declared(model), configuration(), model)

    assert resolution.members("encoder.layer.0") == {
        "encoder.layers.0.dense.weight",
        "encoder.layers.0.dense.bias",
    }
    assert resolution.members("encoder.adapters.0") == {
        f"encoder.layers.0.adapters.adapters.{position}.{module}.{field}"
        for position in range(DEPTH)
        for module in ("down", "up")
        for field in ("weight", "bias")
    }
    assert resolution.members("head.weights") == {
        "head.score.weight",
        "head.score.bias",
    }
    assert resolution.active == configuration()


def test_a_resolution_answers_for_declared_groups_only() -> None:
    model = wired()

    resolution = resolve_groups(declared(model), configuration(), model)

    with pytest.raises(GroupError, match=re.escape("no group named 'encoder.layer.9'")):
        resolution.members("encoder.layer.9")

    assert repr(resolution) == (
        f"GroupResolution(groups={len(declared(model))}, "
        f"trainable={len(resolution.parameters)}, frozen={len(resolution.frozen)})"
    )


def test_resolving_is_independent_of_the_order_the_groups_arrive_in() -> None:
    model = wired()
    groups = declared(model)
    reversed_groups = ParameterGroups(reversed(list(groups)))

    first = resolve_groups(groups, configuration(), model)
    again = resolve_groups(reversed_groups, configuration(), model)

    assert first.trainable == again.trainable
    assert first.frozen == again.frozen
    assert [id(parameter) for parameter in first.parameters] == [
        id(parameter) for parameter in again.parameters
    ]


def test_resolving_sets_gradients_from_the_declaration_alone() -> None:
    model = wired()
    trainable = unfrozen(model)

    for parameter in model.parameters():
        parameter.requires_grad = False

    resolution = resolve_groups(declared(model), configuration(), model)

    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == trainable
    assert len(resolution.parameters) == len(trainable)


def test_a_configuration_naming_an_undeclared_group_is_refused() -> None:
    model = wired()

    with pytest.raises(GroupError, match=re.escape("encoder.layer.9")):
        resolve_groups(declared(model), ActiveGroups(["encoder.layer.9"]), model)


def test_a_group_naming_a_module_the_model_does_not_register_is_refused() -> None:
    model = wired()
    groups = ParameterGroups([ParameterGroup("decoder", "decoder")])

    with pytest.raises(GroupError, match="registers no module named 'decoder'"):
        resolve_groups(groups, ActiveGroups(), model)


def test_a_group_naming_a_layer_the_module_does_not_register_is_refused() -> None:
    model = wired()
    groups = ParameterGroups(
        [ParameterGroup("beyond", "encoder.layers", range(LAYERS + 1))]
    )

    with pytest.raises(GroupError, match=f"no layer at position {LAYERS}"):
        resolve_groups(groups, ActiveGroups(), model)


def test_a_group_that_selects_no_parameter_is_refused() -> None:
    model = wired()
    groups = ParameterGroups([ParameterGroup("none", "encoder.layers", range(0))])

    with pytest.raises(GroupError, match="'none' selects nothing"):
        resolve_groups(groups, ActiveGroups(), model)


def test_a_parameter_no_group_selects_is_refused() -> None:
    model = wired()
    groups = model.encoder.parameter_groups.under("encoder")

    with pytest.raises(GroupError, match="no declared group holds"):
        resolve_groups(groups, ActiveGroups(), model)


def test_two_groups_that_overlap_without_nesting_are_refused() -> None:
    model = wired()
    groups = ParameterGroups(
        [
            ParameterGroup("lower", "encoder.layers", range(2)),
            ParameterGroup("upper", "encoder.layers", range(1, 3)),
        ]
    )

    with pytest.raises(GroupError, match="'lower' and 'upper' share"):
        resolve_groups(groups, ActiveGroups(), model)


def test_two_groups_that_select_the_same_parameters_are_refused() -> None:
    model = wired()
    groups = ParameterGroups(
        [
            ParameterGroup("head", "head"),
            ParameterGroup("score", "head.score"),
        ]
    )

    with pytest.raises(GroupError, match="select the same parameters"):
        resolve_groups(groups, ActiveGroups(), model)
