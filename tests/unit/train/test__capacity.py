"""Tests for exact model-parameter capacity padding."""

import pytest
import torch
from torch import nn

from lofi.run._modules import ParameterCount, parameter_count
from lofi.train._capacity import CAPACITY_ROLE, with_parameter_capacity


def test_capacity_padding_reaches_the_target_without_adding_trainable_weights() -> None:
    encoder = nn.Linear(2, 3)
    modules = with_parameter_capacity({"encoder": encoder}, 14)

    assert tuple(modules) == ("encoder", CAPACITY_ROLE)
    assert parameter_count(
        parameter for module in modules.values() for parameter in module.parameters()
    ) == ParameterCount(total=14, trainable=9)
    padding = next(iter(modules[CAPACITY_ROLE].parameters()))
    assert padding.numel() == 5
    assert not padding.requires_grad


def test_capacity_padding_refuses_a_target_below_the_built_model() -> None:
    with pytest.raises(
        ValueError,
        match="target parameter count 8 is below the built model's count 9",
    ):
        with_parameter_capacity({"encoder": nn.Linear(2, 3)}, 8)


def test_capacity_padding_is_checkpointed_and_restored_exactly() -> None:
    first = with_parameter_capacity({"encoder": nn.Linear(2, 3)}, 14)
    padding = next(iter(first[CAPACITY_ROLE].parameters()))

    with torch.no_grad():
        padding.fill_(7.0)

    state = first[CAPACITY_ROLE].state_dict()
    second = with_parameter_capacity({"encoder": nn.Linear(2, 3)}, 14)
    second[CAPACITY_ROLE].load_state_dict(state)

    restored = next(iter(second[CAPACITY_ROLE].parameters()))
    assert torch.equal(restored, torch.full_like(restored, 7.0))
