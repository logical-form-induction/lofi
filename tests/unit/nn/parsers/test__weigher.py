"""Tests for the two declared split weighers."""

from typing import override

import pytest
import torch

from lofi.nn.parsers._reduction import LogSumExpReduction
from lofi.nn.parsers._weigher import (
    AttendSplitWeigher,
    MarginalSplitWeigher,
    build_split_weigher,
)
from lofi.nn.parsers.functions import SpanFunctionModule
from lofi.spec.parsers.split import (
    AttendSplitWeigherSettings,
    MarginalSplitWeigherSettings,
    SplitWeigherDeclaration,
)
from lofi.spec.span_functions import (
    MlpSpanFunctionSettings,
    SpanFunctionDeclaration,
)


class DotScore(SpanFunctionModule):
    """Score a pair by its inner product for exact attention tests."""

    def __init__(self, width: int) -> None:
        super().__init__((width, width), 1)

    @override
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Return one inner-product logit per pair."""
        self.joined(inputs)
        return (inputs[0] * inputs[1]).sum(dim=-1, keepdim=True)


def test_marginal_weighs_a_split_by_children_inside_and_local_score() -> None:
    joined = torch.tensor([[[1.0, 3.0]]])
    local = torch.tensor([[[2.0, 0.0]]])
    real = torch.ones_like(joined, dtype=torch.bool)

    found = MarginalSplitWeigher().weights(
        joined, local, None, None, real, LogSumExpReduction(), dim=-1
    )

    assert found.exp().flatten().tolist() == pytest.approx([0.11920292, 0.88079708])


def test_attend_weighs_a_split_only_by_the_learned_child_pair_score() -> None:
    joined = torch.tensor([[[1.0, 3.0]]])
    local = torch.tensor([[[2.0, 0.0]]])
    left = torch.tensor([[[[1.0], [1.0]]]])
    right = torch.tensor([[[[2.0], [0.0]]]])
    real = torch.ones_like(joined, dtype=torch.bool)

    found = AttendSplitWeigher(DotScore(1)).weights(
        joined, local, left, right, real, LogSumExpReduction(), dim=-1
    )

    assert found.exp().flatten().tolist() == pytest.approx([0.88079708, 0.11920292])


def test_each_declaration_builds_through_its_installed_plugin() -> None:
    assert isinstance(
        build_split_weigher(
            SplitWeigherDeclaration("marginal", MarginalSplitWeigherSettings()),
            3,
        ),
        MarginalSplitWeigher,
    )
    assert isinstance(
        build_split_weigher(
            SplitWeigherDeclaration(
                "attend",
                AttendSplitWeigherSettings(
                    SpanFunctionDeclaration("mlp", MlpSpanFunctionSettings(1, "tanh"))
                ),
            ),
            3,
        ),
        AttendSplitWeigher,
    )
