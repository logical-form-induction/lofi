"""Derivation-entropy objective tests."""

from typing import TYPE_CHECKING

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._entropy import DerivationEntropyObjective
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)

if TYPE_CHECKING:
    from lofi.spec.losses.derivation_entropy import DerivationEntropyDirection


def outputs(values: torch.Tensor) -> ObjectiveOutputs[object]:
    index = SpanIndex(2)
    inside = LogDistribution(
        torch.zeros(int(values.shape[0]), index.num_spans),
        ("item", "span"),
    )
    return ObjectiveOutputs(
        spans=index,
        chart=CellChart(inside),
        batch=object(),
        entropy=Encoded(values, ("item",)),
    )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("minimize", 2.0), ("maximize", -2.0)],
)
def test_the_declared_direction_signs_mean_entropy(
    direction: DerivationEntropyDirection, expected: float
) -> None:
    objective = DerivationEntropyObjective(direction, name="tree-entropy")
    term = objective.evaluate(outputs(torch.tensor([1.0, 3.0])))

    assert float(term.value) == expected
    assert term.count == 2
    assert float(term.diagnostics["entropy"]) == 2.0


def test_entropy_keeps_its_gradient() -> None:
    values = torch.tensor([1.0, 3.0], requires_grad=True)
    term = DerivationEntropyObjective("minimize", name="entropy").evaluate(
        outputs(values)
    )

    term.value.backward()

    assert values.grad is not None
    assert torch.equal(values.grad, torch.full((2,), 0.5))
