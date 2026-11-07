from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from tests.language import language

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._form_supervision import FormSupervisionObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._readout import CELL_AXES
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.terms import Term

TERM: Final = language(("rain",)).constant("rain")
SPANS: Final = SpanIndex(1)


class Batch(NamedTuple):
    words: tuple[tuple[str, ...], ...]
    gold_forms: tuple[Term | None, ...]


class TrainableReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def score_terms(self, inputs: Encoded, terms: Sequence[Term], /) -> LogDistribution:
        if inputs.tensor.shape[0] != len(terms):
            msg = "one input is required per term"
            raise ValueError(msg)

        return LogDistribution(
            torch.nn.functional.logsigmoid(self.logit).expand(len(terms)),
            ("term",),
        )


def outputs(*forms: Term | None) -> ObjectiveOutputs[Batch]:
    count = len(forms)
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(count, 1), SPAN_AXES)),
        batch=Batch((("rain",),) * count, forms),
        cells=Encoded(torch.ones(count, 1, 1), CELL_AXES),
    )


def test_form_supervision_trains_the_gold_form_toward_higher_probability() -> None:
    readout = TrainableReadout()
    objective = FormSupervisionObjective(readout, name="form-supervision")
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.5)
    initial = float(readout.logit.detach().sigmoid())

    for _ in range(4):
        optimizer.zero_grad()
        objective.evaluate(outputs(TERM)).value.backward()
        optimizer.step()

    measured = objective.evaluate(outputs(TERM, None))
    assert float(readout.logit.detach().sigmoid()) > initial
    assert measured.count == 1
    assert float(
        measured.diagnostics["gold_form_probability"].detach()
    ) == pytest.approx(float(readout.logit.detach().sigmoid()))


def test_form_supervision_is_inapplicable_without_gold_forms() -> None:
    term = FormSupervisionObjective(
        TrainableReadout(), name="form-supervision"
    ).evaluate(outputs(None))

    assert term.count == 0
    assert float(term.value) == 0.0
