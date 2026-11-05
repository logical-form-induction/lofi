from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import pytest
import torch
from tests.language import language

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._form_typing import FormTypingObjective
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


class Target(NamedTuple):
    term: Term
    observed_label: Literal["well-typed", "ill-typed"]


class Batch(NamedTuple):
    words: tuple[tuple[str, ...], ...]
    form_typing: tuple[tuple[Target, ...], ...]


class TrainableReadout(torch.nn.Module):
    def __init__(self, logit: float = 0.0) -> None:
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(logit))

    def score_terms(self, inputs: Encoded, terms: Sequence[Term], /) -> LogDistribution:
        if inputs.tensor.shape[0] != len(terms):
            msg = "one input is required per term"
            raise ValueError(msg)

        return LogDistribution(
            torch.nn.functional.logsigmoid(self.logit).expand(len(terms)),
            ("term",),
        )


def outputs(label: Literal["well-typed", "ill-typed"]) -> ObjectiveOutputs[Batch]:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(1, 1), SPAN_AXES)),
        batch=Batch((("rain",),), ((Target(TERM, label),),)),
        cells=Encoded(torch.ones(1, 1, 1), CELL_AXES),
    )


def test_labeled_ill_typed_forms_lose_probability_after_optimization() -> None:
    readout = TrainableReadout()
    objective = FormTypingObjective(readout, name="form-typing")
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.5)
    initial = float(readout.logit.detach().sigmoid())

    for _ in range(4):
        optimizer.zero_grad()
        objective.evaluate(outputs("ill-typed")).value.backward()
        optimizer.step()

    assert float(readout.logit.detach().sigmoid()) < initial


def test_well_typed_forms_are_trained_toward_higher_probability() -> None:
    readout = TrainableReadout()
    objective = FormTypingObjective(readout, name="form-typing")
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.5)
    initial = float(readout.logit.detach().sigmoid())

    for _ in range(4):
        optimizer.zero_grad()
        objective.evaluate(outputs("well-typed")).value.backward()
        optimizer.step()

    assert float(readout.logit.detach().sigmoid()) > initial


def test_form_typing_reports_each_observed_class_probability() -> None:
    readout = TrainableReadout()
    objective = FormTypingObjective(readout, name="typing")
    batch = Batch(
        (("rain",),),
        ((Target(TERM, "well-typed"), Target(TERM, "ill-typed")),),
    )
    published = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(1, 1), SPAN_AXES)),
        batch=batch,
        cells=Encoded(torch.ones(1, 1, 1), CELL_AXES),
    )

    term = objective.evaluate(published)

    assert term.count == 2
    assert float(term.value.detach()) == pytest.approx(
        -torch.log(torch.tensor(0.5)).item()
    )
    assert float(term.diagnostics["well_typed_probability"].detach()) == pytest.approx(
        0.5
    )
    assert float(term.diagnostics["ill_typed_probability"].detach()) == pytest.approx(
        0.5
    )


def test_a_batch_without_form_typing_is_inapplicable() -> None:
    objective = FormTypingObjective(TrainableReadout(), name="typing")
    published = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(torch.zeros(1, 1), SPAN_AXES)),
        batch=Batch((("rain",),), ((),)),
        cells=Encoded(torch.ones(1, 1, 1), CELL_AXES),
    )

    term = objective.evaluate(published)

    assert term.count == 0
    assert float(term.value) == 0.0
