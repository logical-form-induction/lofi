"""Declared decoded and proof-term logical-form induction train as one axis."""

import math
from typing import TYPE_CHECKING, cast

import pytest
from didactic.settings import load_document
from tests.acceptance.driver import DOCUMENTS, leaf, selection, train

from lofi.run import GridSweepExpander, Sweep, composer, read_metrics

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def test_a_meaning_induction_sweep_expands_both_current_routes() -> None:
    document = cast(
        "dict[str, JsonValue]",
        load_document(DOCUMENTS / "proof-term-meanings-sweep.yaml"),
    )
    sweep = Sweep.model_validate(document)
    points = GridSweepExpander(composer(selection("proof-term-meanings.yaml"))).expand(
        sweep
    )

    assert len(points) == 2
    assert {
        leaf(point.config, "model.readouts.forms.meanings.kind") for point in points
    } == {"decoded", "proof-term"}
    (proof_term,) = (
        point
        for point in points
        if leaf(point.config, "model.readouts.forms.meanings.kind") == "proof-term"
    )
    assert leaf(
        proof_term.config, "model.readouts.forms.meanings.residual_mass"
    ) != leaf(proof_term.config, "approximation.residual_mass")


@pytest.mark.parametrize(
    ("route", "overrides"),
    [
        ("decoded", ('model.readouts.forms.meanings={"kind":"decoded"}',)),
        ("proof-term", ()),
    ],
)
def test_each_meaning_induction_route_trains_a_tiny_point(
    route: str,
    overrides: tuple[str, ...],
    tmp_path: Path,
) -> None:
    point = train("proof-term-meanings.yaml", tmp_path / route, *overrides)
    recorded = read_metrics(point.directory)

    assert leaf(point.resolved, "model.readouts.forms.meanings.kind") == route
    assert recorded is not None
    assert recorded.fitting is not None
    steps = tuple(step for epoch in recorded.fitting.epochs for step in epoch.steps)
    losses = tuple(
        value
        for step in steps
        if (value := step.metrics.get("loss/form-supervision")) is not None
    )
    assert losses
    assert all(math.isfinite(value) for value in losses)
