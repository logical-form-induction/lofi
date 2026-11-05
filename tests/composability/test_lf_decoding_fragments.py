"""Compose the packaged logical-form decoding strategies."""

import math
from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import composed, train

from lofi.run import packaged_options, read_metrics

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("strategy", ["sample", "beam", "exhaustive"])
def test_each_logical_form_decoding_strategy_composes(strategy: str) -> None:
    """Each named strategy replaces the complete decoding declaration."""
    run = composed(
        "inference-point.yaml",
        f"model/readouts/forms/decoding={strategy}",
    )
    forms = run.model.readouts["forms"].model_dump()
    decoding = forms["decoding"]

    if not isinstance(decoding, dict):
        msg = f"the {strategy!r} strategy did not compose as a mapping"
        raise TypeError(msg)

    assert decoding["strategy"] == strategy


def test_the_logical_form_decoding_group_has_one_current_strategy_set() -> None:
    """The package lists the complete current strategy library."""
    assert packaged_options("model.readouts.forms.decoding") == (
        "beam",
        "exhaustive",
        "sample",
    )


@pytest.mark.parametrize("strategy", ["sample", "beam", "exhaustive"])
def test_each_logical_form_decoding_strategy_takes_a_finite_step(
    tmp_path: Path,
    strategy: str,
) -> None:
    """Each packaged strategy reaches optimization through a run document."""
    point = train(
        "lf-category-metrics.yaml",
        tmp_path / strategy,
        f"model/readouts/forms/decoding={strategy}",
        "model.readouts.category-terms.decoding.residual_mass=0.999999",
        "approximation.residual_mass=0.999999",
    )
    record = read_metrics(point.directory)

    assert record is not None
    assert record.fitting is not None
    assert record.fitting.optimizer_steps > 0
    steps = tuple(step for epoch in record.fitting.epochs for step in epoch.steps)
    assert steps
    assert all(
        math.isfinite(value) for step in steps for value in step.metrics.values()
    )
