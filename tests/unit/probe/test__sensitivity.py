"""Tests for the complete ablation-semantics sensitivity report."""

import json
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._interventions import MeanFrom, Resample, Zero
from lofi.probe._sensitivity import (
    AblationMetric,
    AblationSemantics,
    AblationSensitivityCell,
    AblationSensitivityReport,
    ablation_sensitivity,
)
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.probe._interventions import Intervention

POINT: Final = Site(ComponentId("decoder"), Role.SYMBOL_DISTRIBUTION)


def matched() -> Trace:
    trace = Trace()
    trace.record(
        Observation(
            POINT,
            torch.tensor(((0.8, 0.2), (0.3, 0.7))),
            axes=("item", "symbol"),
        )
    )
    trace.close()

    return trace


def cell(
    semantics: AblationSemantics,
    metric: AblationMetric,
    value: float,
    *,
    conclusion: bool = True,
) -> AblationSensitivityCell:
    return AblationSensitivityCell(semantics, metric, value, conclusion=conclusion)


def core(*, flipped: bool = False) -> list[AblationSensitivityCell]:
    return [
        cell(AblationSemantics.RESAMPLE, AblationMetric.LOGIT_DIFFERENCE, 0.8),
        cell(AblationSemantics.RESAMPLE, AblationMetric.PROBABILITY, 0.4),
        cell(AblationSemantics.MEAN, AblationMetric.LOGIT_DIFFERENCE, 0.6),
        cell(
            AblationSemantics.MEAN,
            AblationMetric.PROBABILITY,
            -0.1 if flipped else 0.2,
            conclusion=not flipped,
        ),
    ]


def test_the_core_grid_agrees_and_reports_its_conclusion() -> None:
    report = AblationSensitivityReport(reversed(core()))

    assert tuple(one.key for one in report.cells) == (
        (AblationSemantics.RESAMPLE, AblationMetric.LOGIT_DIFFERENCE),
        (AblationSemantics.RESAMPLE, AblationMetric.PROBABILITY),
        (AblationSemantics.MEAN, AblationMetric.LOGIT_DIFFERENCE),
        (AblationSemantics.MEAN, AblationMetric.PROBABILITY),
    )
    assert report.default_intervention is AblationSemantics.RESAMPLE
    assert report.core == report.cells
    assert report.zero == ()
    assert not report.flip
    assert report.conclusion is True
    assert report.status == "reported"


def test_a_flipped_core_grid_makes_the_conclusion_unreported() -> None:
    report = AblationSensitivityReport(core(flipped=True))

    assert report.flip
    assert report.conclusion is None
    assert report.status == "unreported"
    assert report.to_json()["conclusion"] is None


def test_a_false_conclusion_is_reported_when_all_four_cells_agree() -> None:
    report = AblationSensitivityReport(
        cell(
            one.semantics,
            one.metric,
            -abs(one.value),
            conclusion=False,
        )
        for one in core()
    )

    assert not report.flip
    assert report.conclusion is False
    assert report.status == "reported"


def test_a_report_snapshots_cells_and_both_value_types_are_immutable() -> None:
    cells = core()
    report = AblationSensitivityReport(cells)
    held = report.cells[0]
    cells.clear()

    assert len(report.cells) == 4
    assert copy(report) is report
    assert deepcopy(report) is report
    assert copy(held) is held
    assert deepcopy(held) is held

    with pytest.raises(
        AttributeError, match="ablation sensitivity report is immutable"
    ):
        report.__setattr__("_flip", True)

    with pytest.raises(AttributeError, match="ablation sensitivity cell is immutable"):
        held.__delattr__("_value")


def test_cells_and_reports_compare_hash_and_render_by_their_data() -> None:
    first = AblationSensitivityReport(core())
    same = AblationSensitivityReport(core())
    changed = AblationSensitivityReport(core(flipped=True))

    assert first == same
    assert hash(first) == hash(same)
    assert first != changed
    assert first != "report"
    assert first.cells[0] == core()[0]
    assert hash(first.cells[0]) == hash(core()[0])
    assert first.cells[0] != "cell"
    assert (
        eval(  # noqa: S307
            repr(first.cells[0]),
            {
                "AblationSensitivityCell": AblationSensitivityCell,
                "AblationSemantics": AblationSemantics,
                "AblationMetric": AblationMetric,
            },
        )
        == first.cells[0]
    )
    assert repr(first) == (
        "AblationSensitivityReport(cells=4, flip=False, status='reported')"
    )


def test_the_json_report_template_carries_the_complete_grid_and_convention() -> None:
    report = AblationSensitivityReport(core(flipped=True))
    record = report.to_json()

    assert record == {
        "default_intervention": "resample",
        "cells": [one.to_json() for one in report.cells],
        "flip": True,
        "status": "unreported",
        "conclusion": None,
    }
    assert len(cast("list[object]", record["cells"])) == 4
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    "cells",
    [
        [],
        [cell(AblationSemantics.RESAMPLE, AblationMetric.LOGIT_DIFFERENCE, 0.5)],
        core()[:3],
    ],
)
def test_a_missing_or_single_cell_record_is_refused(
    cells: list[AblationSensitivityCell],
) -> None:
    with pytest.raises(ValueError, match=r"complete core two-by-two grid.*missing"):
        AblationSensitivityReport(cells)


def test_a_repeated_cell_is_refused_by_its_coordinate() -> None:
    repeated = [*core(), core()[0]]

    with pytest.raises(ValueError, match=r"repeats.*resample.*logit-difference"):
        AblationSensitivityReport(repeated)


def test_zero_ablation_is_only_a_complete_explicit_additional_arm() -> None:
    incomplete = [
        *core(),
        cell(
            AblationSemantics.ZERO,
            AblationMetric.LOGIT_DIFFERENCE,
            -10.0,
            conclusion=False,
        ),
    ]

    with pytest.raises(ValueError, match=r"complete zero arm.*zero.*probability"):
        AblationSensitivityReport(incomplete)

    complete = AblationSensitivityReport(
        [
            *incomplete,
            cell(
                AblationSemantics.ZERO,
                AblationMetric.PROBABILITY,
                -5.0,
                conclusion=False,
            ),
        ]
    )

    assert tuple(one.semantics for one in complete.zero) == (
        AblationSemantics.ZERO,
        AblationSemantics.ZERO,
    )
    assert not complete.flip
    assert complete.conclusion is True


def test_the_builder_defaults_to_one_seeded_resample_from_the_matched_trace() -> None:
    source = matched()
    calls: list[tuple[object, AblationMetric]] = []
    values = iter((0.8, 0.4, 0.6, 0.2))

    def measure(intervention: object, metric: AblationMetric) -> float:
        calls.append((intervention, metric))

        return next(values)

    report = ablation_sensitivity(
        source,
        measure,
        lambda _metric, value: value > 0.0,
        seed=17,
    )
    resample = calls[0][0]
    mean = calls[2][0]

    assert type(resample) is Resample
    assert resample.source is source
    assert resample.seed == 17
    assert calls[1][0] is resample
    assert type(mean) is MeanFrom
    assert mean.trace is source
    assert calls[3][0] is mean
    assert tuple(metric for _intervention, metric in calls) == (
        AblationMetric.LOGIT_DIFFERENCE,
        AblationMetric.PROBABILITY,
        AblationMetric.LOGIT_DIFFERENCE,
        AblationMetric.PROBABILITY,
    )
    assert report.default_intervention is AblationSemantics.RESAMPLE
    assert report.status == "reported"
    assert source.closed
    assert torch.equal(
        source[POINT].tensor,
        torch.tensor(((0.8, 0.2), (0.3, 0.7))),
    )


def test_zero_ablation_runs_only_when_the_caller_explicitly_requests_it() -> None:
    calls: list[object] = []

    def measure(intervention: object, _metric: AblationMetric) -> float:
        calls.append(intervention)

        return -1.0 if type(intervention) is Zero else 1.0

    report = ablation_sensitivity(
        matched(),
        measure,
        lambda _metric, value: value > 0.0,
        seed=3,
        include_zero=True,
    )

    assert len(calls) == 6
    assert type(calls[4]) is Zero
    assert calls[5] is calls[4]
    assert len(report.zero) == 2
    assert not report.flip
    assert report.conclusion is True


def test_the_builder_exposes_a_real_flip_from_measured_values() -> None:
    values = iter((0.8, 0.4, 0.6, -0.2))
    report = ablation_sensitivity(
        matched(),
        lambda _intervention, _metric: next(values),
        lambda _metric, value: value > 0.0,
        seed=5,
    )

    assert report.flip
    assert report.conclusion is None
    assert report.status == "unreported"


def test_an_open_matched_trace_is_refused_before_any_measurement() -> None:
    source = Trace()
    called = False

    def measure(_intervention: object, _metric: AblationMetric) -> float:
        nonlocal called
        called = True

        return 1.0

    with pytest.raises(ValueError, match="still open"):
        ablation_sensitivity(source, measure, lambda _metric, _value: True, seed=1)

    assert not called


@pytest.mark.parametrize(
    ("measure", "conclude", "include_zero", "match"),
    [
        (None, lambda _metric, _value: True, False, "measure is not callable"),
        (lambda _arm, _metric: 1.0, None, False, "rule is not callable"),
        (lambda _arm, _metric: 1.0, lambda _metric, _value: True, 1, "not a bool"),
    ],
)
def test_builder_controls_are_typed_before_any_arm_runs(
    measure: object,
    conclude: object,
    include_zero: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        ablation_sensitivity(
            matched(),
            cast("Callable[[Intervention, AblationMetric], float]", measure),
            cast("Callable[[AblationMetric, float], bool]", conclude),
            seed=1,
            include_zero=cast("bool", include_zero),
        )


def test_nonfinite_measurements_and_nonboolean_conclusions_are_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        ablation_sensitivity(
            matched(),
            lambda _arm, _metric: float("nan"),
            lambda _metric, _value: True,
            seed=1,
        )

    with pytest.raises(TypeError, match="conclusion is not a bool"):
        ablation_sensitivity(
            matched(),
            lambda _arm, _metric: 1.0,
            lambda _metric, _value: cast("bool", 1),
            seed=1,
        )


def test_cell_fields_and_report_members_are_exactly_typed() -> None:
    with pytest.raises(TypeError, match="semantics is not one"):
        cell(cast("AblationSemantics", "resample"), AblationMetric.PROBABILITY, 0.5)

    with pytest.raises(TypeError, match="metric is not one"):
        cell(AblationSemantics.RESAMPLE, cast("AblationMetric", "probability"), 0.5)

    with pytest.raises(TypeError, match="conclusion is not a bool"):
        cell(
            AblationSemantics.RESAMPLE,
            AblationMetric.PROBABILITY,
            0.5,
            conclusion=cast("bool", 1),
        )

    with pytest.raises(TypeError, match="contains cells"):
        AblationSensitivityReport([cast("AblationSensitivityCell", "cell")])
