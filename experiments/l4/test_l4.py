"""Exercise logical-form supervision campaigns through public boundaries."""

from __future__ import annotations

import json
from collections import Counter
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from lofi.run import (
    ExperimentSeed,
    PointExecutor,
    RunStore,
    SerialLauncher,
    SweepPoint,
    expand_experiment,
    read_experiment,
    read_metrics,
    read_rows,
)
from lofi.run.operations import TRAINING_OPERATIONS
from lofi.spec import RunSpecCodec, config_search_path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
RUN = EXPERIMENTS / "l4-run.yaml"


class _Axis(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def points(self) -> tuple[JsonValue, ...]: ...


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mapping(value: JsonValue, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        msg = f"{label} is not a mapping"
        raise TypeError(msg)
    return value


def _number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{label} is not numeric"
        raise TypeError(msg)
    return float(value)


def _dataset(point: SweepPoint) -> Mapping[str, JsonValue]:
    data = _mapping(point.config["data"], "data")
    datasets = _mapping(data["datasets"], "data.datasets")
    train = _mapping(datasets["train"], "data.datasets.train")
    return _mapping(train["dataset"], "data.datasets.train.dataset")


def _proportion(point: SweepPoint) -> tuple[float, float, float]:
    proportion = _mapping(_dataset(point)["proportion"], "proportion")
    return (
        _number(proportion["entails"], "proportion.entails"),
        _number(proportion["contradicts"], "proportion.contradicts"),
        _number(proportion["neutral"], "proportion.neutral"),
    )


def _noise_profile(point: SweepPoint) -> tuple[str, float]:
    noise = _dataset(point).get("label_noise")
    if noise is None:
        return ("clean", 0.0)
    held = _mapping(noise, "label_noise")
    confusion = _mapping(held["confusion"], "label_noise.confusion")
    rates = held["rates"]
    if not isinstance(rates, list) or not rates:
        msg = "label_noise.rates is not a nonempty list"
        raise TypeError(msg)
    first = _mapping(rates[0], "label_noise.rates[0]")
    return (cast("str", confusion["kind"]), _number(first["rate"], "rate"))


def _path_points(axis: _Axis) -> tuple[SweepPoint, ...]:
    codec = RunSpecCodec(search_path=config_search_path(RUN.parent))
    points: list[SweepPoint] = []
    for value in axis.points:
        run = codec(RUN, extra={axis.key: value})
        points.append(
            SweepPoint(cast("dict[str, JsonValue]", json.loads(run.model_dump_json())))
        )
    return tuple(points)


def test_campaign_documents_cover_relation_and_confusion_designs() -> None:
    """Every declared regime and confusion treatment composes exactly once."""
    regimes = read_experiment(EXPERIMENTS / "regimes.yaml")
    noise = read_experiment(EXPERIMENTS / "label-noise.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    regime_points = _path_points(regimes.sweep.grid[0])
    mixture_points = _path_points(noise.sweep.grid[0])
    noise_points = _path_points(noise.sweep.grid[1])
    reduced_points = expand_experiment(reduced, ROOT)

    _check(
        regimes.sweep.expected_points == 20, "the relation-regime grid is incomplete"
    )
    _check(noise.sweep.expected_points == 16, "the label-noise grid is incomplete")
    _check(
        Counter(round(1.0 - _proportion(point)[0], 2) for point in regime_points)
        == Counter({0.0: 1, 0.05: 1, 0.1: 1, 0.25: 1, 0.5: 1}),
        "the non-entailment contamination ladder is incomplete",
    )
    _check(
        all(
            isclose(contradicts, neutral)
            for _, contradicts, neutral in map(_proportion, regime_points)
        ),
        "contradiction and neutrality do not split the non-entailment share",
    )
    _check(
        {_proportion(point) for point in mixture_points} == {(0.5, 0.25, 0.25)},
        "the confusion campaign changes the oracle relation mixture",
    )
    _check(
        Counter(map(_noise_profile, noise_points))
        == Counter(
            {
                ("clean", 0.0): 1,
                ("uniform", 0.1): 1,
                ("uniform", 0.2): 1,
                ("tabular", 0.2): 1,
            }
        ),
        "the confusion campaign omitted or duplicated a treatment",
    )
    _check(len(reduced_points) == 4, "the reduced freeze did not expand")
    _check(
        Counter((_proportion(point), _noise_profile(point)) for point in reduced_points)
        == Counter(
            {
                ((1.0, 0.0, 0.0), ("clean", 0.0)): 1,
                ((0.5, 0.25, 0.25), ("clean", 0.0)): 1,
                ((0.5, 0.25, 0.25), ("uniform", 0.2)): 1,
                ((0.5, 0.25, 0.25), ("tabular", 0.2)): 1,
            }
        ),
        "the reduced freeze lost a relation or confusion treatment",
    )

    for point in (*regime_points, *mixture_points, *noise_points, *reduced_points):
        model = _mapping(point.config["model"], "model")
        capacity = _mapping(model["capacity"], "model.capacity")
        _check(
            capacity["kind"] == "parameter-count" and capacity["target"] == 65536,
            "an L4 point is not capacity matched",
        )
        _check(
            model["language"] == {"from": "data"},
            "an L4 point does not learn in the source language",
        )


@pytest.mark.integration
def test_reduced_freeze_reports_recovery_per_regime_and_confusion(
    tmp_path: Path,
) -> None:
    """The production path reports recovery and achieved label confusion."""
    experiment = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    points = expand_experiment(experiment, ROOT)
    store = RunStore(
        tmp_path / "runs",
        freeze_id=None,
        seed=ExperimentSeed(experiment.seed_policy.root),
    )
    executor = PointExecutor(
        store,
        codec=RunSpecCodec(),
        operations=TRAINING_OPERATIONS,
        infrastructure_attempts=2,
        failure_policy=experiment.failure,
    )
    handles = SerialLauncher(store, execute=executor).submit(points)

    _check(len(handles) == 4, "the launcher omitted a reduced treatment")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced treatment did not complete",
    )

    achieved: list[tuple[tuple[float, float, float], bool]] = []
    for point in points:
        directory = store.directory(point)
        recorded = read_rows(directory)
        metrics = read_metrics(directory)
        if recorded is None or metrics is None or metrics.inference is None:
            msg = f"completed point {point.point_id!r} has no inference result"
            raise AssertionError(msg)
        named = {
            cast("str", row["metric"])
            for row in recorded.rows
            if row["task"] == "logical-forms"
        }
        _check(
            {
                "many-to-one-accuracy",
                "random-relabeling-floor",
                "fiber-relative-recovery",
            }
            <= named,
            "a reduced point omitted recovery or its empirical floors",
        )

        inference = metrics.inference
        proportions = inference.inference.proportions
        achieved_proportion = (
            proportions.get("entails", 0.0),
            proportions.get("contradicts", 0.0),
            proportions.get("neutral", 0.0),
        )
        _check(
            isclose(sum(achieved_proportion), 1.0),
            "the achieved decided-label proportions do not sum to one",
        )
        confusion = inference.confusion
        noisy = _noise_profile(point)[0] != "clean"
        _check(
            (confusion is not None) == noisy,
            "the achieved confusion report disagrees with the declared treatment",
        )
        if confusion is not None:
            _check(confusion.satisfied, "the achieved confusion missed its tolerance")
            _check(
                set(confusion.matrix) == {"entails", "contradicts", "neutral"},
                "the achieved confusion matrix lost an oracle relation",
            )
        achieved.append((achieved_proportion, noisy))
        _check(metrics.stage == "closed", "a reduced metrics record is not closed")
        _check(
            metrics.point.parameter_count == 65536,
            "a reduced point missed the matched-capacity target",
        )

    _check(len(achieved) == 4, "the reduced freeze lost an achieved treatment report")
