"""Exercise logical-form distribution campaigns through public boundaries."""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from math import isclose, isfinite
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


def _structure(point: SweepPoint) -> Mapping[str, JsonValue]:
    source = _mapping(_dataset(point)["source"], "source")
    interpretation = _mapping(source["interpretation"], "source.interpretation")
    meanings = _mapping(interpretation["meanings"], "meanings")
    meaning_source = _mapping(meanings["source"], "meanings.source")
    return _mapping(meaning_source["structure"], "structure")


def _length_kind(point: SweepPoint, side: str) -> str:
    length = _mapping(_dataset(point)[side], side)
    return cast("str", length["kind"])


def _semantic_profile(point: SweepPoint) -> tuple[int, str, float, str]:
    structure = _structure(point)
    joint = _mapping(structure["joint"], "joint")
    class_sizes = _mapping(joint["class_sizes"], "joint.class_sizes")
    rank_law = _mapping(structure["rank_law"], "rank_law")
    largest = joint["largest_class"]
    if type(largest) is not int:
        msg = "largest_class is not an integer"
        raise TypeError(msg)
    return (
        largest,
        cast("str", class_sizes["kind"]),
        _number(joint["coupling"], "joint.coupling"),
        cast("str", rank_law["kind"]),
    )


def _ambiguity_profile(point: SweepPoint) -> tuple[float, float, float]:
    polysemy = _mapping(_structure(point)["polysemy"], "polysemy")
    return (
        _number(polysemy["mean"], "polysemy.mean"),
        _number(polysemy["dispersion"], "polysemy.dispersion"),
        _number(polysemy["placement"], "polysemy.placement"),
    )


def _scope_target(point: SweepPoint) -> float:
    scope = _mapping(_dataset(point)["scope"], "scope")
    readings = _mapping(scope["readings"], "scope.readings")
    return _number(readings["value"], "scope.readings.value")


def _group_points(axis: _Axis) -> tuple[SweepPoint, ...]:
    codec = RunSpecCodec(search_path=config_search_path(EXPERIMENTS))
    points: list[SweepPoint] = []
    for value in axis.points:
        if not isinstance(value, str):
            msg = f"group axis {axis.key!r} contains a non-text point"
            raise TypeError(msg)
        run = codec(
            EXPERIMENTS / "l3-run.yaml",
            overrides=(f"{axis.key.replace('.', '/')}={value}",),
        )
        points.append(
            SweepPoint(cast("dict[str, JsonValue]", json.loads(run.model_dump_json())))
        )
    return tuple(points)


def _path_points(axis: _Axis) -> tuple[SweepPoint, ...]:
    codec = RunSpecCodec(search_path=config_search_path(EXPERIMENTS))
    points: list[SweepPoint] = []
    for value in axis.points:
        run = codec(EXPERIMENTS / "l3-run.yaml", extra={axis.key: value})
        points.append(
            SweepPoint(cast("dict[str, JsonValue]", json.loads(run.model_dump_json())))
        )
    return tuple(points)


def test_campaign_documents_cover_every_semantic_factor_cell() -> None:
    """The production grids resolve every declared semantic treatment once."""
    distribution = read_experiment(EXPERIMENTS / "distribution.yaml")
    ambiguity = read_experiment(EXPERIMENTS / "ambiguity.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")

    premise_points = _path_points(distribution.sweep.grid[0])
    hypothesis_points = _path_points(distribution.sweep.grid[1])
    semantic_points = _group_points(distribution.sweep.grid[2])
    rank_points = _group_points(distribution.sweep.grid[3])
    ambiguity_points = _group_points(ambiguity.sweep.grid[0])
    scope_points = _path_points(ambiguity.sweep.grid[1])
    reduced_points = expand_experiment(reduced, ROOT)

    expected_semantics = set(
        product((6, 12), ("uniform", "power-law"), (-0.75, 0.0, 0.75))
    )
    semantic_cells = Counter(
        profile[:3] for profile in map(_semantic_profile, semantic_points)
    )
    expected_ambiguity = set(product((1.25, 2.0), (0.0, 1.0), (-0.75, 0.0, 0.75)))
    ambiguity_cells = Counter(map(_ambiguity_profile, ambiguity_points))

    _check(
        distribution.sweep.expected_points == 384, "the distribution grid is incomplete"
    )
    _check(ambiguity.sweep.expected_points == 96, "the ambiguity grid is incomplete")
    _check(
        {_length_kind(point, "premise") for point in premise_points}
        == {"uniform", "negative-binomial"},
        "the premise-length laws are incomplete",
    )
    _check(
        {_length_kind(point, "hypothesis") for point in hypothesis_points}
        == {"uniform", "negative-binomial"},
        "the hypothesis-length laws are incomplete",
    )
    _check(set(semantic_cells) == expected_semantics, "a semantic profile is absent")
    _check(set(semantic_cells.values()) == {1}, "a semantic profile is duplicated")
    _check(
        {_semantic_profile(point)[3] for point in rank_points}
        == {"uniform", "power-law"},
        "a within-class rank law is absent",
    )
    _check(set(ambiguity_cells) == expected_ambiguity, "an ambiguity profile is absent")
    _check(set(ambiguity_cells.values()) == {1}, "an ambiguity profile is duplicated")
    _check(
        {_scope_target(point) for point in scope_points} == {1.0, 3.0},
        "a scope-reading target is absent",
    )
    _check(len(reduced_points) == 4, "the reduced freeze did not expand")
    _check(
        Counter(_semantic_profile(point)[:3] for point in reduced_points)
        == Counter(
            {
                (6, "power-law", -0.75): 1,
                (6, "power-law", 0.75): 1,
                (12, "power-law", -0.75): 1,
                (12, "power-law", 0.75): 1,
            }
        ),
        "the reduced freeze lost a semantic treatment",
    )
    _check(
        Counter(_ambiguity_profile(point) for point in reduced_points)
        == Counter(
            {
                (1.25, 0.0, -0.75): 1,
                (1.25, 0.0, 0.75): 1,
                (2.0, 1.0, -0.75): 1,
                (2.0, 1.0, 0.75): 1,
            }
        ),
        "the reduced freeze lost an ambiguity treatment",
    )
    _check(
        Counter(_scope_target(point) for point in reduced_points)
        == Counter({1.0: 2, 3.0: 2}),
        "the reduced freeze lost a scope-reading treatment",
    )

    for point in (
        *premise_points,
        *hypothesis_points,
        *semantic_points,
        *rank_points,
        *ambiguity_points,
        *scope_points,
        *reduced_points,
    ):
        model = _mapping(point.config["model"], "model")
        capacity = _mapping(model["capacity"], "model.capacity")
        _check(
            capacity["kind"] == "parameter-count" and capacity["target"] == 65536,
            "an L3 point is not capacity matched",
        )
        _check(
            model["language"] == {"from": "data"},
            "an L3 point does not learn in the source language",
        )


@pytest.mark.integration
def test_reduced_freeze_reports_recovery_and_achieved_semantics(tmp_path: Path) -> None:
    """The production path emits recovery and measured semantic factors."""
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
        infrastructure_attempts=1,
        failure_policy=experiment.failure,
    )
    handles = SerialLauncher(store, execute=executor).submit(points)

    _check(len(handles) == 4, "the launcher omitted a reduced treatment")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced treatment did not complete",
    )

    reports = 0
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
        semantic = metrics.inference.semantic
        if semantic is None:
            msg = "a generated logical-form lexicon has no achieved semantic report"
            raise AssertionError(msg)
        reports += 1

        _check(semantic.joint.realized_size > 0, "the semantic inventory is empty")
        _check(
            set(semantic.joint.class_sizes) == {"predicate", "constant"},
            "the semantic report lost a logical-form class",
        )
        _check(
            semantic.joint.coupling is not None and isfinite(semantic.joint.coupling),
            "the achieved class-frequency coupling was not measured",
        )
        _check(
            semantic.per_form_meanings.total > 0,
            "the achieved polysemy distribution is empty",
        )
        _check(
            sum(semantic.symbols_used.values()) > 0,
            "the pair sample used no logical-form symbol",
        )
        target = _scope_target(point)
        for side in (metrics.inference.premise, metrics.inference.hypothesis):
            if side.scope is None:
                msg = "a scope treatment has no achieved scope report"
                raise AssertionError(msg)
            mean = side.scope.readings.mean
            if mean is None:
                msg = "an achieved scope-reading histogram is empty"
                raise AssertionError(msg)
            _check(
                isclose(mean, target),
                "an achieved scope-reading mean missed its treatment",
            )
        _check(metrics.stage == "closed", "a reduced metrics record is not closed")
        _check(
            metrics.point.parameter_count == 65536,
            "a reduced point missed the matched-capacity target",
        )

    _check(reports == 4, "the reduced freeze lost an achieved semantic report")
