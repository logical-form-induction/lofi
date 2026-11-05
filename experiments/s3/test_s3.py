"""Exercise the lexical-distribution campaigns through their public boundaries."""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from math import isfinite
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
from lofi.spec import RunSpecCodec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent


class _ProfileAxis(Protocol):
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


def _integer(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        msg = f"{label} is not an integer"
        raise AssertionError(msg)
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


def _lexical_profile(point: SweepPoint) -> tuple[int, int, str, str, float]:
    grammar = _mapping(_dataset(point)["grammar"], "grammar")
    lexicon = _mapping(grammar["lexicon"], "grammar.lexicon")
    category_shares = _mapping(lexicon["category_shares"], "category_shares")
    rank_law = _mapping(lexicon["rank_law"], "rank_law")
    return (
        _integer(lexicon["forms"], "forms"),
        _integer(lexicon["categories"], "categories"),
        cast("str", category_shares["kind"]),
        cast("str", rank_law["kind"]),
        _number(lexicon["coupling"], "coupling"),
    )


def _ambiguity_profile(point: SweepPoint) -> tuple[float, float, float]:
    grammar = _mapping(_dataset(point)["grammar"], "grammar")
    lexicon = _mapping(grammar["lexicon"], "grammar.lexicon")
    ambiguity = _mapping(lexicon["ambiguity"], "ambiguity")
    return (
        _number(ambiguity["mean"], "ambiguity.mean"),
        _number(ambiguity["dispersion"], "ambiguity.dispersion"),
        _number(ambiguity["placement"], "ambiguity.placement"),
    )


def _length_law(point: SweepPoint) -> str:
    length = _mapping(_dataset(point)["length"], "length")
    return cast("str", length["kind"])


def _composed_profile_points(axis: _ProfileAxis) -> tuple[SweepPoint, ...]:
    key = axis.key
    values = axis.points
    codec = RunSpecCodec()
    points: list[SweepPoint] = []
    for value in values:
        if not isinstance(value, str):
            msg = "a profile-group point is not a fragment name"
            raise TypeError(msg)
        run = codec(
            EXPERIMENTS / "s3-run.yaml",
            overrides=(f"{key.replace('.', '/')}={value}",),
        )
        config = cast("dict[str, JsonValue]", json.loads(run.model_dump_json()))
        points.append(SweepPoint(config))
    return tuple(points)


def _composed_length_points(axis: _ProfileAxis) -> tuple[SweepPoint, ...]:
    codec = RunSpecCodec()
    points: list[SweepPoint] = []
    for value in axis.points:
        run = codec(
            EXPERIMENTS / "s3-run.yaml",
            extra={axis.key: value},
        )
        config = cast("dict[str, JsonValue]", json.loads(run.model_dump_json()))
        points.append(SweepPoint(config))
    return tuple(points)


def test_campaign_documents_load_the_complete_factor_grids() -> None:
    """Every declared distribution treatment expands through the run boundary."""
    lexical = read_experiment(EXPERIMENTS / "lexical-distribution.yaml")
    ambiguity = read_experiment(EXPERIMENTS / "ambiguity.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    length_points = _composed_length_points(lexical.sweep.grid[0])
    lexical_points = _composed_profile_points(lexical.sweep.grid[1])
    ambiguity_points = _composed_profile_points(ambiguity.sweep.grid[0])
    reduced_points = expand_experiment(reduced, ROOT)

    expected_lexical = set(
        product(
            (16, 32),
            (4, 8),
            ("uniform", "power-law"),
            ("uniform", "power-law"),
            (-0.75, 0.0, 0.75),
        )
    )
    lexical_counts = Counter(_lexical_profile(point) for point in lexical_points)
    expected_ambiguity = set(product((1.25, 2.0), (0.0, 1.0), (-0.75, 0.0, 0.75)))
    ambiguity_counts = Counter(_ambiguity_profile(point) for point in ambiguity_points)

    _check(
        lexical.sweep.expected_points
        == len(lexical.sweep.grid[0].points)
        * len(lexical.sweep.grid[1].points)
        * len(lexical.sweep.seeds)
        == 384,
        "the lexical campaign does not declare the complete crossed design",
    )
    _check(
        set(lexical_counts) == expected_lexical,
        "the lexical campaign omitted a crossed factor profile",
    )
    _check(
        set(lexical_counts.values()) == {1},
        "a lexical profile resolves to a duplicate factor cell",
    )
    _check(
        {_length_law(point) for point in length_points}
        == {"uniform", "negative-binomial"},
        "the lexical campaign omits a sentence-length law",
    )
    _check(
        ambiguity.sweep.expected_points
        == len(ambiguity.sweep.grid[0].points) * len(ambiguity.sweep.seeds)
        == 48,
        "the ambiguity campaign does not declare the complete crossed design",
    )
    _check(
        set(ambiguity_counts) == expected_ambiguity,
        "the ambiguity campaign omitted a fixed-marginal treatment",
    )
    _check(
        set(ambiguity_counts.values()) == {1},
        "an ambiguity profile resolves to a duplicate factor cell",
    )
    _check(len(reduced_points) == 8, "the reduced freeze did not expand")
    _check(
        Counter(_ambiguity_profile(point)[2] for point in reduced_points)
        == {-0.75: 4, 0.75: 4},
        "the reduced freeze does not balance the two frequency placements",
    )
    _check(
        lexical.preregistration.grid["lexical-profile"] == lexical.sweep.grid[1].points,
        "the preregistration does not enumerate the frozen lexical profiles",
    )
    _check(
        ambiguity.preregistration.grid["ambiguity-profile"]
        == ambiguity.sweep.grid[0].points,
        "the preregistration does not enumerate the frozen ambiguity profiles",
    )

    for point in (*length_points, *lexical_points, *ambiguity_points, *reduced_points):
        model = _mapping(point.config["model"], "model")
        capacity = _mapping(model["capacity"], "model.capacity")
        _check(
            capacity["kind"] == "parameter-count" and capacity["target"] == 250000,
            "an S3 point is not bound to the matched-capacity target",
        )
        _check(
            model["language"] == {"from": "data"},
            "an S3 point does not learn the grammar drawn with its data",
        )


@pytest.mark.integration
def test_reduced_freeze_emits_recovery_floors_and_achieved_factors(
    tmp_path: Path,
) -> None:
    """The reduced production path measures both outcomes and achieved factors."""
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

    _check(len(handles) == 8, "the launcher omitted a reduced point")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced point did not complete",
    )

    recovery_excesses: list[float] = []
    achieved_placements: list[float] = []
    achieved_correlations: list[float] = []
    achieved_factors: list[dict[str, int | float | None]] = []
    for point in points:
        directory = store.directory(point)
        recorded = read_rows(directory)
        record = read_metrics(directory)
        if recorded is None or record is None:
            msg = f"completed point {point.point_id!r} has no results"
            raise AssertionError(msg)

        named = {
            cast("str", row["metric"]): _number(row["value"], "reported value")
            for row in recorded.rows
            if row["task"] == "lexical-recovery"
            and row["metric"]
            in {
                "primitive-many-to-one-accuracy",
                "primitive-random-relabeling-floor",
            }
        }
        _check(
            set(named)
            == {
                "primitive-many-to-one-accuracy",
                "primitive-random-relabeling-floor",
            },
            "a reduced point omitted its observed recovery or empirical floor",
        )
        recovery_excesses.append(
            named["primitive-many-to-one-accuracy"]
            - named["primitive-random-relabeling-floor"]
        )

        corpus = record.corpus
        if corpus is None:
            msg = f"completed point {point.point_id!r} has no corpus metrics"
            raise AssertionError(msg)
        placement = _ambiguity_profile(point)[2]
        correlation = corpus.lexicon.ambiguity_frequency_correlation
        _check(
            correlation is not None, "an ambiguity treatment achieved no correlation"
        )
        achieved_placements.append(placement)
        achieved_correlations.append(cast("float", correlation))
        achieved_factors.append(
            {
                "length-total": corpus.length.achieved.total,
                "lexicon-size": corpus.lexicon.realized_size,
                "vocabulary-size": corpus.lexicon.realized_vocabulary_size,
                "primitive-count": corpus.inventory.realized_primitive_count,
                "ambiguity-total": corpus.ambiguity.lexical.total,
                "ambiguity-frequency-correlation": correlation,
                "rank-frequency-exponent": corpus.lexicon.rank_frequency_exponent,
                "coupling": corpus.lexicon.coupling,
                "concordance": corpus.lexicon.concordance,
            }
        )

        _check(record.stage == "closed", "a reduced metrics record is not closed")
        _check(
            record.point.parameter_count == 250000,
            "a reduced point did not achieve the matched parameter count",
        )
        _check(
            record.point.trainable_count < record.point.parameter_count,
            "the frozen capacity pad is not excluded from optimization",
        )
        _check(
            corpus.length.achieved.total == 64, "the reduced corpus has the wrong size"
        )
        _check(
            corpus.lexicon.realized_size == 16,
            "the realized lexicon has the wrong size",
        )
        _check(
            0 < corpus.lexicon.realized_vocabulary_size <= 16,
            "the realized vocabulary falls outside its declared lexicon",
        )
        _check(
            corpus.inventory.realized_primitive_count > 0,
            "the achieved category inventory has no primitive",
        )
        _check(corpus.ambiguity.lexical.total > 0, "no lexical ambiguity was measured")
        _check(
            corpus.lexicon.rank_frequency_exponent is not None,
            "the realized rank-frequency law was not measured",
        )
    _check(len(recovery_excesses) == 8, "the reduced run lost a recovery contrast")
    _check(len(achieved_factors) == 8, "the reduced run lost an achieved-factor report")
    _check(
        Counter(achieved_placements) == {-0.75: 4, 0.75: 4},
        "the achieved-factor report lost a declared placement",
    )
    _check(
        all(isfinite(value) for value in achieved_correlations),
        "an achieved ambiguity-frequency correlation is not finite",
    )
