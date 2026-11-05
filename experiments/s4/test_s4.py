"""Exercise the supervision-and-noise campaigns through public boundaries."""

from __future__ import annotations

import json
import runpy
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from lofi.run import (
    ExperimentSeed,
    RunStore,
    SweepPoint,
    expand_experiment,
    read_experiment,
    read_metrics,
    read_rows,
)
from lofi.spec import RunSpecCodec, config_search_path

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
LEXICAL_PROFILES = frozenset(
    {
        "baseline",
        "large-diverse",
        "power-coupling-negative",
        "power-coupling-zero",
        "power-coupling-positive",
        "ambiguous-frequent",
    }
)


class _Axis(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def points(self) -> tuple[JsonValue, ...]: ...

    def with_points(self, points: tuple[JsonValue, ...], /) -> _Axis: ...


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


def _corruption_share(point: SweepPoint) -> float:
    corruption = _dataset(point).get("corruption")
    if corruption is None:
        return 0.0
    held = _mapping(corruption, "corruption")
    proportion = _mapping(held["proportion"], "corruption.proportion")
    return _number(proportion["value"], "corruption.proportion.value")


def _noise_rates(point: SweepPoint) -> tuple[float, float]:
    noise = _dataset(point).get("noise")
    if noise is None:
        return (0.0, 0.0)
    held = _mapping(noise, "noise")
    grammatical = _mapping(held["grammatical"], "noise.grammatical")
    ungrammatical = _mapping(held["ungrammatical"], "noise.ungrammatical")
    return (
        _number(grammatical["rate"], "noise.grammatical.rate"),
        _number(ungrammatical["rate"], "noise.ungrammatical.rate"),
    )


def _lexical_profile(point: SweepPoint) -> tuple[int, int, str, str, float, float]:
    grammar = _mapping(_dataset(point)["grammar"], "grammar")
    lexicon = _mapping(grammar["lexicon"], "grammar.lexicon")
    category_shares = _mapping(lexicon["category_shares"], "category_shares")
    rank_law = _mapping(lexicon["rank_law"], "rank_law")
    ambiguity = _mapping(lexicon["ambiguity"], "ambiguity")
    forms = lexicon["forms"]
    categories = lexicon["categories"]
    if type(forms) is not int or type(categories) is not int:
        msg = "lexical profile sizes are not integers"
        raise TypeError(msg)
    return (
        forms,
        categories,
        cast("str", category_shares["kind"]),
        cast("str", rank_law["kind"]),
        _number(lexicon["coupling"], "coupling"),
        _number(ambiguity["mean"], "ambiguity.mean"),
    )


def _composed_profile_points(axis: _Axis) -> tuple[SweepPoint, ...]:
    key = axis.key
    values = axis.points
    codec = RunSpecCodec(search_path=config_search_path(ROOT))
    points: list[SweepPoint] = []
    for value in values:
        if not isinstance(value, str):
            msg = "a lexical-profile point is not a fragment name"
            raise TypeError(msg)
        run = codec(
            EXPERIMENTS / "s4-positive-only-run.yaml",
            overrides=(f"{key.replace('.', '/')}={value}",),
        )
        points.append(
            SweepPoint(cast("dict[str, JsonValue]", json.loads(run.model_dump_json())))
        )
    return tuple(points)


def _composed_path_points(config: Path, *axes: _Axis) -> tuple[SweepPoint, ...]:
    codec = RunSpecCodec(search_path=config_search_path(ROOT))
    values = product(*(axis.points for axis in axes))
    points: list[SweepPoint] = []
    for cell in values:
        run = codec(
            config,
            extra={axis.key: value for axis, value in zip(axes, cell, strict=True)},
        )
        points.append(
            SweepPoint(cast("dict[str, JsonValue]", json.loads(run.model_dump_json())))
        )
    return tuple(points)


def test_campaign_documents_expand_the_complete_regime_specific_grids() -> None:
    """Every noise ladder is crossed with every representative lexical profile."""
    positive = read_experiment(EXPERIMENTS / "positive-only.yaml")
    certified = read_experiment(EXPERIMENTS / "certified-negatives.yaml")
    labeled = read_experiment(EXPERIMENTS / "labeled-strings.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    positive_paths = positive.sweep.yoked[0]
    positive_points = tuple(
        point
        for values in zip(*(axis.points for axis in positive_paths), strict=True)
        for point in _composed_path_points(
            positive.config.path,
            *(
                axis.with_points((value,))
                for axis, value in zip(positive_paths, values, strict=True)
            ),
        )
    )
    certified_points = _composed_path_points(
        certified.config.path, certified.sweep.grid[0]
    )
    labeled_points = _composed_path_points(
        labeled.config.path, labeled.sweep.grid[0], labeled.sweep.grid[1]
    )
    lexical_points = _composed_profile_points(positive.sweep.grid[0])
    reduced_points = expand_experiment(reduced, ROOT)

    _check(
        positive.sweep.expected_points == 5 * 6 * 4 == 120,
        "the positive-only grid is incomplete",
    )
    _check(
        certified.sweep.expected_points == 4 * 6 * 4 == 96,
        "the certified-negative grid is incomplete",
    )
    _check(
        labeled.sweep.expected_points == 1 * 6 * 6 * 4 == 144,
        "the labeled-string grid is incomplete",
    )
    _check(len(positive_points) == 5, "a positive-only treatment does not compose")
    _check(
        len(certified_points) == 4, "a certified-negative treatment does not compose"
    )
    _check(len(labeled_points) == 6, "a labeled-string treatment does not compose")
    _check(len(lexical_points) == 6, "an S3 lexical profile does not compose")
    _check(len(reduced_points) == 5, "the reduced freeze is not minimal")
    _check(
        Counter(
            (_corruption_share(point), _noise_rates(point)) for point in reduced_points
        )
        == Counter(
            {
                (0.0, (0.0, 0.0)): 1,
                (0.25, (0.0, 0.0)): 2,
                (0.25, (0.0, 1.0)): 1,
                (0.25, (0.2, 0.2)): 1,
            }
        ),
        "the reduced freeze does not represent every regime and noise level",
    )

    positive_cells = Counter(
        (_corruption_share(point), _noise_rates(point), _lexical_profile(point))
        for point in positive_points
    )
    certified_cells = Counter(
        (_corruption_share(point), _noise_rates(point), _lexical_profile(point))
        for point in certified_points
    )
    labeled_cells = Counter(
        (_corruption_share(point), _noise_rates(point), _lexical_profile(point))
        for point in labeled_points
    )
    profiles = {_lexical_profile(point) for point in lexical_points}

    _check(len(profiles) == 6, "the S3 profile contraction has the wrong size")
    _check(
        {cell[0] for cell in positive_cells} == {0.0, 0.05, 0.1, 0.25, 0.5},
        "the positive-only contamination ladder is incomplete",
    )
    _check(
        {cell[1] for cell in positive_cells} == {(0.0, 0.0), (0.0, 1.0)},
        "positive-only contamination is not one-sided",
    )
    _check(
        {cell[0] for cell in certified_cells} == {0.05, 0.1, 0.25, 0.5},
        "the certified-negative proportion ladder is incomplete",
    )
    _check(
        {cell[1] for cell in certified_cells} == {(0.0, 0.0)},
        "a certified-negative point flips labels",
    )
    _check(
        {cell[0] for cell in labeled_cells} == {0.25},
        "the labeled regime changes its negative proportion",
    )
    _check(
        {cell[1] for cell in labeled_cells}
        == {
            (0.0, 0.0),
            (0.05, 0.05),
            (0.1, 0.1),
            (0.2, 0.2),
            (0.3, 0.3),
            (0.4, 0.4),
        },
        "the symmetric label-flip ladder is incomplete",
    )
    _check(set(positive_cells.values()) == {1}, "positive cells are duplicated")
    _check(set(certified_cells.values()) == {1}, "certified cells are duplicated")
    _check(set(labeled_cells.values()) == {1}, "labeled cells are duplicated")

    for experiment in (positive, certified, labeled):
        _check(
            set(experiment.preregistration.grid["lexical-profile"]) == LEXICAL_PROFILES,
            "a production preregistration omits an S3 profile",
        )

    for point in (
        *positive_points,
        *certified_points,
        *labeled_points,
        *lexical_points,
        *reduced_points,
    ):
        model = _mapping(point.config["model"], "model")
        capacity = _mapping(model["capacity"], "model.capacity")
        _check(
            capacity["kind"] == "parameter-count" and capacity["target"] == 250000,
            "an S4 point is not capacity matched",
        )
        _check(
            model["language"] == {"from": "data"},
            "an S4 point does not learn the grammar drawn with its data",
        )


@pytest.mark.integration
def test_reduced_freeze_trains_and_reports_recovery_per_treatment(
    tmp_path: Path,
) -> None:
    """The real reduced path emits recovery and achieved noise for every arm."""
    experiment = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    points = expand_experiment(experiment, ROOT)
    store = RunStore(
        tmp_path / "runs",
        freeze_id=None,
        seed=ExperimentSeed(experiment.seed_policy.root),
    )
    statuses: list[int] = []
    for index, point in enumerate(points):
        document = tmp_path / f"point-{index}.json"
        document.write_text(json.dumps(point.config), encoding="utf-8")
        arguments = (
            "lofi",
            "run",
            "--config",
            str(document),
            "--seed",
            str(point.seed),
            "--experiment-seed",
            str(experiment.seed_policy.root),
            "--infrastructure-attempts",
            "1",
            "--root",
            str(store.root),
        )
        stdout = StringIO()
        stderr = StringIO()
        previous = sys.argv
        status = 0
        try:
            sys.argv = list(arguments)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                runpy.run_module("lofi.run", run_name="__main__")
        except SystemExit as stopped:
            status = stopped.code if type(stopped.code) is int else 1
        finally:
            sys.argv = previous
        if status != 0:
            msg = stderr.getvalue() or stdout.getvalue()
            raise AssertionError(msg)
        statuses.append(status)

    _check(len(statuses) == 5, "the public runner omitted a reduced treatment")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced treatment did not complete",
    )

    recovered = 0
    achieved: list[tuple[float, tuple[float, float]]] = []
    for point in points:
        directory = store.directory(point)
        recorded = read_rows(directory)
        record = read_metrics(directory)
        if recorded is None or record is None or record.corpus is None:
            msg = f"completed point {point.point_id!r} has no closed result"
            raise AssertionError(msg)
        named = {
            cast("str", row["metric"])
            for row in recorded.rows
            if row["task"] == "lexical-recovery"
        }
        _check(
            {
                "primitive-many-to-one-accuracy",
                "primitive-random-relabeling-floor",
            }
            <= named,
            "a reduced treatment omitted recovery or its empirical floor",
        )
        recovered += 1
        corruption = record.corpus.corruption
        noise = record.corpus.noise
        achieved.append(
            (
                0.0 if corruption is None else corruption.achieved_proportion,
                (
                    0.0
                    if noise is None
                    else noise.achieved_rates.get("grammatical", 0.0),
                    0.0
                    if noise is None
                    else noise.achieved_rates.get("ungrammatical", 0.0),
                ),
            )
        )
        _check(record.stage == "closed", "a reduced metrics record is not closed")
        _check(
            record.point.parameter_count == 250000,
            "a reduced point missed the matched-capacity target",
        )

    _check(recovered == 5, "the reduced freeze lost a treatment report")
    _check(len(achieved) == 5, "the reduced freeze lost an achieved-noise report")
    _check(
        all(
            0.0 <= proportion <= 1.0 and all(0.0 <= rate <= 1.0 for rate in rates)
            for proportion, rates in achieved
        ),
        "an achieved corruption or noise rate is outside the probability bounds",
    )
