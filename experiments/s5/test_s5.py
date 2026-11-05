"""Exercise syntactic anchoring through experiment and run boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

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


def _check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _mapping(value: JsonValue, label: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        msg = f"{label} is not a mapping"
        raise TypeError(msg)
    return value


def _policy(point: SweepPoint) -> tuple[str, float, str | None]:
    data = _mapping(point.config["data"], "data")
    datasets = _mapping(data["datasets"], "data.datasets")
    train = _mapping(datasets["train"], "data.datasets.train")
    policies = _mapping(train["constraint_policy"], "constraint_policy")
    syntactic = _mapping(policies["syntactic"], "constraint_policy.syntactic")
    selector = _mapping(syntactic["selector"], "selector")
    kind = cast("str", selector["kind"])
    if kind == "rank":
        return kind, float(cast("int | float", selector["last"])), None
    if kind == "quantile":
        return kind, float(cast("int | float", selector["share"])), None
    return (
        kind,
        float(cast("int | float", selector["share"])),
        cast("str", selector["role"]),
    )


def _coupling(point: SweepPoint) -> float:
    data = _mapping(point.config["data"], "data")
    datasets = _mapping(data["datasets"], "data.datasets")
    train = _mapping(datasets["train"], "data.datasets.train")
    dataset = _mapping(train["dataset"], "data.datasets.train.dataset")
    grammar = _mapping(dataset["grammar"], "grammar")
    lexicon = _mapping(grammar["lexicon"], "grammar.lexicon")
    return float(cast("int | float", lexicon["coupling"]))


def test_anchoring_documents_load_the_complete_share_valued_design() -> None:
    """Every selection mechanism and share expands at every coupling level."""
    production = read_experiment(EXPERIMENTS / "anchoring.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    production_points = expand_experiment(production, ROOT)
    reduced_points = expand_experiment(reduced, ROOT)

    policies = {_policy(point) for point in production_points}
    expected = {
        (kind, share, role)
        for kind, role in (
            ("rank", None),
            ("quantile", None),
            ("category-role", "closed"),
            ("category-role", "open"),
        )
        for share in (0.25, 0.5, 0.75)
    }
    _check(len(production_points) == 144, "the production design did not expand")
    _check(policies == expected, "the production design omits a selection budget")
    _check(
        {_coupling(point) for point in production_points} == {-0.75, 0.0, 0.75},
        "the production design omits a frequency-class coupling",
    )
    _check(len(reduced_points) == 6, "the reduced design did not expand")
    _check(
        {_policy(point) for point in reduced_points}
        == {
            ("rank", 0.5, None),
            ("quantile", 0.5, None),
            ("category-role", 0.5, "closed"),
        },
        "the reduced design omits a selection mechanism",
    )
    for point in (*production_points, *reduced_points):
        model = _mapping(point.config["model"], "model")
        capacity = _mapping(model["capacity"], "model.capacity")
        _check(
            capacity["kind"] == "parameter-count" and capacity["target"] == 250000,
            "an anchoring point is not capacity matched",
        )
        _check(
            model["language"] == {"from": "data"},
            "an anchoring point does not learn its generated language",
        )


@pytest.mark.integration
def test_reduced_freeze_reports_recovery_against_achieved_constraint_budgets(
    tmp_path: Path,
) -> None:
    """Real fits persist achieved budgets beside recovery and its floor."""
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

    _check(len(handles) == 6, "the launcher omitted a reduced anchoring point")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced anchoring point did not complete",
    )
    achieved: dict[tuple[str, float, str | None], set[float]] = {}
    for point in points:
        directory = store.directory(point)
        record = read_metrics(directory)
        reported = read_rows(directory)
        if record is None or reported is None:
            msg = f"completed point {point.point_id!r} has no results"
            raise AssertionError(msg)
        selection = record.selection
        if selection is None:
            msg = f"completed point {point.point_id!r} has no selection budget"
            raise AssertionError(msg)
        _check(selection.types_pinned > 0, "an anchoring policy pinned no type")
        _check(selection.tokens_covered > 0, "an anchoring policy covered no token")
        _check(
            0.0 < selection.mass_share <= 1.0,
            "an anchoring policy reported an invalid mass share",
        )
        _check(
            selection.concordance is not None,
            "an anchoring point omitted frequency-class concordance",
        )
        achieved.setdefault(_policy(point), set()).add(selection.mass_share)
        rows = {
            cast("str", row["metric"]): row
            for row in reported.rows
            if row["task"] == "lexical-recovery"
        }
        required = {
            "primitive-many-to-one-accuracy",
            "primitive-random-relabeling-floor",
        }
        _check(
            required <= set(rows),
            "an anchoring point omitted recovery or its floor",
        )
        _check(
            all(rows[metric]["value"] is not None for metric in required),
            "an anchoring point emitted an unscored recovery row",
        )
    _check(
        len(achieved) == 3 and all(values for values in achieved.values()),
        "the reduced freeze did not record an achieved budget per policy",
    )
