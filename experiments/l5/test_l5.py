"""Exercise logical-form anchoring through experiment and run boundaries."""

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


def _is_form_anchor(value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    objective = value.get("objective")
    return (
        isinstance(objective, dict)
        and objective.get("kind") == "vector-anchor"
        and objective.get("readout") == "forms"
    )


def _policy(point: SweepPoint) -> tuple[str, float, tuple[str, ...]]:
    data = _mapping(point.config["data"], "data")
    datasets = _mapping(data["datasets"], "data.datasets")
    train = _mapping(datasets["train"], "data.datasets.train")
    policies = _mapping(train["constraint_policy"], "constraint_policy")
    semantic = _mapping(policies["semantic"], "constraint_policy.semantic")
    selector = _mapping(semantic["selector"], "selector")
    kind = cast("str", selector["kind"])
    if kind == "rank":
        return kind, float(cast("int | float", selector["last"])), ()
    if kind == "quantile":
        return kind, float(cast("int | float", selector["share"])), ()

    categories = selector["categories"]
    if not isinstance(categories, list) or not all(
        isinstance(category, str) for category in categories
    ):
        msg = "the logical-constant-class selector does not name text classes"
        raise TypeError(msg)
    named = tuple(cast("str", category) for category in categories)
    return kind, len(named) / 4.0, named


def test_anchoring_documents_load_the_complete_share_valued_design() -> None:
    """Rank, quantile, and constant-class selection span the three budgets."""
    production = read_experiment(EXPERIMENTS / "anchoring.yaml")
    reduced = read_experiment(EXPERIMENTS / "reduced-freeze.yaml")
    production_points = expand_experiment(production, ROOT)
    reduced_points = expand_experiment(reduced, ROOT)

    expected = {
        *(("rank", share, ()) for share in (0.25, 0.5, 0.75)),
        *(("quantile", share, ()) for share in (0.25, 0.5, 0.75)),
        ("category-class", 0.25, ("constant",)),
        ("category-class", 0.5, ("constant", "predicate")),
        (
            "category-class",
            0.75,
            ("connective", "constant", "predicate"),
        ),
    }

    _check(len(production_points) == 36, "the production design did not expand")
    _check(
        {_policy(point) for point in production_points} == expected,
        "the production design omits a logical-form selection budget",
    )
    _check(len(reduced_points) == 3, "the reduced design did not expand")
    _check(
        {_policy(point) for point in reduced_points}
        == {
            ("rank", 0.5, ()),
            ("quantile", 0.5, ()),
            ("category-class", 0.5, ("constant", "predicate")),
        },
        "the reduced design omits a selection mechanism",
    )

    for point in (*production_points, *reduced_points):
        model = _mapping(point.config["model"], "model")
        capacity = _mapping(model["capacity"], "model.capacity")
        spaces = _mapping(model["spaces"], "model.spaces")
        readouts = _mapping(model["readouts"], "model.readouts")
        data = _mapping(point.config["data"], "data")
        datasets = _mapping(data["datasets"], "data.datasets")
        train = _mapping(datasets["train"], "data.datasets.train")
        source = _mapping(train["constraint_source"], "constraint_source")
        loss = _mapping(point.config["loss"], "loss")
        terms = loss["terms"]

        _check(
            capacity["kind"] == "parameter-count" and capacity["target"] == 250000,
            "a logical-form anchoring point is not capacity matched",
        )
        _check(
            model["language"] == {"from": "data"},
            "an anchoring point does not learn its generated language",
        )
        _check(
            "semantics" in spaces and "forms" in readouts,
            "an anchoring point lacks a logical-form beam",
        )
        _check(
            source["kind"] == "gold-logical-form" and source["policy"] == "semantic",
            "an anchoring point does not select semantic gold constraints",
        )
        if not isinstance(terms, list):
            msg = "the loss terms are not a list"
            raise TypeError(msg)
        _check(
            any(_is_form_anchor(term) for term in terms),
            "an anchoring point does not anchor the logical-form readout",
        )


@pytest.mark.integration
def test_reduced_freeze_trains_and_reports_recovery_by_achieved_lf_budget(
    tmp_path: Path,
) -> None:
    """Real reduced fits persist achieved LF budgets beside recovery rows."""
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

    _check(len(handles) == 3, "the launcher omitted a reduced anchoring point")
    _check(
        all(store.is_completed(point) for point in points),
        "a reduced logical-form anchoring point did not complete",
    )

    achieved: dict[tuple[str, float, tuple[str, ...]], float] = {}
    for point in points:
        directory = store.directory(point)
        record = read_metrics(directory)
        reported = read_rows(directory)
        if record is None or reported is None:
            msg = f"completed point {point.point_id!r} has no ingested results"
            raise AssertionError(msg)
        selection = record.selection
        if selection is None:
            msg = f"completed point {point.point_id!r} has no LF selection budget"
            raise AssertionError(msg)

        _check(selection.types_pinned > 0, "an LF policy pinned no lexical form")
        _check(selection.tokens_covered > 0, "an LF policy covered no token")
        _check(
            0.0 < selection.mass_share <= 1.0,
            "an LF policy reported an invalid achieved mass share",
        )
        _check(
            bool(selection.by_category),
            "an LF policy omitted its logical-constant-class budget",
        )
        achieved[_policy(point)] = selection.mass_share

        rows = {
            cast("str", row["metric"]): row
            for row in reported.rows
            if row["task"] == "logical-forms"
        }
        required = {
            "fiber-relative-recovery",
            "fiber-floor",
            "logical-form-normal-form",
        }
        _check(required <= set(rows), "an LF point omitted a recovery row")
        _check(
            all(rows[metric]["value"] is not None for metric in required),
            "an LF point emitted an unscored recovery row",
        )

    _check(
        set(achieved)
        == {
            ("rank", 0.5, ()),
            ("quantile", 0.5, ()),
            ("category-class", 0.5, ("constant", "predicate")),
        },
        "the reduced freeze omitted an achieved LF budget",
    )
