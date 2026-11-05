"""Derived and fixed data seeds as values of one sweep axis."""

import json
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.run import (
    Axis,
    ExecutionEnvironment,
    ExperimentSeed,
    GridSweepExpander,
    Sweep,
    open_training_manifest,
    plan_training,
)
from lofi.spec import (
    DERIVED_SEED,
    PACKAGED_CONFIG_ROOT,
    RunSpec,
    RunSpecCodec,
    UnresolvedSeedError,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
    from lofi.run import SweepPoint

CONFIG: Final = PACKAGED_CONFIG_ROOT / "run/synthetic-parser.yaml"
EXPERIMENT: Final = ExperimentSeed(5)
ENVIRONMENT: Final = ExecutionEnvironment({"lofi": "test"}, "test", "cpu")


def composed(overrides: tuple[str, ...], _seed: int) -> dict[str, JsonValue]:
    """Return the parser run a sweep point's overrides compose.

    Parameters
    ----------
    overrides : tuple[str, ...]
        The leaf assignments the sweep renders.
    """
    run_spec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))(
        CONFIG, overrides=overrides
    )

    return json.loads(run_spec.model_dump_json())


def expanded() -> tuple[SweepPoint, ...]:
    """Return the points of one experiment crossing the data seed with two seeds."""
    sweep = Sweep(
        grid=(Axis(path="data.datasets.train.dataset.seed", values=(DERIVED_SEED, 5)),),
        seeds=(0, 1),
        expected_points=4,
    )

    return GridSweepExpander(composed).expand(sweep)


def test_derived_seeds_resample_the_corpus_and_a_fixed_seed_replicates_it() -> None:
    revisions: dict[bool, set[str]] = {True: set(), False: set()}
    points = expanded()

    assert len(points) == 4

    for point in points:
        run_spec = RunSpec.model_validate(point.config)
        derivation = EXPERIMENT.point(point.point_id, point.seed)
        blueprint = plan_training(run_spec, point=derivation)
        manifest = open_training_manifest(blueprint, derivation, ENVIRONMENT)
        declared = cast(
            "SyntheticCorpusDecl", manifest.run_spec.data.datasets["train"].dataset
        ).seed

        assert (
            declared
            == cast("SyntheticCorpusDecl", run_spec.data.datasets["train"].dataset).seed
        )
        assert manifest.corpus_revision == blueprint.corpus_revision

        revisions[declared == DERIVED_SEED].add(blueprint.corpus_revision)

    assert len(revisions[True]) == 2
    assert len(revisions[False]) == 1
    assert revisions[True].isdisjoint(revisions[False])


def test_replanning_one_point_draws_the_same_corpus() -> None:
    point = next(
        point
        for point in expanded()
        if cast(
            "SyntheticCorpusDecl",
            RunSpec.model_validate(point.config).data.datasets["train"].dataset,
        ).seed
        == DERIVED_SEED
    )
    run_spec = RunSpec.model_validate(point.config)
    derivation = EXPERIMENT.point(point.point_id, point.seed)

    assert (
        plan_training(run_spec, point=derivation).corpus_revision
        == plan_training(run_spec, point=derivation).corpus_revision
    )


def test_a_derived_seed_without_a_point_is_refused_at_planning() -> None:
    run_spec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))(
        CONFIG, overrides=("data.datasets.train.dataset.seed=derived",)
    )

    with pytest.raises(UnresolvedSeedError, match="derived from the point"):
        plan_training(run_spec)
