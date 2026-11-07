import json
from typing import TYPE_CHECKING, Final, Literal, cast

from didactic.settings import compose

from lofi.data import UnionRule
from lofi.run import (
    Axis,
    ExperimentManifest,
    ExperimentSeed,
    GridSweepExpander,
    ManifestStatus,
    RunOutcome,
    RunRecord,
    RunStore,
    Sweep,
    SweepPoint,
    open_manifest,
    resolve_constraint_policy,
)
from lofi.spec import (
    PACKAGED_CONFIG_ROOT,
    ConfigSource,
    DatasetSpec,
    FeaturizerSpec,
    LeafOrigin,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    Provenance,
    RunSpec,
    TrainerSpec,
    fingerprint_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.train import RestorableRandomStreams


class PolicyParser(ParserSpec):
    kind: Literal["constraint-policy-config-parser"] = "constraint-policy-config-parser"


class PolicyLoss(LossSpec):
    kind: Literal["constraint-policy-config-loss"] = "constraint-policy-config-loss"


class PolicyDataset(DatasetSpec):
    kind: Literal["constraint-policy-config-dataset"] = (
        "constraint-policy-config-dataset"
    )


class PolicyFeaturizer(FeaturizerSpec):
    kind: Literal["constraint-policy-config-featurizer"] = (
        "constraint-policy-config-featurizer"
    )


class PolicyTrainer(TrainerSpec):
    kind: Literal["constraint-policy-config-trainer"] = (
        "constraint-policy-config-trainer"
    )


class PolicyOptimizer(OptimizerSpec):
    kind: Literal["constraint-policy-config-optimizer"] = (
        "constraint-policy-config-optimizer"
    )


BASE: Final[dict[str, JsonValue]] = {
    "model": {
        "language": {"from": "data"},
        "parser": {"kind": "constraint-policy-config-parser"},
    },
    "loss": {"terms": [{"objective": {"kind": "constraint-policy-config-loss"}}]},
    "data": {
        "datasets": {
            "train": {
                "dataset": {"kind": "constraint-policy-config-dataset"},
                "featurizer": {"kind": "constraint-policy-config-featurizer"},
            }
        }
    },
    "trainer": {"kind": "constraint-policy-config-trainer"},
    "parse": {"retain": "forest", "reachability": "all"},
    "release": {"checkpoint": "resume-latest"},
    "optimizer": {"kind": "constraint-policy-config-optimizer"},
    "device_policy": {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    },
}


PACKAGES: Final = {
    "didactic": "test",
    "panproto": "test",
    "quivers": "test",
    "torch": "test",
    "lairs": "test",
    "numpy": "test",
    "python": "test",
    "transformers": "test",
}


def _composed(overrides: tuple[str, ...], _seed: int) -> dict[str, JsonValue]:
    """Return one resolved run document under constraint-policy group selections.

    Parameters
    ----------
    overrides : tuple[str, ...]
        Group selections rendered by the sweep.
    """
    specification = compose(
        None,
        schema=RunSpec,
        base=BASE,
        overrides=overrides,
        search_path=(PACKAGED_CONFIG_ROOT,),
    )

    return cast("dict[str, JsonValue]", json.loads(specification.model_dump_json()))


def _manifest(point: SweepPoint) -> ExperimentManifest:
    """Return the running manifest for one constraint-policy point.

    Parameters
    ----------
    point : SweepPoint
        Point whose resolved configuration the manifest records.
    """
    derived = ExperimentSeed(0).point(point.point_id, point.seed)

    return open_manifest(
        run_spec=RunSpec.model_validate(point.config),
        component_fingerprints={
            role: f"fingerprint:{role}"
            for role in RunSpec.model_validate(point.config).component_roles()
        },
        grammar_fingerprint="grammar:constraint-policy-config",
        signature_fingerprint="signature:constraint-policy-config",
        corpus_revision="corpus:constraint-policy-config",
        seed_policy=derived.policy(accelerator=None, samplers=()),
        seed_derivation=derived,
        package_versions=PACKAGES,
        git_revision="constraint-policy-config",
        accelerator="cpu",
    )


def _record() -> RunRecord:
    """Return a record for the configuration-boundary persistence test."""
    return RunRecord(
        provenance=Provenance(
            origins={
                "data.datasets.train.constraint_policy": LeafOrigin(
                    source=ConfigSource.GROUP, layer="constraint-policy"
                )
            }
        ),
        fingerprint=fingerprint_run({}),
        declarations={},
    )


def _complete(
    _point: SweepPoint,
    _directory: Path,
    _streams: RestorableRandomStreams,
    _policy: DevicePolicy,
) -> ManifestStatus:
    """Return the completed status for a persistence-only run."""
    return ManifestStatus.COMPLETED


def test_syntactic_and_semantic_policy_groups_cross_and_persist(
    tmp_path: Path,
) -> None:
    sweep = Sweep(
        grid=(
            Axis(
                group="data.datasets.train.constraint_policy.syntactic",
                values=("none", "all"),
            ),
            Axis(
                group="data.datasets.train.constraint_policy.semantic",
                values=("none", "all"),
            ),
        ),
        expected_points=4,
    )
    points = GridSweepExpander(_composed).expand(sweep)
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(0))

    assert len(points) == 4
    assert len({point.point_id for point in points}) == 4
    assert {
        (
            RunSpec.model_validate(point.config)
            .data.datasets["train"]
            .constraint_policy.syntactic.coverage,
            RunSpec.model_validate(point.config)
            .data.datasets["train"]
            .constraint_policy.semantic.coverage,
        )
        for point in points
    } == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}

    for point in points:
        resolved = RunSpec.model_validate(point.config)
        selected = {
            expression.rsplit("/", 1)[1].split("=", 1)[0]: expression.rsplit("=", 1)[1]
            for expression in point.overrides
        }
        syntactic = resolve_constraint_policy(
            resolved.data.datasets["train"].constraint_policy.syntactic
        )
        semantic = resolve_constraint_policy(
            resolved.data.datasets["train"].constraint_policy.semantic
        )

        assert syntactic.coverage == (1.0 if selected["syntactic"] == "all" else 0.0)
        assert semantic.coverage == (1.0 if selected["semantic"] == "all" else 0.0)
        assert syntactic.ambiguity == UnionRule()
        assert semantic.ambiguity == UnionRule()

        outcome = store.execute(
            point,
            _manifest(point),
            _complete,
            record=_record(),
        )
        stored = store.record(point)

        assert outcome == RunOutcome(
            store.directory(point), ManifestStatus.COMPLETED, cached=False
        )
        assert stored is not None
        assert (
            stored.resolved.data.datasets["train"].constraint_policy
            == resolved.data.datasets["train"].constraint_policy
        )
        assert stored.fingerprint == _record().fingerprint
