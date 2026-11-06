import json
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx
import pytest
import torch
from tests.conformance.experiment_manifest import (
    ExperimentManifestConformance,
    ManifestFixture,
)
from tests.interrupts import NEW_GROUP, interrupt
from tests.unit.run import conftest as launcher
from tests.unit.run.test__experiment import an_experiment_document, write_experiment

from lofi.eval import UNFROZEN
from lofi.run import (
    MANIFEST_NAME,
    RUNS_NAME,
    Axis,
    ExperimentIdentity,
    ExperimentManifest,
    ExperimentSeed,
    ExperimentStore,
    FailureKind,
    GridSweepExpander,
    ManifestIso,
    ManifestStatus,
    PointFailure,
    PointSeed,
    RecordedRows,
    ReseededRunError,
    RunOutcome,
    RunStatus,
    RunStore,
    SeedPolicy,
    Sweep,
    SweepPoint,
    TamperedRecordError,
    UncommittedRunError,
    close_manifest,
    decode_seed_derivation,
    encode_seed_derivation,
    expand_experiment,
    open_manifest,
    point_id,
    read_binding,
    read_experiment,
    resolve_device_policy,
    unique_points,
    write_failure,
    write_rows,
)
from lofi.run._manifest import (
    RESOLVED_NAME,
    SCHEMA_FINGERPRINT_NAME,
    SOURCES_NAME,
    RunRecord,
    StoredRecord,
    read_record,
    write_manifest,
    write_record,
)
from lofi.spec import (
    ArchivedLanguage,
    ConfigSource,
    DatasetSpec,
    FeaturizerSpec,
    IndexedLanguage,
    LeafOrigin,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    Provenance,
    RunFingerprint,
    RunSpec,
    SinkSpec,
    SpecLoader,
    TrainerSpec,
    fingerprint_run,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from lofi._types import JsonValue
    from lofi.nn.policy import DevicePolicy
    from lofi.run import Experiment
    from lofi.train import RestorableRandomStreams

TREE: Final = Path(__file__).resolve().parents[3]
ROOT: Final = ExperimentSeed(0)


class StoreParser(ParserSpec):
    kind: Literal["store-unit-parser"] = "store-unit-parser"


class StoreLoss(LossSpec):
    kind: Literal["store-unit-loss"] = "store-unit-loss"


class StoreDataset(DatasetSpec):
    kind: Literal["store-unit-dataset"] = "store-unit-dataset"


class StoreFeaturizer(FeaturizerSpec):
    kind: Literal["store-unit-featurizer"] = "store-unit-featurizer"


class StoreTrainer(TrainerSpec):
    kind: Literal["store-unit-trainer"] = "store-unit-trainer"
    epochs: int = 1


class StoreOptimizer(OptimizerSpec):
    kind: Literal["store-unit-optimizer"] = "store-unit-optimizer"


class StoreSink(SinkSpec):
    kind: Literal["store-unit-sink"] = "store-unit-sink"


def a_document(epochs: int = 1) -> dict[str, JsonValue]:
    return {
        "parse": {"retain": "forest", "reachability": "all"},
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "store-unit-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "store-unit-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "store-unit-dataset"},
                    "featurizer": {"kind": "store-unit-featurizer"},
                }
            }
        },
        "trainer": {"kind": "store-unit-trainer", "epochs": epochs},
        "optimizer": {"kind": "store-unit-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
        "release": {"checkpoint": "resume-latest"},
        "sinks": {"metrics": {"kind": "store-unit-sink"}},
    }


PACKAGES: Final = {
    "didactic": "0.17.1",
    "panproto": "0.74.3",
    "quivers": "0.17.0",
    "torch": "2.14.0",
    "lairs": "1.0",
    "numpy": "1.0",
    "python": "1.0",
    "transformers": "1.0",
}

GRAMMAR_SPEC: Final = (
    b'{"route":{"source":"decl-1","target":"native-1"},"theory":"S NP"}'
)


def a_manifest(epochs: int = 1) -> ExperimentManifest:
    derived = ExperimentSeed(0).point(a_point(epochs).point_id, 0)

    return open_manifest(
        run_spec=RunSpec.model_validate(a_document(epochs)),
        component_fingerprints={
            role: f"fingerprint:{role}"
            for role in RunSpec.model_validate(a_document(epochs)).component_roles()
        },
        grammar_fingerprint="grammar:ccg",
        signature_fingerprint="signature:ccg",
        corpus_revision="ccgbank:LDC2005T13:auto-v1",
        seed_policy=derived.policy(accelerator=None, samplers=()),
        seed_derivation=derived,
        package_versions=PACKAGES,
        git_revision="0123456789abcdef",
        accelerator="cpu",
    )


def seeded(
    manifest: ExperimentManifest, derived: PointSeed, policy: SeedPolicy
) -> ExperimentManifest:
    document = json.loads(manifest.model_dump_json())
    document["seed_policy"] = dict(policy.record)
    document["seed_derivation"] = encode_seed_derivation(derived).model_dump()

    return ExperimentManifest.model_validate(document)


def a_manifest_for(point: SweepPoint) -> ExperimentManifest:
    derived = ExperimentSeed(0).point(point.point_id, point.seed)

    return seeded(a_manifest(), derived, derived.policy(accelerator=None, samplers=()))


def a_provenance(epochs: int = 1) -> Provenance:
    return Provenance(
        origins={
            "trainer.epochs": LeafOrigin(
                source=ConfigSource.OVERRIDE, layer=f"trainer.epochs={epochs}"
            ),
            "trainer.kind": LeafOrigin(source=ConfigSource.FILE, layer="run.json"),
        }
    )


def a_fingerprint() -> RunFingerprint:
    return fingerprint_run(
        {
            "grammar": IndexedLanguage(
                "decl-1->native-1",
                GRAMMAR_SPEC,
                sha256(GRAMMAR_SPEC).hexdigest(),
                dx.LoweringRoute("decl-1", "native-1"),
            )
        }
    )


DECLARATION: Final = b'{"name":"store-grammar"}'


def a_record(epochs: int = 1) -> RunRecord:
    return RunRecord(
        provenance=a_provenance(epochs),
        fingerprint=a_fingerprint(),
        declarations={"grammar": DECLARATION},
    )


def a_point(epochs: int = 1, *, seed: int = 0) -> SweepPoint:
    return SweepPoint(a_document(epochs), seed=seed)


class Trainer:
    def __init__(self, status: ManifestStatus = ManifestStatus.COMPLETED) -> None:
        self.calls: list[tuple[SweepPoint, Path]] = []
        self._status = status

    def __call__(
        self,
        point: SweepPoint,
        directory: Path,
        _streams: RestorableRandomStreams,
        _policy: DevicePolicy,
    ) -> ManifestStatus:
        self.calls.append((point, directory))

        return self._status


def compose(overrides: tuple[str, ...], _seed: int) -> dict[str, JsonValue]:
    resolved = a_document()

    for expression in overrides:
        key, _, text = expression.partition("=")
        *branches, leaf = key.split(".")
        node = resolved

        for branch in branches:
            child = node[branch]

            if not isinstance(child, dict):
                msg = f"{branch!r} is a leaf"
                raise TypeError(msg)

            node = child

        node[leaf] = json.loads(text)

    return resolved


def test_a_run_directory_is_named_by_the_point_identifier(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()

    assert store.root == tmp_path
    assert store.directory(point) == tmp_path / RUNS_NAME / point.point_id
    assert store.directory(point).name == point_id(point.config, seed=point.seed)
    assert store.directory(a_point(seed=1)) != store.directory(point)
    assert store.directory(a_point(epochs=2)) != store.directory(point)


def test_equal_points_from_different_sweeps_share_one_directory(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    expander = GridSweepExpander(compose)
    first = expander.expand(
        Sweep(grid=(Axis(path="trainer.epochs", values=(1, 2)),), seeds=(0,))
    )
    second = expander.expand(
        Sweep(grid=(Axis(path="trainer.epochs", values=(2, 3)),), seeds=(0,))
    )
    directories = {store.directory(point) for point in (*first, *second)}

    assert len(unique_points((*first, *second))) == 3
    assert len(directories) == 3
    assert store.directory(first[1]) == store.directory(second[0])


def test_a_fresh_point_has_no_manifest_and_is_not_completed(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)

    assert store.manifest(a_point()) is None
    assert store.is_completed(a_point()) is False
    assert not (tmp_path / RUNS_NAME).exists()


def test_executing_a_point_trains_it_in_its_directory(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    trainer = Trainer()

    outcome = store.execute(point, a_manifest(), trainer, record=a_record())

    assert outcome == RunOutcome(
        store.directory(point), ManifestStatus.COMPLETED, cached=False
    )
    assert trainer.calls == [(point, store.directory(point))]
    assert store.is_completed(point)
    assert store.manifest(point) == close_manifest(a_manifest())


def test_execution_passes_the_exact_policy_yielded_by_its_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    manifest = a_manifest()
    policy = resolve_device_policy(manifest.run_spec)
    seen: list[DevicePolicy] = []

    @contextmanager
    def exact_scope(_run_spec: RunSpec) -> Iterator[DevicePolicy]:
        yield policy

    def train(
        _point: SweepPoint,
        _directory: Path,
        _streams: RestorableRandomStreams,
        received: DevicePolicy,
    ) -> ManifestStatus:
        seen.append(received)

        return ManifestStatus.COMPLETED

    monkeypatch.setattr("lofi.run._store.device_scope", exact_scope)
    store.execute(point, manifest, train, record=a_record())

    assert seen == [policy]
    assert seen[0] is policy


def test_a_run_applies_and_supplies_its_single_policy_to_every_stream(
    tmp_path: Path,
) -> None:
    def execute(root: int, directory: Path) -> tuple[float | int, ...]:
        derived = ExperimentSeed(root).point(a_point().point_id, 0)
        policy = SeedPolicy(
            derived.root,
            samplers=("fit", "validate"),
            loaders=("rows", "batches"),
            workers=("rows", "batches"),
        )
        manifest = seeded(a_manifest(), derived, policy)
        seen: list[tuple[float | int, ...]] = []

        def train(
            _point: SweepPoint,
            _directory: Path,
            streams: RestorableRandomStreams,
            _device_policy: DevicePolicy,
        ) -> ManifestStatus:
            global_draws = (
                random.random(),
                float(torch.rand(())),
            )
            loader_draws = (
                float(torch.rand((), generator=streams.loader_generator("rows"))),
                float(torch.rand((), generator=streams.loader_generator("batches"))),
            )
            sampler_seeds = (
                streams.sampler_seed("fit"),
                streams.sampler_seed("validate"),
            )
            streams.worker_initializer("rows")(3)
            row_worker_draw = (
                random.random(),
                float(torch.rand(())),
            )
            streams.worker_initializer("batches")(3)
            batch_worker_draw = (
                random.random(),
                float(torch.rand(())),
            )
            seen.append(
                (
                    *global_draws,
                    *loader_draws,
                    *sampler_seeds,
                    *row_worker_draw,
                    *batch_worker_draw,
                )
            )

            assert dict(streams.seed_policy) == dict(policy.record)
            assert loader_draws[0] != loader_draws[1]
            assert sampler_seeds[0] != sampler_seeds[1]
            assert row_worker_draw != batch_worker_draw

            return ManifestStatus.COMPLETED

        RunStore(directory, freeze_id=None, seed=ExperimentSeed(root)).execute(
            a_point(), manifest, train, record=a_record()
        )

        return seen[0]

    first = execute(17, tmp_path / "first")
    repeated = execute(17, tmp_path / "repeated")
    changed = execute(19, tmp_path / "changed")

    assert repeated == first
    assert all(left != right for left, right in zip(first, changed, strict=True))


def test_the_running_manifest_is_on_disk_while_training_runs(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    seen: list[ExperimentManifest | None] = []

    def train(
        current: SweepPoint,
        directory: Path,
        _streams: RestorableRandomStreams,
        _policy: DevicePolicy,
    ) -> ManifestStatus:
        seen.append(store.manifest(current))
        seen.append(ManifestIso().backward((directory / MANIFEST_NAME).read_bytes()))

        return ManifestStatus.COMPLETED

    store.execute(point, a_manifest(), train, record=a_record())

    assert seen == [a_manifest(), a_manifest()]
    assert seen[0] is not None
    assert seen[0].status is ManifestStatus.RUNNING


def test_rerunning_a_completed_point_performs_no_training(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    first = Trainer()
    second = Trainer()

    store.execute(point, a_manifest(), first, record=a_record())
    outcome = store.execute(
        SweepPoint(a_point().config), a_manifest(), second, record=a_record()
    )

    assert outcome == RunOutcome(
        store.directory(point), ManifestStatus.COMPLETED, cached=True
    )
    assert second.calls == []
    assert len(first.calls) == 1


def test_a_run_refuses_a_seed_derived_for_another_point(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    trainer = Trainer()
    derived = ExperimentSeed(0).point(a_point(seed=1).point_id, 1)
    manifest = seeded(
        a_manifest(), derived, derived.policy(accelerator=None, samplers=())
    )

    with pytest.raises(ValueError, match=r"seed_derivation.*run point"):
        store.execute(a_point(), manifest, trainer, record=a_record())

    assert trainer.calls == []
    assert not store.directory(a_point()).exists()


def test_a_cached_run_refuses_a_different_seed_policy(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    store.execute(point, a_manifest(), Trainer(), record=a_record())
    derived = decode_seed_derivation(a_manifest().seed_derivation)
    changed = seeded(
        a_manifest(), derived, SeedPolicy(derived.root, loaders=("replacement",))
    )

    with pytest.raises(ValueError, match="seed_policy"):
        store.execute(point, changed, Trainer(), record=a_record())


def test_a_cached_run_refuses_a_seed_derived_from_another_experiment_root(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    store.execute(point, a_manifest(), Trainer(), record=a_record())
    derived = ExperimentSeed(1).point(point.point_id, point.seed)
    foreign = seeded(
        a_manifest(), derived, derived.policy(accelerator=None, samplers=())
    )
    trainer = Trainer()

    with pytest.raises(ValueError, match="seed_derivation"):
        store.execute(point, foreign, trainer, record=a_record())

    assert trainer.calls == []


@pytest.mark.parametrize("status", [ManifestStatus.KILLED, ManifestStatus.RUNNING])
def test_an_unfinished_run_refuses_a_seed_derived_from_another_experiment_root(
    tmp_path: Path, status: ManifestStatus
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    directory = store.directory(point)
    directory.mkdir(parents=True)
    recorded = ManifestIso().forward(a_manifest().with_(status=status))
    (directory / MANIFEST_NAME).write_bytes(recorded)
    derived = ExperimentSeed(1).point(point.point_id, point.seed)
    foreign = seeded(
        a_manifest(), derived, derived.policy(accelerator=None, samplers=())
    )
    trainer = Trainer()

    with pytest.raises(ReseededRunError, match="seed_derivation") as refused:
        store.execute(point, foreign, trainer, record=a_record())

    assert refused.value.point_id == point.point_id
    assert trainer.calls == []
    assert (directory / MANIFEST_NAME).read_bytes() == recorded


def test_a_store_derives_each_point_seed_from_its_root(tmp_path: Path) -> None:
    point = a_point(seed=2)
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(5))

    assert store.seed == ExperimentSeed(5)
    assert store.derivation(point) == ExperimentSeed(5).point(point.point_id, 2)
    assert store.derivation(point) != ROOT.point(point.point_id, 2)


@pytest.mark.parametrize("status", list(ManifestStatus))
def test_a_run_recorded_under_another_root_is_neither_served_nor_settled(
    tmp_path: Path, status: ManifestStatus
) -> None:
    point = a_point()
    recorder = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    directory = recorder.directory(point)
    directory.mkdir(parents=True)
    (directory / MANIFEST_NAME).write_bytes(
        ManifestIso().forward(a_manifest().with_(status=status))
    )
    reseeded = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(1))

    assert recorder.is_final(point) is (status is ManifestStatus.COMPLETED)

    with pytest.raises(ReseededRunError) as served:
        reseeded.is_completed(point)

    with pytest.raises(ReseededRunError) as settled:
        reseeded.is_final(point)

    assert served.value.point_id == settled.value.point_id == point.point_id


@pytest.mark.parametrize("kind", [FailureKind.NUMERICAL, FailureKind.INFRASTRUCTURE])
def test_a_failure_recorded_under_another_root_is_not_settled(
    tmp_path: Path, kind: FailureKind
) -> None:
    point = a_point()
    recorder = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    write_failure(
        recorder.directory(point),
        PointFailure(kind, "the loss is not finite", 1, recorder.derivation(point)),
    )
    reseeded = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(1))

    assert recorder.is_final(point)

    with pytest.raises(ReseededRunError):
        reseeded.is_final(point)


def test_a_fresh_directory_refuses_a_manifest_derived_from_another_root(
    tmp_path: Path,
) -> None:
    point = a_point()
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(1))
    trainer = Trainer()

    with pytest.raises(ReseededRunError):
        store.execute(point, a_manifest(), trainer, record=a_record())

    assert trainer.calls == []
    assert not store.directory(point).exists()


def test_a_run_refuses_a_seed_policy_for_a_different_accelerator(
    tmp_path: Path,
) -> None:
    derived = decode_seed_derivation(a_manifest().seed_derivation)
    manifest = seeded(
        a_manifest(), derived, SeedPolicy(derived.root, accelerator="cuda:0")
    )

    with pytest.raises(ValueError, match="seed_policy accelerator"):
        RunStore(tmp_path, freeze_id=None, seed=ROOT).execute(
            a_point(), manifest, Trainer(), record=a_record()
        )


def test_a_point_shared_by_two_overlapping_sweeps_trains_once(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    expander = GridSweepExpander(compose)
    first = expander.expand(
        Sweep(grid=(Axis(path="trainer.epochs", values=(1, 2)),), seeds=(0,))
    )
    second = expander.expand(
        Sweep(grid=(Axis(path="trainer.epochs", values=(2, 3)),), seeds=(0,))
    )
    trainer = Trainer()

    outcomes = [
        store.execute(point, a_manifest_for(point), trainer, record=a_record())
        for point in (*first, *second)
    ]

    assert [outcome.cached for outcome in outcomes] == [False, False, True, False]
    assert [point for point, _ in trainer.calls] == [*first, second[1]]
    assert len({outcome.directory for outcome in outcomes}) == 3


@pytest.mark.parametrize("status", [ManifestStatus.KILLED, ManifestStatus.RUNNING])
def test_a_point_not_closed_as_completed_trains_again(
    tmp_path: Path, status: ManifestStatus
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    directory = store.directory(point)
    directory.mkdir(parents=True)
    (directory / MANIFEST_NAME).write_bytes(
        ManifestIso().forward(a_manifest().with_(status=status))
    )
    trainer = Trainer()

    outcome = store.execute(point, a_manifest(), trainer, record=a_record())

    assert trainer.calls == [(point, directory)]
    assert outcome.cached is False
    assert store.is_completed(point)


def test_a_killed_training_closes_the_manifest_as_killed(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()

    outcome = store.execute(
        point, a_manifest(), Trainer(ManifestStatus.KILLED), record=a_record()
    )
    manifest = store.manifest(point)

    assert outcome.status is ManifestStatus.KILLED
    assert outcome.cached is False
    assert manifest is not None
    assert manifest.status is ManifestStatus.KILLED
    assert store.is_completed(point) is False


def test_a_run_refuses_to_open_with_a_closed_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    trainer = Trainer()

    with pytest.raises(ValueError, match="running manifest"):
        store.execute(
            a_point(), close_manifest(a_manifest()), trainer, record=a_record()
        )

    assert trainer.calls == []
    assert store.manifest(a_point()) is None


def test_a_training_that_reports_running_is_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()

    with pytest.raises(ValueError, match="cannot remain running"):
        store.execute(
            point, a_manifest(), Trainer(ManifestStatus.RUNNING), record=a_record()
        )

    assert store.is_completed(point) is False


def test_identity_is_invariant_under_refactoring_one_fragment_into_two(
    tmp_path: Path,
) -> None:
    class ModelSection(dx.Model, extra="forbid"):
        width: int
        depth: int

    class TrainerSection(dx.Model, extra="forbid"):
        epochs: int

    class Schema(dx.Model, extra="forbid"):
        model: ModelSection
        trainer: TrainerSection

    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "run.json").write_text(
        json.dumps(
            {"trainer": {"epochs": 3}, "model": {"depth": 2, "width": 64}},
        ),
        encoding="utf-8",
        newline="\n",
    )
    (two / "run.json").write_text(
        json.dumps({"defaults": ["base"], "trainer": {"epochs": 3}}),
        encoding="utf-8",
        newline="\n",
    )
    (two / "base.json").write_text(
        json.dumps({"model": {"width": 64, "depth": 2}}), encoding="utf-8", newline="\n"
    )
    loader = SpecLoader()
    whole = loader.compose_traced(one / "run.json", schema=Schema).spec
    split = loader.compose_traced(two / "run.json", schema=Schema).spec
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    resolved = (
        SweepPoint(json.loads(whole.model_dump_json()), overrides=()),
        SweepPoint(json.loads(split.model_dump_json()), overrides=("defaults=base",)),
    )

    assert resolved[0].point_id == resolved[1].point_id
    assert store.directory(resolved[0]) == store.directory(resolved[1])
    assert (
        resolved[0].point_id
        != SweepPoint(
            json.loads(whole.with_(trainer=TrainerSection(epochs=4)).model_dump_json())
        ).point_id
    )


def test_a_fresh_point_holds_no_record(tmp_path: Path) -> None:
    assert RunStore(tmp_path, freeze_id=None, seed=ROOT).record(a_point()) is None


def test_executing_a_point_records_the_three_files_beside_the_manifest(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point(epochs=2)
    manifest = a_manifest(epochs=2)

    store.execute(point, manifest, Trainer(), record=a_record(epochs=2))
    directory = store.directory(point)

    assert sorted(path.name for path in directory.iterdir()) == sorted(
        (
            MANIFEST_NAME,
            RESOLVED_NAME,
            SCHEMA_FINGERPRINT_NAME,
            SOURCES_NAME,
        )
    )
    stored = store.record(point)
    assert stored == read_record(directory)
    assert stored == StoredRecord(
        manifest.run_spec,
        a_provenance(epochs=2),
        a_fingerprint(),
    )
    assert stored.resolved.trainer.model_dump()["epochs"] == 2
    assert stored.fingerprint.digest == a_fingerprint().digest


def test_the_record_is_on_disk_before_training_starts(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    seen: list[StoredRecord | None] = []

    def train(
        current: SweepPoint,
        _directory: Path,
        _streams: RestorableRandomStreams,
        _policy: DevicePolicy,
    ) -> ManifestStatus:
        seen.append(store.record(current))

        return ManifestStatus.COMPLETED

    store.execute(point, a_manifest(), train, record=a_record())

    assert seen == [store.record(point)]
    assert seen[0] is not None


def test_a_manifest_missing_a_recorded_distribution_opens_no_run(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    trainer = Trainer()
    manifest = a_manifest().with_(package_versions={"torch": "2.14.0"})

    with pytest.raises(
        ValueError, match="didactic, lairs, numpy, panproto, python, quivers"
    ):
        store.execute(point, manifest, trainer, record=a_record())

    assert trainer.calls == []
    assert store.manifest(point) is None
    assert not store.directory(point).exists()


def test_an_interrupted_training_closes_the_manifest_as_killed(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()

    def interrupted(
        _point: SweepPoint,
        _directory: Path,
        _streams: RestorableRandomStreams,
        _policy: DevicePolicy,
    ) -> ManifestStatus:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.execute(point, a_manifest(), interrupted, record=a_record())

    manifest = store.manifest(point)
    assert manifest is not None
    assert manifest.status is ManifestStatus.KILLED
    assert manifest != close_manifest(a_manifest())
    assert store.is_completed(point) is False


def test_a_training_that_raises_closes_the_manifest_as_killed_and_propagates(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()

    def failing(
        _point: SweepPoint,
        _directory: Path,
        _streams: RestorableRandomStreams,
        _policy: DevicePolicy,
    ) -> ManifestStatus:
        msg = "the loss is not finite"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="not finite"):
        store.execute(point, a_manifest(), failing, record=a_record())

    manifest = store.manifest(point)
    assert manifest is not None
    assert manifest.status is ManifestStatus.KILLED
    assert store.record(point) is not None


def test_a_killed_point_trains_again_and_its_record_is_rewritten(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()

    store.execute(
        point, a_manifest(), Trainer(ManifestStatus.KILLED), record=a_record()
    )
    outcome = store.execute(
        point,
        a_manifest(),
        Trainer(),
        record=RunRecord(
            provenance=a_provenance(),
            fingerprint=a_fingerprint(),
            declarations={"grammar": DECLARATION},
        ),
    )
    stored = store.record(point)

    assert outcome.status is ManifestStatus.COMPLETED
    assert outcome.cached is False
    assert stored == read_record(store.directory(point))


KILLED_BY_SIGNAL: Final = """
import json
import signal
import sys
import time
from pathlib import Path
from typing import Literal

from lofi.run import (
    ExperimentSeed,
    ManifestStatus,
    RunStore,
    SweepPoint,
    open_manifest,
)
from lofi.run._manifest import RunRecord
from lofi.spec import (
    ConfigSource,
    DatasetSpec,
    FeaturizerSpec,
    LeafOrigin,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    Provenance,
    RunFingerprint,
    RunSpec,
    TrainerSpec,
)
from lofi.train import RestorableRandomStreams

from tests.interrupts import INTERRUPT

# An ignored interrupt is inherited through exec and CPython leaves it
# ignored, so this child sets the disposition rather than taking it from
# whoever spawned it.
signal.signal(INTERRUPT, signal.default_int_handler)


class SignalParser(ParserSpec):
    kind: Literal["store-unit-parser"] = "store-unit-parser"


class SignalLoss(LossSpec):
    kind: Literal["store-unit-loss"] = "store-unit-loss"


class SignalDataset(DatasetSpec):
    kind: Literal["store-unit-dataset"] = "store-unit-dataset"


class SignalFeaturizer(FeaturizerSpec):
    kind: Literal["store-unit-featurizer"] = "store-unit-featurizer"


class SignalTrainer(TrainerSpec):
    kind: Literal["store-unit-trainer"] = "store-unit-trainer"


class SignalOptimizer(OptimizerSpec):
    kind: Literal["store-unit-optimizer"] = "store-unit-optimizer"


root = Path(sys.argv[1])
document = {
    "parse": {"retain": "forest", "reachability": "all"},
    "model": {
        "language": {"from": "data"},
        "parser": {"kind": "store-unit-parser"},
    },
    "loss": {"terms": [{"objective": {"kind": "store-unit-loss"}}]},
    "data": {
        "datasets": {
            "train": {
                "dataset": {"kind": "store-unit-dataset"},
                "featurizer": {"kind": "store-unit-featurizer"},
            }
        }
    },
    "trainer": {"kind": "store-unit-trainer"},
    "optimizer": {"kind": "store-unit-optimizer"},
    "device_policy": {
        "device": "cpu",
        "compute_dtype": "float32",
        "deterministic": True,
        "matmul_precision": "highest",
        "activation_memory_bytes": 8_000_000_000,
    },
    "release": {"checkpoint": "resume-latest"},
}
packages = dict.fromkeys(
    (
        "didactic",
        "lairs",
        "numpy",
        "panproto",
        "python",
        "quivers",
        "torch",
        "transformers",
    ),
    "1",
)
derived = ExperimentSeed(0).point(SweepPoint(document).point_id, 0)
manifest = open_manifest(
    run_spec=RunSpec.model_validate(document),
    component_fingerprints={
        role: "fingerprint"
        for role in RunSpec.model_validate(document).component_roles()
    },
    grammar_fingerprint="grammar",
    signature_fingerprint="signature",
    corpus_revision="corpus",
    seed_policy=derived.policy(accelerator=None, samplers=()),
    seed_derivation=derived,
    package_versions=packages,
    git_revision="revision",
    accelerator="cpu",
)
record = RunRecord(
    provenance=Provenance(
        origins={"trainer.kind": LeafOrigin(source=ConfigSource.FILE, layer="run")}
    ),
    fingerprint=RunFingerprint({"run": "urn:store-signal"}, {}, packages),
    declarations={},
)


def train(
    point: SweepPoint,
    directory: Path,
    _streams: RestorableRandomStreams,
    _policy: DevicePolicy,
) -> ManifestStatus:
    (directory / "training").write_text(str(point.seed))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        time.sleep(0.05)

    return ManifestStatus.COMPLETED


store = RunStore(root, freeze_id=None, seed=ExperimentSeed(0))
store.execute(SweepPoint(document), manifest, train, record=record)
print(json.dumps({"point_id": SweepPoint(document).point_id}))
"""


def _wait_for_training(runs: Path, *, seconds: float) -> Path:
    deadline = time.monotonic() + seconds

    while True:
        marker = next(runs.glob("*/training"), None)

        if marker is not None:
            return marker.parent

        if time.monotonic() > deadline:
            pytest.fail(f"Training did not start in {runs} within {seconds} seconds")

        time.sleep(0.05)


def _terminate(process: subprocess.Popen[bytes], /) -> None:
    process.terminate()


def _interrupt(process: subprocess.Popen[bytes], /) -> None:
    interrupt(process)


@pytest.mark.parametrize(
    ("stop", "expected"),
    [(_terminate, ManifestStatus.RUNNING), (_interrupt, ManifestStatus.KILLED)],
)
def test_a_run_stopped_by_a_signal_leaves_a_manifest_that_is_not_completed(
    tmp_path: Path,
    stop: Callable[[subprocess.Popen[bytes]], None],
    expected: ManifestStatus,
) -> None:
    runs = tmp_path / RUNS_NAME
    runs.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-c", KILLED_BY_SIGNAL, str(tmp_path)],
        cwd=TREE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NEW_GROUP,
    )

    try:
        directory = _wait_for_training(runs, seconds=60)
        stop(process)
        process.wait(timeout=60)

    finally:
        process.kill()
        process.wait()

    manifest = ManifestIso().backward((directory / MANIFEST_NAME).read_bytes())

    assert process.returncode != 0
    assert manifest.status is expected
    assert manifest.status is not ManifestStatus.COMPLETED
    assert (directory / RESOLVED_NAME).is_file()
    assert (directory / SOURCES_NAME).is_file()
    assert (directory / SCHEMA_FINGERPRINT_NAME).is_file()


class TestStoredManifest(ExperimentManifestConformance):
    @pytest.fixture
    def manifest_fixture(self, tmp_path: Path) -> ManifestFixture:
        point = a_point()

        def close(
            manifest: ExperimentManifest, status: ManifestStatus
        ) -> ExperimentManifest:
            store = RunStore(tmp_path / status.value, freeze_id=None, seed=ROOT)
            store.execute(point, manifest, Trainer(status), record=a_record())
            stored = store.manifest(point)

            if stored is None:
                pytest.fail("the store recorded no manifest")

            return stored

        return ManifestFixture(a_manifest, close)


def languages() -> dict[str, ArchivedLanguage]:
    return {
        "grammar": ArchivedLanguage(DECLARATION, a_fingerprint().languages["grammar"])
    }


def executed(store: RunStore, point: SweepPoint) -> Path:
    store.execute(point, a_manifest_for(point), Trainer(), record=a_record())

    return store.directory(point)


def recorded(store: RunStore, point: SweepPoint) -> Path:
    """Return a run directory holding its files but neither trained nor committed."""
    directory = store.directory(point)
    write_record(directory, a_manifest(), a_record())
    write_manifest(directory, a_manifest())

    return directory


def test_the_store_history_is_kept_at_the_runs_root(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)

    assert store.history().root == tmp_path / RUNS_NAME


def test_an_executed_run_commits_its_persisted_bytes(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    directory = executed(store, point)
    history = store.history()
    replayed = history.replay(point.point_id)
    (commit,) = history.committed().values()

    assert history.commit(store.committable(point, languages())) == commit
    assert history.committed() == {point.point_id: commit}
    assert replayed.point_id == point.point_id
    assert replayed.configuration == (directory / RESOLVED_NAME).read_bytes()
    assert replayed.fingerprint == (directory / SCHEMA_FINGERPRINT_NAME).read_bytes()
    assert replayed.languages["grammar"].theory.specification == GRAMMAR_SPEC
    assert replayed.boundary == a_fingerprint()
    assert store.verify(point, history) == replayed


def test_a_run_directory_without_its_files_is_not_committable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RunStore(tmp_path, freeze_id=None, seed=ROOT).committable(
            a_point(), languages()
        )


def test_languages_the_fingerprint_does_not_name_are_not_committable(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    executed(store, point)

    with pytest.raises(ValueError, match="commits languages"):
        store.committable(point, {})


def test_verifying_an_uncommitted_point_is_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    recorded(store, point)

    with pytest.raises(UncommittedRunError):
        store.verify(point, store.history())


@pytest.mark.parametrize(
    "status", [ManifestStatus.KILLED, ManifestStatus.COMPLETED], ids=str
)
def test_only_a_completed_run_is_committed(
    tmp_path: Path, status: ManifestStatus
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    store.execute(point, a_manifest(), Trainer(status), record=a_record())
    committed = store.history().committed()

    if status is ManifestStatus.COMPLETED:
        assert list(committed) == [point.point_id]
        assert store.verify(point, store.history()).languages == languages()
    else:
        assert committed == {}


def test_a_cached_run_adds_no_commit(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    executed(store, point)
    committed = store.history().committed()
    outcome = store.execute(point, a_manifest(), Trainer(), record=a_record())

    assert outcome.cached is True
    assert store.history().committed() == committed
    assert len(dx.Repository.open(tmp_path / RUNS_NAME).log()) == 1


def test_a_run_whose_commit_is_refused_is_left_running(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    directory = recorded(store, point)
    resolved = directory / RESOLVED_NAME
    original = resolved.read_bytes()
    resolved.write_bytes(original.replace(b'"epochs": 1', b'"epochs": 1 '))
    store.history().commit(store.committable(point, languages()))
    resolved.write_bytes(original)

    with pytest.raises(ValueError, match="different record"):
        store.execute(point, a_manifest(), Trainer(), record=a_record())

    manifest = store.manifest(point)

    assert manifest is not None
    assert manifest.status is ManifestStatus.RUNNING


def test_a_directory_altered_after_its_commit_is_refused_and_left_as_found(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    directory = executed(store, point)
    history = store.history()
    history.commit(store.committable(point, languages()))
    committed = history.committed()
    resolved = directory / RESOLVED_NAME
    altered = resolved.read_bytes().replace(b'"epochs": 1', b'"epochs": 2')
    resolved.write_bytes(altered)

    with pytest.raises(TamperedRecordError) as refused:
        store.verify(point, history)

    assert refused.value.subject == "configuration"
    assert refused.value.computed == sha256(altered).hexdigest()
    assert resolved.read_bytes() == altered
    assert history.committed() == committed
    assert history.replay(point.point_id).configuration != altered


def test_an_altered_fingerprint_file_is_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = a_point()
    directory = executed(store, point)
    history = store.history()
    history.commit(store.committable(point, languages()))
    stored = directory / SCHEMA_FINGERPRINT_NAME
    stored.write_bytes(stored.read_bytes() + b"\n")

    with pytest.raises(TamperedRecordError) as refused:
        store.verify(point, history)

    assert refused.value.subject == "fingerprint"


def test_a_refused_check_leaves_later_writes_and_commits_unchanged(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "checked", freeze_id=None, seed=ROOT)
    first = a_point()
    directory = executed(store, first)
    history = store.history()
    history.commit(store.committable(first, languages()))
    (directory / RESOLVED_NAME).write_bytes(b"{}")

    with pytest.raises(TamperedRecordError):
        store.verify(first, history)

    second = a_point(epochs=2)
    executed(store, second)
    history.commit(store.committable(second, languages()))
    control = RunStore(tmp_path / "control", freeze_id=None, seed=ROOT)
    executed(control, second)
    control.history().commit(control.committable(second, languages()))

    assert history.replay(second.point_id) == control.history().replay(second.point_id)
    assert store.verify(second, history) == history.replay(second.point_id)


def test_a_point_bound_under_another_experiment_root_of_its_freeze_is_refused(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path, freeze_id="freeze/1", seed=ROOT)
    point = SweepPoint({"width": 64}, seed=0)
    store.claim(point)

    with pytest.raises(ReseededRunError):
        RunStore(tmp_path, freeze_id="freeze/1", seed=ExperimentSeed(8)).claim(point)

    bound = read_binding(store.directory(point))
    assert bound is not None
    assert bound.freeze_id == "freeze/1"
    assert bound.seed_derivation == store.derivation(point)


def test_a_store_addressed_by_a_freeze_keeps_its_directories_under_it(
    tmp_path: Path,
) -> None:
    point = SweepPoint({"width": 64}, seed=0)
    first = RunStore(tmp_path, freeze_id="freeze/1", seed=ROOT)
    second = RunStore(tmp_path, freeze_id="freeze/7", seed=ROOT)
    unscoped = RunStore(tmp_path, freeze_id=None, seed=ROOT)

    assert first.scope == tmp_path / "freeze" / "1"
    assert first.directory(point) == first.scope / RUNS_NAME / point.point_id
    assert unscoped.directory(point) == tmp_path / RUNS_NAME / point.point_id
    assert len({first.directory(point), second.directory(point)}) == 2
    assert unscoped.directory(point) not in {
        first.directory(point),
        second.directory(point),
    }


def test_claiming_a_point_for_another_freeze_leaves_the_first_freezes_run(
    tmp_path: Path,
) -> None:
    point = SweepPoint({"width": 64}, seed=0)
    first = RunStore(tmp_path, freeze_id="freeze/1", seed=ROOT)
    first.claim(point)
    held = first.directory(point) / "checkpoints"
    held.mkdir()

    second = RunStore(tmp_path, freeze_id="freeze/7", seed=ExperimentSeed(8))
    second.claim(point)

    bound = read_binding(first.directory(point))
    claimed = read_binding(second.directory(point))
    assert held.is_dir()
    assert bound is not None
    assert bound.freeze_id == "freeze/1"
    assert claimed is not None
    assert claimed.freeze_id == "freeze/7"


def test_a_store_addressed_by_no_freeze_binds_no_point(tmp_path: Path) -> None:
    point = SweepPoint({"width": 64}, seed=0)

    with pytest.raises(ValueError, match="binds no point"):
        RunStore(tmp_path, freeze_id=None, seed=ROOT).claim(point)

    assert not RunStore(tmp_path, freeze_id=None, seed=ROOT).directory(point).exists()


def test_a_store_refuses_a_malformed_freeze_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="freeze/<n>"):
        RunStore(tmp_path, freeze_id="../escape", seed=ROOT)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, RunStatus.SUBMITTED),
        (ManifestStatus.RUNNING, RunStatus.RUNNING),
        (ManifestStatus.COMPLETED, RunStatus.COMPLETED),
        (ManifestStatus.KILLED, RunStatus.FAILED),
    ],
)
def test_a_run_store_reports_the_status_its_manifest_records(
    tmp_path: Path, status: ManifestStatus | None, expected: RunStatus
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = launcher.a_point(1)

    if status is not None:
        launcher.record_manifest(store, point, status)

    assert store.status(point) is expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (FailureKind.INTERRUPTED, RunStatus.RUNNING),
        (FailureKind.INFRASTRUCTURE, RunStatus.FAILED),
        (FailureKind.NUMERICAL, RunStatus.FAILED),
        (FailureKind.PREPARATION, RunStatus.FAILED),
    ],
)
def test_a_run_left_running_reports_the_failure_that_settles_it(
    tmp_path: Path, kind: FailureKind, expected: RunStatus
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ROOT)
    point = launcher.a_point(1)
    launcher.record_manifest(store, point, ManifestStatus.RUNNING)
    write_failure(
        store.directory(point),
        PointFailure(
            kind, "the attempt reached no outcome", 1, store.derivation(point)
        ),
    )

    assert store.status(point) is expected


def _pilot_points(root: Path) -> tuple[Experiment, tuple[SweepPoint, ...]]:
    experiment = read_experiment(write_experiment(root, an_experiment_document()))

    return experiment, expand_experiment(experiment, root)


def test_a_pilot_store_binds_its_points_outside_any_freeze(tmp_path: Path) -> None:
    experiment, points = _pilot_points(tmp_path / "repository")
    store = ExperimentStore(tmp_path / "pilot", experiment, pilot=True)
    point = points[0]

    store.claim(point)

    assert store.pilot
    assert store.scope == tmp_path / "pilot"
    assert store.runs.freeze_id is None
    assert store.freeze_id == UNFROZEN
    assert store.identity == ExperimentIdentity(experiment.name, UNFROZEN)
    assert store.directory(point) == tmp_path / "pilot" / RUNS_NAME / point.point_id
    assert read_binding(store.directory(point)) == store.binding(point)
    assert store.binding(point).freeze_id == UNFROZEN
    assert store.entry(point) == {"status": "submitted"}
    assert store.entry(points[1]) == {"status": "not-submitted"}


def test_a_pilot_point_completes_on_rows_recorded_outside_any_freeze(
    tmp_path: Path,
) -> None:
    experiment, points = _pilot_points(tmp_path / "repository")
    store = ExperimentStore(tmp_path / "pilot", experiment, pilot=True)
    point = points[0]
    store.claim(point)
    launcher.record_manifest(store.runs, point, ManifestStatus.COMPLETED)
    derived = store.binding(point).seed_derivation

    write_rows(store.directory(point), RecordedRows("freeze/1", derived, ()))

    assert store.entry(point) == {"status": "killed"}

    write_rows(store.directory(point), RecordedRows(None, derived, ()))

    assert store.is_completed(point)


def test_a_frozen_store_does_not_see_a_pilot_point(tmp_path: Path) -> None:
    experiment, points = _pilot_points(tmp_path / "repository")
    pilot = ExperimentStore(tmp_path / "store", experiment, pilot=True)
    frozen = ExperimentStore(tmp_path / "store", experiment)
    pilot.claim(points[0])

    assert not frozen.pilot
    assert frozen.freeze_id == experiment.freeze_id
    assert frozen.directory(points[0]) != pilot.directory(points[0])
    assert frozen.entry(points[0]) == {"status": "not-submitted"}
