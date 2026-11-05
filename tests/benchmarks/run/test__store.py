from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal

import didactic.api as dx
import pytest

from lofi.run import (
    ExperimentSeed,
    ManifestStatus,
    RunStore,
    SweepPoint,
    open_manifest,
)
from lofi.run._manifest import RunRecord, read_manifest, write_manifest, write_record
from lofi.spec import (
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
    TrainerSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi._types import JsonValue
    from lofi.run import ExperimentManifest

# the P-38 budget: one experiment-manifest operation, of which a run performs two
BUDGET_MS: Final = 100.0


class BenchParser(ParserSpec):
    kind: Literal["bench-store-parser"] = "bench-store-parser"


class BenchLoss(LossSpec):
    kind: Literal["bench-store-loss"] = "bench-store-loss"


class BenchDataset(DatasetSpec):
    kind: Literal["bench-store-dataset"] = "bench-store-dataset"


class BenchFeaturizer(FeaturizerSpec):
    kind: Literal["bench-store-featurizer"] = "bench-store-featurizer"


class BenchTrainer(TrainerSpec):
    kind: Literal["bench-store-trainer"] = "bench-store-trainer"
    epochs: int = 1


class BenchOptimizer(OptimizerSpec):
    kind: Literal["bench-store-optimizer"] = "bench-store-optimizer"


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
    b'{"route":{"source":"decl-1","target":"native-1"},"theory":"S NP S\\\\NP"}'
)


def _document() -> dict[str, JsonValue]:
    return {
        "parse": {"retain": "forest", "reachability": "all"},
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "bench-store-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "bench-store-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "bench-store-dataset"},
                    "featurizer": {"kind": "bench-store-featurizer"},
                }
            }
        },
        "trainer": {"kind": "bench-store-trainer", "epochs": 3},
        "release": {"checkpoint": "resume-latest"},
        "optimizer": {"kind": "bench-store-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
    }


def _manifest() -> ExperimentManifest:
    derived = ExperimentSeed(7).point(SweepPoint(_document()).point_id, 0)

    return open_manifest(
        run_spec=RunSpec.model_validate(_document()),
        component_fingerprints={
            role: sha256(role.encode()).hexdigest()
            for role in RunSpec.model_validate(_document()).component_roles()
        },
        grammar_fingerprint=sha256(GRAMMAR_SPEC).hexdigest(),
        signature_fingerprint="signature:benchmark",
        corpus_revision="ccgbank:LDC2005T13:auto-v1",
        seed_policy=derived.policy(accelerator=None, samplers=()),
        seed_derivation=derived,
        package_versions=PACKAGES,
        git_revision="0123456789abcdef0123456789abcdef01234567",
        accelerator="cpu",
    )


def _record() -> RunRecord:
    origins = {
        path: LeafOrigin(source=ConfigSource.FILE, layer="run.json")
        for path in (
            "model.parser.kind",
            "loss.terms.0.objective.kind",
            "data.datasets.train.dataset.kind",
            "data.datasets.train.featurizer.kind",
            "trainer.kind",
            "trainer.epochs",
            "optimizer.kind",
            "device_policy.device",
            "device_policy.compute_dtype",
            "device_policy.deterministic",
            "device_policy.matmul_precision",
        )
    }

    return RunRecord(
        provenance=Provenance(origins=origins),
        fingerprint=RunFingerprint(
            {"run": "urn:bench-store:run-spec"},
            {
                "grammar": IndexedLanguage(
                    "decl-1->native-1",
                    GRAMMAR_SPEC,
                    sha256(GRAMMAR_SPEC).hexdigest(),
                    dx.LoweringRoute("decl-1", "native-1"),
                )
            },
            PACKAGES,
        ),
        declarations={"grammar": b'{"name":"bench-grammar"}'},
    )


def _median_ms(benchmark: BenchmarkFixture) -> float:
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the manifest operation ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "P-38"
    benchmark.extra_info["budget_ms"] = BUDGET_MS
    benchmark.extra_info["median_ms"] = median_ms

    return median_ms


def _open_run(directory: Path, manifest: ExperimentManifest, record: RunRecord) -> None:
    write_record(directory, manifest, record)
    write_manifest(directory, manifest)


def _close_run(directory: Path, manifest: ExperimentManifest) -> ManifestStatus | None:
    write_manifest(directory, manifest.with_(status=ManifestStatus.COMPLETED))
    stored = read_manifest(directory)

    return None if stored is None else stored.status


def test_opening_a_run_persists_its_manifest_and_record(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(7))
    point = SweepPoint(_document())
    manifest = _manifest()
    record = _record()

    _open_run(store.directory(point), manifest, record)

    assert store.manifest(point) == manifest
    assert store.record(point) is not None


def test_opening_a_run_meets_its_budget(
    benchmark: BenchmarkFixture, tmp_path: Path, wall_clock_budget: WallClockBudget
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(7))
    point = SweepPoint(_document())
    manifest = _manifest()
    record = _record()
    directory = store.directory(point)

    benchmark.pedantic(
        _open_run,
        args=(directory, manifest, record),
        rounds=50,
        warmup_rounds=2,
        iterations=1,
    )

    if wall_clock_budget.enforced:
        assert _median_ms(benchmark) <= BUDGET_MS


def test_closing_a_run_completes_it(tmp_path: Path) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(7))
    point = SweepPoint(_document())
    manifest = _manifest()
    directory = store.directory(point)
    _open_run(directory, manifest, _record())

    status = _close_run(directory, manifest)

    assert status is ManifestStatus.COMPLETED
    assert store.is_completed(point)


def test_closing_a_run_meets_its_budget(
    benchmark: BenchmarkFixture, tmp_path: Path, wall_clock_budget: WallClockBudget
) -> None:
    store = RunStore(tmp_path, freeze_id=None, seed=ExperimentSeed(7))
    point = SweepPoint(_document())
    manifest = _manifest()
    directory = store.directory(point)
    _open_run(directory, manifest, _record())
    benchmark.pedantic(
        _close_run,
        args=(directory, manifest),
        rounds=50,
        warmup_rounds=2,
        iterations=1,
    )

    if wall_clock_budget.enforced:
        assert _median_ms(benchmark) <= BUDGET_MS
