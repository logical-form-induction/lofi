from typing import Literal

import pytest
from tests.conformance.experiment_manifest import (
    EXPERIMENT_MANIFEST_CONFORMANCE_LAWS,
    EXPERIMENT_MANIFEST_LAWS,
    ExperimentManifestConformance,
    ManifestFixture,
)

from lofi.run import (
    ExperimentManifest,
    ExperimentSeed,
    ManifestStatus,
    close_manifest,
    open_manifest,
)
from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    TrainerSpec,
)


class ConformanceParser(ParserSpec):
    kind: Literal["manifest-conformance-parser"] = "manifest-conformance-parser"


class ConformanceLoss(LossSpec):
    kind: Literal["manifest-conformance-loss"] = "manifest-conformance-loss"


class ConformanceDataset(DatasetSpec):
    kind: Literal["manifest-conformance-dataset"] = "manifest-conformance-dataset"


class ConformanceFeaturizer(FeaturizerSpec):
    kind: Literal["manifest-conformance-featurizer"] = "manifest-conformance-featurizer"


class ConformanceTrainer(TrainerSpec):
    kind: Literal["manifest-conformance-trainer"] = "manifest-conformance-trainer"


class ConformanceOptimizer(OptimizerSpec):
    kind: Literal["manifest-conformance-optimizer"] = "manifest-conformance-optimizer"


def a_manifest() -> ExperimentManifest:
    run_spec = RunSpec.model_validate(
        {
            "model": {
                "language": {"from": "data"},
                "parser": {"kind": "manifest-conformance-parser"},
            },
            "loss": {"terms": [{"objective": {"kind": "manifest-conformance-loss"}}]},
            "data": {
                "datasets": {
                    "train": {
                        "dataset": {"kind": "manifest-conformance-dataset"},
                        "featurizer": {"kind": "manifest-conformance-featurizer"},
                    }
                }
            },
            "trainer": {"kind": "manifest-conformance-trainer"},
            "optimizer": {"kind": "manifest-conformance-optimizer"},
            "parse": {"retain": "forest", "reachability": "all"},
            "release": {"checkpoint": "resume-latest"},
            "device_policy": {
                "device": "cpu",
                "compute_dtype": "float32",
                "deterministic": True,
                "matmul_precision": "highest",
                "activation_memory_bytes": 8_000_000_000,
            },
        }
    )
    derived = ExperimentSeed(7).point("0123456789abcdef", 0)

    return open_manifest(
        run_spec=run_spec,
        component_fingerprints={
            role: f"fingerprint:{role}" for role in run_spec.component_roles()
        },
        grammar_fingerprint="grammar:test",
        signature_fingerprint="signature:test",
        corpus_revision="corpus:test",
        seed_policy=derived.policy(accelerator=None, samplers=()),
        seed_derivation=derived,
        package_versions={"didactic": "0.0.2"},
        git_revision="0123456789abcdef",
        accelerator="cpu",
    )


class NeverClosingManifestPath:
    @staticmethod
    def open() -> ExperimentManifest:
        return a_manifest()

    @staticmethod
    def close(
        manifest: ExperimentManifest, _status: ManifestStatus
    ) -> ExperimentManifest:
        return manifest


class TestExperimentManifestPath(ExperimentManifestConformance):
    @pytest.fixture
    def manifest_fixture(self) -> ManifestFixture:
        return ManifestFixture(a_manifest, close_manifest)


@pytest.mark.xfail(strict=True, reason="the manifest never closes")
class TestNeverClosingManifestPath(ExperimentManifestConformance):
    @pytest.fixture
    def manifest_fixture(self) -> ManifestFixture:
        stale = NeverClosingManifestPath()

        return ManifestFixture(stale.open, stale.close)


def test_the_suite_binds_one_method_to_each_manifest_law() -> None:
    bound = {
        method.__name__.removeprefix("test_")
        for method in EXPERIMENT_MANIFEST_CONFORMANCE_LAWS
    }

    assert bound == {law.__name__ for law in EXPERIMENT_MANIFEST_LAWS}
