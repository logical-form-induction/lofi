from typing import Literal

from lofi.spec import (
    DatasetSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    SinkSpec,
    TermEncoderSpec,
    TrainerSpec,
)


class ManifestParser(ParserSpec):
    kind: Literal["manifest-unit-parser"] = "manifest-unit-parser"


class ManifestLoss(LossSpec):
    kind: Literal["manifest-unit-loss"] = "manifest-unit-loss"


class ManifestDataset(DatasetSpec):
    kind: Literal["manifest-unit-dataset"] = "manifest-unit-dataset"


class ManifestFeaturizer(FeaturizerSpec):
    kind: Literal["manifest-unit-featurizer"] = "manifest-unit-featurizer"


class ManifestTrainer(TrainerSpec):
    kind: Literal["manifest-unit-trainer"] = "manifest-unit-trainer"


class ManifestOptimizer(OptimizerSpec):
    kind: Literal["manifest-unit-optimizer"] = "manifest-unit-optimizer"


class ManifestSink(SinkSpec):
    kind: Literal["manifest-unit-sink"] = "manifest-unit-sink"


class MeasuredTermEncoder(TermEncoderSpec):
    kind: Literal["measured-unit-encoder"] = "measured-unit-encoder"
    width: int


class WidthlessTermEncoder(TermEncoderSpec):
    kind: Literal["widthless-unit-encoder"] = "widthless-unit-encoder"


class NamedWidthTermEncoder(TermEncoderSpec):
    kind: Literal["named-width-unit-encoder"] = "named-width-unit-encoder"
    width: str


ROLES = frozenset(
    {
        "model.parser",
        "loss.terms.0",
        "data.datasets.train.dataset",
        "data.datasets.train.featurizer",
        "training.trainer",
        "training.optimizer",
        "output.sinks.metrics",
    }
)


def a_run_spec() -> RunSpec:
    return RunSpec.model_validate(
        {
            "model": {
                "language": {"from": "data"},
                "parser": {"kind": "manifest-unit-parser"},
            },
            "loss": {"terms": [{"objective": {"kind": "manifest-unit-loss"}}]},
            "data": {
                "datasets": {
                    "train": {
                        "dataset": {"kind": "manifest-unit-dataset"},
                        "featurizer": {"kind": "manifest-unit-featurizer"},
                    }
                }
            },
            "trainer": {"kind": "manifest-unit-trainer"},
            "optimizer": {"kind": "manifest-unit-optimizer"},
            "parse": {"retain": "forest", "reachability": "all"},
            "release": {"checkpoint": "resume-latest"},
            "device_policy": {
                "device": "cpu",
                "compute_dtype": "float32",
                "deterministic": True,
                "matmul_precision": "highest",
                "activation_memory_bytes": 8_000_000_000,
            },
            "sinks": {"metrics": {"kind": "manifest-unit-sink"}},
        }
    )
