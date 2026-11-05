from typing import TYPE_CHECKING, Literal, override

import didactic.api as dx

from lofi.spec.components.data import DatasetSpec, FeaturizerSpec
from lofi.spec.components.loss import LossSection, LossSpec, WeightedLossSpec
from lofi.spec.components.model import (
    ControllerSpec,
    DataModelLanguageSpec,
    ModelSection,
    ParserSpec,
    TermDecoderSpec,
    TermEncoderSpec,
)
from lofi.spec.components.run import (
    BootstrapIntervalSpec,
    EvaluatorSpec,
    ProbeSpec,
    RunSpec,
    SinkSpec,
)
from lofi.spec.components.trainer import (
    OptimizerSpec,
    PairSamplerSpec,
    SchedulerSpec,
    TrainerSpec,
    TypeSamplerSpec,
)
from lofi.spec.outputs import DeclaredOutput, DtypeClass
from lofi.spec.readouts import ReadoutSource, TargetSource, TermsReadoutSpec
from lofi.spec.spaces import CombinatorFamilySpec, Merge, SpaceSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


class SizedEncoder(TermEncoderSpec):
    kind: Literal["sized"] = "sized"
    width: int = 32

    @dx.computed
    @override
    def declared_output(self) -> DeclaredOutput:
        return DeclaredOutput(width=self.width, rank=2, dtype_class=DtypeClass.REAL)


class SilentParser(ParserSpec):
    kind: Literal["silent"] = "silent"


class ChartParser(ParserSpec):
    kind: Literal["chart-parser"] = "chart-parser"
    beam: int = 4

    def chart_outputs(self) -> frozenset[str]:
        """Return the outputs consumed by this fixture's chart readout."""
        return frozenset({"cells", "alpha", "mu"})


class ApplicationDecoder(TermDecoderSpec):
    kind: Literal["application-decoder"] = "application-decoder"
    rank: int = 1


class UniformController(ControllerSpec):
    kind: Literal["uniform-controller"] = "uniform-controller"
    input: Literal["inside", "span"] = "inside"


class CoherenceLoss(LossSpec):
    kind: Literal["coherence-loss"] = "coherence-loss"


class PinningLoss(LossSpec):
    kind: Literal["pinning-loss"] = "pinning-loss"
    margin: float = 0.5


class SampleDataset(DatasetSpec):
    kind: Literal["sample-dataset"] = "sample-dataset"
    split: str = "train"


class WindowFeaturizer(FeaturizerSpec):
    kind: Literal["window-featurizer"] = "window-featurizer"


class StepTrainer(TrainerSpec):
    kind: Literal["step-trainer"] = "step-trainer"
    steps: int = 10


class FiniteSampler(TypeSamplerSpec):
    kind: Literal["finite-sampler"] = "finite-sampler"


class RuleSampler(PairSamplerSpec):
    kind: Literal["rule-sampler"] = "rule-sampler"


class AdamOptimizer(OptimizerSpec):
    kind: Literal["adam-optimizer"] = "adam-optimizer"


class CosineScheduler(SchedulerSpec):
    kind: Literal["cosine-scheduler"] = "cosine-scheduler"


class ConsoleSink(SinkSpec):
    kind: Literal["console-sink"] = "console-sink"


class ReadoutProbe(ProbeSpec):
    kind: Literal["readout-probe"] = "readout-probe"


class ExactEvaluator(EvaluatorSpec):
    kind: Literal["exact-evaluator"] = "exact-evaluator"


INTERVAL = BootstrapIntervalSpec(replicates=50, level=0.95)


def declaration() -> dict[str, JsonValue]:
    return {
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "chart-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "coherence-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "sample-dataset"},
                    "featurizer": {"kind": "window-featurizer"},
                }
            }
        },
        "trainer": {"kind": "step-trainer"},
        "release": {"checkpoint": "resume-latest"},
        "parse": {"retain": "forest", "reachability": "all"},
        "optimizer": {"kind": "adam-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
    }


def minimal() -> RunSpec:
    return RunSpec.model_validate(declaration())


def complete() -> RunSpec:
    return minimal().with_(
        model=ModelSection(
            language=DataModelLanguageSpec(),
            parser=ChartParser(beam=6),
            encoders={"terms": SizedEncoder(width=16)},
            controller=UniformController(),
            decoders={"forward": ApplicationDecoder(rank=1)},
            spaces={
                "types": SpaceSpec(
                    sort="Ty",
                    encoder="terms",
                    identity="forward",
                    combinators=CombinatorFamilySpec(
                        merge=Merge.CONCATENATE,
                        decoders={"apply": "forward"},
                    ),
                )
            },
            readouts={
                "publishing": TermsReadoutSpec(
                    space="types",
                    reads=ReadoutSource.INSIDE,
                    targets=TargetSource.GRAMMAR_GOAL,
                    output="grammar-goal",
                )
            },
        ),
        loss=LossSection(
            terms=(
                WeightedLossSpec(objective=CoherenceLoss(), weight=0.1),
                WeightedLossSpec(objective=PinningLoss(margin=0.25), weight=0.2),
            )
        ),
        scheduler=CosineScheduler(),
        sinks={"console": ConsoleSink()},
        probes={"readout": ReadoutProbe()},
        evaluators={"exact": ExactEvaluator(interval=INTERVAL)},
    )
