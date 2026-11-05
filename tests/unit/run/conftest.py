import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, override

import pytest

from lofi.data import BatchShape
from lofi.registry import Registry
from lofi.run import (
    MANIFEST_NAME,
    ExperimentIdentity,
    ExperimentManifest,
    ExperimentPrice,
    ExperimentSeed,
    HardwareDescriptor,
    ManifestIso,
    ManifestStatus,
    MemoryBudget,
    RunStatus,
    RunStore,
    SiteAllocation,
    SiteEnvironment,
    SiteInterpreter,
    SitePlacement,
    SiteProfile,
    SiteQuota,
    SiteStorage,
    SiteSubmission,
    SiteTransport,
    StackPrice,
    StackTiming,
    SweepPoint,
    close_manifest,
    open_manifest,
)
from lofi.spec import (
    DatasetSpec,
    EvaluatorSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    RunSpecCodec,
    SinkSpec,
    TrainerSpec,
)
from lofi.training_plugins import TrainerPlugin, TrainingLength, TrainingUnit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.data import SupervisionArm
    from lofi.run import RunDirectories
    from lofi.training_plugins import TrainerBuildContext

PACKAGED_RUNS: Final = Path(__file__).resolve().parents[3] / "lofi/configs/run"
"""The directory of the packaged, self-contained run documents."""

_PACKAGED_RUN_CODEC: Final = RunSpecCodec(
    search_path=(PACKAGED_RUNS, PACKAGED_RUNS.parent)
)


class LauncherParser(ParserSpec):
    kind: Literal["launcher-unit-parser"] = "launcher-unit-parser"


class LauncherLoss(LossSpec):
    kind: Literal["launcher-unit-loss"] = "launcher-unit-loss"


class LauncherDataset(DatasetSpec):
    kind: Literal["launcher-unit-dataset"] = "launcher-unit-dataset"


class LauncherFeaturizer(FeaturizerSpec):
    kind: Literal["launcher-unit-featurizer"] = "launcher-unit-featurizer"


class LauncherTrainer(TrainerSpec):
    kind: Literal["launcher-unit-trainer"] = "launcher-unit-trainer"
    epochs: int = 1


class OpenEndedTrainer(TrainerSpec):
    """A trainer that runs until its patience is spent and declares no horizon."""

    kind: Literal["open-ended-unit-trainer"] = "open-ended-unit-trainer"
    patience: int


def _plan_open_ended(
    declaration: OpenEndedTrainer, run_spec: RunSpec, /
) -> tuple[str, ...]:
    del declaration, run_spec

    return ()


def _present_open_ended(
    declaration: OpenEndedTrainer, run_spec: RunSpec, /
) -> Mapping[str, frozenset[SupervisionArm]]:
    del declaration, run_spec

    return {}


def _build_open_ended(
    declaration: OpenEndedTrainer, context: TrainerBuildContext, /
) -> None:
    del declaration, context


def _price_open_ended(declaration: OpenEndedTrainer, /) -> None:
    del declaration


OPEN_ENDED = TrainerPlugin(
    spec=OpenEndedTrainer,
    plan=_plan_open_ended,
    present=_present_open_ended,
    build=_build_open_ended,
    price=_price_open_ended,
)
"""The plugin of the trainer that states no training length."""


@pytest.fixture
def open_ended_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the open-ended trainer's plugin under its tag."""
    original = Registry.get

    def installed[MemberT](registry: Registry[MemberT], tag: str) -> MemberT:
        if registry.family == "trainer" and tag == "open-ended-unit-trainer":
            return cast("MemberT", OPEN_ENDED)

        return original(registry, tag)

    monkeypatch.setattr(Registry, "get", installed)


class LauncherOptimizer(OptimizerSpec):
    kind: Literal["launcher-unit-optimizer"] = "launcher-unit-optimizer"


class LauncherSink(SinkSpec):
    kind: Literal["launcher-unit-sink"] = "launcher-unit-sink"


class LauncherEvaluator(EvaluatorSpec):
    """An evaluator reporting the two metrics the experiment fixtures name."""

    kind: Literal["launcher-unit-evaluator"] = "launcher-unit-evaluator"

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        del labels
        return ("recovery", "parity")


def a_document(epochs: int = 1) -> dict[str, JsonValue]:
    """Return a run document the launcher tests vary by epoch count."""
    return {
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "launcher-unit-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "launcher-unit-loss"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "launcher-unit-dataset"},
                    "featurizer": {"kind": "launcher-unit-featurizer"},
                }
            }
        },
        "trainer": {"kind": "launcher-unit-trainer", "epochs": epochs},
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "resume-latest"},
        "optimizer": {"kind": "launcher-unit-optimizer"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
        "sinks": {"metrics": {"kind": "launcher-unit-sink"}},
        "evaluators": {
            "held-out": {
                "kind": "launcher-unit-evaluator",
                "interval": {"replicates": 50, "level": 0.95},
            }
        },
    }


def a_point(epochs: int = 1, *, seed: int = 0) -> SweepPoint:
    """Return a point over the launcher document."""
    return SweepPoint(
        a_document(epochs), overrides=(f"trainer.epochs={epochs}",), seed=seed
    )


EXPERIMENT_SEED = ExperimentSeed(7)
"""The experiment root the launcher tests seed their points under."""


def a_running_manifest(
    point: SweepPoint, *, seed: ExperimentSeed = EXPERIMENT_SEED
) -> ExperimentManifest:
    """Return a running manifest for a point seeded under an experiment root."""
    derived = seed.point(point.point_id, point.seed)

    return open_manifest(
        run_spec=RunSpec.model_validate(point.config),
        component_fingerprints={
            role: f"fingerprint:{role}"
            for role in RunSpec.model_validate(point.config).component_roles()
        },
        grammar_fingerprint="grammar:ccg",
        signature_fingerprint="signature:ccg",
        corpus_revision="ccgbank:LDC2005T13:auto-v1",
        seed_policy=derived.policy(accelerator=None, samplers=()),
        seed_derivation=derived,
        package_versions={"torch": "2.8.0"},
        git_revision="0123456789abcdef",
        accelerator="cpu",
    )


def record_manifest(
    store: RunDirectories, point: SweepPoint, status: ManifestStatus
) -> None:
    """Write a manifest in a point's run directory, closed unless running."""
    manifest = a_running_manifest(point)

    if status is not ManifestStatus.RUNNING:
        manifest = close_manifest(manifest, status)

    directory = store.directory(point)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_bytes(ManifestIso().forward(manifest))


def a_profile(
    scratch: Path, *, tag: str, accelerators_per_node: int = 1
) -> SiteProfile:
    """Return a profile whose every string value is marked with a tag."""
    return SiteProfile(
        submission=SiteSubmission(
            command=f"submit-{tag}",
            directive_prefix=f"#DIRECTIVE-{tag}",
            array_syntax=f"--range-{tag}={{first}}..{{last}}",
            throttle_syntax=f"/{tag}{{throttle}}",
            account_syntax=f"--account-{tag}={{account}}",
            partition_syntax=f"--partition-{tag}={{partition}}",
            accelerator_syntax=(
                f"--accel-{tag}={{accelerator}}:{{accelerators_per_node}}"
            ),
            task_index_variable=f"TASK_INDEX_{tag}",
            walltime_syntax=f"--time-{tag}={{minutes}}",
            memory_syntax=f"--mem-{tag}={{megabytes}}",
            cpus_syntax=f"--cpus-{tag}={{cpus}}",
            log_syntax=f"--log-{tag}={{directory}}/task.log",
            name_syntax=f"--name-{tag}={{name}}",
            job_prefix=f"{tag}-",
            cancel_command=(f"cancel-{tag}", f"--name-{tag}={{name}}"),
        ),
        placement=SitePlacement(
            account=f"account-{tag}",
            partition=f"partition-{tag}",
            accelerator=f"accelerator-{tag}",
            accelerators_per_node=accelerators_per_node,
            cpus_per_task=3,
        ),
        storage=SiteStorage(
            scratch_root=str(scratch / f"scratch-{tag}"),
            cache_root=str(scratch / f"cache-{tag}"),
            quota=SiteQuota(
                command=("echo", "used 0 bytes"),
                used_pattern=r"used (?P<used>\d+)",
                unit_bytes=1,
                limit_bytes=10**12,
            ),
            artifacts=(),
        ),
        allocation=SiteAllocation(accelerator_hours=1000.0, share=0.6),
        environment=SiteEnvironment(
            preamble=(f"module load toolkit-{tag}", f"export SITE={tag}")
        ),
        interpreter=SiteInterpreter(
            command=(f"container-{tag}", "exec", f"/images/{tag}.sif", "python")
        ),
        transport=SiteTransport(
            host=f"host-{tag}", command=(f"shell-{tag}", "-o", f"Option={tag}")
        ),
    )


def a_price(
    points: Sequence[SweepPoint],
    *,
    seconds: tuple[float, ...] = (1.0, 2.0, 3.0),
    peak: int | None = 3 * 1024**3,
    refused: Sequence[SweepPoint] = (),
) -> ExperimentPrice:
    """Return a price whose one stack trains the points, beside an over-budget one.

    Each point trains two epochs of ten steps, and the refused points share a
    stack whose peak lies above its budget.
    """

    def stack(name: str, members: Sequence[SweepPoint], held: int | None) -> StackPrice:
        return StackPrice(
            stack=name,
            points=[point.point_id for point in members],
            length=TrainingLength(2, TrainingUnit.EPOCHS),
            timing=StackTiming(
                steps_per_unit=10,
                positions=list(range(len(seconds))),
                seconds=seconds,
                peak_allocated_bytes=held,
                shape=BatchShape(12.0, 8.0),
            ),
            budget=MemoryBudget(8 * 1024**3),
        )

    stacks = [stack("admitted", points, peak)]

    if refused:
        stacks.append(stack("refused", refused, 9 * 1024**3))

    return ExperimentPrice(
        experiment="headline",
        freeze_id="freeze/1",
        sweep_id="0123456789abcdef",
        device="cuda:0",
        hardware=HardwareDescriptor(
            device_name="NVIDIA A40",
            driver=None,
            cuda="12.8",
            torch="2.8.0",
            host_cpu="x86_64",
        ),
        warmup=1,
        repeats=len(seconds),
        resamples=200,
        bootstrap_seed=0,
        stacks=stacks,
        grid=None,
    )


@pytest.fixture
def make_profile(tmp_path: Path) -> Callable[[str], SiteProfile]:
    """Return a factory for profiles whose values differ by tag."""
    return lambda tag: a_profile(tmp_path, tag=tag)


def a_packaged_run(name: str) -> dict[str, JsonValue]:
    """Return one composed packaged run document as a mapping."""
    spec = _PACKAGED_RUN_CODEC(PACKAGED_RUNS / f"{name}.yaml")

    return cast("dict[str, JsonValue]", json.loads(spec.model_dump_json()))


BRACKETING: dict[str, JsonValue] = {
    "kind": "bracketing",
    "interval": {"replicates": 50, "level": 0.95},
    "conventions": {
        "punctuation": "strip",
        "unary_chains": "collapse",
        "trivial_spans": "drop",
        "minimum_length": 3,
        "aggregation": "sentence",
        "coverage": "zero",
        "self_arcs": "allow",
        "gold_derivation": "canonical",
    },
    "punctuation": {"source": "none", "forms": []},
    "decoder": "minimum-risk",
}
"""A held-out bracketing evaluator a priced experiment's row names."""


def write_priced_experiment(
    root: Path, run: dict[str, JsonValue], sweep: dict[str, JsonValue]
) -> Path:
    """Write an unpriced experiment over a run document and sweep; return its path.

    The run document is given a held-out bracketing evaluator, whose metric is
    the experiment's one row.
    """
    folder = root / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    scored = {**run, "evaluators": {"held-out": BRACKETING}}
    (folder / "run.json").write_text(json.dumps(scored), encoding="utf-8", newline="\n")
    preregistration = {
        "grid": {},
        "hypotheses": [
            {"name": "recovery", "left": "induced", "right": "gold", "family": "main"}
        ],
    }
    (folder / "preregistration.json").write_text(
        json.dumps(preregistration), encoding="utf-8", newline="\n"
    )
    row: dict[str, JsonValue] = {
        "evaluator": "held-out",
        "metric": "unlabeled-bracketing-f1",
    }
    experiment = {
        "role": "preflight",
        "freeze_id": "freeze/1",
        "config": {"kind": "run", "path": "run.json"},
        "launch": {"kind": "serial"},
        "failure": {"numerical": "final", "out_of_memory": "retry"},
        "sweep": sweep,
        "seed_policy": {"root": 7},
        "rows": [row],
        "primary": row,
        "preregistration": "preregistration.json",
        "compute": {"kind": "unpriced"},
        "contraction": {"rungs": [{"kind": "fallback"}], "factorial": "none"},
        "stop_rules": [],
        "decision_point": {
            "date": "2026-12-01",
            "signal": {
                "kind": "metric",
                "row": row,
                "statistic": "mean",
                "direction": "above",
                "threshold": 0.5,
            },
            "fallback": "The synthetic stack trains.",
        },
        "evaluation": {
            "alpha": 0.05,
            "correction": {"kind": "benjamini-hochberg"},
            "dependence": "prds",
            "minimum_seeds": 3,
            "seed_budget": {
                "metric": "recovery",
                "target": "mean",
                "precision": 0.05,
                "level": 0.95,
            },
        },
    }
    path = folder / "pricing.yaml"
    path.write_text(json.dumps(experiment), encoding="utf-8", newline="\n")

    return path


SCRATCH_IDENTITY: Final = ExperimentIdentity("headline", "freeze/1")
"""The experiment a scratch store's points are submitted for."""


class ScratchStore:
    """The run store of one experiment's freeze at a profile's scratch root."""

    def __init__(self, root: Path) -> None:
        self.runs = RunStore(
            root, freeze_id=SCRATCH_IDENTITY.freeze_id, seed=EXPERIMENT_SEED
        )

    @property
    def root(self) -> Path:
        return self.runs.root

    @property
    def scope(self) -> Path:
        return self.runs.scope

    @property
    def identity(self) -> ExperimentIdentity:
        return SCRATCH_IDENTITY

    def directory(self, point: SweepPoint, /) -> Path:
        return self.runs.directory(point)

    def is_completed(self, point: SweepPoint, /) -> bool:
        return self.runs.is_completed(point)

    def is_final(self, point: SweepPoint, /) -> bool:
        return self.runs.is_final(point)

    def status(self, point: SweepPoint, /) -> RunStatus:
        return self.runs.status(point)


def a_scratch_store(profile: SiteProfile) -> ScratchStore:
    """Return the store of the scratch experiment at a profile's scratch root."""
    return ScratchStore(Path(profile.storage.scratch_root))
