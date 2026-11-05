"""Point and experiment manifests, their lifecycle, and judgment report rows."""

import enum
import json
from importlib.metadata import version
from typing import TYPE_CHECKING, NamedTuple, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi._types import JsonValue
from lofi.data import BatchShape
from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    ReportIdentity,
    ReportRow,
    Split,
)
from lofi.eval._values import copy_validated_metric_result
from lofi.run._modules import parameter_count
from lofi.run._seed import ExperimentSeed, PointSeed, SeedPolicy
from lofi.run._values import (
    ExperimentPrice,
    GridCell,
    GridTiming,
    HardwareDescriptor,
    MemoryBudget,
    StackPrice,
    StackTiming,
)
from lofi.spec import (
    Provenance,
    RunSpec,
    SeedDerivationRecord,
    load_plugins,
)
from lofi.stats._values import copy_validated_interval
from lofi.training_plugins import TrainingLength, TrainingUnit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from lofi.judgment import Measurement
    from lofi.nn.grounding import GroundingMetadata, GroundingTelemetry
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._environment import ExecutionEnvironment
    from lofi.run._ports import CountedParameter
    from lofi.run._training import TrainingBlueprint
    from lofi.spec import DevicePolicySpec
    from lofi.train import (
        CheckpointManifest,
        ComponentRecord,
        GroundingRecord,
        PrecisionRecord,
        PreparedTraining,
        RestorableRandomStreams,
        RngRecord,
        StateReference,
        StateTree,
    )

__all__ = [
    "BatchShapeRecord",
    "BindingIso",
    "ExperimentManifest",
    "ExperimentPriceRecord",
    "FailureIso",
    "FailureKind",
    "GridCellRecord",
    "HardwareRecord",
    "ManifestIso",
    "ManifestStatus",
    "MeasuredRole",
    "MemoryBudgetRecord",
    "PointBinding",
    "PointBindingRecord",
    "PointFailure",
    "PointFailureRecord",
    "PointManifest",
    "PriceIso",
    "RecordedRows",
    "RecordedRowsRecord",
    "RoleParameterRecord",
    "RowsIso",
    "SeedPolicyRecord",
    "StackPriceRecord",
    "TrainingLengthRecord",
    "close_manifest",
    "decode_file",
    "decode_manifest",
    "decode_provenance",
    "decode_run_spec",
    "decode_seed_derivation",
    "encode_seed_derivation",
    "indented_json",
    "judgment_report_row",
    "measure_point",
    "open_manifest",
    "open_training_manifest",
    "read_file",
]


def decode_run_spec(document: bytes, /) -> RunSpec:
    """Validate one persisted run declaration at the manifest boundary."""
    return RunSpec.model_validate_json(document)


def decode_provenance(document: bytes, /) -> Provenance:
    """Validate one persisted configuration provenance at the manifest boundary."""
    return Provenance.model_validate_json(document)


def encode_seed_derivation(seed: PointSeed, /) -> SeedDerivationRecord:
    """Encode one interior point seed at the manifest boundary."""
    return SeedDerivationRecord(
        experiment_root=seed.experiment.root,
        point_id=seed.point_id,
        seed=seed.seed,
        root=seed.root,
    )


def decode_seed_derivation(record: SeedDerivationRecord, /) -> PointSeed:
    """Decode one validated manifest derivation as an interior point seed."""
    return PointSeed(
        ExperimentSeed(record.experiment_root), record.point_id, record.seed
    )


def judgment_report_row(
    measurement: Measurement,
    identity: ReportIdentity,
    /,
    *,
    suite: str,
    task: str,
    dataset: str,
    split: Split,
    metric: str,
    seed: int,
    normal_form: str | None,
    specification_hash: str,
    fold: int,
    rung: str,
    held_out_convention: str | None,
) -> ReportRow:
    """Lower one validated judgment measurement into a complete report row.

    Parameters
    ----------
    measurement : Measurement
        Validated judgment measurement to copy.
    identity : ReportIdentity
        Content identity for the run, grammar and scoring conventions.
    suite : str
        Evaluation suite name.
    task : str
        Evaluation task name.
    dataset : str
        Evaluated dataset name.
    split : Split
        Evaluated dataset split.
    metric : str
        Metric name.
    seed : int
        Run seed.
    normal_form : str | None
        Normal-form name, if applicable.
    specification_hash : str
        Judgment specification digest.
    fold : int
        Zero-based held-out fold index.
    rung : str
        Judgment-model rung.
    held_out_convention : str | None
        Participant-effect treatment for held-out participants, if needed.
    """
    if not isinstance(identity, ReportIdentity):
        msg = f"a judgment report identity is not one: {identity!r}"
        raise TypeError(msg)

    interval = (
        None
        if measurement.lower is None
        else copy_validated_interval(
            measurement.lower,
            cast("float", measurement.upper),
            cast("float", measurement.level),
        )
    )

    return ReportRow(
        identity=identity,
        method=rung,
        metric=metric,
        result=copy_validated_metric_result(
            measurement.value,
            measurement.n,
            measurement.unit,
            ci=interval,
        ),
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        suite=suite,
        task=task,
        dataset=dataset,
        split=split,
        seed=seed,
        normal_form=normal_form,
        specification_hash=specification_hash,
        fold=fold,
        rung=rung,
        held_out_convention=held_out_convention,
    )


class _CheckpointRecordTypes(NamedTuple):
    """Concrete checkpoint record types and state serialization at composition."""

    checkpoint_manifest: type[CheckpointManifest]
    component_record: type[ComponentRecord]
    precision_record: type[PrecisionRecord]
    rng_record: type[RngRecord]
    state_reference: type[StateReference]
    state_digest: Callable[[StateTree], str]
    grounding_record: Callable[
        [GroundingMetadata, GroundingTelemetry | None], GroundingRecord
    ]


class RoleParameterRecord(dx.Model, extra="forbid"):
    """Records the declared kind and measured size of one component role.

    Attributes
    ----------
    kind : str
        The registry tag of the component the role declares.
    parameters : int
        How many parameter elements the role's own module holds, leaving out
        those of any role nested inside it.
    trainable : int
        How many of those elements take a gradient in the run.
    """

    kind: str
    parameters: int
    trainable: int

    __axioms__ = (
        dx.axiom("parameters >= 0", message="a count is not negative"),
        dx.axiom("trainable >= 0", message="a count is not negative"),
        dx.axiom(
            "trainable <= parameters",
            message="a role trains no more elements than it holds",
        ),
    )

    @dx.validates("kind")
    @classmethod
    def _kind_is_present(cls, value: str) -> str:
        """Return a nonempty registry tag."""
        return ensure_name("a component kind", value)


class PointManifest(dx.Model, extra="forbid"):
    """Records what one sweep point declares and how large a model it builds.

    Attributes
    ----------
    point_id : str
        The identifier of the point's resolved configuration and seed.
    seed : int
        The seed the point was expanded with.
    corpus_fingerprint : str
        The fingerprint of the corpus declaration the point's data section
        builds.
    roles : dict[str, RoleParameterRecord]
        The declared kind and measured size of each component role the
        point's model builds, keyed by role.
    parameter_count : int
        How many parameter elements the built model holds, counted once per
        distinct parameter. It is measured from the built modules and read
        from no declaration.
    trainable_count : int
        How many of those elements take a gradient in the run.
    """

    point_id: str
    seed: int
    corpus_fingerprint: str
    roles: dict[str, RoleParameterRecord]
    parameter_count: int
    trainable_count: int

    __axioms__ = (
        dx.axiom("seed >= 0", message="a seed counts from zero"),
        dx.axiom("parameter_count >= 0", message="a count is not negative"),
        dx.axiom("trainable_count >= 0", message="a count is not negative"),
        dx.axiom(
            "trainable_count <= parameter_count",
            message="a model trains no more elements than it holds",
        ),
    )

    @dx.validates("point_id", "corpus_fingerprint")
    @classmethod
    def _identity_is_present(cls, value: str) -> str:
        """Return a nonempty identity string."""
        return ensure_name("point manifest identity", value)


class MeasuredRole(NamedTuple):
    """One built component role: the kind it declares and the parameters it owns.

    Attributes
    ----------
    kind : str
        The registry tag of the component the role declares.
    parameters : tuple[CountedParameter, ...]
        The parameters the role's own module holds.
    """

    kind: str
    parameters: tuple[CountedParameter, ...]


def measure_point(
    point: SweepPoint,
    roles: Mapping[str, MeasuredRole],
    /,
    *,
    corpus_fingerprint: str,
) -> PointManifest:
    """Return one point's manifest, with the built model's parameters counted.

    The seed and the identifier are read back from the point, so a manifest
    describes the point it was built for and nothing else. Every count comes
    from the built modules, so a declaration cannot state it.

    Parameters
    ----------
    point : SweepPoint
        The expanded point.
    roles : Mapping[str, MeasuredRole]
        Each component role the point's model builds, with the kind it
        declares and the parameters its own module holds.
    corpus_fingerprint : str
        The fingerprint of the corpus the point's data section builds.

    Returns
    -------
    PointManifest
        The point's identity and the measured size of each role and of the
        whole model.
    """
    whole = parameter_count(
        parameter for role in roles.values() for parameter in role.parameters
    )
    records: dict[str, RoleParameterRecord] = {}

    for name in sorted(roles):
        role = roles[name]
        counted = parameter_count(role.parameters)
        records[name] = RoleParameterRecord(
            kind=role.kind, parameters=counted.total, trainable=counted.trainable
        )

    return PointManifest(
        point_id=point.point_id,
        seed=point.seed,
        corpus_fingerprint=corpus_fingerprint,
        roles=records,
        parameter_count=whole.total,
        trainable_count=whole.trainable,
    )


class ManifestStatus(enum.StrEnum):
    """Identifies the lifecycle state recorded in an experiment manifest."""

    RUNNING = "running"
    COMPLETED = "completed"
    KILLED = "killed"


class SeedPolicyRecord(dx.Model, extra="forbid"):
    """Records the seed policy a run derived its random streams from.

    Attributes
    ----------
    root : int
        The run seed every stream seed is derived from.
    accelerator : str | None
        The accelerator whose default stream participates, or None.
    samplers : tuple[str, ...]
        The sampler stream names, in canonical order.
    loaders : tuple[str, ...]
        The data-loader generator names, in canonical order.
    workers : tuple[str, ...]
        The data-loader worker-family names, in canonical order.
    """

    root: int
    accelerator: str | None
    samplers: tuple[str, ...]
    loaders: tuple[str, ...]
    workers: tuple[str, ...]

    @classmethod
    def of(cls, policy: SeedPolicy, /) -> SeedPolicyRecord:
        """Return the record of a seed policy.

        Parameters
        ----------
        policy : SeedPolicy
            The policy a run derives its streams from.
        """
        return cls.model_validate(dict(policy.record))

    @dx.model_validator()
    def _names_a_policy(self) -> Self:
        """Refuse a record that encodes no seed policy.

        Raises
        ------
        ValueError
            If the root is negative, a stream name is empty or repeated, or
            the stream names are out of canonical order.
        """
        policy = self.policy()

        if (policy.samplers, policy.loaders, policy.workers) != (
            self.samplers,
            self.loaders,
            self.workers,
        ):
            msg = "a seed policy record lists its streams out of canonical order"
            raise ValueError(msg)

        return self

    def policy(self) -> SeedPolicy:
        """Return the seed policy the record encodes."""
        return SeedPolicy(
            self.root,
            accelerator=self.accelerator,
            samplers=self.samplers,
            loaders=self.loaders,
            workers=self.workers,
        )


class ExperimentManifest(dx.Model, extra="forbid"):
    """Records the configuration and environment of one experiment run.

    ``component_fingerprints`` holds one declaration fingerprint per component
    role of the run, keyed as ``RunSpec.component_roles`` names the role, so two roles
    filled by one kind are recorded apart. ``seed_derivation`` records the
    experiment root seed, the rule and the point the run seed was derived
    from, and ``seed_policy`` is rooted at that derived seed. ``package_versions``,
    ``git_revision`` and ``accelerator`` are the environment the run executed
    under, the last as ``accelerator_runtime`` reports it.
    """

    run_spec: RunSpec
    component_fingerprints: dict[str, str]
    grammar_fingerprint: str
    signature_fingerprint: str
    corpus_revision: str
    seed_policy: SeedPolicyRecord
    seed_derivation: SeedDerivationRecord
    package_versions: dict[str, str]
    git_revision: str
    accelerator: str
    status: ManifestStatus

    @property
    def device_policy(self) -> DevicePolicySpec:
        """Return the device policy declared by the run specification."""
        return self.run_spec.device_policy

    @dx.validates("component_fingerprints", "package_versions")
    @classmethod
    def _names_and_values_are_present(cls, values: dict[str, str]) -> dict[str, str]:
        """Return a name-ordered mapping with nonempty keys and values."""
        if not values:
            msg = "a manifest mapping cannot be empty"
            raise ValueError(msg)

        return {
            ensure_name("manifest mapping key", key): ensure_name(
                "manifest mapping value", value
            )
            for key, value in sorted(values.items())
        }

    @dx.validates(
        "grammar_fingerprint",
        "signature_fingerprint",
        "corpus_revision",
        "git_revision",
        "accelerator",
    )
    @classmethod
    def _identity_is_present(cls, value: str) -> str:
        """Return a nonempty identity string."""
        return ensure_name("manifest identity", value)

    @dx.model_validator()
    def _fingerprints_cover_the_run(self) -> Self:
        """Return a manifest with one fingerprint per declared component role."""
        policy = self.seed_policy.policy()
        derived = decode_seed_derivation(self.seed_derivation)

        if policy.root != derived.root:
            msg = (
                f"seed_policy root {policy.root} is not the seed {derived.root} "
                f"its seed_derivation derives"
            )
            raise ValueError(msg)

        expected = frozenset(self.run_spec.component_roles())
        found = frozenset(self.component_fingerprints)
        missing = sorted(expected - found)
        extra = sorted(found - expected)

        if missing or extra:
            details = []

            if missing:
                details.append(f"missing roles: {', '.join(missing)}")

            if extra:
                details.append(f"undeclared roles: {', '.join(extra)}")

            msg = (
                "component fingerprints do not match the run specification ("
                + "; ".join(details)
                + ")"
            )
            raise ValueError(msg)

        return self


class ManifestIso(dx.Iso[ExperimentManifest, bytes]):
    """Converts an experiment manifest to and from its JSON bytes."""

    @override
    def forward(self, a: ExperimentManifest, /) -> bytes:
        """Encode a manifest."""
        return a.model_dump_json().encode()

    @override
    def backward(self, b: bytes, /) -> ExperimentManifest:
        """Decode a manifest once every installed declaration is registered.

        A manifest records the run document's components by their declared
        kinds, so the plugins that declare those kinds load first.
        """
        load_plugins()
        return ExperimentManifest.model_validate_json(b)


def decode_manifest(data: bytes, /) -> ExperimentManifest:
    """Decode a manifest's JSON bytes, refusing any that do not validate.

    Parameters
    ----------
    data : bytes
        The bytes a manifest was written as.

    Returns
    -------
    ExperimentManifest
        The validated manifest.

    Raises
    ------
    ValueError
        If the bytes are not JSON or do not validate as a manifest.
    """
    try:
        return ManifestIso().backward(data)

    except dx.ValidationError as refused:
        msg = f"the bytes do not validate as an experiment manifest: {refused}"
        raise ValueError(msg) from refused


def open_manifest(
    *,
    run_spec: RunSpec,
    component_fingerprints: Mapping[str, str],
    grammar_fingerprint: str,
    signature_fingerprint: str,
    corpus_revision: str,
    seed_policy: SeedPolicy,
    seed_derivation: PointSeed,
    package_versions: Mapping[str, str],
    git_revision: str,
    accelerator: str,
) -> ExperimentManifest:
    """Open a running experiment manifest."""
    return ExperimentManifest(
        run_spec=run_spec,
        component_fingerprints=dict(component_fingerprints),
        grammar_fingerprint=grammar_fingerprint,
        signature_fingerprint=signature_fingerprint,
        corpus_revision=corpus_revision,
        seed_policy=SeedPolicyRecord.of(seed_policy),
        seed_derivation=encode_seed_derivation(seed_derivation),
        package_versions=dict(package_versions),
        git_revision=git_revision,
        accelerator=accelerator,
        status=ManifestStatus.RUNNING,
    )


def open_training_manifest(
    blueprint: TrainingBlueprint,
    seed: PointSeed,
    environment: ExecutionEnvironment,
    /,
) -> ExperimentManifest:
    """Open the complete experiment manifest for one planned training run.

    The run's stream policy is rooted at the point's derived seed, with the
    sampler streams the blueprint declares and the accelerator its device
    names.
    """
    device = blueprint.run_spec.device_policy.device
    family = device.partition(":")[0]
    policy = seed.policy(
        accelerator=None if family in {"cpu", "meta"} else device,
        samplers=blueprint.sampler_streams,
    )

    return open_manifest(
        run_spec=blueprint.run_spec,
        component_fingerprints=blueprint.component_fingerprints,
        grammar_fingerprint=blueprint.grammar_fingerprint,
        signature_fingerprint=blueprint.codec_fingerprint,
        corpus_revision=blueprint.corpus_revision,
        seed_policy=policy,
        seed_derivation=seed,
        package_versions=environment.packages,
        git_revision=environment.git_revision,
        accelerator=environment.accelerator,
    )


def _training_checkpoint_manifest(
    blueprint: TrainingBlueprint,
    prepared: PreparedTraining,
    streams: RestorableRandomStreams,
    records: _CheckpointRecordTypes,
    /,
    *,
    step: int,
    data_position: int,
) -> CheckpointManifest:
    """Return complete role and run-state metadata for configured training."""
    component_state, optimization = prepared.state()
    model = blueprint.run_spec.model
    declarations = model.checkpointed_components()
    paths = model.checkpointed_paths()

    if frozenset(declarations) != frozenset(component_state):
        msg = (
            f"configured declarations name {sorted(declarations)}, but component "
            f"state names {sorted(component_state)}"
        )
        raise ValueError(msg)

    signature = prepared.model.codec.codec.index.digest
    components: dict[str, ComponentRecord] = {}

    for role, state in component_state.items():
        declaration = declarations[role]
        components[role] = records.component_record(
            tag=cast("str", declaration.model_dump()["kind"]),
            spec_fingerprint=blueprint.component_fingerprints[paths[role]],
            signature_fingerprint=signature,
            codec_identity=blueprint.grammar_fingerprint,
            state=records.state_reference(
                key=role,
                digest=records.state_digest(cast("StateTree", state)),
            ),
            labels=(
                None
                if role not in prepared.model.checkpoint_labels
                else dict(prepared.model.checkpoint_labels[role])
            ),
            pinned=prepared.model.pinned.get(role),
        )

    policy = prepared.model.policy
    compute_dtype = policy.autocast_dtype or policy.dtype

    return records.checkpoint_manifest(
        run_id=blueprint.declaration_fingerprint,
        step=step,
        data_position=data_position,
        code_version=version("lofi"),
        action_set=blueprint.action_set_fingerprint,
        components=components,
        rng=records.rng_record(streams=dict(streams.snapshot())),
        precision=records.precision_record(
            compute_dtype=str(compute_dtype).removeprefix("torch."),
            autocast_device_type=(
                policy.device.type
                if policy.autocast_device_type is None
                else policy.autocast_device_type
            ),
            gradient_scaling=optimization.scaler is not None,
        ),
        optimizer=records.state_reference(
            key="optimizer",
            digest=records.state_digest(cast("StateTree", optimization.optimizer)),
        ),
        scaler=(
            None
            if optimization.scaler is None
            else records.state_reference(
                key="scaler",
                digest=records.state_digest(cast("StateTree", optimization.scaler)),
            )
        ),
        scheduler=(
            None
            if optimization.scheduler is None
            else records.state_reference(
                key="scheduler",
                digest=records.state_digest(cast("StateTree", optimization.scheduler)),
            )
        ),
        grounding=(
            None
            if prepared.grounding_metadata is None
            else records.grounding_record(
                prepared.grounding_metadata,
                prepared.grounding_telemetry,
            )
        ),
    )


def close_manifest(
    manifest: ExperimentManifest,
    status: ManifestStatus = ManifestStatus.COMPLETED,
    /,
) -> ExperimentManifest:
    """Return a manifest closed with a terminal status.

    Raises
    ------
    ValueError
        If ``status`` is ``RUNNING``.
    """
    if status is ManifestStatus.RUNNING:
        msg = "a closed manifest cannot remain running"
        raise ValueError(msg)

    return manifest.with_(status=status)


class PointBindingRecord(dx.Model, extra="forbid"):
    """Records the freeze a point was submitted under and its seed derivation.

    Attributes
    ----------
    freeze_id : str
        The freeze the point was submitted under.
    seed_derivation : SeedDerivationRecord
        How the point's run seed was derived.
    """

    freeze_id: str
    seed_derivation: SeedDerivationRecord

    @dx.validates("freeze_id")
    @classmethod
    def _freeze_is_named(cls, value: str) -> str:
        """Return a nonempty freeze identifier."""
        return ensure_name("a freeze identifier", value)


class RecordedRowsRecord(dx.Model, extra="forbid"):
    """Records the report rows a completed point emitted and what produced them.

    Attributes
    ----------
    freeze_id : str | None
        The freeze the rows were produced under, or None for an unfrozen run.
    seed_derivation : SeedDerivationRecord
        How the point's run seed was derived.
    rows : tuple[dict[str, JsonValue], ...]
        The rows, in emission order.
    """

    freeze_id: str | None
    seed_derivation: SeedDerivationRecord
    rows: tuple[dict[str, JsonValue], ...]


class PointFailureRecord(dx.Model, extra="forbid"):
    """Records the attempts a point has spent and how the last one ended.

    Attributes
    ----------
    status : str
        The failure kind, as ``FailureKind`` spells it.
    diagnostic : str
        What diagnosed the failure.
    attempts : int
        How many attempts the point has spent, at least one.
    seed_derivation : SeedDerivationRecord
        How the point's run seed was derived.
    """

    status: str
    diagnostic: str
    attempts: int
    seed_derivation: SeedDerivationRecord

    __axioms__ = (dx.axiom("attempts > 0", message="a failure spends an attempt"),)


class MemoryBudgetRecord(dx.Model, extra="forbid"):
    """Records the ceiling on the bytes one priced training step may hold.

    Attributes
    ----------
    max_bytes : int
        The ceiling in bytes, at least one.
    """

    max_bytes: int

    __axioms__ = (
        dx.axiom("max_bytes > 0", message="a memory budget leaves room for a byte"),
    )


class TrainingLengthRecord(dx.Model, extra="forbid"):
    """Records the training length a priced stack's run document declares.

    Attributes
    ----------
    count : int
        How many units the run trains for.
    unit : str
        The unit, as ``TrainingUnit`` spells it.
    """

    count: int
    unit: str


class BatchShapeRecord(dx.Model, extra="forbid"):
    """Records the mean sentence length and batch size of a stack's batches.

    Attributes
    ----------
    sentence_length : float
        The mean sentence length.
    batch_size : float
        The mean batch size.
    """

    sentence_length: float
    batch_size: float


class StackPriceRecord(dx.Model, extra="forbid"):
    """Records what the points sharing one built stack cost, with raw timings.

    Attributes
    ----------
    stack : str
        The identifier of the stack's resolved run document.
    points : tuple[str, ...]
        The points that train the stack, in sweep order.
    length : TrainingLengthRecord
        The declared training length.
    steps_per_unit : int
        The training steps one unit of the length holds.
    positions : tuple[int, ...]
        The sampled position of each timed step.
    seconds : tuple[float, ...]
        The wall seconds of each timed step.
    peak_allocated_bytes : int | None
        The peak the allocator held, or None where none was read.
    shape : BatchShapeRecord | None
        The mean batch shape, or None for a stack that reads no sentences.
    budget : MemoryBudgetRecord
        The activation-memory ceiling the peak is read against.
    """

    stack: str
    points: tuple[str, ...]
    length: TrainingLengthRecord
    steps_per_unit: int
    positions: tuple[int, ...]
    seconds: tuple[float, ...]
    peak_allocated_bytes: int | None
    shape: BatchShapeRecord | None
    budget: MemoryBudgetRecord


class HardwareRecord(dx.Model, extra="forbid"):
    """Records the machine an experiment's training steps were timed on.

    Attributes
    ----------
    device_name : str
        The name of the device the steps ran on.
    driver : str | None
        The accelerator driver version, or None where none was read.
    cuda : str | None
        The CUDA runtime the framework was built against, or None.
    torch : str
        The framework version.
    host_cpu : str
        The host processor.
    """

    device_name: str
    driver: str | None
    cuda: str | None
    torch: str
    host_cpu: str


class GridCellRecord(dx.Model, extra="forbid"):
    """Records the timed steps of the reference stack at one grid cell.

    Attributes
    ----------
    sentence_length : int
        The cell's sentence length.
    batch_size : int
        The cell's batch size.
    seconds : tuple[float, ...]
        The wall seconds of each timed step.
    """

    sentence_length: int
    batch_size: int
    seconds: tuple[float, ...]


class ExperimentPriceRecord(dx.Model, extra="forbid"):
    """Records what an experiment costs to train, with every raw timing.

    Attributes
    ----------
    experiment : str
        The name of the priced experiment.
    freeze_id : str
        The freeze the experiment runs against.
    sweep_id : str
        The identifier of the experiment's expanded points.
    cost_model : str
        The cost model that produced the figures.
    device : str
        The device every stack was timed on.
    hardware : HardwareRecord
        The machine the device belongs to.
    warmup : int
        The untimed steps each stack ran first.
    repeats : int
        The timed steps each stack and grid cell ran.
    resamples : int
        The bootstrap replicates the quantiles are read off.
    bootstrap_seed : int
        The seed of the bootstrap generator.
    stacks : tuple[StackPriceRecord, ...]
        One price per distinct stack, in sweep order.
    grid : tuple[GridCellRecord, ...] | None
        The reference stack's grid cells, or None to price every stack off its
        own timed steps.
    """

    experiment: str
    freeze_id: str
    sweep_id: str
    cost_model: str
    device: str
    hardware: HardwareRecord
    warmup: int
    repeats: int
    resamples: int
    bootstrap_seed: int
    stacks: tuple[StackPriceRecord, ...]
    grid: tuple[GridCellRecord, ...] | None


class FailureKind(enum.StrEnum):
    """What a point's failure record says of the attempts it has spent."""

    NUMERICAL = "numerical-failure"
    PREPARATION = "preparation-failed"
    INFRASTRUCTURE = "infra-failed"
    INTERRUPTED = "interrupted"


class PointFailure(NamedTuple):
    """How a point failed, what diagnosed it, after how many attempts, and its seed.

    An interrupted record names an attempt that started and has not reached
    an outcome, because it is still running or its process ended first. A
    numerical, preparation and infrastructure failures are final. An
    infrastructure failure is written once the site's declared attempts have
    been spent or the declared retry transformation cannot be applied.
    """

    kind: FailureKind
    diagnostic: str
    attempts: int
    seed_derivation: PointSeed

    @property
    def final(self) -> bool:
        """Return whether the record settles the point's outcome."""
        match self.kind:
            case (
                FailureKind.NUMERICAL
                | FailureKind.PREPARATION
                | FailureKind.INFRASTRUCTURE
            ):
                return True

            case FailureKind.INTERRUPTED:
                return False


class PointBinding(NamedTuple):
    """The freeze a point was submitted under and its run seed's derivation."""

    freeze_id: str
    seed_derivation: PointSeed


class RecordedRows(NamedTuple):
    """The report rows a completed point emitted and what produced them."""

    freeze_id: str | None
    seed_derivation: PointSeed
    rows: tuple[Mapping[str, JsonValue], ...]


class BindingIso(dx.Iso[PointBinding, bytes]):
    """Converts a point binding to and from the JSON bytes ``point.json`` holds."""

    @override
    def forward(self, a: PointBinding, /) -> bytes:
        """Encode a binding."""
        return indented_json(
            PointBindingRecord(
                freeze_id=a.freeze_id,
                seed_derivation=encode_seed_derivation(a.seed_derivation),
            )
        )

    @override
    def backward(self, b: bytes, /) -> PointBinding:
        """Decode a binding."""
        record = PointBindingRecord.model_validate_json(b)

        return PointBinding(
            record.freeze_id, decode_seed_derivation(record.seed_derivation)
        )


class RowsIso(dx.Iso[RecordedRows, bytes]):
    """Converts recorded rows to and from the JSON bytes ``rows.json`` holds."""

    @override
    def forward(self, a: RecordedRows, /) -> bytes:
        """Encode recorded rows, refusing a value JSON cannot hold."""
        return indented_json(
            RecordedRowsRecord(
                freeze_id=a.freeze_id,
                seed_derivation=encode_seed_derivation(a.seed_derivation),
                rows=tuple(dict(row) for row in a.rows),
            )
        )

    @override
    def backward(self, b: bytes, /) -> RecordedRows:
        """Decode recorded rows, each JSON array in a row held as a list."""
        record = RecordedRowsRecord.model_validate_json(b)
        document = cast("dict[str, JsonValue]", json.loads(record.model_dump_json()))

        return RecordedRows(
            record.freeze_id,
            decode_seed_derivation(record.seed_derivation),
            tuple(cast("list[dict[str, JsonValue]]", document["rows"])),
        )


class FailureIso(dx.Iso[PointFailure, bytes]):
    """Converts a point failure to and from the JSON bytes ``failure.json`` holds."""

    @override
    def forward(self, a: PointFailure, /) -> bytes:
        """Encode a failure."""
        return indented_json(
            PointFailureRecord(
                status=a.kind.value,
                diagnostic=a.diagnostic,
                attempts=a.attempts,
                seed_derivation=encode_seed_derivation(a.seed_derivation),
            )
        )

    @override
    def backward(self, b: bytes, /) -> PointFailure:
        """Decode a failure.

        Raises
        ------
        ValueError
            If the recorded status names no failure kind.
        """
        record = PointFailureRecord.model_validate_json(b)

        return PointFailure(
            FailureKind(record.status),
            record.diagnostic,
            record.attempts,
            decode_seed_derivation(record.seed_derivation),
        )


class PriceIso(dx.Iso[ExperimentPrice, bytes]):
    """Converts an experiment price to and from the JSON bytes ``price.json`` holds.

    The bytes hold every raw timing the price was read off, so decoding
    rebuilds the price exactly and its figures are recomputed from them.
    """

    @override
    def forward(self, a: ExperimentPrice, /) -> bytes:
        """Encode a price."""
        hardware = a.hardware

        return indented_json(
            ExperimentPriceRecord(
                experiment=a.experiment,
                freeze_id=a.freeze_id,
                sweep_id=a.sweep_id,
                cost_model=a.cost_model,
                device=a.device,
                hardware=HardwareRecord(
                    device_name=hardware.device_name,
                    driver=hardware.driver,
                    cuda=hardware.cuda,
                    torch=hardware.torch,
                    host_cpu=hardware.host_cpu,
                ),
                warmup=a.warmup,
                repeats=a.repeats,
                resamples=a.resamples,
                bootstrap_seed=a.bootstrap_seed,
                stacks=tuple(_stack_record(stack) for stack in a.stacks),
                grid=(
                    None
                    if a.grid is None
                    else tuple(
                        GridCellRecord(
                            sentence_length=cell.sentence_length,
                            batch_size=cell.batch_size,
                            seconds=cell.seconds,
                        )
                        for cell in a.grid.cells
                    )
                ),
            )
        )

    @override
    def backward(self, b: bytes, /) -> ExperimentPrice:
        """Decode a price.

        Raises
        ------
        ValueError
            If a field fails its value's checks, or the recorded cost model is
            not the one the recorded grid prices under.
        """
        record = ExperimentPriceRecord.model_validate_json(b)
        hardware = record.hardware
        price = ExperimentPrice(
            experiment=record.experiment,
            freeze_id=record.freeze_id,
            sweep_id=record.sweep_id,
            device=record.device,
            hardware=HardwareDescriptor(
                device_name=hardware.device_name,
                driver=hardware.driver,
                cuda=hardware.cuda,
                torch=hardware.torch,
                host_cpu=hardware.host_cpu,
            ),
            warmup=record.warmup,
            repeats=record.repeats,
            resamples=record.resamples,
            bootstrap_seed=record.bootstrap_seed,
            stacks=tuple(_stack_price(stack) for stack in record.stacks),
            grid=(
                None
                if record.grid is None
                else GridTiming(
                    tuple(
                        GridCell(cell.sentence_length, cell.batch_size, cell.seconds)
                        for cell in record.grid
                    )
                )
            ),
        )

        if record.cost_model != price.cost_model:
            msg = (
                f"an experiment price record names the cost model "
                f"{record.cost_model!r}, and its grid prices under "
                f"{price.cost_model!r}"
            )
            raise ValueError(msg)

        return price


def _stack_record(stack: StackPrice, /) -> StackPriceRecord:
    """Return the record of one stack's price."""
    timing = stack.timing
    shape = timing.shape

    return StackPriceRecord(
        stack=stack.stack,
        points=stack.points,
        length=TrainingLengthRecord(
            count=stack.length.count, unit=str(stack.length.unit)
        ),
        steps_per_unit=timing.steps_per_unit,
        positions=timing.positions,
        seconds=timing.seconds,
        peak_allocated_bytes=timing.peak_allocated_bytes,
        shape=(
            None
            if shape is None
            else BatchShapeRecord(
                sentence_length=shape.sentence_length, batch_size=shape.batch_size
            )
        ),
        budget=MemoryBudgetRecord(max_bytes=stack.budget.max_bytes),
    )


def _stack_price(record: StackPriceRecord, /) -> StackPrice:
    """Return the stack price one record holds."""
    shape = record.shape

    return StackPrice(
        stack=record.stack,
        points=record.points,
        length=TrainingLength(record.length.count, TrainingUnit(record.length.unit)),
        timing=StackTiming(
            steps_per_unit=record.steps_per_unit,
            positions=record.positions,
            seconds=record.seconds,
            peak_allocated_bytes=record.peak_allocated_bytes,
            shape=(
                None
                if shape is None
                else BatchShape(shape.sentence_length, shape.batch_size)
            ),
        ),
        budget=MemoryBudget(record.budget.max_bytes),
    )


def indented_json(record: dx.Model, /) -> bytes:
    """Return a record as indented JSON ending in a newline, refusing NaN.

    Raises
    ------
    ValueError
        If the record holds a number JSON cannot hold.
    """
    document = json.loads(record.model_dump_json())

    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()


def read_file[V](path: Path, iso: dx.Iso[V, bytes], /) -> V | None:
    """Return the value a file holds, or None when there is no file.

    Raises
    ------
    ValueError
        If the file does not validate as the record the codec reads.
    """
    if not path.is_file():
        return None

    return decode_file(path, iso)


def decode_file[V](path: Path, iso: dx.Iso[V, bytes], /) -> V:
    """Return the value a file holds, refusing one that does not validate.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ValueError
        If the file does not validate as the record the codec reads.
    """
    try:
        return iso.backward(path.read_bytes())

    except dx.ValidationError as refused:
        msg = f"the record {path} is malformed: {refused}"
        raise ValueError(msg) from refused
