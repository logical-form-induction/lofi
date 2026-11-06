"""Typed training records and their codecs.

The records describe checkpoint manifests, fitting stages and groundings, and
each crosses the boundary as validated JSON.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.nn.grounding import GradientPath, GroundingMetadata, GroundingTelemetry

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


__all__ = [
    "CheckpointManifest",
    "ComponentRecord",
    "FittingRecord",
    "GroundingRecord",
    "ParserFittingRecord",
    "PrecisionRecord",
    "RngRecord",
    "StageRecord",
    "StateReference",
    "StepFittingRecord",
    "disagreements",
    "grounding_record",
    "read_manifest",
    "staged_manifest",
    "write_manifest",
]


class StateReference(dx.Model, extra="forbid"):
    """Identifies a saved state block by key and digest.

    Parameters
    ----------
    key : str
        The name the state was written under.
    digest : str
        A content hash used to detect a truncated or overwritten block.
    """

    key: str
    digest: str


class PrecisionRecord(dx.Model, extra="forbid"):
    """Records checkpoint precision settings.

    Parameters
    ----------
    compute_dtype : str
        The dtype the forward pass ran in, under the name the tensor library gives it.
    autocast_device_type : str
        The device type autocasting was entered for.
    gradient_scaling : bool
        Whether a gradient scaler was in use.
    """

    compute_dtype: str
    autocast_device_type: str
    gradient_scaling: bool


class RngRecord(dx.Model, extra="forbid"):
    """Records random-stream state.

    Parameters
    ----------
    streams : dict[str, str]
        The encoded state of each stream, keyed by the name the run draws from it under.
    """

    streams: dict[str, str]


class ComponentRecord(dx.Model, extra="forbid"):
    """Records component checkpoint metadata.

    Parameters
    ----------
    tag : str
        The declared string tag of the implementation, as its own package registered it.
    spec_fingerprint : str
        A fingerprint of the specification the component was constructed from.
    signature_fingerprint : str
        A fingerprint of the symbol signature the component was built against.
    codec_identity : str
        The identity of the codec whose linearization the saved tensors are indexed by.
    state : StateReference
        Where the component's own state was written.
    labels : dict[str, int] | None
        The column each of the component's output labels occupies, for a component
        whose outputs are indexed by label, or None for one whose outputs are not.
    pinned : str | None
        The pinned pretrained checkpoint a frozen component's parameters are
        loaded from, written in place of those parameters, or None for a
        component whose parameters the checkpoint holds.
    """

    tag: str
    spec_fingerprint: str
    signature_fingerprint: str
    codec_identity: str
    state: StateReference
    labels: dict[str, int] | None = None
    pinned: str | None = None

    @dx.model_validator()
    def _the_labels_fill_the_columns(self) -> ComponentRecord:
        """Validate the label map.

        Returns
        -------
        ComponentRecord
            This record after validation.

        Raises
        ------
        ValueError
            If the label map is empty, or its columns are not exactly the first
            positions counted from zero.
        """
        if self.labels is None:
            return self

        if not self.labels:
            msg = "a label map names no label"

            raise ValueError(msg)

        columns = sorted(self.labels.values())

        if columns != list(range(len(columns))):
            msg = (
                f"a label map over {len(columns)} labels fills the columns "
                f"{list(range(len(columns)))}, not {columns}"
            )

            raise ValueError(msg)

        return self


class GroundingRecord(dx.Model, extra="forbid"):
    """Persist the grounding identity, proof budget, and measured step telemetry."""

    variant: str
    scorer: str
    source: str
    gradient_path: str
    prover: str | None
    prover_version: str | None
    budget_calls: int | None
    budget_wall_seconds: float | None
    budget_cpu_seconds: float | None
    calls: int
    questions: int
    questions_per_second: float
    memo_hits: int
    memo_hit_rate: float
    cpu_seconds: float
    wall_seconds: float
    process_starts: int | None
    label_agreement: float | None
    abstentions: dict[str, int]
    degenerate_mass: float
    malformed_mass: float
    reward_mean: float | None
    variance: float | None
    verified_share: float | None
    out_of_fragment_share: float

    @dx.validates("variant", "scorer", "source")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Validate the grounding tag.

        Parameters
        ----------
        value : str
            The declared tag.

        Returns
        -------
        str
            The tag.
        """
        return ensure_name("a grounding variant", value)

    @dx.validates("gradient_path")
    @classmethod
    def _is_a_declared_path(cls, value: str) -> str:
        """Validate the gradient path.

        Parameters
        ----------
        value : str
            The name of the path's member.

        Returns
        -------
        str
            The name.

        Raises
        ------
        ValueError
            If no gradient path carries that name.
        """
        if value not in GradientPath.__members__:
            named = ", ".join(sorted(GradientPath.__members__))
            msg = f"a gradient path is one of {named}, not {value!r}"

            raise ValueError(msg)

        return value

    @dx.model_validator()
    def _telemetry_is_valid(self) -> GroundingRecord:
        """Validate nonnegative counts, clocks, rates, and masses."""
        for name, value in (
            ("proof calls", self.calls),
            ("proof questions", self.questions),
            ("memo hits", self.memo_hits),
        ):
            ensure_non_negative(name, value)

        if self.budget_calls is not None:
            ensure_non_negative("proof calls per step", self.budget_calls)

        if self.process_starts is not None:
            ensure_non_negative("proof process starts", self.process_starts)

        for cause, count in self.abstentions.items():
            ensure_name("an abstention cause", cause)
            ensure_non_negative("an abstention count", count)

        for name, value in (
            ("questions per second", self.questions_per_second),
            ("memo hit rate", self.memo_hit_rate),
            ("proof CPU seconds", self.cpu_seconds),
            ("proof wall seconds", self.wall_seconds),
            ("degenerate mass", self.degenerate_mass),
            ("malformed mass", self.malformed_mass),
        ):
            held = ensure_finite(name, value)

            if held < 0.0:
                msg = f"{name} is non-negative, not {value!r}"
                raise ValueError(msg)

        for name, value in (
            ("proof wall seconds per step", self.budget_wall_seconds),
            ("proof CPU budget", self.budget_cpu_seconds),
            ("reward mean", self.reward_mean),
            ("estimator variance", self.variance),
            ("verified share", self.verified_share),
            ("label agreement", self.label_agreement),
            ("out-of-fragment share", self.out_of_fragment_share),
        ):
            if value is not None:
                ensure_finite(name, value)

        for name, value in (
            ("memo hit rate", self.memo_hit_rate),
            ("degenerate mass", self.degenerate_mass),
            ("malformed mass", self.malformed_mass),
            ("verified share", self.verified_share),
            ("label agreement", self.label_agreement),
            ("out-of-fragment share", self.out_of_fragment_share),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                msg = f"{name} lies in the unit interval, not {value!r}"
                raise ValueError(msg)

        return self

    @property
    def path(self) -> GradientPath:
        """Return the gradient path this record names.

        Returns
        -------
        GradientPath
            The member the recorded name carries.
        """
        return GradientPath[self.gradient_path]

    @property
    def compatibility_identity(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str | None,
        str | None,
        int | None,
        float | None,
        float | None,
    ]:
        """Return stable grounding fields that constrain checkpoint recovery."""
        return (
            self.variant,
            self.scorer,
            self.source,
            self.prover,
            self.prover_version,
            self.budget_calls,
            self.budget_wall_seconds,
            self.budget_cpu_seconds,
        )


def grounding_record(
    metadata: GroundingMetadata,
    telemetry: GroundingTelemetry | None = None,
    /,
) -> GroundingRecord:
    """Lower grounding identity and latest detached telemetry to the manifest."""
    budget = metadata.budget

    return GroundingRecord(
        variant=metadata.variant,
        scorer=metadata.scorer,
        source=metadata.source,
        gradient_path=(
            GradientPath.PATHWISE.name
            if telemetry is None
            else telemetry.gradient_path.name
        ),
        prover=metadata.prover,
        prover_version=metadata.prover_version,
        budget_calls=None if budget is None else budget.calls_per_step,
        budget_wall_seconds=None if budget is None else budget.wall_per_step,
        budget_cpu_seconds=None if budget is None else budget.cpu_seconds,
        calls=0 if telemetry is None else telemetry.calls,
        questions=0 if telemetry is None else telemetry.questions,
        questions_per_second=(
            0.0 if telemetry is None else telemetry.questions_per_second
        ),
        memo_hits=0 if telemetry is None else telemetry.memo_hits,
        memo_hit_rate=0.0 if telemetry is None else telemetry.memo_hit_rate,
        cpu_seconds=0.0 if telemetry is None else telemetry.cpu_seconds,
        wall_seconds=0.0 if telemetry is None else telemetry.wall_seconds,
        process_starts=None if telemetry is None else telemetry.process_starts,
        label_agreement=None if telemetry is None else telemetry.label_agreement,
        abstentions={} if telemetry is None else dict(telemetry.abstentions),
        degenerate_mass=0.0 if telemetry is None else telemetry.degenerate_mass,
        malformed_mass=0.0 if telemetry is None else telemetry.malformed_mass,
        reward_mean=None if telemetry is None else telemetry.reward_mean,
        variance=None if telemetry is None else telemetry.variance,
        verified_share=None if telemetry is None else telemetry.verified_share,
        out_of_fragment_share=(
            0.0 if telemetry is None else telemetry.out_of_fragment_share
        ),
    )


class StageRecord(dx.Model, extra="forbid"):
    """Records the stage of a staged protocol a checkpoint was written at.

    A staged run writes one checkpoint per stage against one set of weights, so
    this says how far through its protocol a set of weights is and which of the
    declared parameter groups were still being trained when it was written.

    Parameters
    ----------
    name : str
        What the stage is called.
    active : tuple[str, ...]
        The declared parameter groups the stage trained.
    position : int
        Where the stage falls in the protocol, counted from the first.
    """

    name: str
    active: tuple[str, ...]
    position: int

    @dx.validates("name")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Validate the stage name.

        Parameters
        ----------
        value : str
            What the stage is called.

        Returns
        -------
        str
            The name.
        """
        return ensure_name("a stage name", value)

    @dx.validates("position")
    @classmethod
    def _is_a_position(cls, value: int) -> int:
        """Validate the stage's position.

        Parameters
        ----------
        value : int
            Where the stage falls in the protocol.

        Returns
        -------
        int
            The position.
        """
        return ensure_non_negative("a stage position", value)

    @dx.model_validator()
    def _the_groups_are_named_once(self) -> StageRecord:
        """Validate the active-group names.

        Returns
        -------
        StageRecord
            This record after validation.

        Raises
        ------
        ValueError
            If a group name is empty or two entries name one group.
        """
        for name in self.active:
            ensure_name("an active group name", name)

        repeated = sorted({name for name in self.active if self.active.count(name) > 1})

        if repeated:
            msg = f"the stage {self.name!r} names the groups {repeated} twice"

            raise ValueError(msg)

        return self


class FittingRecord(dx.TaggedUnion, discriminator="unit", extra="forbid"):
    """Base for checkpoint fitting records, one per training unit."""


class ParserFittingRecord(FittingRecord):
    """Records validation state at a configured parser checkpoint.

    Parameters
    ----------
    unit : Literal["epochs"]
        The training unit the record counts.
    kind : Literal["validation-best", "resume-latest"]
        Whether the checkpoint holds a newly best model or the latest resumable state.
    completed_epochs : int
        Number of training and validation epochs completed.
    metric : str
        Validation metric used to select the best checkpoint.
    best : float | None
        Best finite observed value through this epoch, or None before the
        validation-owning stage.
    failures : int
        Consecutive nonimproving validations through this epoch.
    best_checkpoint : str | None
        Current global-best checkpoint handle, or None before validation and
        on the best checkpoint itself because a manifest cannot name its own
        assigned handle.
    """

    unit: Literal["epochs"] = "epochs"
    kind: Literal["validation-best", "resume-latest"]
    completed_epochs: int
    metric: str
    best: float | None
    failures: int
    best_checkpoint: str | None
    stop: Literal["target-met", "target-not-met"] | None = None
    achieved_accuracy: float | None = None

    @dx.validates("completed_epochs")
    @classmethod
    def _has_completed_an_epoch(cls, value: int) -> int:
        """Validate the completed epoch count."""
        return ensure_positive("completed_epochs", value)

    @dx.validates("metric")
    @classmethod
    def _metric_is_named(cls, value: str) -> str:
        """Validate the monitored metric name."""
        return ensure_name("metric", value)

    @dx.validates("best")
    @classmethod
    def _best_is_finite(cls, value: float | None) -> float | None:
        """Validate the best metric value, when validation has begun."""
        return None if value is None else ensure_finite("best", value)

    @dx.validates("failures")
    @classmethod
    def _failures_are_a_count(cls, value: int) -> int:
        """Validate the consecutive failure count."""
        return ensure_non_negative("failures", value)

    @dx.validates("achieved_accuracy")
    @classmethod
    def _accuracy_is_a_proportion(cls, value: float | None) -> float | None:
        """Validate the terminal held-out accuracy, when recorded."""
        if value is None:
            return None

        checked = ensure_finite("achieved_accuracy", value)

        if not 0.0 <= checked <= 1.0:
            msg = f"achieved_accuracy is outside [0, 1]: {value!r}"
            raise ValueError(msg)

        return checked

    @dx.model_validator()
    def _handle_matches_the_checkpoint_kind(self) -> ParserFittingRecord:
        """Validate the best-checkpoint handle."""
        if self.kind == "validation-best":
            if self.best is None:
                msg = "a validation-best checkpoint must record a finite best value"
                raise ValueError(msg)

            if self.best_checkpoint is not None:
                msg = "a validation-best checkpoint cannot name its own handle"
                raise ValueError(msg)

            if self.stop is not None or self.achieved_accuracy is not None:
                msg = "a validation-best checkpoint cannot record target stopping"
                raise ValueError(msg)

            return self

        if self.best is None:
            if self.best_checkpoint is not None:
                msg = "a pre-validation checkpoint cannot name a best checkpoint"
                raise ValueError(msg)

            if self.failures != 0:
                msg = "a pre-validation checkpoint cannot record validation failures"
                raise ValueError(msg)

        else:
            if self.best_checkpoint is None:
                msg = "a validated checkpoint must name the global best checkpoint"
                raise ValueError(msg)

            ensure_name("best_checkpoint", self.best_checkpoint)

        if (self.stop is None) != (self.achieved_accuracy is None):
            msg = "target stopping records both stop and achieved_accuracy"
            raise ValueError(msg)

        return self


class StepFittingRecord(FittingRecord):
    """Record optimizer-step progress and any held-out target outcome.

    Parameters
    ----------
    unit : Literal["steps"]
        The training unit the record counts.
    kind : Literal["resume-latest"]
        Resume-only checkpoint purpose.
    completed_steps : int
        Number of optimizer steps completed.
    stop : Literal["target-met", "target-not-met"] | None
        Why an accuracy target ended fitting, if it did.
    achieved_accuracy : float | None
        The most recent held-out accuracy, when a target is evaluated.
    """

    unit: Literal["steps"] = "steps"
    kind: Literal["resume-latest"] = "resume-latest"
    completed_steps: int
    stop: Literal["target-met", "target-not-met"] | None = None
    achieved_accuracy: float | None = None

    @dx.validates("completed_steps")
    @classmethod
    def _has_completed_a_step(cls, value: int) -> int:
        """Validate the completed optimizer-step count."""
        return ensure_positive("completed_steps", value)

    @dx.validates("achieved_accuracy")
    @classmethod
    def _accuracy_is_a_proportion(cls, value: float | None) -> float | None:
        """Validate the most recent held-out accuracy, when evaluated."""
        if value is None:
            return None

        checked = ensure_finite("achieved_accuracy", value)

        if not 0.0 <= checked <= 1.0:
            msg = f"achieved_accuracy is outside [0, 1]: {value!r}"
            raise ValueError(msg)

        return checked

    @dx.model_validator()
    def _stopping_reason_has_an_accuracy(self) -> StepFittingRecord:
        """Require every terminal target result to carry its observation."""
        if self.stop is not None and self.achieved_accuracy is None:
            msg = f"{self.stop} requires an achieved_accuracy"
            raise ValueError(msg)

        return self


class CheckpointManifest(dx.Model, extra="forbid"):
    """Records a checkpoint manifest.

    Parameters
    ----------
    run_id : str
        The run the checkpoint was written from.
    step : int
        The number of optimizer steps the run had taken.
    code_version : str
        The version of the code that wrote it.
    action_set : str
        The digest of the controller action set, in column order, the run was
        built with.
    components : dict[str, ComponentRecord]
        One record per component, keyed by the role it fills in the wiring.
    rng : RngRecord
        The state of every random stream the run was drawing from.
    precision : PrecisionRecord
        The numeric regime the run was executing under.
    data_position : int
        The next item position the resumed run reads.
    grounding : GroundingRecord | None
        The grounding the inference term was trained under, or None for a run
        whose objective grounds no inference.
    stage : StageRecord | None
        The stage of a staged protocol the checkpoint was written at, or None
        for a run that fits in one stage.
    fitting : FittingRecord | None
        Closed parser-epoch or distance-step progress, or None outside configured
        fitting.
    optimizer : StateReference | None
        Where the optimizer state was written, or None for weights-only checkpoints.
    scaler : StateReference | None
        Where the gradient scaler state was written, or None.
    scheduler : StateReference | None
        Where the learning rate scheduler state was written, or None.
    """

    run_id: str
    step: int
    code_version: str
    action_set: str
    components: dict[str, ComponentRecord]
    rng: RngRecord
    precision: PrecisionRecord
    data_position: int
    grounding: GroundingRecord | None = None
    stage: StageRecord | None = None
    fitting: FittingRecord | None = None
    optimizer: StateReference | None = None
    scaler: StateReference | None = None
    scheduler: StateReference | None = None

    @dx.validates("data_position")
    @classmethod
    def _is_a_data_position(cls, value: int) -> int:
        """Validate the next data position.

        Parameters
        ----------
        value : int
            The position counted from the first item.

        Returns
        -------
        int
            The position.
        """
        return ensure_non_negative("a checkpoint data position", value)

    @dx.computed
    def roles(self) -> frozenset[str]:
        """Return the roles this checkpoint holds a component for.

        Returns
        -------
        frozenset[str]
            The role of every component in the manifest.
        """
        return frozenset(self.components)

    @property
    def references(self) -> Mapping[str, StateReference]:
        """Return the run state references, keyed by the name each was written under.

        Returns
        -------
        Mapping[str, StateReference]
            The optimizer, scaler and scheduler references that are present.
        """
        return MappingProxyType(
            {
                reference.key: reference
                for reference in (self.optimizer, self.scaler, self.scheduler)
                if reference is not None
            }
        )

    @dx.model_validator()
    def _the_references_are_keyed_apart(self) -> CheckpointManifest:
        """Validate the run state reference keys.

        Returns
        -------
        CheckpointManifest
            This manifest after validation.

        Raises
        ------
        ValueError
            If two references share a key, or a reference is keyed by a role.
        """
        keys = [
            reference.key
            for reference in (self.optimizer, self.scaler, self.scheduler)
            if reference is not None
        ]
        repeated = sorted({key for key in keys if keys.count(key) > 1})

        if repeated:
            msg = f"two run-state references are written under {repeated}"

            raise ValueError(msg)

        shadowed = sorted(frozenset(keys) & frozenset(self.components))

        if shadowed:
            msg = f"a run-state reference is keyed by the role {shadowed}"

            raise ValueError(msg)

        return self


def _fields_that_differ(
    expected: ComponentRecord,
    found: ComponentRecord,
) -> Iterable[tuple[str, str | dict[str, int] | None, str | dict[str, int] | None]]:
    """Yield differing identity fields and values.

    Parameters
    ----------
    expected : ComponentRecord
        What the process reading the checkpoint was built with.
    found : ComponentRecord
        What the checkpoint was written with.

    Yields
    ------
    tuple[str, str | dict[str, int] | None, str | dict[str, int] | None]
        The field name, the value the reader was built with and the value the checkpoint
        was written with, for each of the six fields that differ.
    """
    stated = (
        ("tag", expected.tag, found.tag),
        ("spec_fingerprint", expected.spec_fingerprint, found.spec_fingerprint),
        (
            "signature_fingerprint",
            expected.signature_fingerprint,
            found.signature_fingerprint,
        ),
        ("codec_identity", expected.codec_identity, found.codec_identity),
        ("labels", expected.labels, found.labels),
        ("pinned", expected.pinned, found.pinned),
    )

    for field, mine, theirs in stated:
        if mine != theirs:
            yield field, mine, theirs


def disagreements(
    expected: CheckpointManifest,
    found: CheckpointManifest,
) -> tuple[str, ...]:
    """Return checkpoint incompatibility reasons.

    Parameters
    ----------
    expected : CheckpointManifest
        The manifest the reading process would write for itself.
    found : CheckpointManifest
        The manifest the checkpoint was written with.

    Returns
    -------
    tuple[str, ...]
        One sentence per disagreement, in role order, empty when the checkpoint may be
        read.
    """
    reasons: list[str] = [
        f"the checkpoint holds no component for the role {role!r}"
        for role in sorted(expected.roles - found.roles)
    ]

    reasons.extend(
        f"the checkpoint holds a component for the unwired role {role!r}"
        for role in sorted(found.roles - expected.roles)
    )

    for role in sorted(expected.roles & found.roles):
        mine_record = expected.components[role]
        their_record = found.components[role]

        for field, mine, theirs in _fields_that_differ(mine_record, their_record):
            reasons.append(
                f"the role {role!r} was built with {field} {mine!r} "
                f"and saved with {theirs!r}"
            )

    if expected.run_id != found.run_id:
        reasons.append(
            f"the run identity is {expected.run_id!r} and the checkpoint was "
            f"written for {found.run_id!r}"
        )

    if expected.code_version != found.code_version:
        reasons.append(
            f"the code version is {expected.code_version!r} and the checkpoint "
            f"was written by {found.code_version!r}"
        )

    if expected.action_set != found.action_set:
        reasons.append(
            f"the action set digest is {expected.action_set!r} and the checkpoint "
            f"was written under {found.action_set!r}"
        )

    if expected.precision != found.precision:
        reasons.append(
            f"the precision is {expected.precision!r} and the checkpoint was "
            f"written under {found.precision!r}"
        )

    mine_grounding = (
        None
        if expected.grounding is None
        else expected.grounding.compatibility_identity
    )
    their_grounding = (
        None if found.grounding is None else found.grounding.compatibility_identity
    )

    if mine_grounding != their_grounding:
        reasons.append(
            f"the grounding is {expected.grounding!r} and the checkpoint was "
            f"written under {found.grounding!r}"
        )

    mine_topology = _run_state_topology(expected)
    their_topology = _run_state_topology(found)

    if mine_topology != their_topology:
        reasons.append(
            f"the run-state topology is {mine_topology!r} and the checkpoint "
            f"was written with {their_topology!r}"
        )

    return tuple(reasons)


def _run_state_topology(manifest: CheckpointManifest) -> tuple[tuple[str, str], ...]:
    """Return the present run state parts and the keys holding them.

    Parameters
    ----------
    manifest : CheckpointManifest
        The manifest whose topology is read.
    """
    return tuple(
        (part, reference.key)
        for part, reference in (
            ("optimizer", manifest.optimizer),
            ("scaler", manifest.scaler),
            ("scheduler", manifest.scheduler),
        )
        if reference is not None
    )


def staged_manifest(
    manifest: CheckpointManifest,
    /,
    *,
    name: str,
    active: Iterable[str],
    position: int,
) -> CheckpointManifest:
    """Return the manifest one stage of a staged protocol writes its checkpoint under.

    Parameters
    ----------
    manifest : CheckpointManifest
        The manifest the run would write for itself.
    name : str
        What the stage is called.
    active : Iterable[str]
        The declared parameter groups the stage trained.
    position : int
        Where the stage falls in the protocol, counted from the first.

    Returns
    -------
    CheckpointManifest
        The same manifest, naming the stage it was written at.

    Raises
    ------
    TypeError
        If the name is not a string or the position is not an integer.
    ValueError
        If the name or a group name is empty, if the position is negative, or if
        two entries name one group.
    """
    return manifest.with_(
        stage=StageRecord(name=name, active=tuple(sorted(active)), position=position)
    )


def write_manifest(manifest: CheckpointManifest, /) -> str:
    """Return the manifest as the text a checkpoint stores.

    Parameters
    ----------
    manifest : CheckpointManifest
        The manifest to write.

    Returns
    -------
    str
        One JSON object from which ``read_manifest`` rebuilds the manifest exactly.
    """
    return manifest.model_dump_json()


def read_manifest(text: str, /) -> CheckpointManifest:
    """Return the manifest a checkpoint stored.

    Parameters
    ----------
    text : str
        What ``write_manifest`` returned.

    Returns
    -------
    CheckpointManifest
        The manifest, equal to the one written.

    Raises
    ------
    ValueError
        If the text is not a JSON object describing a manifest, naming what is
        wrong with it.
    """
    try:
        return CheckpointManifest.model_validate_json(text)

    except dx.ValidationError as error:
        msg = f"the text does not describe a checkpoint manifest: {error}"

        raise ValueError(msg) from error
