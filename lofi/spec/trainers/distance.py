"""Distance-pretraining declaration."""

from types import MappingProxyType
from typing import Literal, override

import didactic.api as dx

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi.spec.components.base import RunRole
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.trainers.checkpoints import CheckpointPolicySpec
from lofi.spec.type_samplers.distribution import (
    _DEFAULT_BRANCHING,
    _DEFAULT_CHILD_DEPTH,
    ArityBranchingSpec,
    ChildDepthSpec,
)

__all__ = ["DepthCurriculumSpec", "DistanceTrainerSpec"]


class DepthCurriculumSpec(dx.Model, extra="forbid"):
    """Declares linear growth from one depth through the finite inventory.

    Attributes
    ----------
    start_depth : int
        Positive first available depth.
    steps : int
        Positive number of optimizer positions over which depth grows.
    """

    start_depth: int
    steps: int

    @dx.validates("start_depth", "steps")
    @classmethod
    def _counts_are_positive(cls, value: int) -> int:
        """Validate a positive depth or step count."""
        return ensure_positive("a depth-curriculum count", value)


class DistanceTrainerSpec(TrainerSpec):
    """Declares step-based type-distance pretraining.

    Attributes
    ----------
    kind : Literal["distance"]
        Registry tag.
    steps : int
        Positive optimizer-step horizon.
    batch_size : int
        Positive number of type pairs in one step.
    checkpoints : CheckpointPolicySpec
        Resumable checkpoint cadence, counted in completed steps, and retention.
    ball_radius : int | None
        Optional closed edit-distance radius for the second type of each pair.
    depth_curriculum : DepthCurriculumSpec | None
        Optional linear depth curriculum.
    inverse_distance_power : float
        Finite nonnegative exponent for inverse distance pair weights.
    embedding_penalty : float
        Finite nonnegative coefficient on each pair's embedding norm.
    child_depth : ChildDepthSpec
        Relative mass over feasible nondeepest child depths.
    branching : ArityBranchingSpec
        Deepest-child weights declared independently by constructor arity.
    constructor_weights : dict[str, float]
        Finite nonnegative weights keyed by constructor name.
    max_grad_norm : float | None
        Positive finite clipping norm, or ``None`` for no clipping.
    """

    kind: Literal["distance"] = "distance"
    dataset: str
    encoder: str
    sort: str
    steps: int
    batch_size: int
    checkpoints: CheckpointPolicySpec
    ball_radius: int | None = None
    depth_curriculum: DepthCurriculumSpec | None = None
    inverse_distance_power: float
    embedding_penalty: float
    child_depth: ChildDepthSpec = _DEFAULT_CHILD_DEPTH
    branching: ArityBranchingSpec = _DEFAULT_BRANCHING
    constructor_weights: dict[str, float] = dx.field(default_factory=dict)
    max_grad_norm: float | None = None

    @dx.validates("dataset", "encoder", "sort")
    @classmethod
    def _roles_are_named(cls, value: str) -> str:
        """Return a nonempty encoder role or represented sort."""
        return ensure_name("a distance-training binding", value)

    @dx.validates("steps", "batch_size")
    @classmethod
    def _counts_are_positive(cls, value: int) -> int:
        """Validate a positive step or batch count."""
        return ensure_positive("a distance-training count", value)

    @override
    def halve_batch(self) -> TrainerSpec | None:
        """Return a copy whose batch is halved, down to one."""
        return (
            None
            if self.batch_size == 1
            else self.with_(batch_size=max(1, self.batch_size // 2))
        )

    @dx.validates("ball_radius")
    @classmethod
    def _radius_is_nonnegative(cls, value: int | None) -> int | None:
        """Validate the optional closed-ball radius."""
        if value is None:
            return None

        return ensure_non_negative("ball_radius", value)

    @dx.validates("constructor_weights")
    @classmethod
    def _constructor_weights_are_nonnegative(
        cls, values: dict[str, float]
    ) -> dict[str, float]:
        """Validate finite nonnegative constructor weights."""
        checked: dict[str, float] = {}

        for name, value in values.items():
            if not name:
                msg = "a constructor weight has an empty name"
                raise ValueError(msg)

            held = ensure_finite(f"constructor_weights[{name!r}]", value)

            if held < 0.0:
                msg = f"constructor_weights[{name!r}] is negative: {value!r}"
                raise ValueError(msg)

            checked[name] = held

        return checked

    @dx.validates("inverse_distance_power", "embedding_penalty")
    @classmethod
    def _coefficients_are_nonnegative(cls, value: float) -> float:
        """Validate a finite nonnegative objective coefficient."""
        checked = ensure_finite("a distance-training coefficient", value)

        if checked < 0.0:
            msg = f"a distance-training coefficient is negative: {value!r}"
            raise ValueError(msg)

        return checked

    @dx.validates("max_grad_norm")
    @classmethod
    def _clipping_is_positive(cls, value: float | None) -> float | None:
        """Validate the optional gradient clipping norm."""
        if value is None:
            return None

        checked = ensure_finite("max_grad_norm", value)

        if checked <= 0.0:
            msg = f"max_grad_norm is not positive: {value!r}"
            raise ValueError(msg)

        return checked

    @override
    def required_roles(self) -> frozenset[RunRole]:
        """Return the term encoder it fits and the sampler it draws terms from."""
        return frozenset({RunRole.TERM_ENCODERS, RunRole.TYPE_SAMPLER})

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        """Return every parsing, decoding, loss and evaluation role.

        Distance training fits the term encoder alone to its own objective.
        """
        return frozenset(
            {
                RunRole.PARSER,
                RunRole.TOKENS_ENCODER,
                RunRole.SUBWORD_POOLER,
                RunRole.CONTROLLER,
                RunRole.PAIR_SAMPLER,
                RunRole.TERM_DECODERS,
                RunRole.READOUTS,
                RunRole.SPAN_DECODERS,
                RunRole.LOSS,
                RunRole.EVALUATORS,
            }
        )

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Require a sentence dataset supplying the learner language."""
        return MappingProxyType(
            {self.dataset: frozenset({TrainingBatchKind.SENTENCES})}
        )
