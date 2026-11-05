"""Term autoencoding and combinator-pretraining declaration."""

from types import MappingProxyType
from typing import Literal, Self, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.components.base import RunRole
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.trainers.checkpoints import CheckpointPolicySpec
from lofi.spec.trainers.early_stopping import AccuracyStoppingSpec
from lofi.spec.trainers.generation import TermGenerationSpec

__all__ = ["TermsTrainerSpec"]


class TermsTrainerSpec(TrainerSpec):
    """Declare generated term and combinator-pair pretraining."""

    kind: Literal["terms"] = "terms"
    dataset: str
    spaces: tuple[str, ...]
    steps: int | None = None
    stop: AccuracyStoppingSpec | None = None
    batch_size: int
    generation: TermGenerationSpec
    checkpoints: CheckpointPolicySpec
    max_grad_norm: float | None = None

    @dx.validates("dataset")
    @classmethod
    def _dataset_is_named(cls, value: str) -> str:
        """Validate the dataset supplying the learner language."""
        return ensure_name("a terms-training dataset", value)

    @dx.validates("spaces")
    @classmethod
    def _spaces_are_named_once(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate a nonempty ordered set of represented-space roles."""
        if not values:
            msg = "terms training requires at least one represented space"
            raise ValueError(msg)

        held = tuple(ensure_name("a terms-training space", value) for value in values)
        repeated = sorted({value for value in held if held.count(value) > 1})

        if repeated:
            msg = f"terms training names spaces more than once: {repeated}"
            raise ValueError(msg)

        return held

    @dx.validates("batch_size")
    @classmethod
    def _batch_is_positive(cls, value: int) -> int:
        """Validate a positive generated-term batch size."""
        return ensure_positive("a terms-training batch size", value)

    @override
    def halve_batch(self) -> TrainerSpec | None:
        """Return a copy whose generated-term batch is halved, down to one."""
        return (
            None
            if self.batch_size == 1
            else self.with_(batch_size=max(1, self.batch_size // 2))
        )

    @dx.validates("steps")
    @classmethod
    def _optional_steps_are_positive(cls, value: int | None) -> int | None:
        """Validate the optional fixed optimizer-step horizon."""
        return None if value is None else ensure_positive("steps", value)

    @dx.validates("max_grad_norm")
    @classmethod
    def _norm_is_positive(cls, value: float | None) -> float | None:
        """Validate an optional positive finite gradient-norm bound."""
        if value is None:
            return None

        checked = ensure_finite("a terms-training gradient-norm bound", value)

        if checked <= 0.0:
            msg = f"a terms-training gradient-norm bound is not positive: {value!r}"
            raise ValueError(msg)

        return checked

    @dx.model_validator()
    def _schedule_fits_the_run(self) -> Self:
        """Require exactly one schedule that visits every represented space."""
        if (self.steps is None) == (self.stop is None):
            msg = "terms training declares exactly one of steps and stop"
            raise ValueError(msg)

        horizon = self.step_horizon()

        if horizon < len(self.spaces):
            msg = (
                f"{horizon} terms-training steps do not visit all "
                f"{len(self.spaces)} represented spaces"
            )
            raise ValueError(msg)

        return self

    def step_horizon(self) -> int:
        """Return the fixed budget or accuracy target's optimizer-step ceiling."""
        if self.stop is not None:
            return self.stop.max_steps

        if self.steps is None:
            msg = "terms training has neither a fixed horizon nor an accuracy target"
            raise RuntimeError(msg)

        return self.steps

    @override
    def required_roles(self) -> frozenset[RunRole]:
        """Return the neural and sampling roles term pretraining consumes."""
        return frozenset(
            {
                RunRole.TERM_ENCODERS,
                RunRole.TERM_DECODERS,
                RunRole.PAIR_SAMPLER,
                RunRole.TYPE_SAMPLER,
            }
        )

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        """Return sentence-model roles that generated terms do not consume."""
        return frozenset(
            {
                RunRole.PARSER,
                RunRole.TOKENS_ENCODER,
                RunRole.SUBWORD_POOLER,
                RunRole.CONTROLLER,
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
