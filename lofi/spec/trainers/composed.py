"""Alternating and staged trainer declarations."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self, override

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.mixing import MixingSpec
from lofi.spec.components.trainer import EpochPartitionSpec, TrainerSpec
from lofi.spec.trainers.parser import ParserTrainerSpec
from lofi.spec.trainers.steps import (
    StepSpec,
)

if TYPE_CHECKING:
    from lofi.spec.components.base import RunRole
    from lofi.spec.components.data import TrainingBatchKind

__all__ = ["AlternatingTrainerSpec", "StageSpec", "StagedTrainerSpec"]


def _repeated(values: tuple[str, ...], noun: str) -> None:
    """Refuse repeated names in one declared ordered collection."""
    repeated = sorted({value for value in values if values.count(value) > 1})

    if repeated:
        msg = f"a {noun} names entries more than once: {repeated}"
        raise ValueError(msg)


class StageSpec(dx.Model, extra="forbid"):
    """Declare one task interval and the parameter groups it leaves trainable."""

    name: str
    step: StepSpec
    active_groups: tuple[str, ...]
    epochs: int

    @dx.validates("name")
    @classmethod
    def _name_is_nonempty(cls, value: str) -> str:
        """Validate the stage name written into its checkpoint manifest."""
        return ensure_name("a training-stage name", value)

    @dx.validates("active_groups")
    @classmethod
    def _groups_are_named_once(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate a nonempty, distinct collection of active roles."""
        held = tuple(ensure_name("an active group", value) for value in values)

        if not held:
            msg = "a training stage declares at least one active group"
            raise ValueError(msg)

        _repeated(held, "training stage")
        return held

    @dx.validates("epochs")
    @classmethod
    def _epochs_are_positive(cls, value: int) -> int:
        """Validate the number of passes in this stage."""
        return ensure_positive("a training-stage epoch count", value)


class AlternatingTrainerSpec(TrainerSpec):
    """Compose named objective views under one declared mixing policy."""

    kind: Literal["alternating"] = "alternating"
    base: ParserTrainerSpec
    tasks: tuple[StepSpec, ...]
    mixing: MixingSpec

    @dx.model_validator()
    def _tasks_are_distinct(self) -> Self:
        """Require at least two uniquely named tasks."""
        if len(self.tasks) < 2:
            msg = "alternating training declares at least two tasks"
            raise ValueError(msg)

        _repeated(tuple(task.name for task in self.tasks), "alternating trainer")

        if self.base.accumulation_steps != 1:
            msg = "alternating training requires one microbatch per optimizer step"
            raise ValueError(msg)

        sampler = self.base.batch_sampler
        if not isinstance(sampler, EpochPartitionSpec):
            msg = (
                "alternating training needs a sampler that declares epoch partitioning"
            )
            raise TypeError(msg)

        if sampler.reshuffles():
            msg = "alternating training requires one fixed declared task epoch"
            raise ValueError(msg)

        return self

    @override
    def refuse_epoch(self, occurrences: int, /) -> None:
        """Delegate epoch shape validation to the parser-training base."""
        self.base.refuse_epoch(occurrences)

    @override
    def refuse_loss_terms(self, terms: frozenset[str], /) -> None:
        """Refuse a monitored metric or task objective the run's loss lacks."""
        self.base.refuse_loss_terms(terms)

        for task in self.tasks:
            task.refuse_objectives(terms)

    @override
    def required_roles(self) -> frozenset[RunRole]:
        """Return the roles required by the parser-training base."""
        return frozenset(
            {
                *self.base.required_roles(),
                *(role for task in self.tasks for role in task.sampler_roles()),
            }
        )

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        """Return the roles forbidden by the parser-training base."""
        return self.base.forbidden_roles() - self.required_roles()

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Return each task's batch requirements grouped by dataset role."""
        required = {
            role: set(kinds) for role, kinds in self.base.dataset_requirements().items()
        }

        for task in self.tasks:
            for role, kinds in task.dataset_requirements().items():
                required.setdefault(role, set()).update(kinds)

        return MappingProxyType(
            {name: frozenset(kinds) for name, kinds in required.items()}
        )

    @override
    def evaluation_dataset(self) -> str:
        """Return the shared parser dataset evaluated after fitting."""
        return self.base.dataset

    @override
    def featurized_datasets(self) -> frozenset[str]:
        """Return datasets consumed through parser-style featurizers."""
        return frozenset(
            {
                *self.base.featurized_datasets(),
                *(role for task in self.tasks for role in task.featurized_datasets()),
            }
        )

    @override
    def halve_batch(self) -> TrainerSpec | None:
        """Return this composition with its parser batch halved."""
        base = self.base.halve_batch()
        return None if base is None else self.with_(base=base)


class StagedTrainerSpec(TrainerSpec):
    """Fit one parser graph through declared active-group stages."""

    kind: Literal["staged"] = "staged"
    base: ParserTrainerSpec
    validation_stage: str | None
    stages: tuple[StageSpec, ...]

    @dx.validates("validation_stage")
    @classmethod
    def _validation_stage_is_named(cls, value: str | None) -> str | None:
        """Validate the stage that owns parser validation, when one does."""
        return None if value is None else ensure_name("a validation stage", value)

    @dx.model_validator()
    def _stages_are_distinct_and_cover_the_horizon(self) -> Self:
        """Validate one uniquely named stage per interval of the base horizon."""
        if not self.stages:
            msg = "staged training declares at least one stage"
            raise ValueError(msg)

        _repeated(tuple(stage.name for stage in self.stages), "staged trainer")

        names = tuple(stage.name for stage in self.stages)

        if self.validation_stage is not None:
            if self.validation_stage not in names:
                msg = (
                    f"validation stage {self.validation_stage!r} is not declared by "
                    "the staged trainer"
                )
                raise ValueError(msg)

            if self.validation_stage != names[-1]:
                msg = (
                    f"validation stage {self.validation_stage!r} is not the final "
                    f"stage {names[-1]!r}"
                )
                raise ValueError(msg)

        epochs = sum(stage.epochs for stage in self.stages)

        if epochs != self.base.epochs:
            msg = (
                f"staged training declares {epochs} stage epochs, not its "
                f"base horizon of {self.base.epochs}"
            )
            raise ValueError(msg)

        if self.base.active_groups is not None:
            msg = "staged training declares active groups only within its stages"
            raise ValueError(msg)

        return self

    @override
    def refuse_epoch(self, occurrences: int, /) -> None:
        """Delegate epoch shape validation to the parser-training base."""
        self.base.refuse_epoch(occurrences)

    @override
    def refuse_loss_terms(self, terms: frozenset[str], /) -> None:
        """Refuse a monitored metric or stage objective the run's loss lacks."""
        self.base.refuse_loss_terms(terms)

        for stage in self.stages:
            stage.step.refuse_objectives(terms)

    @override
    def required_roles(self) -> frozenset[RunRole]:
        """Return the roles required by the parser-training base."""
        return frozenset(
            {
                *self.base.required_roles(),
                *(role for stage in self.stages for role in stage.step.sampler_roles()),
            }
        )

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        """Return the roles forbidden by the parser-training base."""
        return self.base.forbidden_roles() - self.required_roles()

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Return every stage task's requirements grouped by dataset role."""
        required = {
            role: set(kinds) for role, kinds in self.base.dataset_requirements().items()
        }

        for stage in self.stages:
            for role, kinds in stage.step.dataset_requirements().items():
                required.setdefault(role, set()).update(kinds)

        return MappingProxyType(
            {role: frozenset(kinds) for role, kinds in required.items()}
        )

    @override
    def evaluation_dataset(self) -> str:
        """Return the shared parser dataset evaluated after fitting."""
        return self.base.dataset

    @override
    def featurized_datasets(self) -> frozenset[str]:
        """Return every stage dataset lowered through a featurizer."""
        return frozenset(
            {
                *self.base.featurized_datasets(),
                *(
                    role
                    for stage in self.stages
                    for role in stage.step.featurized_datasets()
                ),
            }
        )

    @override
    def halve_batch(self) -> TrainerSpec | None:
        """Return this composition with its parser batch halved."""
        base = self.base.halve_batch()
        return None if base is None else self.with_(base=base)
