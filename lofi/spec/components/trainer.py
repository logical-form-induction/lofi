"""The trainer, optimizer, scheduler and type-sampler families of a run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import didactic.api as dx

from lofi.spec.components.base import ComponentSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.components.base import RunRole
    from lofi.spec.components.data import TrainingBatchKind

_DEFAULT_ADAM_LEARNING_RATE: Final = 1e-3
_DEFAULT_ADAM_BETAS: Final = (0.9, 0.999)
_DEFAULT_ADAM_EPSILON: Final = 1e-8
_DEFAULT_UNREGULARIZED_WEIGHT_DECAY: Final = 0.0
_DEFAULT_WARMUP_STEPS: Final = 0
_DEFAULT_WARMUP_START_FACTOR: Final = 1e-8
__all__ = [
    "BatchSamplerSpec",
    "EpochPartitionSpec",
    "OptimizerSpec",
    "PairSamplerSpec",
    "SchedulerSpec",
    "TrainerSpec",
    "TypeSamplerSpec",
]


@runtime_checkable
class EpochPartitionSpec(Protocol):
    """Publish whether a batch declaration changes its partition across epochs."""

    def reshuffles(self) -> bool:
        """Return whether each epoch uses a distinct partition seed."""


class BatchSamplerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares how a trainer partitions one epoch into batches."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists batch samplers."""
        return "batch_sampler"


class TrainerSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a trainer."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "trainer"

    def refuse_epoch(self, occurrences: int, /) -> None:
        """Refuse a dataset whose epochs this trainer cannot divide into its steps.

        A trainer that divides no epoch accepts every dataset.

        Parameters
        ----------
        occurrences : int
            How many item occurrences the dataset holds.
        """

    def refuse_loss_terms(self, terms: frozenset[str], /) -> None:
        """Refuse a trainer that names a loss term the run does not declare.

        A trainer that names no loss term, or no metric of one, accepts every
        loss section.

        Parameters
        ----------
        terms : frozenset[str]
            The names the run's loss terms are reported under.
        """

    def required_roles(self) -> frozenset[RunRole]:
        """Return the run roles the trainer cannot train without."""
        return frozenset()

    def forbidden_roles(self) -> frozenset[RunRole]:
        """Return the run roles the trainer does not consume.

        A run that declares one of them would train no differently, so it is
        refused rather than run with an inert component.
        """
        return frozenset()

    def dataset_requirements(self) -> Mapping[str, frozenset[TrainingBatchKind]]:
        """Return required batch capabilities keyed by training-dataset role."""
        return {}

    def evaluation_dataset(self) -> str | None:
        """Return the dataset role held-out evaluation reads, if one exists."""
        return None

    def featurized_datasets(self) -> frozenset[str]:
        """Return dataset roles whose batches this trainer featurizes."""
        return frozenset()

    def halve_batch(self) -> TrainerSpec | None:
        """Return this trainer with every effective batch halved, if supported."""
        return None


class OptimizerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares an optimizer."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "optimizer"


class SchedulerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a learning-rate scheduler."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "scheduler"


class TypeSamplerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a type sampler."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "type_sampler"


class PairSamplerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a labeled rule-instance sampler."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "pair_sampler"
