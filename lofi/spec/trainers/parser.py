"""Parser-trainer declaration."""

from math import ceil
from types import MappingProxyType
from typing import Literal, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive
from lofi.spec.components.base import RunRole
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.components.trainer import BatchSamplerSpec, TrainerSpec
from lofi.spec.trainers import early_stopping as _early_stopping
from lofi.spec.trainers.checkpoints import CheckpointPolicySpec

__all__ = ["ParserTrainerSpec"]


class ParserTrainerSpec(TrainerSpec):
    """Declares one parser-training run.

    Attributes
    ----------
    kind : Literal["parser"]
        Registry tag.
    epochs : int
        Positive number of passes over the training data.
    batch_size : int
        Positive number of items in one microbatch.
    accumulation_steps : int
        Positive number of microbatches in one optimizer step.
    max_grad_norm : float | None
        Positive finite clipping norm, or ``None`` for no clipping.
    active_groups : tuple[str, ...] | None
        Distinct parameter-group names to train, or ``None`` to train every
        declared group. A group whose component is declared frozen is not
        trained even when named.
    batch_sampler : BatchSamplerSpec
        Declared batch partition and epoch-reshuffle policy.
    early_stopping : _early_stopping.EarlyStoppingSpec
        Stopping rule and retained-best checkpoint retention.
    checkpoints : CheckpointPolicySpec
        Resumable checkpoint cadence, counted in completed epochs, and retention.
    """

    kind: Literal["parser"] = "parser"
    dataset: str
    epochs: int
    batch_size: int
    accumulation_steps: int = 1
    max_grad_norm: float | None = None
    active_groups: tuple[str, ...] | None = None
    batch_sampler: BatchSamplerSpec
    early_stopping: _early_stopping.EarlyStoppingSpec
    checkpoints: CheckpointPolicySpec

    @dx.validates("epochs")
    @classmethod
    def _epochs_are_positive(cls, value: int) -> int:
        """Validate the epoch count."""
        return ensure_positive("epochs", value)

    @dx.validates("batch_size")
    @classmethod
    def _batch_size_is_positive(cls, value: int) -> int:
        """Validate the microbatch size."""
        return ensure_positive("batch_size", value)

    @override
    def halve_batch(self) -> TrainerSpec | None:
        """Return a copy whose microbatch is halved, down to one."""
        return (
            None
            if self.batch_size == 1
            else self.with_(batch_size=max(1, self.batch_size // 2))
        )

    @dx.validates("accumulation_steps")
    @classmethod
    def _accumulation_is_positive(cls, value: int) -> int:
        """Validate the number of microbatches in an optimizer step."""
        return ensure_positive("accumulation_steps", value)

    @dx.validates("dataset")
    @classmethod
    def _dataset_is_named(cls, value: str) -> str:
        """Validate the training dataset role."""
        return ensure_name("a parser-training dataset", value)

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

    @dx.validates("active_groups")
    @classmethod
    def _groups_are_named_once(
        cls, names: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Validate the optional active parameter-group names."""
        if names is None:
            return None

        for name in names:
            ensure_name("an active group name", name)

        repeated = sorted({name for name in names if names.count(name) > 1})

        if repeated:
            msg = f"active_groups names groups more than once: {repeated}"

            raise ValueError(msg)

        return names

    @override
    def refuse_epoch(self, occurrences: int, /) -> None:
        """Refuse a dataset whose epoch cannot close whole accumulation windows.

        The occurrences assigned to the training partition are taken in
        microbatches of ``batch_size``, the last possibly short, and every
        ``accumulation_steps`` microbatches close one optimizer step.

        Parameters
        ----------
        occurrences : int
            How many item occurrences the dataset holds.

        Raises
        ------
        ValueError
            If an epoch's microbatches do not divide into whole windows, naming
            ``trainer.accumulation_steps``.
        """
        microbatches = ceil(occurrences / self.batch_size)

        if microbatches % self.accumulation_steps:
            msg = (
                f"trainer.accumulation_steps: {self.accumulation_steps} cannot divide "
                f"an epoch of {occurrences} training occurrences, taken "
                f"{self.batch_size} "
                f"at a time, whose {microbatches} microbatches close no whole "
                "accumulation windows"
            )
            raise ValueError(msg)

    @override
    def refuse_loss_terms(self, terms: frozenset[str], /) -> None:
        """Refuse a monitored metric that epoch validation never reports.

        Validation reports each active loss term under ``loss/{term}`` and each
        of its diagnostics under ``loss/{term}/{diagnostic}``, so the monitored
        metric names a declared term.

        Parameters
        ----------
        terms : frozenset[str]
            The names the run's loss terms are reported under.

        Raises
        ------
        ValueError
            If the monitored metric is not a metric of a declared loss term, naming
            ``trainer.early_stopping.metric``.
        """
        metric = self.early_stopping.metric
        namespace, _, rest = metric.partition("/")
        term = rest.partition("/")[0]

        if namespace != "loss" or term not in terms:
            msg = (
                f"trainer.early_stopping.metric: validation reports no {metric!r}; "
                f"it reports loss/{{term}} and loss/{{term}}/{{diagnostic}} for the "
                f"declared loss terms {sorted(terms)}"
            )
            raise ValueError(msg)

    @override
    def required_roles(self) -> frozenset[RunRole]:
        """Return the sentence front end, parser and loss it trains.

        Parser-specific dependencies such as a controller or represented term space
        are declared by that parser and its objectives. A scalar chart therefore does
        not carry unused vector components merely to satisfy the trainer kind.
        """
        return frozenset(
            {
                RunRole.PARSER,
                RunRole.TOKENS_ENCODER,
                RunRole.SUBWORD_POOLER,
                RunRole.LOSS,
            }
        )

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        """Return samplers from which parser training draws nothing."""
        return frozenset({RunRole.PAIR_SAMPLER, RunRole.TYPE_SAMPLER})

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        """Require sentence batches from the named parser dataset."""
        return MappingProxyType(
            {self.dataset: frozenset({TrainingBatchKind.SENTENCES})}
        )

    @override
    def evaluation_dataset(self) -> str:
        """Return the parser dataset whose held-out split is evaluated."""
        return self.dataset

    @override
    def featurized_datasets(self) -> frozenset[str]:
        """Return the parser dataset lowered through its featurizer."""
        return frozenset({self.dataset})
