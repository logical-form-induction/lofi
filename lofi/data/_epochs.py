"""Seeded training epochs a dataset route builds from its own occurrences.

A dataset plugin owns how its lowered occurrences become training batches.
Training supplies the featurizer, the batch sampler and the seeds; the route
supplies which occurrences fall in which split, how one partition of them is
batched, and which split trains. Every other split is a retained partition.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final

from lofi._checks import ensure_name, ensure_positive
from lofi._frozen import FrozenValue, Unpicklable
from lofi.spec import derive_seed

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data.sampler import BatchSampler

__all__ = [
    "DatasetEpoch",
    "EpochBatching",
    "EpochSource",
    "PartitionBatches",
    "SeededEpochs",
]


@final
class DatasetEpoch[B](FrozenValue):
    """Immutable training batches and exact named retained partitions.

    Parameters
    ----------
    training : tuple[B, ...]
        The epoch's training batches.
    retained : Mapping[str, tuple[B, ...]]
        The batches of each retained split, in declaration order.
    """

    _noun = "dataset epoch"

    __slots__ = ("_retained", "_training")

    def __init__(
        self,
        training: tuple[B, ...],
        retained: Mapping[str, tuple[B, ...]],
    ) -> None:
        object.__setattr__(self, "_training", training)
        object.__setattr__(self, "_retained", MappingProxyType(dict(retained)))

    @property
    def training(self) -> tuple[B, ...]:
        """Return this epoch's seeded training batches."""
        return self._training

    @property
    def retained(self) -> Mapping[str, tuple[B, ...]]:
        """Return exact named retained partitions in declaration order."""
        return self._retained


class EpochSource[B](Protocol):
    """Build deterministic batches for a requested epoch."""

    def at(self, epoch: int, /) -> DatasetEpoch[B]:
        """Return the training and retained batches of one epoch."""
        ...


@final
class EpochBatching(FrozenValue):
    """How training partitions a dataset's occurrences into seeded batches.

    Parameters
    ----------
    batch_sampler : BatchSampler
        The declared strategy that partitions one split's occurrences.
    batch_size : int
        The most occurrences one batch holds.
    training_seed : int
        The root seed of the training batches.
    validation_seed : int
        The root seed of the retained batches.
    """

    _noun = "epoch batching"

    __slots__ = ("_batch_sampler", "_batch_size", "_training_seed", "_validation_seed")

    def __init__(
        self,
        batch_sampler: BatchSampler,
        *,
        batch_size: int,
        training_seed: int,
        validation_seed: int,
    ) -> None:
        object.__setattr__(self, "_batch_sampler", batch_sampler)
        object.__setattr__(
            self, "_batch_size", ensure_positive("batch_size", batch_size)
        )
        object.__setattr__(self, "_training_seed", training_seed)
        object.__setattr__(self, "_validation_seed", validation_seed)

    @property
    def batch_sampler(self) -> BatchSampler:
        """Return the declared batch sampler."""
        return self._batch_sampler

    @property
    def batch_size(self) -> int:
        """Return the most occurrences one batch holds."""
        return self._batch_size

    @property
    def training_seed(self) -> int:
        """Return the root seed of the training batches."""
        return self._training_seed

    @property
    def validation_seed(self) -> int:
        """Return the root seed of the retained batches."""
        return self._validation_seed


class PartitionBatches[B](Protocol):
    """Batch one split's occurrences under one sampler seed."""

    def __call__(self, positions: tuple[int, ...], seed: int, /) -> tuple[B, ...]:
        """Return the complete batch partition of the selected occurrences."""
        ...


@final
class SeededEpochs[B](FrozenValue, Unpicklable):
    """Epochs over split occurrences, reshuffled under epoch-local seeds.

    The first epoch is built once. Later epochs reuse it unless the batch
    sampler reshuffles each epoch, in which case the training batches are
    drawn under a seed derived from the epoch index. Retained partitions keep
    one seed per split.

    Parameters
    ----------
    splits : Mapping[str, tuple[int, ...]]
        The occurrence positions of each split, in first-seen order.
    training_split : str
        The split that trains; every other split is retained.
    batches : PartitionBatches[B]
        How one split's occurrences become batches.
    batching : EpochBatching
        The declared sampler, batch size and seeds.

    Raises
    ------
    ValueError
        If the training split holds no occurrence or no split is retained.
    """

    _noun = "seeded epoch source"
    _because = "holds a configured featurizer"

    __slots__ = ("_batches", "_batching", "_first", "_retained", "_training")

    def __init__(
        self,
        splits: Mapping[str, tuple[int, ...]],
        *,
        training_split: str,
        batches: PartitionBatches[B],
        batching: EpochBatching,
    ) -> None:
        name = ensure_name("a training split", training_split)
        training = splits.get(name, ())

        if not training:
            msg = f"the dataset holds no occurrence in its training split {name!r}"
            raise ValueError(msg)

        retained = {split: held for split, held in splits.items() if split != name}

        if not retained:
            msg = f"the dataset retains no split beside its training split {name!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_training", training)
        object.__setattr__(self, "_retained", MappingProxyType(retained))
        object.__setattr__(self, "_batches", batches)
        object.__setattr__(self, "_batching", batching)
        object.__setattr__(self, "_first", self._build(0))

    def _seed(self, epoch: int, /) -> int:
        """Return the fixed or epoch-local training seed."""
        root = self._batching.training_seed

        if not self._batching.batch_sampler.reshuffle_each_epoch:
            return root

        return derive_seed(root, f"epoch:{epoch}")

    def _build(self, epoch: int, /) -> DatasetEpoch[B]:
        """Build one complete epoch from its split occurrences."""
        validation = self._batching.validation_seed

        return DatasetEpoch(
            self._batches(self._training, self._seed(epoch)),
            {
                split: self._batches(held, derive_seed(validation, f"retained:{split}"))
                for split, held in self._retained.items()
            },
        )

    def at(self, epoch: int, /) -> DatasetEpoch[B]:
        """Return one fixed or deterministically reshuffled epoch.

        Raises
        ------
        TypeError
            If the epoch index is not an integer.
        ValueError
            If the epoch index is negative.
        """
        if type(epoch) is not int:
            msg = f"an epoch index is not an integer: {epoch!r}"
            raise TypeError(msg)

        if epoch < 0:
            msg = f"an epoch index is negative: {epoch!r}"
            raise ValueError(msg)

        if epoch == 0 or not self._batching.batch_sampler.reshuffle_each_epoch:
            return self._first

        return self._build(epoch)
