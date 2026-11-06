"""Declared deterministic batch partitioning."""

from random import Random
from typing import TYPE_CHECKING, Protocol, override

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "BatchPartition",
    "BatchSampler",
    "BucketOccupancy",
    "LengthBucketBatchSampler",
    "RandomBatchSampler",
    "SequentialBatchSampler",
]


def _seed(value: int, /) -> int:
    """Return an exact integer partition seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"seed is not an integer: {value!r}"
        raise TypeError(msg)

    return value


class BucketOccupancy(FrozenValue):
    """Padding occupancy achieved by a batch partition."""

    _noun = "batch occupancy"
    __slots__ = ("_capacity", "_live", "_padded", "_rate")

    def __init__(self, live: int, capacity: int) -> None:
        held_live = ensure_non_negative("live positions", live)
        held_capacity = ensure_non_negative("capacity positions", capacity)

        if held_live > held_capacity:
            msg = (
                f"live positions {held_live} exceed capacity positions {held_capacity}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_live", held_live)
        object.__setattr__(self, "_capacity", held_capacity)
        object.__setattr__(self, "_padded", held_capacity - held_live)
        object.__setattr__(
            self,
            "_rate",
            0.0 if held_capacity == 0 else held_live / held_capacity,
        )

    @property
    def live(self) -> int:
        """Return the number of positions containing data."""
        return self._live

    @property
    def capacity(self) -> int:
        """Return the number of allocated positions."""
        return self._capacity

    @property
    def padded(self) -> int:
        """Return the number of positions holding padding."""
        return self._padded

    @property
    def rate(self) -> float:
        """Return the share of allocated positions containing data."""
        return self._rate

    @property
    def _fields(self) -> tuple[int, int, int, float]:
        return (self._live, self._capacity, self._padded, self._rate)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another occupancy has the same counts and rate."""
        return type(other) is BucketOccupancy and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over the occupancy fields."""
        return hash((BucketOccupancy, self._fields))


class BatchPartition(FrozenValue):
    """An immutable partition of item indices into padded batches."""

    _noun = "batch partition"
    __slots__ = ("_batch_size", "_batches", "_lengths", "_occupancy")

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        batches: Sequence[Sequence[int]],
        /,
    ) -> None:
        held_lengths = tuple(
            ensure_non_negative(f"lengths[{index}]", length)
            for index, length in enumerate(lengths)
        )
        held_size = ensure_positive("batch_size", batch_size)
        held_batches = tuple(tuple(batch) for batch in batches)
        flattened = tuple(index for batch in held_batches for index in batch)

        if sorted(flattened) != list(range(len(held_lengths))):
            msg = "batches do not partition every item index exactly once"
            raise ValueError(msg)

        if any(not batch or len(batch) > held_size for batch in held_batches):
            msg = f"a batch is empty or larger than batch_size {held_size}"
            raise ValueError(msg)

        capacity = sum(
            len(batch) * max(held_lengths[index] for index in batch)
            for batch in held_batches
        )
        object.__setattr__(self, "_lengths", held_lengths)
        object.__setattr__(self, "_batch_size", held_size)
        object.__setattr__(self, "_batches", held_batches)
        object.__setattr__(
            self,
            "_occupancy",
            BucketOccupancy(sum(held_lengths), capacity),
        )

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return item lengths in index order."""
        return self._lengths

    @property
    def batch_size(self) -> int:
        """Return the maximum number of items per batch."""
        return self._batch_size

    @property
    def batches(self) -> tuple[tuple[int, ...], ...]:
        """Return the complete partition."""
        return self._batches

    @property
    def occupancy(self) -> BucketOccupancy:
        """Return the partition's padding occupancy."""
        return self._occupancy

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Return a fresh iterator over the fixed batches."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._batches)

    @property
    def _fields(
        self,
    ) -> tuple[tuple[int, ...], int, tuple[tuple[int, ...], ...], BucketOccupancy]:
        return self._lengths, self._batch_size, self._batches, self._occupancy

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another partition contains the same batches."""
        return type(other) is BatchPartition and self._fields == other._fields

    @override
    def __hash__(self) -> int:
        """Return a hash over the complete partition."""
        return hash((BatchPartition, self._fields))


class BatchSampler(Protocol):
    """Partitions ordered item lengths into one deterministic epoch."""

    @property
    def reshuffle_each_epoch(self) -> bool:
        """Return whether training derives a new seed for each epoch."""
        ...

    def partition(
        self, lengths: Sequence[int], batch_size: int, *, seed: int
    ) -> BatchPartition:
        """Return one complete batch partition."""
        ...


class LengthBucketBatchSampler(FrozenValue):
    """Partition shuffled indices through declared length-local buckets."""

    _noun = "length-bucket batch sampler"
    __slots__ = ("_bucket_factor", "_reshuffle")

    def __init__(self, bucket_factor: int, *, reshuffle_each_epoch: bool) -> None:
        object.__setattr__(
            self, "_bucket_factor", ensure_positive("bucket_factor", bucket_factor)
        )
        object.__setattr__(self, "_reshuffle", reshuffle_each_epoch)

    @property
    def bucket_factor(self) -> int:
        """Return how many batches are sorted together by length."""
        return self._bucket_factor

    @property
    def reshuffle_each_epoch(self) -> bool:
        """Return whether each epoch derives a new partition seed."""
        return self._reshuffle

    def partition(
        self, lengths: Sequence[int], batch_size: int, *, seed: int
    ) -> BatchPartition:
        """Return a shuffled, locally length-sorted partition."""
        held_lengths = tuple(lengths)
        held_size = ensure_positive("batch_size", batch_size)
        generator = Random(_seed(seed))
        indices = list(range(len(held_lengths)))
        generator.shuffle(indices)
        bucket_size = self._bucket_factor * held_size
        batches: list[tuple[int, ...]] = []

        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            bucket.sort(key=held_lengths.__getitem__, reverse=True)
            batches.extend(
                tuple(bucket[offset : offset + held_size])
                for offset in range(0, len(bucket), held_size)
            )

        generator.shuffle(batches)
        return BatchPartition(held_lengths, held_size, batches)


class RandomBatchSampler(FrozenValue):
    """Partition a random permutation of item indices."""

    _noun = "random batch sampler"
    __slots__ = ("_reshuffle",)

    def __init__(self, *, reshuffle_each_epoch: bool) -> None:
        object.__setattr__(self, "_reshuffle", reshuffle_each_epoch)

    @property
    def reshuffle_each_epoch(self) -> bool:
        """Return whether each epoch derives a new partition seed."""
        return self._reshuffle

    def partition(
        self, lengths: Sequence[int], batch_size: int, *, seed: int
    ) -> BatchPartition:
        """Return a partition of one seeded random permutation."""
        held_lengths = tuple(lengths)
        held_size = ensure_positive("batch_size", batch_size)
        indices = list(range(len(held_lengths)))
        Random(_seed(seed)).shuffle(indices)
        batches = tuple(
            tuple(indices[start : start + held_size])
            for start in range(0, len(indices), held_size)
        )
        return BatchPartition(held_lengths, held_size, batches)


class SequentialBatchSampler(FrozenValue):
    """Partition item indices in their declared order."""

    _noun = "sequential batch sampler"
    __slots__ = ()

    @property
    def reshuffle_each_epoch(self) -> bool:
        """Return false because sequential order is fixed."""
        return False

    def partition(
        self, lengths: Sequence[int], batch_size: int, *, seed: int
    ) -> BatchPartition:
        """Return a stable index-order partition."""
        _seed(seed)
        held_lengths = tuple(lengths)
        held_size = ensure_positive("batch_size", batch_size)
        batches = tuple(
            tuple(range(start, min(start + held_size, len(held_lengths))))
            for start in range(0, len(held_lengths), held_size)
        )
        return BatchPartition(held_lengths, held_size, batches)
