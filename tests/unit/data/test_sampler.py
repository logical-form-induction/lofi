"""Tests for declared batch partitioning."""

import pytest

from lofi.data.sampler import (
    BatchPartition,
    BucketOccupancy,
    LengthBucketBatchSampler,
    RandomBatchSampler,
    SequentialBatchSampler,
)


def rich_lengths() -> tuple[int, ...]:
    """Return varied lengths whose partitions expose ordering differences."""
    return tuple((index * 7) % 13 for index in range(47))


def test_one_seed_repeats_across_instances_and_partitions() -> None:
    first = LengthBucketBatchSampler(3, reshuffle_each_epoch=True)
    second = LengthBucketBatchSampler(3, reshuffle_each_epoch=True)

    assert first.partition(rich_lengths(), 4, seed=31) == first.partition(
        rich_lengths(), 4, seed=31
    )
    assert first.partition(rich_lengths(), 4, seed=31) == second.partition(
        rich_lengths(), 4, seed=31
    )


def test_another_seed_changes_a_rich_partition() -> None:
    sampler = LengthBucketBatchSampler(3, reshuffle_each_epoch=True)

    assert sampler.partition(rich_lengths(), 4, seed=31) != sampler.partition(
        rich_lengths(), 4, seed=32
    )


def test_every_strategy_partitions_each_index_once() -> None:
    samplers = (
        LengthBucketBatchSampler(3, reshuffle_each_epoch=False),
        RandomBatchSampler(reshuffle_each_epoch=False),
        SequentialBatchSampler(),
    )

    for sampler in samplers:
        partition = sampler.partition(rich_lengths(), 6, seed=7)
        flattened = tuple(index for batch in partition for index in batch)

        assert sorted(flattened) == list(range(len(rich_lengths())))
        assert len(flattened) == len(set(flattened))
        assert sorted(map(len, partition)) == [5, 6, 6, 6, 6, 6, 6, 6]


def test_random_and_sequential_orders_are_distinct() -> None:
    sequential = SequentialBatchSampler().partition(rich_lengths(), 4, seed=9)
    random = RandomBatchSampler(reshuffle_each_epoch=False).partition(
        rich_lengths(), 4, seed=9
    )

    assert sequential.batches[0] == (0, 1, 2, 3)
    assert random.batches != sequential.batches


def test_a_declared_bucket_factor_changes_padding_occupancy() -> None:
    lengths = tuple(range(1, 65))
    local = LengthBucketBatchSampler(1, reshuffle_each_epoch=False).partition(
        lengths, 4, seed=19
    )
    global_ = LengthBucketBatchSampler(16, reshuffle_each_epoch=False).partition(
        lengths, 4, seed=19
    )

    assert global_.occupancy.padded < local.occupancy.padded
    assert global_.occupancy.rate > local.occupancy.rate


def test_the_occupancy_report_counts_live_capacity_and_padding() -> None:
    lengths = rich_lengths()
    partition = LengthBucketBatchSampler(3, reshuffle_each_epoch=False).partition(
        lengths, 5, seed=19
    )
    expected_capacity = sum(
        len(batch) * max(lengths[index] for index in batch) for batch in partition
    )

    assert partition.occupancy == BucketOccupancy(sum(lengths), expected_capacity)
    assert partition.occupancy.padded == expected_capacity - sum(lengths)


def test_empty_and_zero_length_partitions_have_zero_occupancy() -> None:
    sampler = LengthBucketBatchSampler(2, reshuffle_each_epoch=False)
    empty = sampler.partition((), 3, seed=0)
    zero = sampler.partition((0, 0, 0), 2, seed=0)

    assert empty.batches == ()
    assert empty.occupancy == BucketOccupancy(0, 0)
    assert zero.occupancy == BucketOccupancy(0, 0)


@pytest.mark.parametrize(
    ("lengths", "batch_size", "seed", "message"),
    [
        ((1, -1), 2, 0, "negative"),
        ((1, 2), 0, 0, "not positive"),
        ((1, 2), 2, True, "not an integer"),
    ],
)
def test_invalid_partition_arguments_are_refused(
    lengths: tuple[int, ...], batch_size: int, seed: int, message: str
) -> None:
    sampler = LengthBucketBatchSampler(2, reshuffle_each_epoch=False)
    error = TypeError if isinstance(seed, bool) else ValueError

    with pytest.raises(error, match=message):
        sampler.partition(lengths, batch_size, seed=seed)


def test_invalid_partitions_and_occupancy_are_refused() -> None:
    with pytest.raises(ValueError, match="partition"):
        BatchPartition((1, 2), 2, ((0,),))

    with pytest.raises(ValueError, match="exceed"):
        BucketOccupancy(4, 3)
