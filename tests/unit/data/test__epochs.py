from typing import TYPE_CHECKING, cast

import pytest

from lofi.data._epochs import DatasetEpoch, EpochBatching, SeededEpochs
from lofi.spec import derive_seed

if TYPE_CHECKING:
    from lofi.data.sampler import BatchSampler


class _Sampler:
    def __init__(self, *, reshuffle: bool) -> None:
        self.reshuffle_each_epoch = reshuffle


def _batching(*, reshuffle: bool) -> EpochBatching:
    return EpochBatching(
        cast("BatchSampler", _Sampler(reshuffle=reshuffle)),
        batch_size=2,
        training_seed=11,
        validation_seed=23,
    )


def _record(
    calls: list[tuple[tuple[int, ...], int]],
) -> SeededEpochs[tuple[tuple[int, ...], int]]:
    def batches(
        positions: tuple[int, ...], seed: int, /
    ) -> tuple[tuple[tuple[int, ...], int], ...]:
        calls.append((positions, seed))
        return ((positions, seed),)

    return SeededEpochs(
        {"train": (0, 2), "dev": (1,), "test": (3,)},
        training_split="train",
        batches=batches,
        batching=_batching(reshuffle=True),
    )


def test_the_training_split_trains_and_every_other_split_is_retained() -> None:
    calls: list[tuple[tuple[int, ...], int]] = []
    epoch = _record(calls).at(0)

    assert epoch.training == (((0, 2), derive_seed(11, "epoch:0")),)
    assert dict(epoch.retained) == {
        "dev": (((1,), derive_seed(23, "retained:dev")),),
        "test": (((3,), derive_seed(23, "retained:test")),),
    }


def test_the_first_epoch_is_built_once_and_later_epochs_reshuffle() -> None:
    calls: list[tuple[tuple[int, ...], int]] = []
    source = _record(calls)
    built = len(calls)

    assert source.at(0) is source.at(0)
    assert len(calls) == built

    later = source.at(3)

    assert later.training == (((0, 2), derive_seed(11, "epoch:3")),)
    assert later.retained == source.at(0).retained


def test_a_fixed_sampler_reuses_the_first_epoch() -> None:
    source = SeededEpochs(
        {"train": (0,), "dev": (1,)},
        training_split="train",
        batches=lambda positions, seed: ((positions, seed),),
        batching=_batching(reshuffle=False),
    )

    assert source.at(5) is source.at(0)


def test_a_declared_training_split_name_selects_the_training_occurrences() -> None:
    source = SeededEpochs(
        {"fit": (0,), "train": (1,)},
        training_split="fit",
        batches=lambda positions, seed: ((positions, seed),),
        batching=_batching(reshuffle=False),
    )

    assert source.at(0).training == (((0,), 11),)
    assert tuple(source.at(0).retained) == ("train",)


def test_epochs_refuse_an_empty_training_split_and_no_retained_split() -> None:
    with pytest.raises(ValueError, match="no occurrence in its training split"):
        SeededEpochs(
            {"dev": (0,)},
            training_split="train",
            batches=lambda positions, seed: ((positions, seed),),
            batching=_batching(reshuffle=False),
        )

    with pytest.raises(ValueError, match="retains no split"):
        SeededEpochs(
            {"train": (0,)},
            training_split="train",
            batches=lambda positions, seed: ((positions, seed),),
            batching=_batching(reshuffle=False),
        )


def test_an_epoch_index_is_a_nonnegative_integer() -> None:
    source = _record([])

    with pytest.raises(TypeError, match="not an integer"):
        source.at(cast("int", 1.0))

    with pytest.raises(ValueError, match="negative"):
        source.at(-1)


def test_batching_refuses_a_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EpochBatching(
            cast("BatchSampler", _Sampler(reshuffle=False)),
            batch_size=0,
            training_seed=0,
            validation_seed=0,
        )


def test_a_dataset_epoch_copies_its_retained_partitions() -> None:
    retained = {"dev": (3,)}
    epoch = DatasetEpoch((1, 2), retained)
    retained["test"] = (4,)

    assert epoch.training == (1, 2)
    assert dict(epoch.retained) == {"dev": (3,)}
