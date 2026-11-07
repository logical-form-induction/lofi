from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import lofi.data.synthetic.training._pairs as pair_training
from lofi.data._epochs import EpochBatching
from lofi.data.synthetic.training import InferencePairs, inference_training_batches
from lofi.spec import derive_seed

if TYPE_CHECKING:
    from lofi.data import Relational
    from lofi.data._ports import Featurizer
    from lofi.data.featurizers._core import FeaturizedBatch
    from lofi.data.sampler import BatchSampler
    from lofi.data.synthetic.training import PairCorpus

FEATURIZER = cast("Featurizer[Relational, FeaturizedBatch[Relational]]", object())


def _dataset(splits: tuple[str | None, ...], decided: tuple[bool, ...]) -> PairCorpus:
    return cast(
        "PairCorpus",
        SimpleNamespace(
            corpus_pairs=tuple(
                SimpleNamespace(relation="entailment" if held else None, split=split)
                for split, held in zip(splits, decided, strict=True)
            ),
            labeling="labeling",
            fingerprint="0123456789abcdef",
        ),
    )


def _batching() -> EpochBatching:
    return EpochBatching(
        cast("BatchSampler", SimpleNamespace(reshuffle_each_epoch=False)),
        batch_size=4,
        training_seed=5,
        validation_seed=9,
    )


def test_pair_training_batches_decided_pairs_by_their_declared_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    def batches(
        dataset: object, featurizer: object, /, **settings: object
    ) -> tuple[object, ...]:
        del dataset, featurizer
        positions = cast("tuple[int, ...]", settings["occurrence_positions"])
        seed = cast("int", settings["sampler_seed"])
        calls.append((positions, seed))
        return ((positions, seed),)

    monkeypatch.setattr(pair_training, "inference_training_batches", batches)
    dataset = _dataset(("fit", "dev", "fit", "dev"), (True, True, True, False))
    pairs = InferencePairs(dataset, training_split="fit")
    epoch = pairs.epochs(FEATURIZER, _batching()).at(0)

    assert pairs.corpus is dataset
    assert pairs.labeling == "labeling"
    assert epoch.training == (((0, 2), 5),)
    assert dict(epoch.retained) == {"dev": (((1,), derive_seed(9, "retained:dev")),)}
    assert "training_split='fit'" in repr(pairs)


def test_pair_training_refuses_a_decided_pair_without_a_split() -> None:
    pairs = InferencePairs(_dataset((None,), (True,)), training_split="train")

    with pytest.raises(ValueError, match="no declared split"):
        pairs.epochs(FEATURIZER, _batching())


def test_inference_batches_refuse_an_undecided_pair() -> None:
    dataset = _dataset(("train", "train"), (True, False))

    with pytest.raises(ValueError, match="not decided pairs"):
        inference_training_batches(
            dataset,
            FEATURIZER,
            occurrence_positions=(1,),
            batch_sampler=cast("BatchSampler", object()),
            batch_size=1,
            sampler_seed=0,
            step=0,
        )
