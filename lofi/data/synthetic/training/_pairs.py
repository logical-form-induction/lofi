"""Labeled inference-pair batches and the pair training of a pair corpus."""

from typing import TYPE_CHECKING, Protocol, Self, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._types import ItemId, ReadoutId
from lofi.data._epochs import SeededEpochs
from lofi.data._supervision import Example, Relational
from lofi.lf import ArgumentRenaming, InferenceLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.data._epochs import EpochBatching
    from lofi.data._ports import Featurizer
    from lofi.data._supervision import GoldDerivation
    from lofi.data._values import Item
    from lofi.data.featurizers._core import FeaturizedBatch
    from lofi.data.sampler import BatchSampler
    from lofi.data.synthetic._illtyped import FormTypingExample
    from lofi.data.synthetic._inference import SyntheticInference
    from lofi.lf import InferenceLabeling

__all__ = [
    "CorpusPair",
    "InferencePairs",
    "InferenceTrainingBatch",
    "PairCorpus",
    "inference_training_batches",
]


@final
class CorpusPair(FrozenValue):
    """One premise-hypothesis pair of a pair corpus, with its observed relation.

    Parameters
    ----------
    premise : Item
        The premise item.
    hypothesis : Item
        The hypothesis item.
    relation : InferenceLabel | None
        The logical relation the data records for the pair, or ``None`` when
        the pair is undecided.
    split : str | None
        The declared partition holding the pair.
    judgment : int | None
        The pair's declared ordinal response level, when the data supplies one.
    gold : tuple[str, str] | None
        The gold logical forms of the premise and the hypothesis in the
        logical-form surface syntax, when the data supplies them.
    """

    _noun = "corpus pair"
    __slots__ = ("_gold", "_hypothesis", "_judgment", "_premise", "_relation", "_split")

    def __init__(
        self,
        premise: Item,
        hypothesis: Item,
        /,
        *,
        relation: InferenceLabel | None,
        split: str | None,
        judgment: int | None = None,
        gold: tuple[str, str] | None = None,
    ) -> None:
        object.__setattr__(self, "_premise", premise)
        object.__setattr__(self, "_hypothesis", hypothesis)
        object.__setattr__(self, "_relation", relation)
        object.__setattr__(self, "_split", split)
        object.__setattr__(self, "_judgment", judgment)
        object.__setattr__(self, "_gold", gold)

    @property
    def premise(self) -> Item:
        """Return the premise item."""
        return self._premise

    @property
    def hypothesis(self) -> Item:
        """Return the hypothesis item."""
        return self._hypothesis

    @property
    def relation(self) -> InferenceLabel | None:
        """Return the observed logical relation, or ``None`` when undecided."""
        return self._relation

    @property
    def split(self) -> str | None:
        """Return the declared partition holding the pair."""
        return self._split

    @property
    def judgment(self) -> int | None:
        """Return the declared ordinal response level, when supplied."""
        return self._judgment

    @property
    def gold(self) -> tuple[str, str] | None:
        """Return the premise and hypothesis gold logical forms, when supplied."""
        return self._gold

    @override
    def __repr__(self) -> str:
        """Return the pair's item identifiers, relation and split."""
        return (
            f"CorpusPair({self._premise.item_id!r}, {self._hypothesis.item_id!r}, "
            f"relation={self._relation!r}, split={self._split!r})"
        )


class PairCorpus(Protocol):
    """A lowered dataset of labeled premise-hypothesis pairs a grounding trains on."""

    @property
    def fingerprint(self) -> str:
        """Return the digest of the dataset declaration."""
        ...

    @property
    def labeling(self) -> InferenceLabeling:
        """Return the label vocabulary shared by training and evaluation."""
        ...

    @property
    def corpus_pairs(self) -> tuple[CorpusPair, ...]:
        """Return every pair in source order, decided or not."""
        ...

    @property
    def derivations(self) -> tuple[Example[GoldDerivation[str]], ...]:
        """Return the gold derivations the two sides of the pairs retain."""
        ...

    @property
    def gold_forms(self) -> tuple[tuple[ItemId, str], ...]:
        """Return each side item's gold logical form, premises first."""
        ...

    @property
    def drawn(self) -> SyntheticInference | None:
        """Return the generated draw the pairs came from, when they were drawn."""
        ...


@final
class InferenceTrainingBatch(FrozenValue):
    """One immutable minibatch of featurized, labeled inference pairs."""

    _noun = "synthetic inference training batch"
    _because = "holds featurized tensors"
    __slots__ = (
        "_alignments",
        "_featurized",
        "_form_typing",
        "_hypotheses",
        "_hypothesis_rows",
        "_item_ids",
        "_judgments",
        "_labels",
        "_pair_ids",
        "_premise_rows",
        "_premises",
        "_step",
        "_words",
    )

    def __init__(
        self,
        featurized: FeaturizedBatch[Relational],
        words: Sequence[tuple[str, ...]],
        pair_ids: Sequence[ItemId],
        labels: Sequence[InferenceLabel],
        judgments: Sequence[tuple[int, int]],
        /,
        *,
        step: int,
    ) -> None:
        held_words = tuple(tuple(row) for row in words)
        held_pair_ids = tuple(pair_ids)
        held_labels = tuple(labels)
        member_rows = featurized.member_rows

        if len(held_labels) != len(member_rows):
            msg = (
                f"an inference batch has {len(member_rows)} pairs and "
                f"{len(held_labels)} labels"
            )
            raise ValueError(msg)

        if len(held_pair_ids) != len(member_rows):
            msg = (
                f"an inference batch has {len(member_rows)} pairs and "
                f"{len(held_pair_ids)} pair identifiers"
            )
            raise ValueError(msg)

        if len(set(held_pair_ids)) != len(held_pair_ids):
            msg = f"an inference batch repeats pair identifiers: {held_pair_ids!r}"
            raise ValueError(msg)

        if len(held_words) != len(featurized.batch):
            msg = (
                f"an inference batch has {len(featurized.batch)} side rows and "
                f"{len(held_words)} word rows"
            )
            raise ValueError(msg)

        invalid = tuple(
            label for label in held_labels if not isinstance(label, InferenceLabel)
        )

        if invalid:
            msg = f"an inference training batch has undecided labels {invalid!r}"
            raise ValueError(msg)

        malformed = tuple(rows for rows in member_rows if len(rows) != 2)

        if malformed:
            msg = f"an inference pair has other than two side rows: {malformed!r}"
            raise ValueError(msg)

        premise_rows = tuple(rows[0] for rows in member_rows)
        hypothesis_rows = tuple(rows[1] for rows in member_rows)
        premises = tuple(
            ReadoutId(f"pair-{position}:premise")
            for position in range(len(member_rows))
        )
        hypotheses = tuple(
            ReadoutId(f"pair-{position}:hypothesis")
            for position in range(len(member_rows))
        )
        object.__setattr__(self, "_featurized", featurized)
        object.__setattr__(self, "_words", held_words)
        object.__setattr__(self, "_premise_rows", premise_rows)
        object.__setattr__(self, "_hypothesis_rows", hypothesis_rows)
        object.__setattr__(self, "_premises", premises)
        object.__setattr__(self, "_hypotheses", hypotheses)
        object.__setattr__(self, "_labels", held_labels)
        object.__setattr__(self, "_pair_ids", held_pair_ids)
        held_judgments = tuple(judgments)

        if any(not 0 <= pair < len(member_rows) for pair, _ in held_judgments):
            msg = (
                f"an inference batch has judgment outside its pairs: {held_judgments!r}"
            )
            raise ValueError(msg)

        if any(level < 0 for _, level in held_judgments):
            msg = (
                f"an inference batch has a negative judgment level: {held_judgments!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_judgments", held_judgments)
        object.__setattr__(
            self,
            "_alignments",
            tuple((ArgumentRenaming.identity(),) for _ in member_rows),
        )
        object.__setattr__(self, "_item_ids", featurized.batch.item_ids)
        object.__setattr__(self, "_form_typing", tuple(() for _ in held_words))
        object.__setattr__(self, "_step", ensure_non_negative("step", step))

    @property
    def featurized(self) -> FeaturizedBatch[Relational]:
        return self._featurized

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        return self._words

    @property
    def item_ids(self) -> tuple[ItemId, ...]:
        return self._item_ids

    @property
    def form_typing(self) -> tuple[tuple[FormTypingExample, ...], ...]:
        return self._form_typing

    @property
    def premise_rows(self) -> tuple[int, ...]:
        return self._premise_rows

    @property
    def hypothesis_rows(self) -> tuple[int, ...]:
        return self._hypothesis_rows

    @property
    def premises(self) -> tuple[ReadoutId, ...]:
        return self._premises

    @property
    def hypotheses(self) -> tuple[ReadoutId, ...]:
        return self._hypotheses

    @property
    def alignments(self) -> tuple[tuple[ArgumentRenaming, ...], ...]:
        return self._alignments

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        return self._labels

    @property
    def pair_ids(self) -> tuple[ItemId, ...]:
        """Return one stable occurrence identifier per pair."""
        return self._pair_ids

    @property
    def judgments(self) -> tuple[tuple[int, int], ...]:
        """Return pair positions and their declared ordinal response levels."""
        return self._judgments

    @property
    def step(self) -> int:
        return self._step

    def at_step(self, step: int, /) -> Self:
        """Return the same immutable pairs at another optimizer position."""
        return type(self)(
            self._featurized,
            self._words,
            self._pair_ids,
            self._labels,
            self._judgments,
            step=step,
        )

    def __len__(self) -> int:
        return len(self._labels)


def inference_training_batches(
    dataset: PairCorpus,
    featurizer: Featurizer[Relational, FeaturizedBatch[Relational]],
    /,
    *,
    occurrence_positions: Sequence[int],
    batch_sampler: BatchSampler,
    batch_size: int,
    sampler_seed: int,
    step: int,
) -> tuple[InferenceTrainingBatch, ...]:
    """Featurize decided pairs of one pair corpus into deterministic batches."""
    positions = tuple(occurrence_positions)
    pairs = dataset.corpus_pairs
    labeling = dataset.labeling
    decided = tuple(
        position for position, pair in enumerate(pairs) if pair.relation is not None
    )

    if any(position not in decided for position in positions):
        outside = tuple(position for position in positions if position not in decided)
        msg = f"inference batch positions are not decided pairs: {outside!r}"
        raise ValueError(msg)

    items = {
        **{pair.premise.item_id: pair.premise for pair in pairs},
        **{pair.hypothesis.item_id: pair.hypothesis for pair in pairs},
    }
    lengths = tuple(
        max(len(pairs[position].premise.tokens), len(pairs[position].hypothesis.tokens))
        for position in positions
    )
    partition = batch_sampler.partition(lengths, batch_size, seed=sampler_seed)
    batches = []

    for selected in partition:
        batch_positions = tuple(positions[index] for index in selected)
        labels = tuple(
            labeling.encode(_decided(pairs[position])) for position in batch_positions
        )
        examples = tuple(
            Example(
                pairs[position].premise,
                Relational(pairs[position].hypothesis.item_id, label.value),
            )
            for position, label in zip(batch_positions, labels, strict=True)
        )
        featured = featurizer(examples)
        words = tuple(
            tuple(token.surface for token in items[item_id].tokens)
            for item_id in featured.batch.item_ids
        )
        batches.append(
            InferenceTrainingBatch(
                featured,
                words,
                tuple(
                    ItemId(f"inference-pair/{position}") for position in batch_positions
                ),
                labels,
                tuple(
                    (local, judgment)
                    for local, position in enumerate(batch_positions)
                    if (judgment := pairs[position].judgment) is not None
                ),
                step=step,
            )
        )

    return tuple(batches)


def _decided(pair: CorpusPair, /) -> InferenceLabel:
    """Return the relation of a pair already selected as decided."""
    relation = pair.relation

    if relation is None:
        msg = f"pair {pair!r} is undecided"
        raise ValueError(msg)

    return relation


@final
class InferencePairs(FrozenValue):
    """The decided pairs of one pair corpus, split for training.

    Parameters
    ----------
    dataset : PairCorpus
        The lowered pair corpus.
    training_split : str
        The split whose decided pairs train; every other split is retained.
    """

    _noun = "inference pair training"

    __slots__ = ("_dataset", "_training_split")

    def __init__(self, dataset: PairCorpus, *, training_split: str) -> None:
        object.__setattr__(self, "_dataset", dataset)
        object.__setattr__(self, "_training_split", training_split)

    @property
    def corpus(self) -> PairCorpus:
        """Return the pair corpus the epochs are drawn from."""
        return self._dataset

    @property
    def labeling(self) -> InferenceLabeling:
        """Return the label vocabulary shared by training and evaluation."""
        return self._dataset.labeling

    def epochs(
        self,
        featurizer: Featurizer[Relational, FeaturizedBatch[Relational]],
        batching: EpochBatching,
        /,
    ) -> SeededEpochs[InferenceTrainingBatch]:
        """Return the seeded epochs over the decided pairs of each split.

        Raises
        ------
        ValueError
            If a decided pair has no split, the training split holds no
            decided pair or no split is retained.
        """
        dataset = self._dataset
        splits: dict[str, list[int]] = {}

        for position, pair in enumerate(dataset.corpus_pairs):
            if pair.relation is None:
                continue

            split = pair.split

            if split is None:
                msg = f"inference pair {position} has no declared split"
                raise ValueError(msg)

            splits.setdefault(split, []).append(position)

        def batches(
            positions: tuple[int, ...], seed: int, /
        ) -> tuple[InferenceTrainingBatch, ...]:
            return inference_training_batches(
                dataset,
                featurizer,
                occurrence_positions=positions,
                batch_sampler=batching.batch_sampler,
                batch_size=batching.batch_size,
                sampler_seed=seed,
                step=0,
            )

        return SeededEpochs(
            {split: tuple(held) for split, held in splits.items()},
            training_split=self._training_split,
            batches=batches,
            batching=batching,
        )

    @override
    def __repr__(self) -> str:
        """Return the dataset fingerprint and the training split."""
        return (
            f"InferencePairs(fingerprint={self._dataset.fingerprint[:12]!r}, "
            f"training_split={self._training_split!r})"
        )
