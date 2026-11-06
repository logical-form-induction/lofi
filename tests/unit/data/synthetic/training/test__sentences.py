from __future__ import annotations

from copy import copy, deepcopy
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch
from tests.unit.data.synthetic.conftest import flat_generator

from lofi.data import FeaturizerResources, HashedResources
from lofi.data._epochs import EpochBatching
from lofi.data.datasets import ReservedForms
from lofi.data.featurizers.unlabeled import UnlabeledFeaturizer
from lofi.data.sampler import LengthBucketBatchSampler
from lofi.data.synthetic import (
    Corruption,
    InterpretationFunction,
    LabelNoise,
    NegativeDataPolicy,
    UniformLength,
    WeightedOperator,
)
from lofi.data.synthetic._corrupt import SWAP_PLUGIN, CorruptionContext
from lofi.data.synthetic._injection import ARGUMENT_SWAP_PLUGIN, IllTypingContext
from lofi.data.synthetic.spec.operators import ArgumentSwapSpec, SwapSpec
from lofi.data.synthetic.training import (
    CorpusSentences,
    FormTypingTarget,
    SyntheticTrainingBatch,
    corpus_featurizer_resources,
    synthetic_training_batches,
)
from lofi.nn._tensorize import TorchCodeTensorizer
from lofi.nn.loss._ports import (
    LabeledItems,
    ScheduledBatch,
    SurfaceItems,
)

SAMPLER = LengthBucketBatchSampler(100, reshuffle_each_epoch=False)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi._types import ItemId, JsonValue
    from lofi.data.synthetic import SyntheticCorpus, SyntheticDataset
    from lofi.data.synthetic._injection import IllTypingOperator
    from lofi.spec.grammar import GrammarAssembly

RESERVED = ReservedForms(("<pad>", "<unk>"), unknown="<unk>")
GRAMMAR = cast("GrammarAssembly", object())


class _DatasetView(NamedTuple):
    """The dataset field the lowering reads in focused tests."""

    corpus: SyntheticCorpus
    source_interpretation: InterpretationFunction | None


class _ItemConstraintRows:
    """Return one test constraint row for each selected item."""

    def for_items(self, item_ids: Sequence[ItemId], /) -> tuple[tuple[str, ...], ...]:
        return tuple((str(item_id),) for item_id in item_ids)


class _WeightedConstraint(NamedTuple):
    """One shared constraint with a visible objective weight."""

    name: str
    weight: float


class _SharedConstraintRows:
    """Place one localized multi-item constraint in every contributing row."""

    def __init__(self, constraint: _WeightedConstraint, /) -> None:
        self._constraint = constraint

    def for_items(
        self, item_ids: Sequence[ItemId], /
    ) -> tuple[tuple[_WeightedConstraint, ...], ...]:
        return tuple((self._constraint,) for _ in item_ids)


class _EqualConstraintRows:
    """Place one shared constraint and one distinct equal observation in each row."""

    def __init__(self, shared: _WeightedConstraint, /) -> None:
        self._shared = shared

    def for_items(
        self, item_ids: Sequence[ItemId], /
    ) -> tuple[tuple[_WeightedConstraint, ...], ...]:
        return tuple(
            (self._shared, _WeightedConstraint("equal", 0.5)) for _ in item_ids
        )


def _dataset(
    *,
    noise: LabelNoise | None = None,
    ill_typing_policy: NegativeDataPolicy | None = None,
) -> SyntheticDataset:
    generator = flat_generator()
    corruption_context = CorruptionContext(
        generator.lexicon, generator.grammar.matching, 3
    )
    interpretation = generator.interpretation
    if interpretation is None:
        msg = "the flat fixture needs an interpretation"
        raise ValueError(msg)
    ill_typing: Corruption[IllTypingOperator] | None = None
    if ill_typing_policy is not None:
        ill_typing = Corruption(
            0.43,
            operators=(
                WeightedOperator(
                    ARGUMENT_SWAP_PLUGIN.construct(
                        ArgumentSwapSpec(), IllTypingContext(interpretation)
                    ),
                    1.0,
                ),
            ),
            attempts=100,
            draw="exact",
            policy=ill_typing_policy,
        )
    corpus = generator.sample(
        7,
        seed=2,
        length=UniformLength(2, 3),
        corruption=Corruption(
            0.43,
            operators=(
                WeightedOperator(
                    SWAP_PLUGIN.compile(SwapSpec())(corruption_context), 1.0
                ),
            ),
            attempts=100,
            draw="exact",
        ),
        noise=noise,
        ill_typing=ill_typing,
    )
    return cast("SyntheticDataset", _DatasetView(corpus, generator.interpretation))


def _featurizer(
    resources: HashedResources,
) -> UnlabeledFeaturizer:
    return UnlabeledFeaturizer(
        resources.hasher,
        "<pad>",
        resources.tensorizer,
    )


def _flattened_ids[ConstraintT](
    batches: tuple[SyntheticTrainingBatch[ConstraintT], ...],
) -> tuple[str, ...]:
    return tuple(item_id for batch in batches for item_id in batch.item_ids)


def test_resources_follow_observed_tokens_and_only_gold_derivation_categories() -> None:
    dataset = _dataset()
    reserved = ReservedForms(("<p>", "<bos>", "<u>"), unknown="<u>")
    resources = corpus_featurizer_resources(dataset, reserved, TorchCodeTensorizer())
    observed = dataset.corpus.observed()
    surfaces = tuple(
        dict.fromkeys(
            token.surface for example in observed for token in example.item.tokens
        )
    )
    categories = tuple(
        dict.fromkeys(
            derivation.node(node_id).label
            for example in dataset.corpus.examples
            for derivation in (example.supervision.derivation,)
            for node_id in derivation.preorder
        )
    )

    assert isinstance(resources, FeaturizerResources)
    assert resources.hasher.lexicon.ordinary == surfaces
    assert resources.hasher.lexicon.reserved == ("<p>", "<bos>", "<u>")
    assert resources.hasher.categories.ordinary == categories
    assert tuple(resources.lookup.items.values()) == tuple(
        example.item for example in observed
    )


def test_equal_sampler_seeds_agree_and_a_planted_unequal_seed_changes_order() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    featurizer = _featurizer(resources)

    first = synthetic_training_batches(
        dataset.corpus,
        featurizer,
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=4,
    )
    repeated = synthetic_training_batches(
        dataset.corpus,
        featurizer,
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=4,
    )
    changed = synthetic_training_batches(
        dataset.corpus,
        featurizer,
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=13,
        step=4,
    )

    assert tuple(batch.item_ids for batch in first) == tuple(
        batch.item_ids for batch in repeated
    )
    assert _flattened_ids(first) != _flattened_ids(changed)


def test_batches_cover_every_observation_once_and_keep_a_partial_final_batch() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=8,
    )
    observed_ids = tuple(example.item.item_id for example in dataset.corpus.observed())

    assert sorted(_flattened_ids(batches)) == sorted(observed_ids)
    assert tuple(map(len, batches)) == (1, 3, 3)
    assert all(batch.step == 8 for batch in batches)


def test_a_constraint_batcher_returns_one_row_per_item_and_batches_flatten_it() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=_ItemConstraintRows(),
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )

    for batch in batches:
        assert len(batch.constraint_rows) == len(batch.item_ids)
        assert batch.constraints == tuple(map(str, batch.item_ids))


def test_a_multi_item_constraint_is_flattened_once_without_multiplying_its_weight() -> (
    None
):
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    shared = _WeightedConstraint("shared", 0.7)
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=_SharedConstraintRows(shared),
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )

    for batch in batches:
        assert batch.constraint_rows == tuple((shared,) for _ in batch.item_ids)
        assert batch.constraints == (shared,)
        assert sum(constraint.weight for constraint in batch.constraints) == 0.7


def test_distinct_equal_constraints_survive_flattening_as_separate_evidence() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    shared = _WeightedConstraint("shared", 0.7)
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=_EqualConstraintRows(shared),
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )

    for batch in batches:
        observed = tuple(row[1] for row in batch.constraint_rows)

        assert len({id(constraint) for constraint in observed}) == len(batch.item_ids)
        assert len(batch.constraints) == 1 + len(batch.item_ids)
        assert batch.constraints[0] is shared
        assert all(
            held is expected
            for held, expected in zip(batch.constraints[1:], observed, strict=True)
        )


def test_a_constrained_batch_refuses_rows_not_aligned_to_its_items() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batch = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=7,
        sampler_seed=2,
        step=0,
    )[0]

    with pytest.raises(ValueError, match="constraint rows"):
        SyntheticTrainingBatch(
            batch.featurized,
            batch.words,
            cast("Sequence[bool]", batch.labels),
            batch.item_ids,
            batch.form_typing,
            (("constraint",),),
            step=0,
        )


def test_batches_lower_only_the_required_distinct_occurrence_positions() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    positions = (0, 2, 5)
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=positions,
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=2,
        sampler_seed=11,
        step=0,
    )
    observed = dataset.corpus.observed()

    assert set(_flattened_ids(batches)) == {
        observed[position].item.item_id for position in positions
    }
    assert sum(map(len, batches)) == len(positions)


def test_batches_require_occurrence_selection_and_refuse_repetition() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())

    missing_selection = cast(
        "Callable[[], tuple[SyntheticTrainingBatch[object], ...]]",
        partial(
            synthetic_training_batches,
            dataset.corpus,
            _featurizer(resources),
            dataset.source_interpretation,
            constraint_batcher=None,
            batch_sampler=SAMPLER,
            batch_size=3,
            sampler_seed=11,
            step=0,
        ),
    )

    with pytest.raises(TypeError, match="occurrence_positions"):
        missing_selection()

    with pytest.raises(ValueError, match="repeats"):
        synthetic_training_batches(
            dataset.corpus,
            _featurizer(resources),
            dataset.source_interpretation,
            occurrence_positions=(1, 1),
            constraint_batcher=None,
            batch_sampler=SAMPLER,
            batch_size=3,
            sampler_seed=11,
            step=0,
        )


def test_every_batch_aligns_the_three_consumers_and_real_featurized_batch() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=5,
    )

    for batch in batches:
        assert isinstance(batch, SurfaceItems)
        assert isinstance(batch, LabeledItems)
        assert isinstance(batch, ScheduledBatch)
        assert batch.lengths == tuple(map(len, batch.words))
        assert batch.featurized.batch.item_ids == batch.item_ids
        assert tuple(item.item_id for item in batch.featurized.hashed) == batch.item_ids
        assert batch.featurized.batch.vectors.dtype.is_floating_point is False
        assert batch.featurized.batch.mask.values.dtype is torch.bool


def test_labels_come_from_the_observed_record_even_when_truth_is_flipped() -> None:
    plain = _dataset()
    flipped = _dataset(noise=LabelNoise(grammatical=1.0, ungrammatical=1.0))
    plain_resources = corpus_featurizer_resources(
        plain, RESERVED, TorchCodeTensorizer()
    )
    flipped_resources = corpus_featurizer_resources(
        flipped, RESERVED, TorchCodeTensorizer()
    )
    plain_batches = synthetic_training_batches(
        plain.corpus,
        _featurizer(plain_resources),
        plain.source_interpretation,
        occurrence_positions=tuple(range(len(plain.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )
    flipped_batches = synthetic_training_batches(
        flipped.corpus,
        _featurizer(flipped_resources),
        flipped.source_interpretation,
        occurrence_positions=tuple(range(len(flipped.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )
    plain_labels = {
        item_id: label
        for batch in plain_batches
        for item_id, label in zip(batch.item_ids, batch.labels, strict=True)
    }
    flipped_labels = {
        item_id: label
        for batch in flipped_batches
        for item_id, label in zip(batch.item_ids, batch.labels, strict=True)
    }
    observed = {
        label.item_id: label.observed_label == "grammatical"
        for label in flipped.corpus.labels
    }
    corrupted = {record.item.item_id for record in flipped.corpus.corrupted}

    assert tuple(plain_labels) == tuple(flipped_labels)
    assert all(
        plain_labels[item_id] is not flipped_labels[item_id] for item_id in observed
    )
    assert flipped_labels == observed
    assert corrupted
    assert all(flipped_labels[item_id] is True for item_id in corrupted)


@pytest.mark.parametrize(
    ("policy", "negative_label"),
    [
        (NegativeDataPolicy("labeled-negatives"), "ill-typed"),
        (NegativeDataPolicy("positive-only"), "well-typed"),
        (NegativeDataPolicy("label-flips", noise=1.0), "well-typed"),
    ],
)
def test_batches_carry_form_typing_labels_under_the_declared_negative_regime(
    policy: NegativeDataPolicy, negative_label: str
) -> None:
    dataset = _dataset(ill_typing_policy=policy)
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )
    rows = {
        item_id: row
        for batch in batches
        for item_id, row in zip(batch.item_ids, batch.form_typing, strict=True)
    }

    assert dataset.source_interpretation is not None
    expected = {
        item_id: tuple(
            FormTypingTarget(
                example.item_id,
                dataset.source_interpretation.read_text(example.text),
                example.true_label,
                example.observed_label,
            )
            for example in dataset.corpus.form_typing.get(item_id, ())
        )
        for item_id in rows
    }

    assert rows == expected
    assert all(row[0].observed_label == "well-typed" for row in rows.values() if row)
    injected = tuple(example for row in rows.values() for example in row[1:])
    assert injected
    assert {example.true_label for example in injected} == {"ill-typed"}
    assert {example.observed_label for example in injected} == {negative_label}


def test_batches_project_the_first_certified_form_as_gold() -> None:
    dataset = _dataset(ill_typing_policy=NegativeDataPolicy("labeled-negatives"))
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batches = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=3,
        sampler_seed=11,
        step=0,
    )

    for batch in batches:
        assert batch.gold_forms == tuple(
            row[0].term if row else None for row in batch.form_typing
        )
        assert all(
            row[0].true_label == "well-typed" for row in batch.form_typing if row
        )


def test_a_batch_refuses_a_form_typing_row_for_another_item() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batch = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=7,
        sampler_seed=2,
        step=0,
    )[0]
    first = next(target for row in batch.form_typing for target in row)
    mismatched = (
        (
            FormTypingTarget(
                cast("ItemId", "other"),
                first.term,
                "well-typed",
                "well-typed",
            ),
        ),
        *batch.form_typing[1:],
    )

    with pytest.raises(ValueError, match="logical-form row 0"):
        SyntheticTrainingBatch(
            batch.featurized,
            batch.words,
            cast("Sequence[bool]", batch.labels),
            batch.item_ids,
            mismatched,
            batch.constraint_rows,
            step=0,
        )


def test_resources_and_batches_are_immutable_and_validate_scalar_controls() -> None:
    dataset = _dataset()
    resources = corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
    batch = synthetic_training_batches(
        dataset.corpus,
        _featurizer(resources),
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SAMPLER,
        batch_size=7,
        sampler_seed=2,
        step=1,
    )[0]

    with pytest.raises(AttributeError, match="immutable"):
        resources.__setattr__("hasher", resources.hasher)

    with pytest.raises(AttributeError, match="immutable"):
        resources.__delattr__("hasher")

    with pytest.raises(AttributeError, match="immutable"):
        batch.__setattr__("step", 2)

    with pytest.raises(AttributeError, match="immutable"):
        batch.__delattr__("step")

    assert copy(resources) is resources
    assert deepcopy(resources) is resources
    assert copy(batch) is batch
    assert deepcopy(batch) is batch

    with pytest.raises(ValueError, match="step"):
        synthetic_training_batches(
            dataset.corpus,
            _featurizer(resources),
            dataset.source_interpretation,
            occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
            constraint_batcher=None,
            batch_sampler=SAMPLER,
            batch_size=3,
            sampler_seed=0,
            step=-1,
        )

    invalid_seed: JsonValue = True
    with pytest.raises(TypeError, match="seed"):
        synthetic_training_batches(
            dataset.corpus,
            _featurizer(resources),
            dataset.source_interpretation,
            occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
            constraint_batcher=None,
            batch_sampler=SAMPLER,
            batch_size=3,
            sampler_seed=cast("int", invalid_seed),
            step=0,
        )


def _split_sentences(training_split: str) -> CorpusSentences:
    generator = flat_generator()
    corpus = generator.sample(
        7,
        seed=2,
        length=UniformLength(2, 3),
        splits=("fit", "fit", "dev", "fit", "test", "fit", "dev"),
    )
    return CorpusSentences(
        corpus,
        source_grammar=GRAMMAR,
        source_interpretation=generator.interpretation,
        training_split=training_split,
    )


def test_corpus_sentences_train_on_the_declared_split_and_retain_the_rest() -> None:
    sentences = _split_sentences("fit")
    dataset = _DatasetView(sentences.corpus, sentences.source_interpretation)
    featurizer = _featurizer(
        corpus_featurizer_resources(
            cast("SyntheticDataset", dataset), RESERVED, TorchCodeTensorizer()
        )
    )
    epoch = sentences.epochs(
        featurizer,
        None,
        sentences.source_interpretation,
        EpochBatching(SAMPLER, batch_size=2, training_seed=3, validation_seed=4),
    ).at(0)
    observed = sentences.corpus.observed()

    def held(split: str) -> set[ItemId]:
        return {
            example.item.item_id for example in observed if example.item.split == split
        }

    assert {item for batch in epoch.training for item in batch.item_ids} == held("fit")
    assert set(epoch.retained) == {"dev", "test"}
    assert {item for batch in epoch.retained["dev"] for item in batch.item_ids} == held(
        "dev"
    )


def test_corpus_sentences_refuse_an_item_without_a_split() -> None:
    dataset = _dataset()
    sentences = CorpusSentences(
        dataset.corpus,
        source_grammar=GRAMMAR,
        source_interpretation=None,
        training_split="train",
    )

    with pytest.raises(ValueError, match="has no declared split"):
        sentences.epochs(
            _featurizer(
                corpus_featurizer_resources(dataset, RESERVED, TorchCodeTensorizer())
            ),
            None,
            None,
            EpochBatching(SAMPLER, batch_size=2, training_seed=0, validation_seed=0),
        )
