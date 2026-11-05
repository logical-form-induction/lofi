"""Featurizer resources and parser batches for synthetic corpora."""

from typing import TYPE_CHECKING, Protocol, Self, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data._epochs import SeededEpochs
from lofi.data._hashed import HashedResources, ItemHasher
from lofi.data._inventory import Inventory
from lofi.data._join import ItemLookup
from lofi.data._supervision import Example, Unlabeled
from lofi.data.synthetic._illtyped import FORM_TYPING_LABELS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data._epochs import EpochBatching
    from lofi.data._ports import CodeTensorizer, ConstraintBatcher, Featurizer
    from lofi.data._supervision import Categorical
    from lofi.data.datasets import ReservedForms
    from lofi.data.featurizers._core import FeaturizedBatch
    from lofi.data.sampler import BatchSampler
    from lofi.data.synthetic._illtyped import FormTypingExample, FormTypingLabel
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._values import (
        GrammaticalityLabel,
        ObservedLabel,
        SyntheticCorpus,
    )
    from lofi.data.synthetic.spec.inference import SyntheticInferenceDataset
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term


__all__ = [
    "CorpusSentences",
    "FormTypingTarget",
    "SentenceCorpus",
    "SyntheticTrainingBatch",
    "corpus_featurizer_resources",
    "corpus_sentences",
    "synthetic_inference_featurizer_resources",
    "synthetic_training_batches",
]


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return strings once in first-use order."""
    return tuple(dict.fromkeys(values))


@final
class FormTypingTarget(FrozenValue):
    """One parsed logical form with its certified and observed typing labels."""

    _noun = "logical-form typing target"

    __slots__ = ("_item_id", "_observed_label", "_term", "_true_label")

    def __init__(
        self,
        item_id: ItemId,
        term: Term,
        true_label: FormTypingLabel,
        observed_label: FormTypingLabel,
    ) -> None:
        for field, label in (("true", true_label), ("observed", observed_label)):
            if label not in FORM_TYPING_LABELS:
                msg = (
                    f"a logical-form typing target has unknown {field} label {label!r}"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_item_id", item_id)
        object.__setattr__(self, "_term", term)
        object.__setattr__(self, "_true_label", true_label)
        object.__setattr__(self, "_observed_label", observed_label)

    @property
    def item_id(self) -> ItemId:
        """Return the surface item this form belongs to."""
        return self._item_id

    @property
    def term(self) -> Term:
        """Return the parsed logical form."""
        return self._term

    @property
    def true_label(self) -> FormTypingLabel:
        """Return the form's certified typing label."""
        return self._true_label

    @property
    def observed_label(self) -> FormTypingLabel:
        """Return the label presented to training."""
        return self._observed_label

    def _fields(self) -> tuple[ItemId, Term, FormTypingLabel, FormTypingLabel]:
        """Return the fields this target is equal on."""
        return self._item_id, self._term, self._true_label, self._observed_label

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another target presents the same labeled form."""
        return type(other) is FormTypingTarget and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item, term, and labels."""
        return hash((FormTypingTarget, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return a concise rendering of the labeled form."""
        return (
            f"FormTypingTarget({self._item_id!r}, {self._term!r}, "
            f"{self._true_label!r}, {self._observed_label!r})"
        )


@final
@final
class SyntheticTrainingBatch[ConstraintT](FrozenValue):
    """One immutable synthetic minibatch with parser inputs and observed labels.

    Parameters
    ----------
    featurized : FeaturizedBatch[Unlabeled]
        Concrete lexical features in example order.
    words : Sequence[tuple[str, ...]]
        Surface tokens in example order.
    labels : Sequence[bool]
        Observed grammaticality judgments in example order.
    item_ids : Sequence[ItemId]
        Item identifiers in example order.
    form_typing : Sequence[Sequence[FormTypingTarget]]
        Logical-form typing examples aligned to the surface items. A row is
        empty when the corpus has no interpretation; otherwise it starts with
        the well-typed gold form and may include an injected form.
    constraints : Sequence[Sequence[ConstraintT]]
        Constraints already localized to this batch, one row per item when
        constrained or an empty sequence when unconstrained.
    step : int
        Caller-supplied zero-based training step.

    Raises
    ------
    TypeError
        If a label is not Boolean or the step is not an integer.
    ValueError
        If the step is negative or the batch fields are not aligned.
    """

    _noun = "synthetic training batch"

    __slots__ = (
        "_constraint_rows",
        "_constraints",
        "_featurized",
        "_form_typing",
        "_item_ids",
        "_labels",
        "_lengths",
        "_step",
        "_words",
    )

    def __init__(
        self,
        featurized: FeaturizedBatch[Unlabeled],
        words: Sequence[tuple[str, ...]],
        labels: Sequence[bool],
        item_ids: Sequence[ItemId],
        form_typing: Sequence[Sequence[FormTypingTarget]],
        constraints: Sequence[Sequence[ConstraintT]],
        *,
        step: int,
    ) -> None:
        held_words = tuple(tuple(item) for item in words)
        held_labels = tuple(labels)
        held_ids = tuple(item_ids)
        held_form_typing = tuple(tuple(row) for row in form_typing)
        held_constraint_rows = tuple(tuple(row) for row in constraints)
        count = len(featurized)

        for position, label in enumerate(held_labels):
            if not isinstance(label, bool):
                msg = (
                    f"the observed label at position {position} is not Boolean: "
                    f"{label!r}"
                )
                raise TypeError(msg)

        for name, size in (
            ("word rows", len(held_words)),
            ("labels", len(held_labels)),
            ("item ids", len(held_ids)),
            ("logical-form rows", len(held_form_typing)),
        ):
            if size != count:
                msg = f"a featurized batch of {count} items came with {size} {name}"
                raise ValueError(msg)

        if held_constraint_rows and len(held_constraint_rows) != count:
            msg = (
                f"a synthetic batch of {count} items came with "
                f"{len(held_constraint_rows)} constraint rows"
            )
            raise ValueError(msg)

        for position, (item_id, row) in enumerate(
            zip(held_ids, held_form_typing, strict=True)
        ):
            mismatched = tuple(
                example.item_id for example in row if example.item_id != item_id
            )

            if mismatched:
                msg = (
                    f"logical-form row {position} belongs to item {item_id!r} but "
                    f"names {mismatched!r}"
                )
                raise ValueError(msg)

        feature_ids = tuple(item.item_id for item in featurized.hashed)
        if feature_ids != held_ids or featurized.batch.item_ids != held_ids:
            msg = (
                f"the synthetic batch names item ids {held_ids!r}, but its "
                f"features name {feature_ids!r} and {featurized.batch.item_ids!r}"
            )
            raise ValueError(msg)

        lengths = tuple(map(len, held_words))
        if featurized.batch.lengths != lengths:
            msg = (
                f"the synthetic batch has word lengths {lengths!r}, but its "
                f"features have lengths {featurized.batch.lengths!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_featurized", featurized)
        object.__setattr__(self, "_words", held_words)
        object.__setattr__(self, "_lengths", lengths)
        object.__setattr__(self, "_labels", held_labels)
        object.__setattr__(self, "_item_ids", held_ids)
        object.__setattr__(self, "_form_typing", held_form_typing)
        object.__setattr__(self, "_constraint_rows", held_constraint_rows)
        distinct_constraints: list[ConstraintT] = []
        flattened: set[int] = set()
        for row in held_constraint_rows:
            for constraint in row:
                if id(constraint) not in flattened:
                    flattened.add(id(constraint))
                    distinct_constraints.append(constraint)
        object.__setattr__(
            self,
            "_constraints",
            tuple(distinct_constraints),
        )
        object.__setattr__(self, "_step", ensure_non_negative("step", step))

    @property
    def featurized(self) -> FeaturizedBatch[Unlabeled]:
        """Return the concrete lexical features."""
        return self._featurized

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return each item's surface tokens in item order."""
        return self._words

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return each item's token count in item order."""
        return self._lengths

    @property
    def labels(self) -> tuple[bool | None, ...]:
        """Return each item's observed grammaticality label in item order."""
        return self._labels

    @property
    def item_ids(self) -> tuple[ItemId, ...]:
        """Return each item's identifier in item order."""
        return self._item_ids

    @property
    def form_typing(self) -> tuple[tuple[FormTypingTarget, ...], ...]:
        """Return logical-form typing examples aligned to the surface items."""
        return self._form_typing

    @property
    def gold_forms(self) -> tuple[Term | None, ...]:
        """Return each item's first certified form, or none when it has no form."""
        return tuple(row[0].term if row else None for row in self._form_typing)

    @property
    def constraints(self) -> tuple[ConstraintT, ...]:
        """Return constraints localized to this batch."""
        return self._constraints

    @property
    def constraint_rows(self) -> tuple[tuple[ConstraintT, ...], ...]:
        """Return localized constraints aligned by item, or no rows."""
        return self._constraint_rows

    @property
    def step(self) -> int:
        """Return the caller-supplied zero-based training step."""
        return self._step

    def __len__(self) -> int:
        """Return how many examples the minibatch holds."""
        return len(self._item_ids)

    def at_step(self, step: int, /) -> Self:
        """Return this immutable batch at another optimizer-step position."""
        return type(self)(
            self._featurized,
            self._words,
            self._labels,
            self._item_ids,
            self._form_typing,
            self._constraint_rows,
            step=step,
        )

    @override
    def __repr__(self) -> str:
        """Return the item count and training step."""
        return f"SyntheticTrainingBatch(items={len(self)}, step={self._step})"


class SentenceCorpus(Protocol):
    """A lowered dataset whose gold sentence corpus a parser trains on."""

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the sentence corpus with its gold derivations."""
        ...

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the corpus was read under."""
        ...

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        """Return the interpretation the corpus was read under, when declared."""
        ...


def corpus_featurizer_resources(
    dataset: SentenceCorpus, reserved: ReservedForms, tensorizer: CodeTensorizer, /
) -> HashedResources:
    """Derive concrete featurizer resources from one lowered sentence corpus.

    Surface forms and lookup items follow the exact order of
    :meth:`SyntheticCorpus.observed`. Categories follow first use in the preorder
    traversals of grammatical gold derivations; corrupted items contribute no
    category.

    Parameters
    ----------
    dataset : SentenceCorpus
        Lowered dataset with a gold sentence corpus.
    reserved : ReservedForms
        The reserved token forms the lexicon begins with.
    tensorizer : CodeTensorizer
        The tensor backend the featurizer pads code rows with.

    Returns
    -------
    HashedResources
        Dataset vocabulary, hasher, the given tensorizer and item lookup.
    """
    observed = dataset.corpus.observed()
    surfaces = _unique(
        token.surface for example in observed for token in example.item.tokens
    )
    categories = _unique(
        derivation.node(node_id).label
        for example in dataset.corpus.examples
        for derivation in (example.supervision.derivation,)
        for node_id in derivation.preorder
    )
    lexicon = Inventory(
        "synthetic surface forms",
        surfaces,
        reserved=reserved.forms,
    )
    hasher = ItemHasher(
        lexicon,
        Inventory("synthetic categories", categories),
        unknown=reserved.unknown,
    )

    return HashedResources(
        hasher,
        tensorizer,
        ItemLookup(tuple(example.item for example in observed)),
    )


def synthetic_inference_featurizer_resources(
    dataset: SyntheticInferenceDataset,
    reserved: ReservedForms,
    tensorizer: CodeTensorizer,
    /,
) -> HashedResources:
    """Return pair-featurizer resources over both inference sides."""
    samples = (dataset.inference.premises, dataset.inference.hypotheses)
    examples = tuple(example for sample in samples for example in sample.examples)
    items = {}

    for example in examples:
        item = example.item
        prior = items.get(item.item_id)

        if prior is not None and prior != item:
            msg = f"inference sides disagree about item {item.item_id!r}"
            raise ValueError(msg)

        items[item.item_id] = item

    surfaces = _unique(
        token.surface for item in items.values() for token in item.tokens
    )
    categories = _unique(
        derivation.node(node_id).label
        for example in examples
        for derivation in (example.supervision.derivation,)
        for node_id in derivation.preorder
    )
    lexicon = Inventory(
        "synthetic inference surface forms",
        surfaces,
        reserved=reserved.forms,
    )
    hasher = ItemHasher(
        lexicon,
        Inventory("synthetic inference categories", categories),
        unknown=reserved.unknown,
    )

    return HashedResources(
        hasher,
        tensorizer,
        ItemLookup(tuple(items.values())),
    )


def _observed_label(label: GrammaticalityLabel) -> bool:
    """Return the Boolean representation of one observed label."""
    return label == "grammatical"


def _batch[ConstraintT](
    observed: Sequence[Example[Categorical]],
    labels: Sequence[ObservedLabel],
    form_typing: Mapping[ItemId, tuple[FormTypingExample, ...]],
    interpretation: InterpretationFunction | None,
    positions: tuple[int, ...],
    featurizer: Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]],
    constraint_batcher: ConstraintBatcher[ConstraintT] | None,
    *,
    step: int,
) -> SyntheticTrainingBatch[ConstraintT]:
    """Build one aligned parser batch over selected observed positions."""
    selected = tuple(observed[position] for position in positions)
    examples = tuple(Example(example.item, Unlabeled()) for example in selected)
    item_ids = tuple(example.item.item_id for example in selected)

    return SyntheticTrainingBatch(
        featurizer(examples),
        tuple(
            tuple(token.surface for token in example.item.tokens)
            for example in selected
        ),
        tuple(
            _observed_label(labels[position].observed_label) for position in positions
        ),
        item_ids,
        tuple(
            _form_typing_targets(
                form_typing.get(item_id, ()),
                interpretation,
            )
            for item_id in item_ids
        ),
        () if constraint_batcher is None else constraint_batcher.for_items(item_ids),
        step=step,
    )


def _form_typing_targets(
    examples: Sequence[FormTypingExample],
    interpretation: InterpretationFunction | None,
    /,
) -> tuple[FormTypingTarget, ...]:
    """Parse one aligned row at the dataset-to-training boundary."""
    if not examples:
        return ()

    if interpretation is None:
        msg = "logical-form typing examples require the dataset's interpretation"
        raise ValueError(msg)

    return tuple(
        FormTypingTarget(
            example.item_id,
            interpretation.read_text(example.text),
            example.true_label,
            example.observed_label,
        )
        for example in examples
    )


def synthetic_training_batches[ConstraintT](
    corpus: SyntheticCorpus,
    featurizer: Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]],
    interpretation: InterpretationFunction | None,
    /,
    *,
    occurrence_positions: Sequence[int],
    constraint_batcher: ConstraintBatcher[ConstraintT] | None,
    batch_sampler: BatchSampler,
    batch_size: int,
    sampler_seed: int,
    step: int,
) -> tuple[SyntheticTrainingBatch[ConstraintT], ...]:
    """Lower a sentence corpus into deterministic immutable parser batches.

    Parameters
    ----------
    corpus : SyntheticCorpus
        The sentence corpus of a lowered dataset.
    featurizer : Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]]
        Already-built featurizer for the unlabeled input arm.
    occurrence_positions : Sequence[int]
        Distinct observed occurrence positions included in the partition.
    constraint_batcher : ConstraintBatcher[ConstraintT] | None
        Dataset constraints to localize, or none for an unconstrained batch.
    batch_sampler : BatchSampler
        Declared strategy that partitions the selected occurrences.
    batch_size : int
        Maximum number of examples in a minibatch.
    sampler_seed : int
        Seed controlling the length-bucket sampler's fixed ordering.
    step : int
        Caller-supplied zero-based training step, copied to every batch.

    Returns
    -------
    tuple[SyntheticTrainingBatch[ConstraintT], ...]
        Complete fixed batch partition.

    Raises
    ------
    TypeError
        If the seed, step or size is not an integer.
    ValueError
        If the step is negative, the batch size is not positive or the occurrence
        positions are empty, repeated or outside the observed inventory.
    """
    held_step = ensure_non_negative("step", step)
    observed = corpus.observed()
    positions = tuple(occurrence_positions)

    if not positions:
        msg = "occurrence_positions selects no observed occurrence"
        raise ValueError(msg)

    if any(type(position) is not int for position in positions):
        msg = f"occurrence_positions are not all integers: {positions!r}"
        raise TypeError(msg)

    repeated = sorted(
        {position for position in positions if positions.count(position) > 1}
    )

    if repeated:
        msg = f"occurrence_positions repeats positions {repeated}"
        raise ValueError(msg)

    outside = sorted(
        position for position in positions if position < 0 or position >= len(observed)
    )

    if outside:
        msg = (
            f"occurrence_positions contains positions outside an inventory of "
            f"{len(observed)}: {outside}"
        )
        raise ValueError(msg)

    partition = batch_sampler.partition(
        tuple(len(observed[position].item.tokens) for position in positions),
        batch_size,
        seed=sampler_seed,
    )

    return tuple(
        _batch(
            observed,
            corpus.labels,
            corpus.form_typing,
            interpretation,
            tuple(positions[position] for position in selected),
            featurizer,
            constraint_batcher,
            step=held_step,
        )
        for selected in partition
    )


@final
class CorpusSentences(FrozenValue):
    """The observed sentences of one corpus, split for parser training.

    Parameters
    ----------
    corpus : SyntheticCorpus
        The sentence corpus the epochs are drawn from.
    source_grammar : GrammarAssembly
        The compiled grammar the corpus was read under.
    source_interpretation : InterpretationFunction | None
        The interpretation the corpus was read under, when declared.
    training_split : str
        The split whose occurrences train; every other split is retained.
    """

    _noun = "corpus sentence training"

    __slots__ = ("_corpus", "_grammar", "_interpretation", "_training_split")

    def __init__(
        self,
        corpus: SyntheticCorpus,
        *,
        source_grammar: GrammarAssembly,
        source_interpretation: InterpretationFunction | None,
        training_split: str,
    ) -> None:
        object.__setattr__(self, "_corpus", corpus)
        object.__setattr__(self, "_grammar", source_grammar)
        object.__setattr__(self, "_interpretation", source_interpretation)
        object.__setattr__(self, "_training_split", training_split)

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the sentence corpus the epochs are drawn from."""
        return self._corpus

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the corpus was read under."""
        return self._grammar

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        """Return the interpretation the corpus was read under, when declared."""
        return self._interpretation

    def epochs[ConstraintT](
        self,
        featurizer: Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]],
        constraints: ConstraintBatcher[ConstraintT] | None,
        interpretation: InterpretationFunction | None,
        batching: EpochBatching,
        /,
    ) -> SeededEpochs[SyntheticTrainingBatch[ConstraintT]]:
        """Return the seeded parser epochs over the observed occurrences.

        Raises
        ------
        ValueError
            If an observed item has no split, the training split holds no
            occurrence or no split is retained.
        """
        corpus = self._corpus
        splits: dict[str, list[int]] = {}

        for position, example in enumerate(corpus.observed()):
            split = example.item.split

            if split is None:
                msg = f"corpus item {example.item.item_id!r} has no declared split"
                raise ValueError(msg)

            splits.setdefault(split, []).append(position)

        def batches(
            positions: tuple[int, ...], seed: int, /
        ) -> tuple[SyntheticTrainingBatch[ConstraintT], ...]:
            return synthetic_training_batches(
                corpus,
                featurizer,
                interpretation,
                occurrence_positions=positions,
                constraint_batcher=constraints,
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
        """Return the corpus size and the training split."""
        return (
            f"CorpusSentences(observed={len(self._corpus.observed())}, "
            f"training_split={self._training_split!r})"
        )


def corpus_sentences(
    dataset: SentenceCorpus, training_split: str, /
) -> CorpusSentences:
    """Return the sentence training of one lowered sentence corpus.

    Parameters
    ----------
    dataset : SentenceCorpus
        The lowered dataset.
    training_split : str
        The split its declaration trains on.
    """
    return CorpusSentences(
        dataset.corpus,
        source_grammar=dataset.source_grammar,
        source_interpretation=dataset.source_interpretation,
        training_split=training_split,
    )
