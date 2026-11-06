"""Dataset plugins for collected PMB, NLI, and FraCaS releases.

Each plugin declares its checked route here and loads the release readers in
``lofi.data._collected_reading`` only when a run lowers a release.
"""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast, final

from lofi._frozen import FrozenValue
from lofi.data._hashed import HashedResources, ItemHasher
from lofi.data._inventory import Inventory
from lofi.data._join import ItemLookup
from lofi.data._supervision import Relational, Unlabeled
from lofi.data.datasets import DatasetPlugin, DatasetVocabulary, RefusedTraining
from lofi.data.synthetic.training import (
    CorpusPair,
    CorpusSentences,
    InferencePairs,
    corpus_featurizer_resources,
    corpus_sentences,
)
from lofi.lf import RELATION_LABELS, registered_labeling
from lofi.registry import plugin_for
from lofi.spec.datasets.collected import (
    ChaosNliDatasetSpec,
    FracasDatasetSpec,
    MnliDatasetSpec,
    PmbCcgDatasetSpec,
    PmbDrsDatasetSpec,
    SnliDatasetSpec,
    UnliDatasetSpec,
    canonical_bytes,
)
from lofi.spec.grammar import (
    GrammarAssembly,
    GrammarIso,
)
from lofi.spec.grammar import (
    canonical_bytes as grammar_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data._gold_corpus import GoldCorpus
    from lofi.data._ports import CodeTensorizer
    from lofi.data._report import LoadReport
    from lofi.data._supervision import Example, Supervision
    from lofi.data._values import Item
    from lofi.data.datasets import DatasetLowerer, LoweredDataset, ReservedForms
    from lofi.data.gold import GoldAggregator
    from lofi.data.synthetic import SyntheticCorpus
    from lofi.data.tokenizers import Tokenizer
    from lofi.deduce import ParsePolicy, Retain
    from lofi.lf import InferenceLabeling
    from lofi.spec import DatasetSpec
    from lofi.spec.components.base import ConfiguredSpec
    from lofi.spec.components.data import GoldAggregationSpec, TokenizerSpec
    from lofi.spec.grammar import GrammarDecl

__all__ = [
    "CHAOS_NLI_PLUGIN",
    "FRACAS_PLUGIN",
    "MNLI_PLUGIN",
    "PMB_CCG_PLUGIN",
    "PMB_DRS_PLUGIN",
    "SNLI_PLUGIN",
    "UNLI_PLUGIN",
    "CheckedCorpusDataset",
    "CheckedPairDataset",
    "CollectedCorpus",
    "CorpusDataset",
    "PairCorpusDataset",
    "PmbCcgDataset",
    "collected_featurizer_resources",
]

_RELATIONS: Final = MappingProxyType({label.value: label for label in RELATION_LABELS})
"""Each logical relation keyed by the name a collected release records it under."""


@final
class CollectedCorpus(FrozenValue):
    """Collected examples, stable splits, item lookup, and source load report."""

    __slots__ = ("_examples", "_lookup", "_report", "_splits")
    _noun = "collected corpus"

    def __init__(
        self,
        examples: Mapping[str, Sequence[Example[Supervision[str]]]],
        report: LoadReport,
    ) -> None:
        held = {split: tuple(rows) for split, rows in examples.items()}
        items: dict[ItemId, Item] = {}
        for rows in held.values():
            for example in rows:
                item = example.item
                prior = items.get(item.item_id)
                if prior is not None and prior != item:
                    msg = f"collected examples disagree about item {item.item_id!r}"
                    raise ValueError(msg)
                items[item.item_id] = item
        object.__setattr__(self, "_splits", tuple(held))
        object.__setattr__(self, "_examples", MappingProxyType(held))
        object.__setattr__(self, "_lookup", ItemLookup(tuple(items.values())))
        object.__setattr__(self, "_report", report)

    @property
    def splits(self) -> tuple[str, ...]:
        """Return split names in source order."""
        return self._splits

    @property
    def lookup(self) -> ItemLookup:
        """Return every source item keyed by reader identity."""
        return self._lookup

    @property
    def report(self) -> LoadReport:
        """Return the source reader's load report."""
        return self._report

    def examples(self, split: str, /) -> tuple[Example[Supervision[str]], ...]:
        """Return every example in one split."""
        return self._examples.get(split, ())

    def observed(self, split: str, /) -> tuple[Example[Supervision[str]], ...]:
        """Return supervised examples in one split."""
        return tuple(
            example
            for example in self.examples(split)
            if not isinstance(example.supervision, Unlabeled)
        )

    def items(self, split: str, /) -> tuple[Item, ...]:
        """Return each distinct item used in one split, in first-use order."""
        by_id: dict[ItemId, Item] = {}
        for example in self.examples(split):
            by_id.setdefault(example.item.item_id, example.item)
        return tuple(by_id.values())


@final
class CheckedCorpusDataset[S](FrozenValue):
    """A collected corpus declaration beside its compiled grammar."""

    __slots__ = ("_declaration", "_grammar", "_source")
    _noun = "checked collected dataset"

    def __init__(
        self,
        source: S,
        grammar: GrammarAssembly,
    ) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(
            self, "_declaration", canonical_bytes(cast("DatasetSpec", source))
        )
        object.__setattr__(self, "_grammar", grammar)

    @property
    def source(self) -> S:
        """Return the validated source declaration."""
        return self._source

    @property
    def declaration(self) -> bytes:
        """Return canonical declaration bytes."""
        return self._declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar."""
        return self._grammar

    @property
    def source_interpretation(self) -> None:
        """Return no source interpretation."""
        return None

    @property
    def source_interpretation_declaration(self) -> None:
        """Return no source interpretation declaration."""
        return None


@final
class CheckedPairDataset[S](FrozenValue):
    """A pair-corpus declaration with its compiled boundary components."""

    __slots__ = (
        "_aggregation",
        "_declaration",
        "_grammar",
        "_labeling",
        "_source",
        "_tokenizer",
    )
    _noun = "checked pair dataset"

    def __init__(
        self,
        source: S,
        grammar: GrammarAssembly,
        tokenizer: Tokenizer,
        aggregation: GoldAggregator,
        labeling: InferenceLabeling | None,
    ) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(
            self, "_declaration", canonical_bytes(cast("DatasetSpec", source))
        )
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_aggregation", aggregation)
        object.__setattr__(self, "_labeling", labeling)

    @property
    def source(self) -> S:
        """Return the validated source declaration."""
        return self._source

    @property
    def declaration(self) -> bytes:
        """Return canonical declaration bytes."""
        return self._declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar."""
        return self._grammar

    @property
    def source_interpretation(self) -> None:
        """Return no source interpretation."""
        return None

    @property
    def source_interpretation_declaration(self) -> None:
        """Return no source interpretation declaration."""
        return None

    @property
    def tokenizer(self) -> Tokenizer:
        """Return the declared tokenizer."""
        return self._tokenizer

    @property
    def aggregation(self) -> GoldAggregator:
        """Return the declared gold aggregator."""
        return self._aggregation

    @property
    def labeling(self) -> InferenceLabeling | None:
        """Return the declared label vocabulary, when the release trains pairs."""
        return self._labeling


@final
class CorpusDataset(FrozenValue):
    """A lowered collected corpus and the grammar declared beside it."""

    __slots__ = ("_corpus", "_declaration", "_fingerprint", "_grammar")
    _noun = "collected dataset"

    def __init__(
        self,
        declaration: bytes,
        grammar: GrammarAssembly,
        corpus: CollectedCorpus,
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_fingerprint", sha256(declaration).hexdigest())
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_corpus", corpus)

    @property
    def declaration(self) -> bytes:
        """Return canonical declaration bytes."""
        return self._declaration

    @property
    def fingerprint(self) -> str:
        """Return the declaration digest."""
        return self._fingerprint

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar."""
        return self._grammar

    @property
    def corpus(self) -> CollectedCorpus:
        """Return the loaded corpus."""
        return self._corpus


@final
class PairCorpusDataset(FrozenValue):
    """A lowered collected pair corpus whose decided pairs train a grounding.

    Each relational example names its hypothesis among the corpus items and
    records a logical relation, which the declared labeling reads as an output
    label.

    Raises
    ------
    ValueError
        If a relational example names an absent hypothesis or a relation that
        is not a logical relation.
    """

    __slots__ = (
        "_corpus",
        "_corpus_pairs",
        "_declaration",
        "_fingerprint",
        "_grammar",
        "_labeling",
        "_training_split",
    )
    _noun = "collected pair dataset"

    def __init__(
        self,
        declaration: bytes,
        grammar: GrammarAssembly,
        corpus: CollectedCorpus,
        labeling: InferenceLabeling,
        *,
        training_split: str | None,
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_fingerprint", sha256(declaration).hexdigest())
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_corpus", corpus)
        object.__setattr__(self, "_labeling", labeling)
        object.__setattr__(self, "_training_split", training_split)
        object.__setattr__(self, "_corpus_pairs", _corpus_pairs(corpus))

    @property
    def declaration(self) -> bytes:
        """Return canonical declaration bytes."""
        return self._declaration

    @property
    def fingerprint(self) -> str:
        """Return the declaration digest."""
        return self._fingerprint

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar."""
        return self._grammar

    @property
    def corpus(self) -> CollectedCorpus:
        """Return the loaded corpus."""
        return self._corpus

    @property
    def labeling(self) -> InferenceLabeling:
        """Return the declared label vocabulary."""
        return self._labeling

    @property
    def training_split(self) -> str | None:
        """Return the declared training split, when the dataset trains."""
        return self._training_split

    @property
    def corpus_pairs(self) -> tuple[CorpusPair, ...]:
        """Return every relational pair in split and source order."""
        return self._corpus_pairs

    @property
    def derivations(self) -> tuple[()]:
        """Return no gold derivation, which a collected pair release lacks."""
        return ()

    @property
    def gold_forms(self) -> tuple[()]:
        """Return no gold logical form, which a collected pair release lacks."""
        return ()

    @property
    def drawn(self) -> None:
        """Return no generated draw for a collected release."""
        return None


def _corpus_pairs(corpus: CollectedCorpus, /) -> tuple[CorpusPair, ...]:
    """Return the relational pairs of a collected corpus, refusing a stray one."""
    items = corpus.lookup.items
    pairs: list[CorpusPair] = []

    for split in corpus.splits:
        for example in corpus.examples(split):
            supervision = example.supervision

            if not isinstance(supervision, Relational):
                continue

            hypothesis = items.get(supervision.other)

            if hypothesis is None:
                msg = (
                    f"collected pair {example.item.item_id!r} names absent "
                    f"hypothesis {supervision.other!r}"
                )
                raise ValueError(msg)

            if supervision.relation not in _RELATIONS:
                msg = (
                    f"collected pair {example.item.item_id!r} records "
                    f"{supervision.relation!r}, not one of the logical relations "
                    f"{tuple(_RELATIONS)!r}"
                )
                raise ValueError(msg)

            pairs.append(
                CorpusPair(
                    example.item,
                    hypothesis,
                    relation=_RELATIONS[supervision.relation],
                    split=split,
                )
            )

    return tuple(pairs)


@final
class PmbCcgDataset(FrozenValue):
    """A lowered PMB CCG release as one gold sentence corpus."""

    __slots__ = ("_declaration", "_fingerprint", "_gold", "_grammar", "_training_split")
    _noun = "PMB CCG dataset"

    def __init__(
        self,
        declaration: bytes,
        grammar: GrammarAssembly,
        gold: GoldCorpus,
        *,
        training_split: str | None,
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_fingerprint", sha256(declaration).hexdigest())
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_gold", gold)
        object.__setattr__(self, "_training_split", training_split)

    @property
    def declaration(self) -> bytes:
        """Return canonical declaration bytes."""
        return self._declaration

    @property
    def fingerprint(self) -> str:
        """Return the declaration digest."""
        return self._fingerprint

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar."""
        return self._grammar

    @property
    def source_interpretation(self) -> None:
        """Return no logical form interpretation for syntactic PMB data."""
        return None

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the canonical gold derivations through the sentence interface."""
        return self._gold.corpus

    @property
    def report(self) -> LoadReport[str]:
        """Return skip, reachability and normal form counts."""
        return self._gold.report

    @property
    def source_digest(self) -> str:
        """Return the digest of the canonical source analyses."""
        return self._gold.digest

    @property
    def training_split(self) -> str | None:
        """Return the declared training split, when the dataset trains."""
        return self._training_split


class _CollectedItems(Protocol):
    """A lowered collected dataset whose collected corpus holds its items."""

    @property
    def corpus(self) -> CollectedCorpus:
        """Return the loaded corpus."""
        ...


def collected_featurizer_resources(
    dataset: _CollectedItems, reserved: ReservedForms, tensorizer: CodeTensorizer, /
) -> HashedResources:
    """Return lexical encoding and pair lookup for a collected dataset."""
    forms = tuple(
        dict.fromkeys(
            token.surface
            for split in dataset.corpus.splits
            for item in dataset.corpus.items(split)
            for token in item.tokens
        )
    )
    lexicon = Inventory(
        "collected surface forms",
        forms,
        reserved=reserved.forms,
    )
    return HashedResources(
        ItemHasher(
            lexicon,
            Inventory("collected categories", ()),
            unknown=reserved.unknown,
        ),
        tensorizer,
        dataset.corpus.lookup,
    )


def _built_pair_components(
    tokenizer: TokenizerSpec,
    aggregation: GoldAggregationSpec,
) -> tuple[Tokenizer, GoldAggregator]:
    """Build the two pair-corpus components at the checked boundary."""
    configured_tokenizer = cast("ConfiguredSpec[object]", tokenizer)
    configured_aggregation = cast("ConfiguredSpec[object]", aggregation)
    built_tokenizer = plugin_for(configured_tokenizer).construct(
        configured_tokenizer, None
    )
    built_aggregation = plugin_for(configured_aggregation).construct(
        configured_aggregation, None
    )
    return cast("Tokenizer", built_tokenizer), cast("GoldAggregator", built_aggregation)


class _Reading(Protocol):
    """The collected-release readers, loaded when a run lowers a release."""

    lower_pmb_ccg: DatasetLowerer[
        CheckedCorpusDataset[PmbCcgDatasetSpec], PmbCcgDataset
    ]
    lower_pmb_drs: DatasetLowerer[
        CheckedCorpusDataset[PmbDrsDatasetSpec], CorpusDataset
    ]
    lower_snli: DatasetLowerer[CheckedPairDataset[SnliDatasetSpec], PairCorpusDataset]
    lower_mnli: DatasetLowerer[CheckedPairDataset[MnliDatasetSpec], PairCorpusDataset]
    lower_chaosnli: DatasetLowerer[
        CheckedPairDataset[ChaosNliDatasetSpec], PairCorpusDataset
    ]
    lower_unli: DatasetLowerer[CheckedPairDataset[UnliDatasetSpec], CorpusDataset]
    lower_fracas: DatasetLowerer[
        CheckedPairDataset[FracasDatasetSpec], PairCorpusDataset
    ]


def _reading() -> _Reading:
    """Return the release readers at the lowering site that reads a release.

    The readers decode through the optional corpus codecs, so discovering the
    collected dataset plugins imports neither the readers nor the codecs.
    """
    return cast("_Reading", import_module("lofi.data._collected_reading"))


def _lower_pmb_ccg(
    checked: CheckedCorpusDataset[PmbCcgDatasetSpec], policy: ParsePolicy[Retain], /
) -> PmbCcgDataset:
    """Read one checked PMB CCG release."""
    return _reading().lower_pmb_ccg(checked, policy)


def _lower_pmb_drs(
    checked: CheckedCorpusDataset[PmbDrsDatasetSpec], policy: ParsePolicy[Retain], /
) -> CorpusDataset:
    """Read one checked PMB clausal-DRS release."""
    return _reading().lower_pmb_drs(checked, policy)


def _lower_snli(
    checked: CheckedPairDataset[SnliDatasetSpec], policy: ParsePolicy[Retain], /
) -> PairCorpusDataset:
    """Read one checked SNLI release."""
    return _reading().lower_snli(checked, policy)


def _lower_mnli(
    checked: CheckedPairDataset[MnliDatasetSpec], policy: ParsePolicy[Retain], /
) -> PairCorpusDataset:
    """Read one checked MultiNLI release."""
    return _reading().lower_mnli(checked, policy)


def _lower_chaosnli(
    checked: CheckedPairDataset[ChaosNliDatasetSpec], policy: ParsePolicy[Retain], /
) -> PairCorpusDataset:
    """Read one checked ChaosNLI release."""
    return _reading().lower_chaosnli(checked, policy)


def _lower_unli(
    checked: CheckedPairDataset[UnliDatasetSpec], policy: ParsePolicy[Retain], /
) -> CorpusDataset:
    """Read one checked UNLI release."""
    return _reading().lower_unli(checked, policy)


def _lower_fracas(
    checked: CheckedPairDataset[FracasDatasetSpec], policy: ParsePolicy[Retain], /
) -> PairCorpusDataset:
    """Read one checked FraCaS release as binary inference pairs."""
    return _reading().lower_fracas(checked, policy)


def _validate(grammar: GrammarDecl, dataset: LoweredDataset) -> None:
    """Reject a runtime dataset inconsistent with its canonical declaration."""
    if dataset.source_grammar.declaration != grammar_bytes(grammar):
        msg = "a collected dataset was lowered under another grammar"
        raise ValueError(msg)


def _vocabulary(
    dataset: _CollectedItems, reserved: ReservedForms, /
) -> DatasetVocabulary:
    """Return first-observed forms across every split."""
    forms = tuple(
        dict.fromkeys(
            token.surface
            for split in dataset.corpus.splits
            for item in dataset.corpus.items(split)
            for token in item.tokens
        )
    )
    return DatasetVocabulary(
        forms,
        reserved=reserved.indices,
        unknown=reserved.unknown,
    )


def _streams(declaration: DatasetSpec, /) -> tuple[str, ...]:
    """Return no random streams for a collected release."""
    del declaration
    return ()


def _check_pmb_ccg(
    source: PmbCcgDatasetSpec, /
) -> CheckedCorpusDataset[PmbCcgDatasetSpec]:
    """Compile the grammar declared for one PMB CCG release."""
    return CheckedCorpusDataset(
        source,
        GrammarIso().forward(source.grammar, source.language),
    )


def _check_pmb_drs(
    source: PmbDrsDatasetSpec, /
) -> CheckedCorpusDataset[PmbDrsDatasetSpec]:
    """Compile the grammar declared for one PMB DRS release."""
    return CheckedCorpusDataset(
        source,
        GrammarIso().forward(source.grammar, source.language),
    )


def _checked_pair[S](
    source: S,
    grammar: GrammarAssembly,
    tokenizer: TokenizerSpec,
    aggregation: GoldAggregationSpec,
    labeling: str | None,
) -> CheckedPairDataset[S]:
    """Build one pair corpus's boundary components."""
    built_tokenizer, built_aggregation = _built_pair_components(
        tokenizer,
        aggregation,
    )
    return CheckedPairDataset(
        source,
        grammar,
        built_tokenizer,
        built_aggregation,
        None if labeling is None else registered_labeling(labeling),
    )


def _check_snli(source: SnliDatasetSpec, /) -> CheckedPairDataset[SnliDatasetSpec]:
    """Compile one SNLI declaration."""
    return _checked_pair(
        source,
        GrammarIso().forward(source.grammar, source.language),
        source.tokenizer,
        source.aggregation,
        source.labeling,
    )


def _check_mnli(source: MnliDatasetSpec, /) -> CheckedPairDataset[MnliDatasetSpec]:
    """Compile one MultiNLI declaration."""
    return _checked_pair(
        source,
        GrammarIso().forward(source.grammar, source.language),
        source.tokenizer,
        source.aggregation,
        source.labeling,
    )


def _check_chaosnli(
    source: ChaosNliDatasetSpec, /
) -> CheckedPairDataset[ChaosNliDatasetSpec]:
    """Compile one ChaosNLI declaration."""
    return _checked_pair(
        source,
        GrammarIso().forward(source.grammar, source.language),
        source.tokenizer,
        source.aggregation,
        source.labeling,
    )


def _check_unli(source: UnliDatasetSpec, /) -> CheckedPairDataset[UnliDatasetSpec]:
    """Compile one UNLI declaration."""
    return _checked_pair(
        source,
        GrammarIso().forward(source.grammar, source.language),
        source.tokenizer,
        source.aggregation,
        None,
    )


def _check_fracas(
    source: FracasDatasetSpec, /
) -> CheckedPairDataset[FracasDatasetSpec]:
    """Compile one FraCaS declaration."""
    return _checked_pair(
        source,
        GrammarIso().forward(source.grammar, source.language),
        source.tokenizer,
        source.aggregation,
        source.labeling,
    )


def _validate_pmb_ccg(dataset: PmbCcgDataset, /) -> None:
    source = PmbCcgDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _validate_pmb_drs(dataset: CorpusDataset, /) -> None:
    source = PmbDrsDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _validate_snli(dataset: PairCorpusDataset, /) -> None:
    source = SnliDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _validate_mnli(dataset: PairCorpusDataset, /) -> None:
    source = MnliDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _validate_chaosnli(dataset: PairCorpusDataset, /) -> None:
    source = ChaosNliDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _validate_unli(dataset: CorpusDataset, /) -> None:
    source = UnliDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _validate_fracas(dataset: PairCorpusDataset, /) -> None:
    source = FracasDatasetSpec.model_validate_json(dataset.declaration)
    _validate(source.grammar, dataset)


def _pair_training(dataset: PairCorpusDataset, /) -> InferencePairs:
    """Return the decided pairs of a collected pair corpus, split for training.

    Raises
    ------
    ValueError
        If the declaration names no training split.
    """
    split = dataset.training_split

    if split is None:
        msg = "a collected pair dataset trains only on a declared training split"
        raise ValueError(msg)

    return InferencePairs(dataset, training_split=split)


def _pmb_sentences(dataset: PmbCcgDataset, /) -> CorpusSentences:
    """Return the gold derivations of the declared training split.

    Raises
    ------
    ValueError
        If the declaration names no training split.
    """
    split = dataset.training_split

    if split is None:
        msg = "a PMB CCG dataset trains only on a declared training split"
        raise ValueError(msg)

    return corpus_sentences(dataset, split)


def _pmb_vocabulary(
    dataset: PmbCcgDataset, reserved: ReservedForms, /
) -> DatasetVocabulary:
    """Return corpus forms in their first-observed order."""
    forms = tuple(
        dict.fromkeys(
            token.surface
            for example in dataset.corpus.examples
            for token in example.item.tokens
        )
    )
    return DatasetVocabulary(forms, reserved=reserved.indices, unknown=reserved.unknown)


_SENTENCES_REFUSED: Final = RefusedTraining(
    "a collected pair corpus supplies inference pairs, not sentence training"
)
_PAIRS_REFUSED: Final = RefusedTraining(
    "a collected sentence corpus supplies sentence training, not inference pairs"
)
_GRAPHS_REFUSED: Final = RefusedTraining(
    "a PMB clausal-DRS corpus supplies gold graphs, which no training route reads"
)
_GRADED_REFUSED: Final = RefusedTraining(
    "a UNLI corpus mixes categorical and graded responses, which no pair "
    "training route reads together"
)
PMB_CCG_PLUGIN: Final = DatasetPlugin(
    spec=PmbCcgDatasetSpec,
    checked=CheckedCorpusDataset,
    dataset=PmbCcgDataset,
    check=_check_pmb_ccg,
    lower=_lower_pmb_ccg,
    validate=_validate_pmb_ccg,
    vocabulary=_pmb_vocabulary,
    resources=corpus_featurizer_resources,
    sentences=_pmb_sentences,
    pairs=_PAIRS_REFUSED,
    streams=_streams,
)
PMB_DRS_PLUGIN: Final = DatasetPlugin(
    spec=PmbDrsDatasetSpec,
    checked=CheckedCorpusDataset,
    dataset=CorpusDataset,
    check=_check_pmb_drs,
    lower=_lower_pmb_drs,
    validate=_validate_pmb_drs,
    vocabulary=_vocabulary,
    resources=collected_featurizer_resources,
    sentences=_GRAPHS_REFUSED,
    pairs=_GRAPHS_REFUSED,
    streams=_streams,
)
SNLI_PLUGIN: Final = DatasetPlugin(
    spec=SnliDatasetSpec,
    checked=CheckedPairDataset,
    dataset=PairCorpusDataset,
    check=_check_snli,
    lower=_lower_snli,
    validate=_validate_snli,
    vocabulary=_vocabulary,
    resources=collected_featurizer_resources,
    sentences=_SENTENCES_REFUSED,
    pairs=_pair_training,
    streams=_streams,
)
MNLI_PLUGIN: Final = DatasetPlugin(
    spec=MnliDatasetSpec,
    checked=CheckedPairDataset,
    dataset=PairCorpusDataset,
    check=_check_mnli,
    lower=_lower_mnli,
    validate=_validate_mnli,
    vocabulary=_vocabulary,
    resources=collected_featurizer_resources,
    sentences=_SENTENCES_REFUSED,
    pairs=_pair_training,
    streams=_streams,
)
CHAOS_NLI_PLUGIN: Final = DatasetPlugin(
    spec=ChaosNliDatasetSpec,
    checked=CheckedPairDataset,
    dataset=PairCorpusDataset,
    check=_check_chaosnli,
    lower=_lower_chaosnli,
    validate=_validate_chaosnli,
    vocabulary=_vocabulary,
    resources=collected_featurizer_resources,
    sentences=_SENTENCES_REFUSED,
    pairs=_pair_training,
    streams=_streams,
)
UNLI_PLUGIN: Final = DatasetPlugin(
    spec=UnliDatasetSpec,
    checked=CheckedPairDataset,
    dataset=CorpusDataset,
    check=_check_unli,
    lower=_lower_unli,
    validate=_validate_unli,
    vocabulary=_vocabulary,
    resources=collected_featurizer_resources,
    sentences=_GRADED_REFUSED,
    pairs=_GRADED_REFUSED,
    streams=_streams,
)
FRACAS_PLUGIN: Final = DatasetPlugin(
    spec=FracasDatasetSpec,
    checked=CheckedPairDataset,
    dataset=PairCorpusDataset,
    check=_check_fracas,
    lower=_lower_fracas,
    validate=_validate_fracas,
    vocabulary=_vocabulary,
    resources=collected_featurizer_resources,
    sentences=_SENTENCES_REFUSED,
    pairs=_pair_training,
    streams=_streams,
)
