"""Installed CCGbank dataset lowering and empirical reports."""

from hashlib import sha256
from importlib import import_module
from json import dumps, loads
from math import log
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast, final

from lofi._frozen import FrozenValue
from lofi.data._gold_corpus import gold_corpus, lexical_counts
from lofi.data.corpora.ccgbank.roundtrip import CCGBANK_HEAD_SELECTIONS
from lofi.data.datasets import DatasetPlugin, DatasetVocabulary, RefusedTraining
from lofi.data.synthetic.spec.corpus import LexiconEstimateSpec
from lofi.data.synthetic.training import (
    CorpusSentences,
    corpus_featurizer_resources,
    corpus_sentences,
)
from lofi.lexicon import Entry, StaticLexicon
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.datasets import CcgbankDatasetSpec
from lofi.spec.grammar import (
    GrammarAssembly,
    GrammarIso,
)
from lofi.spec.grammar import (
    canonical_bytes as grammar_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import lofi.data.corpora.ccgbank.reader
    from lofi._types import ItemId
    from lofi.data import LoadReport
    from lofi.data.corpora.ccgbank._parg_parse import PargSentence
    from lofi.data.datasets import ReservedForms
    from lofi.data.synthetic import SyntheticCorpus
    from lofi.deduce import ParsePolicy, Retain

__all__ = [
    "PLUGIN",
    "SOURCE_VERSION",
    "TARGET_VERSION",
    "CcgbankDataset",
    "CheckedCcgbankDataset",
    "canonical_bytes",
    "lexicon_estimate",
]

SOURCE_VERSION: Final = "lofi-ccgbank-dataset-decl"
TARGET_VERSION: Final = "lofi-ccgbank-dataset-native"


def canonical_bytes(declaration: CcgbankDatasetSpec, /) -> bytes:
    """Return the declaration's canonical JSON bytes."""
    document = dumps(
        loads(declaration.model_dump_json()),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return document.encode("utf-8")


@final
class CheckedCcgbankDataset(FrozenValue):
    """A CCGbank declaration beside its compiled grammar."""

    __slots__ = ("_declaration", "_grammar", "_source")
    _noun = "checked CCGbank dataset"

    def __init__(
        self,
        source: CcgbankDatasetSpec,
        declaration: bytes,
        grammar: GrammarAssembly,
    ) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_grammar", grammar)

    @property
    def source(self) -> CcgbankDatasetSpec:
        """Return the validated source declaration."""
        return self._source

    @property
    def declaration(self) -> bytes:
        """Return the canonical declaration bytes."""
        return self._declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar used to read the corpus."""
        return self._grammar

    @property
    def source_interpretation(self) -> None:
        """Return no source interpretation for syntactic CCGbank data."""
        return None

    @property
    def source_interpretation_declaration(self) -> None:
        """Return no source interpretation declaration."""
        return None


@final
class CcgbankDataset(FrozenValue):
    """A lowered CCGbank corpus with its grammar and load report."""

    __slots__ = (
        "_corpus",
        "_declaration",
        "_dependencies",
        "_fingerprint",
        "_grammar",
        "_report",
        "_source_digest",
    )
    _noun = "CCGbank dataset"

    def __init__(
        self,
        declaration: bytes,
        grammar: GrammarAssembly,
        corpus: SyntheticCorpus,
        report: LoadReport[str],
        dependencies: Mapping[ItemId, PargSentence],
        source_digest: str,
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_fingerprint", sha256(declaration).hexdigest())
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_corpus", corpus)
        object.__setattr__(self, "_report", report)
        object.__setattr__(self, "_dependencies", MappingProxyType(dict(dependencies)))
        object.__setattr__(self, "_source_digest", source_digest)

    @property
    def declaration(self) -> bytes:
        """Return the canonical dataset declaration bytes."""
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
    def corpus(self) -> SyntheticCorpus:
        """Return the loaded examples through the shared sentence interface."""
        return self._corpus

    @property
    def report(self) -> LoadReport[str]:
        """Return skip, reachability and normal-form counts."""
        return self._report

    @property
    def dependencies(self) -> Mapping[ItemId, PargSentence]:
        """Return paired PARG sentences by item identifier."""
        return self._dependencies

    @property
    def source_digest(self) -> str:
        """Return the digest of the emitted source analyses."""
        return self._source_digest

    @property
    def source_interpretation(self) -> None:
        """Return no logical-form interpretation for syntactic CCGbank data."""
        return None

    @property
    def semantic(self) -> None:
        """Return no generated semantic report."""
        return None


def lexicon_estimate(dataset: CcgbankDataset, /) -> LexiconEstimateSpec:
    """Measure a document-ready synthetic lexicon estimate from a corpus."""
    counts = lexical_counts(dataset.corpus.examples)
    source = StaticLexicon(
        Entry(form, category, None, log(count))
        for (form, category), count in counts.items()
    )
    return LexiconEstimateSpec.measured(source, str, dataset.source_digest)


class _Reading(Protocol):
    """The CCGbank reader module, loaded only when a dataset reads its corpus."""

    CcgbankCorpusReader: type[lofi.data.corpora.ccgbank.reader.CcgbankCorpusReader]


def _reading() -> _Reading:
    """Return the CCGbank reader at the lowering site that reads the corpus.

    The reader decodes through the optional corpus codecs, so discovering the
    dataset plugin imports neither the reader nor its codecs.
    """
    return cast("_Reading", import_module("lofi.data.corpora.ccgbank.reader"))


def _check(declaration: CcgbankDatasetSpec, /) -> CheckedCcgbankDataset:
    """Compile the dataset's declared grammar without reading its source."""
    return CheckedCcgbankDataset(
        declaration,
        canonical_bytes(declaration),
        GrammarIso().forward(declaration.grammar, declaration.language),
    )


def _lower(
    checked: CheckedCcgbankDataset, parse_policy: ParsePolicy[Retain], /
) -> CcgbankDataset:
    """Read the selected CCGbank splits and measure their reports."""
    del parse_policy
    source = checked.source
    unknown_policy, category = source.unknown_words.policy()
    reader = _reading().CcgbankCorpusReader(
        Path(lower_package_resource(source.source)),
        splits=source.splits,
        unknown_words=unknown_policy,
        unknown_category=category,
    )
    gold = gold_corpus(
        tuple(example for split in source.splits for example in reader.examples(split)),
        checked.source_grammar,
        reader.report(),
        seed=int.from_bytes(sha256(checked.declaration).digest()[:8], "big"),
        selections=CCGBANK_HEAD_SELECTIONS,
    )
    return CcgbankDataset(
        checked.declaration,
        checked.source_grammar,
        gold.corpus,
        gold.report,
        reader.dependencies,
        gold.digest,
    )


def _validate(dataset: CcgbankDataset, /) -> None:
    """Reject a lowered dataset inconsistent with its declaration."""
    source = CcgbankDatasetSpec.model_validate_json(dataset.declaration)
    if dataset.source_grammar.declaration != grammar_bytes(source.grammar):
        msg = "a CCGbank dataset was lowered under another grammar"
        raise ValueError(msg)


def _vocabulary(
    dataset: CcgbankDataset, reserved: ReservedForms, /
) -> DatasetVocabulary:
    """Return corpus forms in their first-observed order."""
    forms = tuple(
        dict.fromkeys(
            token.surface
            for example in dataset.corpus.examples
            for token in example.item.tokens
        )
    )
    return DatasetVocabulary(
        forms,
        reserved=reserved.indices,
        unknown=reserved.unknown,
    )


def _sentences(dataset: CcgbankDataset, /) -> CorpusSentences:
    """Return the gold derivations of the declared training split."""
    declaration = CcgbankDatasetSpec.model_validate_json(dataset.declaration)
    return corpus_sentences(dataset, declaration.training_split)


def _streams(declaration: CcgbankDatasetSpec, /) -> tuple[()]:
    """Return no random stream for deterministic corpus reading."""
    del declaration
    return ()


PLUGIN: Final = DatasetPlugin(
    spec=CcgbankDatasetSpec,
    checked=CheckedCcgbankDataset,
    dataset=CcgbankDataset,
    check=_check,
    lower=_lower,
    validate=_validate,
    vocabulary=_vocabulary,
    resources=corpus_featurizer_resources,
    sentences=_sentences,
    pairs=RefusedTraining(
        "a CCGbank dataset supplies sentence training, not inference pairs"
    ),
    streams=_streams,
)
"""The installed CCGbank dataset route."""
