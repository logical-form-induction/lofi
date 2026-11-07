"""Readers that lower collected PMB, NLI, and FraCaS releases.

The collected dataset plugins load this module only when a run lowers a
collected release, because its readers decode through the optional corpus
codecs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast

import lairs
from lairs import records

from lofi._types import ItemId
from lofi.data._gold_corpus import gold_corpus
from lofi.data._report import LoadReport
from lofi.data._supervision import (
    Example,
    GoldDerivation,
    Relational,
    Supervision,
    Unlabeled,
)
from lofi.data._values import Item, LicenseRef, Licensing, Provenance
from lofi.data.collected import (
    CollectedCorpus,
    CorpusDataset,
    PairCorpusDataset,
    PmbCcgDataset,
)
from lofi.data.corpora import GrammarInventory
from lofi.data.corpora.fracas import FracasAnswer, FracasCodec
from lofi.data.corpora.layers import ComparisonInventory, LayersCorpusReader
from lofi.data.corpora.nli import ChaosNliCodec, MnliCodec, SnliCodec, UnliCodec
from lofi.data.corpora.pmb import PmbCcgCodec, PmbDrsCodec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lairs.integrations.codecs import CorpusFragment, FragmentRecord
    from lairs.records import annotation as annotation_records
    from lairs.records import corpus as corpus_records
    from lairs.records import expression as expression_records
    from lairs.records import graph as graph_records
    from lairs.records import judgment as judgment_records
    from lairs.records import segmentation as segmentation_records

    from lofi.data.collected import CheckedCorpusDataset, CheckedPairDataset
    from lofi.data.corpora.fracas import FracasProblem
    from lofi.data.gold import GoldAggregator
    from lofi.data.tokenizers import Tokenizer
    from lofi.deduce import ParsePolicy, Retain
    from lofi.judge import Rule
    from lofi.spec.datasets.collected import (
        ChaosNliDatasetSpec,
        CorpusReleaseSpec,
        CorpusSourceSpec,
        FracasDatasetSpec,
        MnliDatasetSpec,
        PmbCcgDatasetSpec,
        PmbDrsDatasetSpec,
        SnliDatasetSpec,
        UnliDatasetSpec,
    )
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term, TermStore

    type _LayerModel = (
        annotation_records.AnnotationLayer
        | corpus_records.Corpus
        | corpus_records.Membership
        | expression_records.Expression
        | graph_records.GraphNode
        | graph_records.GraphEdge
        | judgment_records.ExperimentDef
        | judgment_records.JudgmentSet
        | segmentation_records.Segmentation
    )
    type _LayerModelClass = type[_LayerModel]


__all__ = [
    "lower_chaosnli",
    "lower_fracas",
    "lower_mnli",
    "lower_pmb_ccg",
    "lower_pmb_drs",
    "lower_snli",
    "lower_unli",
]


class _PairCodec(Protocol):
    """A codec decoding one NLI release file into a Layers fragment."""

    def decode(self, data: bytes, /) -> CorpusFragment:
        """Return the fragment one release file holds."""
        ...


_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_CORPUS_NSID: Final = "pub.layers.corpus.corpus"
_MEMBERSHIP_NSID: Final = "pub.layers.corpus.membership"
_EXPRESSION_NSID: Final = "pub.layers.expression.expression"


def _layer_models() -> Mapping[str, _LayerModelClass]:
    """Load the record classes admitted by a collected corpus."""
    return MappingProxyType(
        {
            _CORPUS_NSID: records.corpus.Corpus,
            _MEMBERSHIP_NSID: records.corpus.Membership,
            _EXPRESSION_NSID: records.expression.Expression,
            "pub.layers.segmentation.segmentation": records.segmentation.Segmentation,
            "pub.layers.annotation.annotationLayer": records.annotation.AnnotationLayer,
            "pub.layers.graph.graphNode": records.graph.GraphNode,
            "pub.layers.graph.graphEdge": records.graph.GraphEdge,
            "pub.layers.judgment.experimentDef": records.judgment.ExperimentDef,
            "pub.layers.judgment.judgmentSet": records.judgment.JudgmentSet,
        }
    )


def _model_of(
    record: FragmentRecord,
    models: Mapping[str, _LayerModelClass],
    /,
) -> _LayerModel:
    """Validate one supported Layers fragment record."""
    model = models.get(record.nsid)
    if model is None:
        msg = f"a collected corpus contains unsupported record {record.nsid!r}"
        raise ValueError(msg)
    return model.model_validate_json(record.value_json)


def _reader_of(
    fragment: CorpusFragment,
    split: str,
    source_id: str,
    release_id: str,
    inventory: GrammarInventory[TermStore, Term, Rule[Term]] | None = None,
) -> LayersCorpusReader:
    """Build one Layers reader, adding memberships to an unpooled fragment."""
    models = _layer_models()
    corpus_record = next(
        (record for record in fragment.records if record.nsid == _CORPUS_NSID),
        None,
    )
    key = sha256(f"{source_id}:{release_id}:{fragment.source}".encode()).hexdigest()[
        :24
    ]
    corpus_ref = (
        f"at://local/{_CORPUS_NSID}/{key}"
        if corpus_record is None
        else corpus_record.local_id
    )
    corpus = lairs.Corpus(uri=corpus_ref)
    for record in fragment.records:
        corpus.add_record(record.local_id, _model_of(record, models))
    if corpus_record is None:
        expressions = tuple(
            record for record in fragment.records if record.nsid == _EXPRESSION_NSID
        )
        for ordinal, expression in enumerate(expressions):
            membership_ref = f"at://local/{_MEMBERSHIP_NSID}/{key}-{ordinal}"
            corpus.add_record(
                membership_ref,
                records.corpus.Membership(
                    corpusRef=corpus_ref,
                    expressionRef=expression.local_id,
                    createdAt=_EPOCH,
                    split=split,
                    ordinal=ordinal,
                ),
            )
    return LayersCorpusReader(
        corpus,
        inventory,
        comparison=None
        if inventory is None
        else ComparisonInventory(inventory, inventory),
    )


def _provenance(
    source_method: str,
    spdx: str,
    eprint_refs: tuple[str, ...],
) -> Provenance:
    """Return release provenance shared by every item in one declaration."""
    return Provenance(
        source_method,
        Licensing(spdx, (LicenseRef(spdx, "whole"),)),
        eprint_refs,
    )


def _retokenized(
    example: Example[Supervision[str]],
    tokenizer: Tokenizer,
    provenance: Provenance,
) -> Example[Supervision[str]]:
    """Return an example tokenized once from its source text."""
    item = example.item
    if item.text is None:
        msg = f"collected item {item.item_id!r} has no source text to tokenize"
        raise ValueError(msg)
    return Example(
        Item(
            item.item_id,
            tokenizer(item.text),
            text=item.text,
            languages=item.languages,
            split=item.split,
            provenance=provenance,
        ),
        example.supervision,
    )


def _read_pair_fragment(
    fragment: CorpusFragment,
    release: CorpusReleaseSpec,
    tokenizer: Tokenizer,
    aggregation: GoldAggregator,
) -> CollectedCorpus:
    """Read, retokenize, and aggregate one NLI Layers fragment."""
    reader = _reader_of(
        fragment,
        release.split,
        release.source_id,
        release.release_id,
    )
    provenance = _provenance(
        release.source_method,
        release.spdx,
        release.eprint_refs,
    )
    examples = {
        split: aggregation(
            tuple(
                _retokenized(example, tokenizer, provenance)
                for example in reader.examples(split)
            )
        )
        for split in reader.splits()
    }
    return CollectedCorpus(examples, reader.report())


def _read_pmb(
    fragment: CorpusFragment,
    release: CorpusReleaseSpec,
    grammar: GrammarAssembly,
    *,
    ccg: bool,
) -> CollectedCorpus:
    """Read PMB source segmentation and its gold tree or graph carrier."""
    inventory = GrammarInventory(grammar) if ccg else None
    reader = _reader_of(
        fragment,
        release.split,
        release.source_id,
        release.release_id,
        inventory,
    )
    provenance = _provenance(
        release.source_method,
        release.spdx,
        release.eprint_refs,
    )
    examples = {
        split: tuple(
            Example(
                Item(
                    example.item.item_id,
                    example.item.tokens,
                    text=example.item.text,
                    languages=(release.language,),
                    split=example.item.split,
                    provenance=provenance,
                ),
                example.supervision,
            )
            for example in reader.examples(split)
        )
        for split in reader.splits()
    }
    return CollectedCorpus(examples, reader.report())


def _fracas_item(
    item_id: ItemId,
    text: str,
    release: CorpusReleaseSpec,
    tokenizer: Tokenizer,
    provenance: Provenance,
) -> Item:
    """Return one declared FraCaS pair member."""
    return Item(
        item_id,
        tokenizer(text),
        text=text,
        languages=(release.language,),
        split=release.split,
        provenance=provenance,
    )


def _fracas_examples(
    problems: Sequence[FracasProblem],
    release: CorpusReleaseSpec,
    tokenizer: Tokenizer,
    answers: type[FracasAnswer],
) -> tuple[tuple[Example[Supervision[str]], ...], int]:
    """Return binary FraCaS pairs, joining ordered premises into one member."""
    labels = {
        answers.YES: "entails",
        answers.NO: "contradicts",
        answers.UNKNOWN: "neutral",
        answers.UNDEF: "neutral",
    }
    provenance = _provenance(
        release.source_method,
        release.spdx,
        release.eprint_refs,
    )
    examples: list[Example[Supervision[str]]] = []
    skipped = 0
    for problem in problems:
        if problem.hypothesis is None or not problem.premises:
            skipped += 1
            continue
        stem = sha256(
            f"{release.source_id}:{release.release_id}:{problem.problem_id}".encode()
        ).hexdigest()[:24]
        premise = _fracas_item(
            ItemId(f"fracas:{stem}:premise"),
            " ".join(problem.premises),
            release,
            tokenizer,
            provenance,
        )
        hypothesis = _fracas_item(
            ItemId(f"fracas:{stem}:hypothesis"),
            problem.hypothesis,
            release,
            tokenizer,
            provenance,
        )
        examples.extend(
            (
                Example(
                    premise,
                    Relational(hypothesis.item_id, labels[problem.answer]),
                ),
                Example(hypothesis, Unlabeled()),
            )
        )
    return tuple(examples), skipped


def _merged(corpora: Sequence[CollectedCorpus], /) -> CollectedCorpus:
    """Join the corpora read from the releases of one declaration.

    Each release holds its own splits, so the examples join by split and the
    record, emission and skip counts add. Reachability and normal form analyses
    describe one reader's examples, so a joined corpus carries them only when
    one release was read.
    """
    if len(corpora) == 1:
        return corpora[0]

    examples: dict[str, list[Example[Supervision[str]]]] = {}
    skips: dict[str, int] = {}
    for corpus in corpora:
        if corpus.report.analyses is not None:
            msg = (
                "the analyses of several collected releases describe different "
                "readers and are measured over the joined corpus instead"
            )
            raise ValueError(msg)
        for split in corpus.splits:
            examples.setdefault(split, []).extend(corpus.examples(split))
        for reason, count in corpus.report.skips_by_reason.items():
            skips[reason] = skips.get(reason, 0) + count
    return CollectedCorpus(
        examples,
        LoadReport(
            sum(corpus.report.records_seen for corpus in corpora),
            sum(corpus.report.emitted for corpus in corpora),
            skips,
        ),
    )


def _pair_dataset[S](
    checked: CheckedPairDataset[S],
    corpora: Sequence[CollectedCorpus],
    training_split: str | None,
    /,
) -> PairCorpusDataset:
    """Return the pair corpus of one checked declaration's read releases."""
    labeling = checked.labeling

    if labeling is None:
        msg = "a collected pair corpus is read under a declared inference labeling"
        raise ValueError(msg)

    return PairCorpusDataset(
        checked.declaration,
        checked.source_grammar,
        _merged(corpora),
        labeling,
        training_split=training_split,
    )


def _read_pair_releases[S](
    checked: CheckedPairDataset[S],
    sources: Sequence[CorpusSourceSpec],
    codec: Callable[[CorpusReleaseSpec], _PairCodec],
    /,
) -> tuple[CollectedCorpus, ...]:
    """Read every declared NLI release file through its codec."""
    return tuple(
        _read_pair_fragment(
            codec(entry.release).decode(Path(entry.source).read_bytes()),
            entry.release,
            checked.tokenizer,
            checked.aggregation,
        )
        for entry in sources
    )


def lower_pmb_ccg(
    checked: CheckedCorpusDataset[PmbCcgDatasetSpec],
    policy: ParsePolicy[Retain],
) -> PmbCcgDataset:
    """Read the checked PMB CCG releases as one gold sentence corpus.

    Each derivation step is lowered to the compiled rule that licenses it, so
    a parser trains on the PMB derivations through the declared grammar's own
    rule identities.
    """
    del policy
    corpora = tuple(
        _read_pmb(
            PmbCcgCodec().decode(Path(entry.source).read_bytes()),
            entry.release,
            checked.source_grammar,
            ccg=True,
        )
        for entry in checked.source.sources
    )
    examples = tuple(
        cast("Example[GoldDerivation[str]]", example)
        for corpus in corpora
        for split in corpus.splits
        for example in corpus.examples(split)
        if isinstance(example.supervision, GoldDerivation)
    )
    gold = gold_corpus(
        examples,
        checked.source_grammar,
        LoadReport(
            sum(corpus.report.records_seen for corpus in corpora),
            sum(corpus.report.emitted for corpus in corpora),
            _joined_skips(corpora),
        ),
        seed=int.from_bytes(sha256(checked.declaration).digest()[:8], "big"),
        selections=MappingProxyType({}),
    )
    return PmbCcgDataset(
        checked.declaration,
        checked.source_grammar,
        gold,
        training_split=checked.source.training_split,
    )


def _joined_skips(corpora: Sequence[CollectedCorpus], /) -> dict[str, int]:
    """Return the skip counts of several read releases, added by reason."""
    skips: dict[str, int] = {}
    for corpus in corpora:
        for reason, count in corpus.report.skips_by_reason.items():
            skips[reason] = skips.get(reason, 0) + count
    return skips


def lower_pmb_drs(
    checked: CheckedCorpusDataset[PmbDrsDatasetSpec],
    policy: ParsePolicy[Retain],
) -> CorpusDataset:
    """Read the checked PMB clausal-DRS releases."""
    del policy
    corpus = _merged(
        tuple(
            _read_pmb(
                PmbDrsCodec().decode(Path(entry.source).read_bytes()),
                entry.release,
                checked.source_grammar,
                ccg=False,
            )
            for entry in checked.source.sources
        )
    )
    return CorpusDataset(checked.declaration, checked.source_grammar, corpus)


def lower_snli(
    checked: CheckedPairDataset[SnliDatasetSpec],
    policy: ParsePolicy[Retain],
) -> PairCorpusDataset:
    """Read the checked SNLI releases."""
    del policy
    return _pair_dataset(
        checked,
        _read_pair_releases(checked, checked.source.sources, SnliCodec),
        checked.source.training_split,
    )


def lower_mnli(
    checked: CheckedPairDataset[MnliDatasetSpec],
    policy: ParsePolicy[Retain],
) -> PairCorpusDataset:
    """Read the checked MultiNLI releases."""
    del policy
    return _pair_dataset(
        checked,
        _read_pair_releases(checked, checked.source.sources, MnliCodec),
        checked.source.training_split,
    )


def lower_chaosnli(
    checked: CheckedPairDataset[ChaosNliDatasetSpec],
    policy: ParsePolicy[Retain],
) -> PairCorpusDataset:
    """Read the checked ChaosNLI releases."""
    del policy
    return _pair_dataset(
        checked,
        _read_pair_releases(checked, checked.source.sources, ChaosNliCodec),
        checked.source.training_split,
    )


def lower_unli(
    checked: CheckedPairDataset[UnliDatasetSpec],
    policy: ParsePolicy[Retain],
) -> CorpusDataset:
    """Read the checked UNLI releases."""
    del policy
    return CorpusDataset(
        checked.declaration,
        checked.source_grammar,
        _merged(_read_pair_releases(checked, checked.source.sources, UnliCodec)),
    )


def _read_fracas(
    entry: CorpusSourceSpec,
    tokenizer: Tokenizer,
    aggregation: GoldAggregator,
    /,
) -> CollectedCorpus:
    """Read one FraCaS release file as binary inference pairs."""
    release = entry.release
    codec = FracasCodec(release)
    problems = codec.iso.backward(
        codec.decode(Path(entry.source).read_bytes())
    ).problems
    examples, skipped = _fracas_examples(problems, release, tokenizer, FracasAnswer)
    return CollectedCorpus(
        {release.split: aggregation(examples)},
        LoadReport(
            len(problems),
            len(problems) - skipped,
            {} if skipped == 0 else {"a FraCaS problem without a pair": skipped},
        ),
    )


def lower_fracas(
    checked: CheckedPairDataset[FracasDatasetSpec],
    policy: ParsePolicy[Retain],
) -> PairCorpusDataset:
    """Read the checked FraCaS releases as binary inference pairs."""
    del policy
    return _pair_dataset(
        checked,
        tuple(
            _read_fracas(entry, checked.tokenizer, checked.aggregation)
            for entry in checked.source.sources
        ),
        checked.source.training_split,
    )
