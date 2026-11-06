"""Boundary declarations for collected PMB, NLI, and FraCaS corpora."""

from json import dumps, loads
from typing import TYPE_CHECKING, Final, Literal, Self, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.lf import registered_labeling
from lofi.spec.components.data import (
    DatasetSpec,
    GoldAggregationSpec,
    TokenizerSpec,
    TrainingBatchKind,
)
from lofi.spec.gold import MajorityGoldSpec
from lofi.spec.grammar import GrammarDecl
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl
from lofi.spec.tokenizers import RegexTokenizerSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ChaosNliDatasetSpec",
    "CorpusReleaseSpec",
    "CorpusSourceSpec",
    "FracasDatasetSpec",
    "MnliDatasetSpec",
    "PmbCcgDatasetSpec",
    "PmbDrsDatasetSpec",
    "SnliDatasetSpec",
    "UnliDatasetSpec",
    "canonical_bytes",
]

_TOKENIZER_FIELD: Final = RegexTokenizerSpec()
_AGGREGATION_FIELD: Final = MajorityGoldSpec()
_DEFAULT_LABELING: Final = "three-way"
"""The inference label vocabulary a collected release reads when it names none."""


class CorpusReleaseSpec(dx.Model, extra="forbid"):
    """Declare the identity, split, language, and terms of one corpus release."""

    source_id: str
    release_id: str
    split: str
    language: str
    source_method: str
    spdx: str
    eprint_refs: tuple[str, ...] = ()

    @dx.validates(
        "source_id",
        "release_id",
        "split",
        "language",
        "source_method",
        "spdx",
    )
    @classmethod
    def _text_is_present(cls, value: str) -> str:
        """Return a nonempty release field."""
        if not value.strip():
            msg = "corpus release metadata cannot be empty"
            raise ValueError(msg)

        return value

    @dx.validates("eprint_refs")
    @classmethod
    def _references_are_present(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Return nonempty eprint references."""
        if any(not value.strip() for value in values):
            msg = "corpus release eprint references cannot be empty"
            raise ValueError(msg)
        return values


class CorpusSourceSpec(dx.Model, extra="forbid"):
    """Declare one release file of a collected dataset beside its release.

    A collected dataset reads one file per split, so a corpus distributed as
    training, development and test files declares one source for each.
    """

    source: str
    release: CorpusReleaseSpec

    __axioms__ = (dx.axiom("len(source) > 0", message="a release file needs a path"),)


def canonical_bytes(declaration: DatasetSpec, /) -> bytes:
    """Return one collected-dataset declaration's canonical JSON bytes."""
    document = dumps(
        loads(declaration.model_dump_json()),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return document.encode("utf-8")


def _split_names(sources: Sequence[CorpusSourceSpec], /) -> tuple[str, ...]:
    """Return the release splits, refusing an empty or repeated selection."""
    splits = tuple(entry.release.split for entry in sources)

    if not splits:
        msg = "a collected dataset declares at least one release file"
        raise ValueError(msg)

    if len(set(splits)) != len(splits):
        msg = f"a collected dataset reads each split from one release: {splits!r}"
        raise ValueError(msg)

    return splits


def _checked_training_split(
    training_split: str | None, sources: Sequence[CorpusSourceSpec], /
) -> None:
    """Refuse a training split no declared release reads."""
    splits = _split_names(sources)

    if training_split is None:
        return

    ensure_name("a collected dataset's training split", training_split)

    if training_split not in splits:
        msg = (
            f"a collected dataset trains on {training_split!r}, which no "
            f"declared release reads from {splits!r}"
        )
        raise ValueError(msg)


def _held_out_split(
    training_split: str | None, sources: Sequence[CorpusSourceSpec], /
) -> str:
    """Return the first declared release split that does not train.

    A dataset whose one release trains reports that release's split.
    """
    return next(
        (
            entry.release.split
            for entry in sources
            if entry.release.split != training_split
        ),
        sources[0].release.split,
    )


def _retained_splits(
    training_split: str | None, sources: Sequence[CorpusSourceSpec], /
) -> frozenset[str]:
    """Return every declared release split that does not train."""
    return frozenset(
        entry.release.split
        for entry in sources
        if entry.release.split != training_split
    )


def _checked_labeling(value: str, /) -> str:
    """Return the name of an installed inference label vocabulary."""
    registered_labeling(value)
    return value


class PmbCcgDatasetSpec(DatasetSpec):
    """Declare PMB CCG derivations read without changing their segmentation.

    A parser trains on the gold derivations of the declared training split.
    """

    kind: Literal["pmb-ccg"] = "pmb-ccg"
    sources: tuple[CorpusSourceSpec, ...]
    training_split: str | None = None
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD

    @dx.model_validator()
    def _sources_name_the_training_split(self) -> Self:
        """Require distinct release splits that include any training split."""
        _checked_training_split(self.training_split, self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split that does not train."""
        return _held_out_split(self.training_split, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split that does not train."""
        return _retained_splits(self.training_split, self.sources)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the sentence and gold derivation batches this corpus yields."""
        return frozenset({TrainingBatchKind.SENTENCES, TrainingBatchKind.GOLD_FORMS})


class PmbDrsDatasetSpec(DatasetSpec):
    """Declare PMB clausal DRS graphs as gold logical forms."""

    kind: Literal["pmb-drs"] = "pmb-drs"
    sources: tuple[CorpusSourceSpec, ...]
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD

    @dx.model_validator()
    def _sources_name_distinct_splits(self) -> Self:
        """Require one release per declared split."""
        _split_names(self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split."""
        return _held_out_split(None, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split."""
        return _retained_splits(None, self.sources)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the sentence and gold-graph batches this corpus yields."""
        return frozenset({TrainingBatchKind.SENTENCES, TrainingBatchKind.GOLD_FORMS})


class SnliDatasetSpec(DatasetSpec):
    """Declare normalized SNLI JSONL releases.

    A grounding trains on the decided pairs of the declared training split,
    each relation read through the declared labeling.
    """

    kind: Literal["snli"] = "snli"
    sources: tuple[CorpusSourceSpec, ...]
    training_split: str | None = None
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    tokenizer: TokenizerSpec = _TOKENIZER_FIELD
    aggregation: GoldAggregationSpec = _AGGREGATION_FIELD
    labeling: str = _DEFAULT_LABELING

    @dx.model_validator()
    def _sources_name_the_training_split(self) -> Self:
        """Require distinct release splits that include any training split."""
        _checked_training_split(self.training_split, self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split that does not train."""
        return _held_out_split(self.training_split, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split that does not train."""
        return _retained_splits(self.training_split, self.sources)

    @dx.validates("labeling")
    @classmethod
    def _labeling_is_registered(cls, value: str) -> str:
        """Return the name of an installed inference label vocabulary."""
        return _checked_labeling(value)

    def inference_labels(self) -> tuple[str, ...]:
        """Return the declared output vocabulary in model-column order."""
        return tuple(label.value for label in registered_labeling(self.labeling).labels)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return labeled inference pair batches."""
        return frozenset({TrainingBatchKind.INFERENCE_PAIRS})


class MnliDatasetSpec(DatasetSpec):
    """Declare normalized MultiNLI JSONL releases.

    A grounding trains on the decided pairs of the declared training split,
    each relation read through the declared labeling.
    """

    kind: Literal["mnli"] = "mnli"
    sources: tuple[CorpusSourceSpec, ...]
    training_split: str | None = None
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    tokenizer: TokenizerSpec = _TOKENIZER_FIELD
    aggregation: GoldAggregationSpec = _AGGREGATION_FIELD
    labeling: str = _DEFAULT_LABELING

    @dx.model_validator()
    def _sources_name_the_training_split(self) -> Self:
        """Require distinct release splits that include any training split."""
        _checked_training_split(self.training_split, self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split that does not train."""
        return _held_out_split(self.training_split, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split that does not train."""
        return _retained_splits(self.training_split, self.sources)

    @dx.validates("labeling")
    @classmethod
    def _labeling_is_registered(cls, value: str) -> str:
        """Return the name of an installed inference label vocabulary."""
        return _checked_labeling(value)

    def inference_labels(self) -> tuple[str, ...]:
        """Return the declared output vocabulary in model-column order."""
        return tuple(label.value for label in registered_labeling(self.labeling).labels)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return labeled inference pair batches."""
        return frozenset({TrainingBatchKind.INFERENCE_PAIRS})


class ChaosNliDatasetSpec(DatasetSpec):
    """Declare normalized ChaosNLI count releases.

    A grounding trains on the aggregated relation of each pair in the declared
    training split, read through the declared labeling.
    """

    kind: Literal["chaosnli"] = "chaosnli"
    sources: tuple[CorpusSourceSpec, ...]
    training_split: str | None = None
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    tokenizer: TokenizerSpec = _TOKENIZER_FIELD
    aggregation: GoldAggregationSpec = _AGGREGATION_FIELD
    labeling: str = _DEFAULT_LABELING

    @dx.model_validator()
    def _sources_name_the_training_split(self) -> Self:
        """Require distinct release splits that include any training split."""
        _checked_training_split(self.training_split, self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split that does not train."""
        return _held_out_split(self.training_split, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split that does not train."""
        return _retained_splits(self.training_split, self.sources)

    @dx.validates("labeling")
    @classmethod
    def _labeling_is_registered(cls, value: str) -> str:
        """Return the name of an installed inference label vocabulary."""
        return _checked_labeling(value)

    def inference_labels(self) -> tuple[str, ...]:
        """Return the declared output vocabulary in model-column order."""
        return tuple(label.value for label in registered_labeling(self.labeling).labels)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return labeled inference pair batches."""
        return frozenset({TrainingBatchKind.INFERENCE_PAIRS})


class UnliDatasetSpec(DatasetSpec):
    """Declare normalized UNLI probability releases."""

    kind: Literal["unli"] = "unli"
    sources: tuple[CorpusSourceSpec, ...]
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    tokenizer: TokenizerSpec = _TOKENIZER_FIELD
    aggregation: GoldAggregationSpec = _AGGREGATION_FIELD

    @dx.model_validator()
    def _sources_name_distinct_splits(self) -> Self:
        """Require one release per declared split."""
        _split_names(self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split."""
        return _held_out_split(None, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split."""
        return _retained_splits(None, self.sources)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return graded inference pair batches."""
        return frozenset({TrainingBatchKind.INFERENCE_PAIRS})


class FracasDatasetSpec(DatasetSpec):
    """Declare FraCaS XML releases as pairs of premises and hypotheses.

    A grounding trains on the decided pairs of the declared training split,
    each relation read through the declared labeling.
    """

    kind: Literal["fracas"] = "fracas"
    sources: tuple[CorpusSourceSpec, ...]
    training_split: str | None = None
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    tokenizer: TokenizerSpec = _TOKENIZER_FIELD
    aggregation: GoldAggregationSpec = _AGGREGATION_FIELD
    labeling: str = _DEFAULT_LABELING

    @dx.model_validator()
    def _sources_name_the_training_split(self) -> Self:
        """Require distinct release splits that include any training split."""
        _checked_training_split(self.training_split, self.sources)
        return self

    def held_out_split(self) -> str:
        """Return the first declared release split that does not train."""
        return _held_out_split(self.training_split, self.sources)

    def retained_splits(self) -> frozenset[str]:
        """Return every declared release split that does not train."""
        return _retained_splits(self.training_split, self.sources)

    @dx.validates("labeling")
    @classmethod
    def _labeling_is_registered(cls, value: str) -> str:
        """Return the name of an installed inference label vocabulary."""
        return _checked_labeling(value)

    def inference_labels(self) -> tuple[str, ...]:
        """Return the declared output vocabulary in model-column order."""
        return tuple(label.value for label in registered_labeling(self.labeling).labels)

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return labeled inference pair batches."""
        return frozenset({TrainingBatchKind.INFERENCE_PAIRS})
