"""Data-layer protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence

    from lairs.integrations.codecs import CorpusFragment, FragmentRecord
    from torch import Tensor

    from lofi._types import ItemId
    from lofi.data._constraint_sources import ConstraintSelection
    from lofi.data._hashed import ItemHasher
    from lofi.data._join import ItemLookup
    from lofi.data._report import LoadReport
    from lofi.data._sampling import (
        PairDistribution,
        PairSampleReport,
        SampledPair,
        TypeDistribution,
        TypeSampleReport,
    )
    from lofi.data._selection import ScopeKind, SelectionPolicy
    from lofi.data._supervision import Example, Supervision, SupervisionArm
    from lofi.data._values import Item
    from lofi.lexicon import LexicalSource
    from lofi.terms import Signature, Term, TermStore

__all__ = [
    "AmbiguityRule",
    "CodeTensorizer",
    "Codec",
    "ConstraintBatcher",
    "ConstraintSource",
    "CorpusCodec",
    "CorpusReader",
    "CorpusRelease",
    "Featurizer",
    "FeaturizerComponent",
    "FeaturizerResources",
    "InventoriedTypeSampler",
    "Iso",
    "IsoCodec",
    "LabelMembership",
    "PairSampler",
    "Signed",
    "Source",
    "StepRecovery",
    "TypeInventory",
    "TypeSampler",
]


class CorpusRelease(Protocol):
    """Metadata a corpus codec writes into source records."""

    @property
    def source_id(self) -> str:
        """Return the stable source identifier."""
        ...

    @property
    def release_id(self) -> str:
        """Return the source release identifier."""
        ...

    @property
    def split(self) -> str:
        """Return the split receiving decoded records."""
        ...

    @property
    def language(self) -> str:
        """Return the source language tag."""
        ...

    @property
    def source_method(self) -> str:
        """Return how the source expressions were obtained."""
        ...

    @property
    def spdx(self) -> str:
        """Return the release's SPDX license expression."""
        ...

    @property
    def eprint_refs(self) -> tuple[str, ...]:
        """Return publication record references in declaration order."""
        ...


@runtime_checkable
class CodeTensorizer(Protocol):
    """A tensor backend for padded integer rows."""

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[Tensor, Tensor]:
        """Build integer rows and their boolean presence mask.

        Parameters
        ----------
        rows : Sequence[tuple[int, ...]]
            Integer rows before padding.
        padding : int
            Code used to pad shorter rows.

        Returns
        -------
        tuple[Tensor, Tensor]
            Padded integer rows and a boolean presence mask.
        """


@runtime_checkable
class FeaturizerResources(Protocol):
    """Dataset resources used to construct a featurizer."""

    @property
    def hasher(self) -> ItemHasher:
        """Return the item encoder."""

    @property
    def tensorizer(self) -> CodeTensorizer:
        """Return the padded-row tensor backend."""

    @property
    def lookup(self) -> ItemLookup:
        """Return the items addressable by identifier."""


@runtime_checkable
class FeaturizerComponent(Protocol):
    """A featurizer exposed through component wiring."""

    @property
    def accepts(self) -> frozenset[SupervisionArm]:
        """Return the supervision arms this featurizer reads."""

    @property
    def selection(self) -> SupervisionArm | None:
        """Return the selected arm, if one is fixed."""


@runtime_checkable
class TypeSampler(Protocol):
    """A source of interned terms from a declared distribution."""

    @property
    def signature(self) -> Signature:
        """Return the sampled term signature."""

    def sample(
        self,
        count: int,
        distribution: TypeDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[Term, ...], TypeSampleReport]:
        """Return a deterministic batch and its observed counts."""


@runtime_checkable
class TypeInventory(Protocol):
    """A stable finite candidate universe of interned terms."""

    @property
    def store(self) -> TermStore:
        """Return the store that owns every candidate."""

    @property
    def signature(self) -> Signature:
        """Return the candidate term signature."""

    @property
    def max_size(self) -> int:
        """Return the inclusive node bound."""

    @property
    def candidates(self) -> tuple[Term, ...]:
        """Return every candidate in stable inventory order."""

    def at_depth(self, depth: int, /) -> tuple[Term, ...]:
        """Return candidates at one exact one-based depth."""

    def through_depth(self, max_depth: int, /) -> tuple[Term, ...]:
        """Return candidates no deeper than an inclusive bound."""

    def __iter__(self) -> Iterator[Term]:
        """Return candidates in stable inventory order."""

    def __len__(self) -> int:
        """Return the candidate count."""

    @overload
    def __getitem__(self, index: int, /) -> Term: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[Term, ...]: ...

    def __getitem__(self, index: int | slice, /) -> Term | tuple[Term, ...]:
        """Return one candidate or a stable candidate slice."""


@runtime_checkable
class InventoriedTypeSampler(TypeSampler, Protocol):
    """A type sampler publishing its finite candidate universe."""

    @property
    def inventory(self) -> TypeInventory:
        """Return the finite candidate universe."""


@runtime_checkable
class PairSampler[T, Label](Protocol):
    """A source of labeled rule instances of arbitrary arity."""

    def sample(
        self,
        count: int,
        distribution: PairDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[SampledPair[T, Label], ...], PairSampleReport]:
        """Return a deterministic batch and its observed outcomes."""


@runtime_checkable
class Source[T](Protocol):
    """A repeatable finite iterable."""

    def __iter__(self) -> Iterator[T]:
        """Return a fresh iterator over the elements.

        Returns
        -------
        Iterator[T]
            An iterator that starts at the first element.
        """

    def __len__(self) -> int:
        """Return how many elements there are.

        Returns
        -------
        int
            The count.
        """


@runtime_checkable
class StepRecovery[T, Recovered](Protocol):
    """A recovery relation for observed steps."""

    def recover(
        self,
        inputs: Sequence[T],
        output: T,
        /,
    ) -> tuple[Recovered, ...]:
        """Return recovered values in stable order."""


@runtime_checkable
class LabelMembership[T](Protocol):
    """A label inventory used for reachability checks."""

    def contains(self, label: T, /) -> bool:
        """Return whether the inventory contains a label."""


class Signed(Protocol):
    """A value with a grammar signature."""

    @property
    def signature(self) -> Signature:
        """Return the term algebra the grammar's categories are built over."""


@runtime_checkable
class Iso[Parsed, Target](Protocol):
    """A lossless bidirectional translation."""

    def forward(self, source: Parsed, /) -> Target:
        """Translate a source value.

        Parameters
        ----------
        source : Parsed
            The value to translate.

        Returns
        -------
        Target
            Translated value.
        """

    def backward(self, target: Target, /) -> Parsed:
        """Translate a target value back to its parsed form.

        Parameters
        ----------
        target : Target
            The value to translate back.

        Returns
        -------
        Parsed
            Parsed source value.
        """


@runtime_checkable
class Codec[Fragment, Record](Protocol):
    """An encoder and decoder for external records.

    Attributes
    ----------
    name : str
        The name the codec is registered and reported under.
    """

    @property
    def name(self) -> str:
        """Return the name the codec is registered and reported under."""

    def decode(self, src: str | bytes, *, into: Fragment | None = None) -> Fragment:
        """Decode records from an external source.

        Parameters
        ----------
        src : str | bytes
            Encoded text or bytes.
        into : Fragment | None
            Existing fragment to extend, if any.

        Returns
        -------
        Fragment
            Decoded fragment.
        """

    def encode(self, records: Iterable[Record]) -> bytes | str:
        """Encode a batch of records.

        Parameters
        ----------
        records : Iterable[Record]
            The records to write out.

        Returns
        -------
        bytes | str
            Encoded records.
        """


@runtime_checkable
class IsoCodec[Parsed, Fragment, Record](Codec[Fragment, Record], Protocol):
    """A codec backed by an isomorphism.

    Attributes
    ----------
    iso : Iso[Parsed, Fragment]
        The translation between the format's parsed shape and a fragment.
    """

    @property
    def iso(self) -> Iso[Parsed, Fragment]:
        """Return the translation between the format's parsed shape and a fragment."""


@runtime_checkable
class CorpusReader[L: Hashable, G](Protocol):
    """A reader for shared data-layer values."""

    @property
    def name(self) -> str:
        """Return the name the reader is registered and reported under."""

    @property
    def grammar(self) -> G | None:
        """Return the grammar the corpus's analyses are read under, if it has one."""

    @property
    def lexicon(self) -> LexicalSource[L, float] | None:
        """Return the axioms the corpus ships, if it ships any."""

    def splits(self) -> tuple[str, ...]:
        """Return the splits the corpus is divided into.

        Returns
        -------
        tuple[str, ...]
            Split names in source order.
        """

    def items(self, split: str, /) -> Source[Item]:
        """Return the items of one split.

        Parameters
        ----------
        split : str
            The split's name.

        Returns
        -------
        Source[Item]
            Re-readable items in stable order.
        """

    def examples(self, split: str, /) -> Source[Example[Supervision[L]]]:
        """Return supervised examples from one split.

        Parameters
        ----------
        split : str
            The split's name.

        Returns
        -------
        Source[Example[Supervision[L]]]
            Re-readable examples in item order.
        """

    def supervision_kinds(self) -> frozenset[SupervisionArm]:
        """Return the supervision arms emitted by this reader.

        Returns
        -------
        frozenset[SupervisionArm]
            The arms, named by their own types.
        """

    def report(self) -> LoadReport:
        """Return the corpus load report.

        Returns
        -------
        LoadReport
            The report of the most recent load.
        """


@runtime_checkable
class AmbiguityRule[C: Hashable, L: Hashable](Protocol):
    """A resolution of a form observed under several types."""

    @property
    def scope(self) -> ScopeKind:
        """Return the scope a resolved constraint binds at."""

    def permitted(self, masses: Mapping[C, float], /) -> frozenset[L]:
        """Return the labels a form observed under those masses may bear.

        Parameters
        ----------
        masses : Mapping[C, float]
            The mass observed on each label, in observation order.

        Returns
        -------
        frozenset[L]
            The permitted labels, never empty.
        """


@runtime_checkable
class ConstraintSource[C: Hashable, L: Hashable](Protocol):
    """A source of span constraints drawn from a corpus."""

    @property
    def scope(self) -> ScopeKind:
        """Return the scope the emitted constraints bind at."""

    @property
    def policy(self) -> SelectionPolicy[C, L]:
        """Return which constraints are emitted from the evidence read."""

    def select(
        self, examples: Iterable[Example[Supervision[C]]], /
    ) -> ConstraintSelection[C, L]:
        """Return constrained examples and the achieved selection budget.

        Parameters
        ----------
        examples : Iterable[Example[Supervision[C]]]
            The corpus, in item order.

        Returns
        -------
        ConstraintSelection[C, L]
            Constrained examples in corpus order beside the load report and exact
            achieved selection budget.
        """


@runtime_checkable
class ConstraintBatcher[ConstraintT](Protocol):
    """Localizes dataset constraints to one item-id minibatch."""

    def for_items(
        self, item_ids: Sequence[ItemId], /
    ) -> tuple[tuple[ConstraintT, ...], ...]:
        """Return one row of localized constraints per item in batch order."""


@runtime_checkable
class Featurizer[S, B](Protocol):
    """A converter from domain values to model batches."""

    @property
    def accepts(self) -> frozenset[SupervisionArm]:
        """Return the arms of the supervision union this featurizer reads."""

    @property
    def selection(self) -> SupervisionArm | None:
        """Return the selected arm, or None to require a unique compatible arm."""

    def __call__(self, examples: Sequence[Example[S]], /) -> B:
        """Featurize a batch of examples.

        Parameters
        ----------
        examples : Sequence[Example[S]]
            Minibatch examples in item order.

        Returns
        -------
        B
            Featurized batch.
        """


type CorpusCodec[Parsed] = IsoCodec[Parsed, CorpusFragment, FragmentRecord]
"""An encoder and decoder for corpus fragments."""
