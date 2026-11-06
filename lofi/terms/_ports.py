"""Term-layer protocols and encodings."""

from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from lofi._native import RepresentativeKernel, Term, TermStore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.terms._values import Operator, Signature

__all__ = [
    "BatchEncoding",
    "Calculus",
    "CategoryResolution",
    "Constructors",
    "Encoding",
    "ExactMatching",
    "Functors",
    "Interner",
    "Layout",
    "Matching",
    "OpName",
    "Path",
    "PayloadSort",
    "Reifiable",
    "RootMatching",
    "SignatureStableMatching",
    "Sort",
    "SortTag",
    "TermKey",
    "TermMatching",
    "TermSyntax",
]


type Sort = str
"""The kind of value an operator constructs."""

type OpName = str
"""An operator name within a signature."""

type PayloadSort = Literal["none", "constant", "index", "free"]
"""Classifies an operator's non-term payload.

A ``constant`` payload is a stable identifier in its sort's vocabulary, an
``index`` payload is a de Bruijn index, and a ``free`` payload is the store's
handle on a free variable's name.
"""

type Path = tuple[int, ...]
"""Argument positions from a term root."""

type TermKey = tuple[OpName, int | str | None, tuple[TermKey, ...]]
"""A recursive, immutable structural key for a term."""


@runtime_checkable
class CategoryResolution(Protocol):
    """Resolution of category source text within one grammar crossing."""

    def __call__(self, source: str, /) -> Term:
        """Return the store-owned category denoted by source text."""
        ...


class SortTag(Protocol):
    """A static marker for one sort of a language.

    A language declares one subclass per sort it hands to consumers, such as
    ``class Ty(SortTag)``, and a store's view of that sort returns terms typed
    by it. The tag has no runtime role: the store's construction-time sort
    check is what keeps a term of one sort out of a position of another.
    """


@runtime_checkable
class Matching[T](Protocol):
    """A term congruence."""

    @property
    def name(self) -> str:
        """Return the relation's name, as it appears in a grammar fingerprint."""
        ...

    @property
    def has_key(self) -> bool:
        """Return whether representative keys support chart packing."""
        ...

    def key(self, t: T, /) -> T:
        """Return the term's congruence-class representative.

        Parameters
        ----------
        t : T
            The term to canonicalize.
        """
        ...

    def equivalent(self, a: T, b: T, /) -> bool:
        """Return whether two terms share a congruence class.

        Parameters
        ----------
        a : T
            The left term.
        b : T
            The right term.
        """
        ...


@runtime_checkable
class TermMatching(Matching[Term], Protocol):
    """A term congruence compiled for native rule matching."""

    @property
    def representatives(self) -> RepresentativeKernel:
        """Return the native representative function."""
        ...


@runtime_checkable
class ExactMatching(Protocol):
    """A term congruence that declares whether it is syntactic identity.

    An exact congruence relates two terms only when they are identical, so a
    term's head is a sound index of every term it matches. A congruence that
    declares nothing is treated as coarser than identity.
    """

    @property
    def exact(self) -> bool:
        """Report whether two terms match only when they are identical."""
        ...


@runtime_checkable
class RootMatching(Protocol):
    """A term congruence that declares which root operators it identifies.

    Two terms match only when the root operator of one is among the operators
    the congruence identifies with the root operator of the other. A fixed
    premise root together with those operators is thus a sound head index of
    every term the premise matches.
    """

    def identified_roots(
        self, operator: str, signature: Signature, /
    ) -> frozenset[str] | None:
        """Return every root operator a term rooted at ``operator`` may match.

        Parameters
        ----------
        operator : str
            The root operator name of a fixed premise.
        signature : Signature
            The signature whose operators the congruence ranges over.

        Returns
        -------
        frozenset[str] | None
            The identified operator names, ``operator`` among them, or
            ``None`` when the congruence bounds no root.
        """
        ...


@runtime_checkable
class SignatureStableMatching(Protocol):
    """A term congruence whose keys survive signature extension.

    The capability applies when an extension preserves the digest of every
    vocabulary already present in the source signature. It says that adding
    sorts, operators, or vocabularies cannot change the symbolic pattern for
    the representative of a source term.
    """

    @property
    def signature_extension_stable(self) -> bool:
        """Report whether representative patterns survive signature extension."""
        ...


@runtime_checkable
class TermSyntax(Protocol):
    """A text form for terms of one signature."""

    @property
    def signature(self) -> Signature:
        """Return the signature whose terms the syntax reads and writes."""
        ...

    def parse(self, store: TermStore, text: str, /) -> Term:
        """Intern the term a text denotes."""
        ...

    def render(self, store: TermStore, term: Term, /) -> str:
        """Return canonical text that parses back to the term."""
        ...


@runtime_checkable
class Constructors[T](Protocol):
    """Declared constructor operations over terms of one sort."""

    @property
    def names(self) -> tuple[OpName, ...]:
        """Return constructor names in declaration order."""
        ...

    def roles(self, constructor: OpName, /) -> tuple[str, ...]:
        """Return the declared role of every child slot."""
        ...

    def is_constructor(self, term: T, /) -> bool:
        """Return whether a term is headed by a declared constructor."""
        ...

    def constructor(self, term: T, /) -> OpName:
        """Return a constructor term's declared operator name."""
        ...

    def make(self, constructor: OpName, children: Sequence[T], /) -> T:
        """Return the term built by a constructor over its children."""
        ...


@runtime_checkable
class Functors[T](Protocol):
    """Binary term functor operations."""

    @property
    def slashes(self) -> frozenset[OpName]:
        """Return the functor operators."""
        ...

    @property
    def forward_slash(self) -> OpName:
        """Return the operator that takes its argument on the right."""
        ...

    @property
    def backward_slash(self) -> OpName:
        """Return the operator that takes its argument on the left."""
        ...

    @property
    def result_path(self) -> Path:
        """Return the one-child path to a functor's result."""
        ...

    @property
    def argument_path(self) -> Path:
        """Return the one-child path to a functor's argument."""
        ...

    @property
    def category_sort(self) -> Sort:
        """Return the sort the functor operators build and take."""
        ...

    def is_functor(self, term: T, /) -> bool:
        """Return whether a term is headed by a functor operator."""
        ...

    def slash(self, term: T, /) -> OpName:
        """Return a functor's operator."""
        ...

    def make(self, slash: OpName, result: T, argument: T, /) -> T:
        """Return a functor over a result and an argument."""
        ...

    def opposite(self, slash: OpName, /) -> OpName:
        """Return the operator with the opposite orientation."""
        ...


@runtime_checkable
class Layout[T, S, I](Protocol):
    """An integer sequence layout for terms."""

    @property
    def name(self) -> str:
        """Return the layout's name, as it appears in a grammar fingerprint."""
        ...

    @property
    def path_addressed(self) -> bool:
        """Return whether each slot has a term-independent tree path."""
        ...

    @property
    def capacity(self) -> int:
        """Return the fixed number of slots per encoding."""
        ...

    @property
    def skeleton(self) -> Encoding | None:
        """Return fixed slot structure, or None for term-dependent structure."""
        ...

    def static_symbols(self, index: I, /) -> tuple[tuple[bool, ...], ...]:
        """Return the symbols that may occupy each slot."""
        ...

    def encode(self, t: T, index: I, /) -> Encoding:
        """Encode a term as integer channels.

        Parameters
        ----------
        t : T
            The term to encode.
        index : I
            The operator index used for symbol codes.
        """
        ...

    def decode(self, enc: Encoding, store: S, index: I, /) -> T:
        """Decode integer channels into a term.

        Parameters
        ----------
        enc : Encoding
            The encoding to decode.
        store : S
            The store that interns the term.
        index : I
            The operator index used for symbol codes.
        """
        ...

    def legal_next(self, prefix: Sequence[int], index: I, /) -> tuple[bool, ...]:
        """Return symbols allowed after a partial encoding.

        Parameters
        ----------
        prefix : Sequence[int]
            The symbol codes already emitted.
        index : I
            The operator index addressed by the result.
        """
        ...


@runtime_checkable
class Reifiable[S](Protocol):
    """An object with its source declaration."""

    def to_spec(self) -> S:
        """Return the declaration this object was built from."""
        ...


class Encoding(NamedTuple):
    """Parallel integer channels for one term.

    Attributes
    ----------
    symbols : tuple[int, ...]
        The operator code at each slot.
    parents : tuple[int, ...]
        The parent slot for each slot.
    depths : tuple[int, ...]
        The depth of each slot.
    occupied : tuple[bool, ...]
        Whether each slot contains a node.
    payloads : tuple[int, ...] | None
        The optional payload code at each slot.
    """

    symbols: tuple[int, ...]
    parents: tuple[int, ...]
    depths: tuple[int, ...]
    occupied: tuple[bool, ...]
    payloads: tuple[int, ...] | None = None

    NO_PARENT = -1
    """The sentinel for a root slot."""

    @property
    def width(self) -> int:
        """Return the number of slots, occupied or not."""
        return len(self.symbols)

    @property
    def length(self) -> int:
        """Return the occupied slot count."""
        return self.occupied.count(True)

    def ancestors(self, slot: int) -> tuple[int, ...]:
        """Return ancestor slots from parent to root.

        Parameters
        ----------
        slot : int
            The slot whose ancestors are requested.
        """
        walk: list[int] = []
        at = self.parents[slot]

        while at != Encoding.NO_PARENT:
            walk.append(at)
            at = self.parents[at]

        return tuple(walk)


class BatchEncoding(NamedTuple):
    """Equal-width term encodings.

    Attributes
    ----------
    rows : tuple[Encoding, ...]
        The encodings in batch order.
    """

    rows: tuple[Encoding, ...]

    @property
    def width(self) -> int:
        """Return the row width, or zero when empty."""
        return self.rows[0].width if self.rows else 0

    @property
    def symbols(self) -> tuple[tuple[int, ...], ...]:
        """Return the operator codes, one row per encoding."""
        return tuple(row.symbols for row in self.rows)

    @property
    def parents(self) -> tuple[tuple[int, ...], ...]:
        """Return the parent slots, one row per encoding."""
        return tuple(row.parents for row in self.rows)

    @property
    def depths(self) -> tuple[tuple[int, ...], ...]:
        """Return the depths, one row per encoding."""
        return tuple(row.depths for row in self.rows)

    @property
    def mask(self) -> tuple[tuple[bool, ...], ...]:
        """Return occupied slots for each encoding."""
        return tuple(row.occupied for row in self.rows)

    @property
    def payloads(self) -> tuple[tuple[int, ...], ...] | None:
        """Return the payload codes, one row per encoding, or None for no channel."""
        if not self.rows or self.rows[0].payloads is None:
            return None

        return tuple(row.payloads for row in self.rows if row.payloads is not None)

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the occupied count of each row."""
        return tuple(row.length for row in self.rows)


class Calculus(NamedTuple):
    """Abstraction and application operators.

    Attributes
    ----------
    lam : OpName
        The abstraction operator name.
    app : OpName
        The application operator name.
    """

    lam: OpName
    app: OpName


class Interner(Protocol):
    """A term interning interface."""

    def op(
        self,
        op: Operator,
        /,
        *args: Term,
        payload: int | None = None,
    ) -> Term:
        """Return an interned operator application.

        Parameters
        ----------
        op : Operator
            Operator to apply.
        *args : Term
            Arguments in operator order.
        payload : int | None
            Optional constant, index, or free variable payload.

        Returns
        -------
        Term
            Interned application node.
        """
        ...
