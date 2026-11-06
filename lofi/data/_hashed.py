"""Hashed examples and their encoder."""

from typing import TYPE_CHECKING, NamedTuple, final, override

from lofi._frozen import FrozenValue
from lofi.data._constraint import SpanConstraints
from lofi.data._inventory import UnknownSymbolError
from lofi.data._supervision import (
    GoldDerivation,
    Scalar,
    Unlabeled,
)

if TYPE_CHECKING:
    from lofi._types import ItemId
    from lofi.data._derivation import Derivation
    from lofi.data._inventory import Inventory
    from lofi.data._join import ItemLookup
    from lofi.data._ports import CodeTensorizer
    from lofi.data._supervision import Example, Supervision, SupervisionArm
    from lofi.data._values import Item

__all__ = [
    "Alphabet",
    "HashedItem",
    "HashedResources",
    "ItemHasher",
    "UnreadableArmError",
]


class UnreadableArmError(ValueError):
    """An error for an unsupported supervision arm.

    Parameters
    ----------
    arm : str
        The arm's name.
    accepted : frozenset[str]
        The names of the arms the adapter does read.
    """

    def __init__(self, arm: str, accepted: frozenset[str]) -> None:
        super().__init__(
            f"the adapter reads {', '.join(sorted(accepted))} and was given "
            f"{arm}, which it has no dense form for"
        )

        #: The arm's name.
        self.arm = arm

        #: The names of the arms the adapter does read.
        self.accepted = accepted


class Alphabet(NamedTuple):
    """Lexical and category inventory sizes.

    Parameters
    ----------
    lexical : int
        How many codes the surface-form inventory declares.
    categorial : int
        How many codes the category inventory declares.
    """

    lexical: int
    categorial: int


class HashedItem:
    """An encoded corpus item.

    Parameters
    ----------
    item_id : ItemId
        Item identifier, preserved unchanged.
    alphabet : Alphabet
        Lexical and category inventory sizes.
    symbols : tuple[int, ...]
        Surface-form codes in token order.
    analyses : tuple[frozenset[int], ...]
        Alternative category codes by token position, if available.
    response : Scalar | None
        Scalar response, if available.

    Raises
    ------
    ValueError
        If the identifier, codes, analysis width, or category alternatives are
        invalid.
    """

    __slots__ = ("_alphabet", "_analyses", "_item_id", "_response", "_symbols")

    def __init__(
        self,
        item_id: ItemId,
        alphabet: Alphabet,
        symbols: tuple[int, ...],
        analyses: tuple[frozenset[int], ...] = (),
        response: Scalar | None = None,
    ) -> None:
        if not item_id:
            msg = "a hashed item has no id"
            raise ValueError(msg)

        for position, code in enumerate(symbols):
            _in_range(code, alphabet.lexical, f"the surface at position {position}")

        if analyses and len(analyses) != len(symbols):
            msg = (
                f"a hashed item holds {len(symbols)} tokens and analyses over "
                f"{len(analyses)} positions"
            )
            raise ValueError(msg)

        for position, beam in enumerate(analyses):
            if not beam:
                msg = f"position {position} of a hashed item admits no category"
                raise ValueError(msg)

            for code in sorted(beam):
                _in_range(
                    code, alphabet.categorial, f"a category at position {position}"
                )

        self._item_id = item_id
        self._alphabet = alphabet
        self._symbols = symbols
        self._analyses = analyses
        self._response = response

    @property
    def item_id(self) -> ItemId:
        """Return the item identifier."""
        return self._item_id

    @property
    def alphabet(self) -> Alphabet:
        """Return the inventory sizes used for encoding."""
        return self._alphabet

    @property
    def symbols(self) -> tuple[int, ...]:
        """Return the code of each token's surface form, in position order."""
        return self._symbols

    @property
    def analyses(self) -> tuple[frozenset[int], ...]:
        """Return the alternative category codes at each position, unresolved."""
        return self._analyses

    @property
    def response(self) -> Scalar | None:
        """Return the optional scalar response."""
        return self._response

    @property
    def is_ambiguous(self) -> bool:
        """Return whether some position admits more than one category."""
        return any(len(beam) > 1 for beam in self._analyses)

    def _identity(
        self,
    ) -> tuple[
        ItemId, Alphabet, tuple[int, ...], tuple[frozenset[int], ...], Scalar | None
    ]:
        """Return the five fields a hashed item is equal on."""
        return (
            self._item_id,
            self._alphabet,
            self._symbols,
            self._analyses,
            self._response,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same datum in the same codes."""
        if type(other) is not HashedItem:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the id, the alphabet, the codes and the response."""
        return hash((HashedItem, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this hashed item."""
        return (
            f"HashedItem({self._item_id!r}, {self._alphabet!r}, "
            f"{self._symbols!r}, {self._analyses!r}, {self._response!r})"
        )


def _in_range(code: int, ceiling: int, subject: str) -> None:
    """Validate a code against an inventory size.

    Parameters
    ----------
    code : int
        The code to check.
    ceiling : int
        How many codes the inventory declares.
    subject : str
        Subject used in validation messages.

    Raises
    ------
    ValueError
        If the code is negative or at or past the ceiling.
    """
    if not 0 <= code < ceiling:
        msg = (
            f"{subject} is coded {code}, which no inventory of {ceiling} codes "
            "could have minted"
        )
        raise ValueError(msg)


def _leaf_labels(derivation: Derivation[str], tokens: int) -> tuple[str, ...]:
    """Return the category each token position takes under one analysis.

    Parameters
    ----------
    derivation : Derivation[str]
        The analysis.
    tokens : int
        Number of tokens in the item.

    Returns
    -------
    tuple[str, ...]
        One category per position, in position order.

    Raises
    ------
    ValueError
        If the analysis leaves a token position uncovered.
    """
    covered: list[str | None] = [None] * tokens

    for node in derivation.nodes.values():
        if not node.is_lexical:
            continue

        for position in range(node.span.start, node.span.end):
            if 0 <= position < tokens:
                covered[position] = node.label

    if any(label is None for label in covered):
        missing = [at for at, label in enumerate(covered) if label is None]
        msg = (
            f"an analysis leaves positions {missing} of a {tokens}-token item "
            "uncovered, so it analyzes a different item"
        )
        raise ValueError(msg)

    return tuple(label for label in covered if label is not None)


class ItemHasher:
    """An encoder from symbolic examples to hashed examples.

    Parameters
    ----------
    lexicon : Inventory
        Surface-form inventory.
    categories : Inventory
        Category inventory.
    unknown : str | None
        Reserved lexical symbol used for unknown surface forms.

    Raises
    ------
    UnknownSymbolError
        If the stand-in is not a symbol the lexical inventory declares.
    ValueError
        If the stand-in is an ordinary symbol rather than a reserved one.
    """

    __slots__ = ("_categories", "_lexicon", "_unknown")

    def __init__(
        self,
        lexicon: Inventory,
        categories: Inventory,
        unknown: str | None = None,
    ) -> None:
        if unknown is not None and not lexicon.is_reserved(unknown):
            msg = (
                f"{unknown!r} stands in for an undeclared surface form and is "
                f"an ordinary symbol of {lexicon.name}, so its code moves as "
                "the inventory grows"
            )
            raise ValueError(msg)

        self._lexicon = lexicon
        self._categories = categories
        self._unknown = unknown

    @property
    def lexicon(self) -> Inventory:
        """Return the surface forms the run declares."""
        return self._lexicon

    @property
    def categories(self) -> Inventory:
        """Return the categories the run declares."""
        return self._categories

    @property
    def alphabet(self) -> Alphabet:
        """Return how many codes each of the two inventories declares."""
        return Alphabet(self._lexicon.size, self._categories.size)

    @property
    def accepts(self) -> frozenset[SupervisionArm]:
        """Return the supported supervision arms."""
        return _ACCEPTED

    def __call__(self, example: Example[Supervision[str]], /) -> HashedItem:
        """Return the hashed form of one example.

        Parameters
        ----------
        example : Example[Supervision[str]]
            Symbolic example.

        Returns
        -------
        HashedItem
            Encoded item, analyses, and response.

        Raises
        ------
        UnreadableArmError
            If the example's supervision is of an arm this adapter does not
            read.
        UnknownSymbolError
            If a surface form or a category is one the inventories do not
            declare and no stand-in was named.
        ValueError
            If the supervision analyses a different item from the one beside
            it.
        """
        item = example.item
        supervision = example.supervision
        symbols = tuple(self._surface(token.surface) for token in item.tokens)

        if isinstance(supervision, Unlabeled):
            return HashedItem(item.item_id, self.alphabet, symbols)

        if isinstance(supervision, Scalar):
            return HashedItem(item.item_id, self.alphabet, symbols, (), supervision)

        if isinstance(supervision, SpanConstraints):
            return HashedItem(
                item.item_id,
                self.alphabet,
                symbols,
                self._position_codes(supervision, item),
            )

        if isinstance(supervision, GoldDerivation):
            labels = _leaf_labels(supervision.derivation, len(item.tokens))
            analyses = tuple(
                frozenset({self._categories.index(label)}) for label in labels
            )

            return HashedItem(item.item_id, self.alphabet, symbols, analyses)

        raise UnreadableArmError(
            type(supervision).__name__,
            frozenset(arm.__name__ for arm in _ACCEPTED),
        )

    def _surface(self, surface: str) -> int:
        """Return the code a token's surface form is declared under.

        Parameters
        ----------
        surface : str
            The written form.

        Returns
        -------
        int
            The surface code or configured unknown code.

        Raises
        ------
        UnknownSymbolError
            If the inventory does not declare the form and no stand-in was
            named.
        """
        if surface in self._lexicon:
            return self._lexicon.index(surface)

        if self._unknown is None:
            raise UnknownSymbolError(self._lexicon.name, surface)

        return self._lexicon.index(self._unknown)

    def _position_codes(
        self,
        supervision: SpanConstraints[str],
        item: Item,
    ) -> tuple[frozenset[int], ...]:
        """Return category codes for canonical per-position constraints.

        Parameters
        ----------
        supervision : SpanConstraints[str]
            Constraints over the item.
        item : Item
            Associated item.

        Returns
        -------
        tuple[frozenset[int], ...]
            One set per position when the constraints are total and width one,
            otherwise no dense analyses.

        Raises
        ------
        ValueError
            If the constraints name a different item length.
        UnknownSymbolError
            If a permitted label is a category the inventory does not declare.
        """
        if supervision.token_count != len(item.tokens):
            msg = (
                f"constraints run over {supervision.token_count} positions and "
                f"the item holds {len(item.tokens)} tokens"
            )
            raise ValueError(msg)

        constraints = supervision.constraints
        if len(constraints) != len(item.tokens):
            return ()

        for position, constraint in enumerate(constraints):
            span = constraint.span
            if (
                span.start != position
                or span.end != position + 1
                or constraint.scope.form is not None
            ):
                return ()

        return tuple(
            frozenset(self._categories.index(label) for label in constraint.permitted)
            for constraint in constraints
        )


_ACCEPTED: frozenset[SupervisionArm] = frozenset(
    {Unlabeled, Scalar, SpanConstraints, GoldDerivation}
)
"""The supported supervision arms."""


@final
class HashedResources(FrozenValue):
    """The encoder, tensor backend and item lookup a featurizer is built from.

    Parameters
    ----------
    hasher : ItemHasher
        The encoder over a dataset's vocabulary and gold symbols.
    tensorizer : CodeTensorizer
        The padded-row tensor backend.
    lookup : ItemLookup
        Every addressable item, keyed by identifier.
    """

    _noun = "featurizer resources"

    __slots__ = ("_hasher", "_lookup", "_tensorizer")

    def __init__(
        self,
        hasher: ItemHasher,
        tensorizer: CodeTensorizer,
        lookup: ItemLookup,
    ) -> None:
        object.__setattr__(self, "_hasher", hasher)
        object.__setattr__(self, "_tensorizer", tensorizer)
        object.__setattr__(self, "_lookup", lookup)

    @property
    def hasher(self) -> ItemHasher:
        """Return the item encoder."""
        return self._hasher

    @property
    def tensorizer(self) -> CodeTensorizer:
        """Return the padded-row tensor backend."""
        return self._tensorizer

    @property
    def lookup(self) -> ItemLookup:
        """Return the addressable items keyed by identifier."""
        return self._lookup
