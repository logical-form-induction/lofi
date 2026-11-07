"""Corpus items, spans, and provenance."""

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from lofi._types import ItemId

__all__ = [
    "ByteSpan",
    "Item",
    "LicenseRef",
    "Licensing",
    "Provenance",
    "Reproducibility",
    "Token",
    "TokenSpan",
]


class ByteSpan:
    """A half-open UTF-8 byte span.

    Parameters
    ----------
    start : int
        Offset of the first byte covered.
    end : int
        Offset one past the last byte covered.

    Raises
    ------
    ValueError
        If ``start`` is negative or ``end`` precedes ``start``.
    """

    __slots__ = ("_end", "_start")

    def __init__(self, start: int, end: int) -> None:
        if start < 0:
            msg = f"a byte span starts before the text: {start}"
            raise ValueError(msg)

        if end < start:
            msg = f"a byte span ends before it starts: [{start}, {end})"
            raise ValueError(msg)

        self._start = start
        self._end = end

    @property
    def start(self) -> int:
        """Return the first covered byte offset."""
        return self._start

    @property
    def end(self) -> int:
        """Return the exclusive byte end offset."""
        return self._end

    @property
    def length(self) -> int:
        """Return the number of covered bytes."""
        return self._end - self._start

    def substring(self, text: str) -> str:
        """Return the part of a text this span covers.

        Parameters
        ----------
        text : str
            The source text the offsets were taken against.

        Returns
        -------
        str
            The covered substring.

        Raises
        ------
        UnicodeDecodeError
            If either offset falls inside a multi-byte character.
        ValueError
            If the span runs past the end of the text.
        """
        encoded = text.encode()

        if self._end > len(encoded):
            msg = f"a byte span runs past the text: {self._end} > {len(encoded)}"
            raise ValueError(msg)

        return encoded[self._start : self._end].decode()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a byte span with the same offsets."""
        if type(other) is not ByteSpan:
            return NotImplemented

        return self._start == other._start and self._end == other._end

    @override
    def __hash__(self) -> int:
        """Return a hash over the two offsets."""
        return hash((ByteSpan, self._start, self._end))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this span."""
        return f"ByteSpan({self._start}, {self._end})"


class TokenSpan:
    """A half-open token span.

    Parameters
    ----------
    start : int
        Position of the first token covered.
    end : int
        Position one past the last token covered.

    Raises
    ------
    ValueError
        If ``start`` is negative or ``end`` precedes ``start``.
    """

    __slots__ = ("_end", "_start")

    def __init__(self, start: int, end: int) -> None:
        if start < 0:
            msg = f"a span starts before the first token: {start}"
            raise ValueError(msg)

        if end < start:
            msg = f"a span ends before it starts: [{start}, {end})"
            raise ValueError(msg)

        self._start = start
        self._end = end

    @property
    def start(self) -> int:
        """Return the first covered token position."""
        return self._start

    @property
    def end(self) -> int:
        """Return the exclusive token end position."""
        return self._end

    @property
    def length(self) -> int:
        """Return the number of covered token positions."""
        return self._end - self._start

    def concat(self, other: TokenSpan) -> TokenSpan:
        """Return the span reaching from this span's start to another's end.

        Parameters
        ----------
        other : TokenSpan
            The span beginning where this one ends.

        Returns
        -------
        TokenSpan
            The span covering both.

        Raises
        ------
        ValueError
            If the two spans are not adjacent.
        """
        if other.start != self._end:
            msg = (
                f"spans are not adjacent: [{self._start}, {self._end}) "
                f"then [{other.start}, {other.end})"
            )
            raise ValueError(msg)

        return TokenSpan(self._start, other.end)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a token span with the same positions."""
        if type(other) is not TokenSpan:
            return NotImplemented

        return self._start == other._start and self._end == other._end

    @override
    def __hash__(self) -> int:
        """Return a hash over the two positions."""
        return hash((TokenSpan, self._start, self._end))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this span."""
        return f"TokenSpan({self._start}, {self._end})"


class Token:
    """A surface token and its position.

    Parameters
    ----------
    index : int
        The token's position in its item, counting from zero.
    surface : str
        The token's written form.
    span : ByteSpan | None
        Where the form falls in the item's source text, when the corpus
        records it. A corpus that ships tokens without offsets leaves this
        absent.

    Raises
    ------
    ValueError
        If ``index`` is negative.
    """

    __slots__ = ("_index", "_span", "_surface")

    def __init__(self, index: int, surface: str, span: ByteSpan | None = None) -> None:
        if index < 0:
            msg = f"a token position is negative: {index}"
            raise ValueError(msg)

        self._index = index
        self._surface = surface
        self._span = span

    @property
    def index(self) -> int:
        """Return the token's position in its item."""
        return self._index

    @property
    def surface(self) -> str:
        """Return the token's written form."""
        return self._surface

    @property
    def span(self) -> ByteSpan | None:
        """Return where the form falls in the item's source text, if recorded."""
        return self._span

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a token with the same three fields."""
        if type(other) is not Token:
            return NotImplemented

        return (
            self._index == other._index
            and self._surface == other._surface
            and self._span == other._span
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over position, surface and span."""
        return hash((Token, self._index, self._surface, self._span))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this token."""
        return f"Token({self._index}, {self._surface!r}, {self._span!r})"


class LicenseRef:
    """A license identifier and its scope.

    Parameters
    ----------
    identifier : str
        An SPDX license identifier.
    applies_to : str | None
        The component the identifier covers, named as the releasing corpus names
        it. Absent when the identifier covers the whole item.

    Raises
    ------
    ValueError
        If ``identifier`` is empty.
    """

    __slots__ = ("_applies_to", "_identifier")

    def __init__(self, identifier: str, applies_to: str | None = None) -> None:
        if not identifier:
            msg = "a license reference has no identifier"
            raise ValueError(msg)

        self._identifier = identifier
        self._applies_to = applies_to

    @property
    def identifier(self) -> str:
        """Return the SPDX license identifier."""
        return self._identifier

    @property
    def applies_to(self) -> str | None:
        """Return the component the identifier covers, if it covers only one."""
        return self._applies_to

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same license reference."""
        if type(other) is not LicenseRef:
            return NotImplemented

        return (
            self._identifier == other._identifier
            and self._applies_to == other._applies_to
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the identifier and what it applies to."""
        return hash((LicenseRef, self._identifier, self._applies_to))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this reference."""
        return f"LicenseRef({self._identifier!r}, {self._applies_to!r})"


class Licensing:
    """An item's licensing terms.

    Parameters
    ----------
    expression : str
        An SPDX license expression covering the item as a whole.
    licenses : tuple[LicenseRef, ...]
        Per-component refinements of the expression, where a corpus licenses its
        text and its annotation differently.

    Raises
    ------
    ValueError
        If ``expression`` is empty. An unlicensed item is not constructible, so
        no consumer downstream decides what an absent license means.
    """

    __slots__ = ("_expression", "_licenses")

    def __init__(self, expression: str, licenses: tuple[LicenseRef, ...] = ()) -> None:
        if not expression:
            msg = "a licensing record has no SPDX expression"
            raise ValueError(msg)

        self._expression = expression
        self._licenses = licenses

    @property
    def expression(self) -> str:
        """Return the SPDX license expression covering the whole item."""
        return self._expression

    @property
    def licenses(self) -> tuple[LicenseRef, ...]:
        """Return per-component license refinements."""
        return self._licenses

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value licenses the same way."""
        if type(other) is not Licensing:
            return NotImplemented

        return (
            self._expression == other._expression and self._licenses == other._licenses
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the expression and the per-component references."""
        return hash((Licensing, self._expression, self._licenses))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return f"Licensing({self._expression!r}, {self._licenses!r})"


class Reproducibility:
    """Metadata about how an artifact was produced.

    Parameters
    ----------
    code_uri : str | None
        A URI for the code that produced the artifact.
    commit_hash : str | None
        The revision of that code.
    command : str | None
        The command line that ran.
    environment : str | None
        The environment it ran in, as the producer recorded it.
    random_seed : int | None
        The seed the run was given.
    """

    __slots__ = (
        "_code_uri",
        "_command",
        "_commit_hash",
        "_environment",
        "_random_seed",
    )

    def __init__(
        self,
        code_uri: str | None = None,
        commit_hash: str | None = None,
        command: str | None = None,
        environment: str | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._code_uri = code_uri
        self._commit_hash = commit_hash
        self._command = command
        self._environment = environment
        self._random_seed = random_seed

    @property
    def code_uri(self) -> str | None:
        """Return a URI for the code that produced the artifact."""
        return self._code_uri

    @property
    def commit_hash(self) -> str | None:
        """Return the revision of the producing code."""
        return self._commit_hash

    @property
    def command(self) -> str | None:
        """Return the command line that produced the artifact."""
        return self._command

    @property
    def environment(self) -> str | None:
        """Return the environment the producing run took place in."""
        return self._environment

    @property
    def random_seed(self) -> int | None:
        """Return the seed the producing run was given."""
        return self._random_seed

    def _fields(
        self,
    ) -> tuple[str | None, str | None, str | None, str | None, int | None]:
        """Return the five fields in declaration order."""
        return (
            self._code_uri,
            self._commit_hash,
            self._command,
            self._environment,
            self._random_seed,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same five details."""
        if type(other) is not Reproducibility:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five details."""
        return hash((Reproducibility, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return f"Reproducibility{self._fields()!r}"


class Provenance:
    """An item's source and licensing metadata.

    Parameters
    ----------
    source_method : str
        How the annotation was arrived at, in the releasing corpus's own
        vocabulary. It is the gold-versus-silver discriminator, and consumers
        compare it for equality against a value they declare, never by substring.
    licensing : Licensing
        The terms the item is released under.
    eprint_refs : tuple[str, ...]
        References to the publications that describe the item's source.
    reproducibility : Reproducibility | None
        How the item was computed, when it was computed rather than annotated.

    Raises
    ------
    ValueError
        If ``source_method`` is empty.
    """

    __slots__ = ("_eprint_refs", "_licensing", "_reproducibility", "_source_method")

    def __init__(
        self,
        source_method: str,
        licensing: Licensing,
        eprint_refs: tuple[str, ...] = (),
        reproducibility: Reproducibility | None = None,
    ) -> None:
        if not source_method:
            msg = "a provenance record has no source method"
            raise ValueError(msg)

        self._source_method = source_method
        self._licensing = licensing
        self._eprint_refs = eprint_refs
        self._reproducibility = reproducibility

    @property
    def source_method(self) -> str:
        """Return how the annotation was arrived at."""
        return self._source_method

    @property
    def licensing(self) -> Licensing:
        """Return the terms the item is released under."""
        return self._licensing

    @property
    def eprint_refs(self) -> tuple[str, ...]:
        """Return references describing the item's source."""
        return self._eprint_refs

    @property
    def reproducibility(self) -> Reproducibility | None:
        """Return how the item was computed, if it was computed."""
        return self._reproducibility

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same origin."""
        if type(other) is not Provenance:
            return NotImplemented

        return (
            self._source_method == other._source_method
            and self._licensing == other._licensing
            and self._eprint_refs == other._eprint_refs
            and self._reproducibility == other._reproducibility
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over method, licensing, references and reproducibility."""
        return hash(
            (
                Provenance,
                self._source_method,
                self._licensing,
                self._eprint_refs,
                self._reproducibility,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return (
            f"Provenance({self._source_method!r}, {self._licensing!r}, "
            f"{self._eprint_refs!r}, {self._reproducibility!r})"
        )


class Item:
    """An identified token sequence with metadata.

    Parameters
    ----------
    item_id : ItemId
        The reader-assigned handle this datum is joined on.
    tokens : tuple[Token, ...]
        The item's tokens, whose positions must run from zero without a gap.
    text : str | None
        The source text the tokens' byte spans are offsets into.
    languages : tuple[str, ...]
        The languages the item is written in, as BCP 47 tags. More than one is
        normal for code-switched data.
    split : str | None
        The portion of its corpus the item belongs to, when the corpus declares
        portions.
    provenance : Provenance | None
        Where the item came from and under what terms.

    Raises
    ------
    ValueError
        If the id is empty, if the token positions are not ``0`` up to one less
        than the token count, or if a token's byte span runs past ``text``.
    """

    __slots__ = (
        "_item_id",
        "_languages",
        "_provenance",
        "_split",
        "_text",
        "_tokens",
    )

    def __init__(
        self,
        item_id: ItemId,
        tokens: tuple[Token, ...],
        *,
        text: str | None = None,
        languages: tuple[str, ...] = (),
        split: str | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if not item_id:
            msg = "an item has no id"
            raise ValueError(msg)

        for position, token in enumerate(tokens):
            if token.index != position:
                msg = (
                    f"item {item_id!r} has a token at position {position} "
                    f"whose index is {token.index}"
                )
                raise ValueError(msg)

        if text is not None:
            self._check_spans(item_id, tokens, text)

        self._item_id = item_id
        self._tokens = tokens
        self._text = text
        self._languages = languages
        self._split = split
        self._provenance = provenance

    @staticmethod
    def _check_spans(item_id: ItemId, tokens: tuple[Token, ...], text: str) -> None:
        """Raise if any token's byte span runs past the end of the text.

        Parameters
        ----------
        item_id : ItemId
            The id to name in the message.
        tokens : tuple[Token, ...]
            The tokens to check.
        text : str
            The text the spans are offsets into.

        Raises
        ------
        ValueError
            If a span ends past the text's last byte.
        """
        limit = len(text.encode())

        for token in tokens:
            span = token.span

            if span is not None and span.end > limit:
                msg = (
                    f"item {item_id!r} has a token at position {token.index} "
                    f"whose byte span ends at {span.end}, past the text's {limit}"
                )
                raise ValueError(msg)

    @property
    def item_id(self) -> ItemId:
        """Return the reader-assigned handle this datum is joined on."""
        return self._item_id

    @property
    def tokens(self) -> tuple[Token, ...]:
        """Return the item's tokens, in position order."""
        return self._tokens

    @property
    def text(self) -> str | None:
        """Return the source text the tokens' byte spans are offsets into."""
        return self._text

    @property
    def languages(self) -> tuple[str, ...]:
        """Return the languages the item is written in."""
        return self._languages

    @property
    def split(self) -> str | None:
        """Return the portion of its corpus the item belongs to."""
        return self._split

    @property
    def provenance(self) -> Provenance | None:
        """Return where the item came from and under what terms."""
        return self._provenance

    def _identity(
        self,
    ) -> tuple[ItemId, tuple[Token, ...], str | None, tuple[str, ...], str | None]:
        """Return the fields an item is equal on, which exclude its provenance."""
        return (self._item_id, self._tokens, self._text, self._languages, self._split)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same datum."""
        if type(other) is not Item:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields the item is equal on."""
        return hash((Item, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this item."""
        return (
            f"Item({self._item_id!r}, {self._tokens!r}, text={self._text!r}, "
            f"languages={self._languages!r}, split={self._split!r}, "
            f"provenance={self._provenance!r})"
        )
