"""PMB CCG derivation encoding and decoding."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, NamedTuple

from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import segmentation as segmentation_records

from lofi.data.registration import registers_as

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.data import Iso

__all__ = [
    "NAME",
    "MalformedPmbCcgError",
    "PmbCcgCodec",
    "PmbCcgIso",
    "PmbDerivation",
    "PmbFile",
    "PmbLeaf",
    "PmbNode",
    "PmbTree",
]

NAME: Final = "pmb-ccg"
"""The codec name."""

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
"""The timestamp used when a PMB term supplies none."""

_SCHEME: Final = "at://local"
"""The local record URI prefix."""

_EXPRESSION_NSID: Final = "pub.layers.expression.expression"
"""The expression-record collection."""

_SEGMENTATION_NSID: Final = "pub.layers.segmentation.segmentation"
"""The segmentation-record collection."""

_ANNOTATION_NSID: Final = "pub.layers.annotation.annotationLayer"
"""The annotation layer collection."""

_ONTOLOGY: Final = f"{_SCHEME}/pub.layers.ontology.typeDef/pmb-ccg"
"""The type definition prefix for PMB categories."""

_CCG_ID_KEY: Final = "ccgId"
"""The expression feature containing the source CCG identifier."""

_ARGUMENT_COUNT_KEY: Final = "argumentCount"
"""The internal-node feature containing the source argument count."""

_ARGUMENT_PREFIX: Final = "argument."
"""The internal-node feature prefix for non-child arguments."""

_FROM_KEY: Final = "from"
"""The PMB leaf feature containing the character start."""

_TO_KEY: Final = "to"
"""The PMB leaf feature containing the character end."""

_LIST_FEATURES: Final = frozenset({"verbnet"})
"""The PMB metadata fields whose values are Prolog lists."""

_HEADER: Final = (
    ":- op(601, xfx, (/)).",
    ":- op(601, xfx, (\\)).",
)
"""The canonical PMB slash-operator declarations."""

_PRECEDENCE: Final = {"/": 10, "\\": 10, ":": 20}
"""Parser binding powers for PMB infix terms."""

_CCG_ARITY: Final = 2
"""The number of arguments in a top-level CCG term."""

_LEAF_ARITY: Final = 3
"""The number of arguments in a lexical CCG term."""


class MalformedPmbCcgError(ValueError):
    """An error for malformed PMB CCG text or records."""


class PmbLeaf(NamedTuple):
    """A lexical PMB derivation node.

    Attributes
    ----------
    category : str
        Canonical category text.
    text : str
        The lexical spelling the derivation writes. A multiword token joins its
        parts with ``~``, and a normalized token drops the separator, so this
        may differ from the text of its span.
    start : int
        Character start of the token's span in the expression.
    end : int
        Character end of the token's span in the expression.
    features : tuple[tuple[str, str], ...]
        Ordered lexical features other than the two offsets.
    """

    category: str
    text: str
    start: int
    end: int
    features: tuple[tuple[str, str], ...]

    @property
    def surface(self) -> str:
        """Return the text this leaf occupies in its character span.

        Returns
        -------
        str
            The lexical spelling, with the characters it does not record
            written as the spaces that fill out the span.

        Raises
        ------
        MalformedPmbCcgError
            If the spelling is wider than the span.
        """
        spread = _spread(self.text, self.end - self.start)

        if spread is None:
            msg = f"lexical text {self.text!r} overruns its character span"
            raise MalformedPmbCcgError(msg)

        return spread


class PmbTree(NamedTuple):
    """A nonterminal PMB derivation node."""

    rule: str
    category: str
    arguments: tuple[str | PmbLeaf | PmbTree, ...]

    @property
    def children(self) -> tuple[PmbLeaf | PmbTree, ...]:
        """Return child derivations in source argument order."""
        return tuple(
            argument for argument in self.arguments if not isinstance(argument, str)
        )


type PmbNode = PmbLeaf | PmbTree
"""A lexical or nonterminal PMB derivation node."""


class PmbDerivation(NamedTuple):
    """A parsed PMB CCG derivation."""

    identifier: str
    tree: PmbNode


type PmbFile = tuple[PmbDerivation, ...]
"""The derivations in one PMB CCG file."""


class _Token(NamedTuple):
    """One lexical token in a Prolog term."""

    kind: str
    value: str
    offset: int


class _Compound(NamedTuple):
    """A Prolog compound term."""

    functor: str
    arguments: tuple[_Value, ...]


class _Infix(NamedTuple):
    """A Prolog infix term."""

    operator: str
    left: _Value
    right: _Value


class _ListValue(NamedTuple):
    """A Prolog list."""

    values: tuple[_Value, ...]


type _Value = int | str | _Compound | _Infix | _ListValue
"""A parsed Prolog value supported by the PMB format."""


class _AnnotationInput(NamedTuple):
    """Inputs for one generated Layers annotation."""

    node: PmbNode
    position: int
    parent: int | None
    children: tuple[int, ...]
    token: int | None


def _source_text(src: str | bytes) -> str:
    """Return source text decoded as UTF-8.

    Parameters
    ----------
    src : str | bytes
        PMB CCG text or bytes.

    Raises
    ------
    MalformedPmbCcgError
        If byte input is not UTF-8.
    """
    if isinstance(src, str):
        return src

    try:
        return src.decode("utf-8")
    except UnicodeDecodeError as error:
        msg = "PMB CCG input is not UTF-8"
        raise MalformedPmbCcgError(msg) from error


def _spread(spelling: str, width: int) -> str | None:
    """Return the span text one lexical spelling occupies.

    A PMB token's offsets address the original surface while its spelling may
    normalize that surface, so the derivation does not record every character
    the span covers. A multiword token joins its parts with ``~``, and a few
    documents in the English gold split instead drop the separator outright,
    writing ``pawnshop`` over the nine characters of ``pawn shop``. The
    characters the spelling does not record are written here as spaces: at the
    first ``~`` when the spelling has one, and after the spelling when it has
    none. This keeps the token one token and its span the one the release
    declares. A spelling wider than its span records characters the span cannot
    hold and has no reading.

    Parameters
    ----------
    spelling : str
        The lexical text as the derivation writes it.
    width : int
        Character width of the token's span.

    Returns
    -------
    str | None
        The text of the span, or ``None`` when the spelling overruns it.
    """
    parts = spelling.split("~")
    separators = [1] * (len(parts) - 1)
    surplus = width - sum(len(part) for part in parts) - len(separators)

    if surplus < 0:
        return None

    if not separators:
        return spelling + " " * surplus

    separators[0] += surplus
    pieces = [parts[0]]

    for separator, part in zip(separators, parts[1:], strict=True):
        pieces.append(" " * separator)
        pieces.append(part)

    return "".join(pieces)


def _body(text: str) -> str:
    """Return Prolog terms after validating the directive header.

    A PMB file opens with the two slash operator declarations the term reader
    honors, and every released file follows them with ``multifile`` and
    ``discontiguous`` declarations for ``ccg/2`` and ``id/2``. Those two
    declare how a Prolog database is assembled and cannot change how a term
    reads, so they are accepted and dropped. Any other directive is refused.

    Parameters
    ----------
    text : str
        Source text containing local directives.

    Raises
    ------
    MalformedPmbCcgError
        If a directive is malformed or either slash declaration is absent.
    """
    declarations: set[str] = set()
    lines: list[str] = []
    operator = re.compile(
        r":-\s*op\s*\(\s*601\s*,\s*xfx\s*,\s*\(?\s*([/\\])\s*\)?\s*\)\s*\."
    )
    database = re.compile(
        r":-\s*(?:multifile|discontiguous)\s+"
        r"[a-z]\w*/\d+(?:\s*,\s*[a-z]\w*/\d+)*\s*\."
    )

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if stripped.startswith(":-"):
            declaration = operator.fullmatch(stripped)

            if declaration is not None:
                declarations.add(declaration.group(1))
            elif database.fullmatch(stripped) is None:
                msg = f"line {number} is not a supported directive"
                raise MalformedPmbCcgError(msg)
        else:
            lines.append(line)

    missing = {"/", "\\"} - declarations

    if missing:
        msg = f"PMB CCG input does not declare xfx 601 operators {sorted(missing)!r}"
        raise MalformedPmbCcgError(msg)

    return "\n".join(lines)


def _quoted_token(text: str, start: int) -> tuple[_Token, int]:
    """Return one quoted atom and the next source offset.

    Parameters
    ----------
    text : str
        Prolog source text.
    start : int
        Offset of the opening quote.

    Raises
    ------
    MalformedPmbCcgError
        If the quoted atom is unfinished.
    """
    quote = text[start]
    cursor = start + 1
    value: list[str] = []

    while cursor < len(text):
        character = text[cursor]

        if character == quote:
            if cursor + 1 < len(text) and text[cursor + 1] == quote:
                value.append(quote)
                cursor += 2
                continue

            return _Token("atom", "".join(value), start), cursor + 1

        if character == "\\" and cursor + 1 < len(text):
            value.append(text[cursor + 1])
            cursor += 2
            continue

        value.append(character)
        cursor += 1

    msg = f"quoted atom at offset {start} is unfinished"
    raise MalformedPmbCcgError(msg)


def _integer_token(text: str, start: int) -> tuple[_Token, int]:
    """Return one integer token and the next source offset.

    Parameters
    ----------
    text : str
        Prolog source text.
    start : int
        Offset of the integer's first character.
    """
    cursor = start + (text[start] == "-")

    while cursor < len(text) and text[cursor].isdigit():
        cursor += 1

    return _Token("integer", text[start:cursor], start), cursor


def _atom_token(text: str, start: int) -> tuple[_Token, int]:
    """Return one bare atom and the next source offset.

    Parameters
    ----------
    text : str
        Prolog source text.
    start : int
        Offset of the atom's first character.
    """
    cursor = start

    while cursor < len(text):
        character = text[cursor]

        if character.isspace() or character in "()[],:./\\%\"'":
            break

        cursor += 1

    return _Token("atom", text[start:cursor], start), cursor


def _tokens(text: str) -> tuple[_Token, ...]:
    """Return lexical tokens for supported Prolog syntax.

    Parameters
    ----------
    text : str
        Prolog term text without operator directives.

    Raises
    ------
    MalformedPmbCcgError
        If a quoted atom is unfinished.
    """
    tokens: list[_Token] = []
    cursor = 0
    punctuation = {
        "(": "open",
        ")": "close",
        "[": "list_open",
        "]": "list_close",
        ",": "comma",
        ".": "dot",
        ":": "operator",
        "/": "operator",
        "\\": "operator",
    }

    while cursor < len(text):
        character = text[cursor]

        if character.isspace():
            cursor += 1
            continue

        if character == "%":
            newline = text.find("\n", cursor)
            cursor = len(text) if newline < 0 else newline + 1
            continue

        kind = punctuation.get(character)

        if kind is not None:
            tokens.append(_Token(kind, character, cursor))
            cursor += 1
            continue

        if character in {"'", '"'}:
            token, cursor = _quoted_token(text, cursor)
            tokens.append(token)
            continue

        if character == "-" and cursor + 1 < len(text) and text[cursor + 1].isdigit():
            token, cursor = _integer_token(text, cursor)
            tokens.append(token)
            continue

        if character.isdigit():
            token, cursor = _integer_token(text, cursor)
            tokens.append(token)
            continue

        token, cursor = _atom_token(text, cursor)
        tokens.append(token)

    tokens.append(_Token("end", "", len(text)))
    return tuple(tokens)


class _Parser:
    """A parser for the Prolog subset used by PMB CCG files.

    Parameters
    ----------
    tokens : Sequence[_Token]
        Lexical tokens to parse.
    operators : frozenset[str]
        Infix operators enabled for this input.
    """

    __slots__ = ("_at", "_operators", "_tokens")

    def __init__(self, tokens: Sequence[_Token], operators: frozenset[str]) -> None:
        self._tokens = tokens
        self._operators = operators
        self._at = 0

    def file(self) -> PmbFile:
        """Return every CCG derivation in the token stream.

        Raises
        ------
        MalformedPmbCcgError
            If the stream is not a nonempty sequence of well-formed CCG terms.
        """
        derivations: list[PmbDerivation] = []

        while self._peek().kind != "end":
            value = self._value()
            self._take("dot")

            if not isinstance(value, _Compound) or value.functor != "ccg":
                msg = f"term at offset {self._peek().offset} is not a ccg term"
                raise MalformedPmbCcgError(msg)

            if len(value.arguments) != _CCG_ARITY:
                msg = "a ccg term does not have an identifier and derivation"
                raise MalformedPmbCcgError(msg)

            derivations.append(
                PmbDerivation(
                    _write_value(value.arguments[0]),
                    _node_of_value(value.arguments[1]),
                )
            )

        if not derivations:
            msg = "PMB CCG input has no derivation"
            raise MalformedPmbCcgError(msg)

        return tuple(derivations)

    def one(self) -> _Value:
        """Return the only value in the token stream.

        Raises
        ------
        MalformedPmbCcgError
            If tokens remain after the value.
        """
        value = self._value()

        if self._peek().kind != "end":
            token = self._peek()
            msg = f"unexpected {token.value!r} at offset {token.offset}"
            raise MalformedPmbCcgError(msg)

        return value

    def _value(self, minimum: int = 0) -> _Value:
        """Return one value at or above a binding power.

        Parameters
        ----------
        minimum : int
            Minimum infix binding power admitted here.

        Raises
        ------
        MalformedPmbCcgError
            If the value is malformed or an xfx chain lacks parentheses.
        """
        left = self._primary()
        used_xfx: set[int] = set()

        while True:
            token = self._peek()

            if token.kind != "operator" or token.value not in self._operators:
                break

            power = _PRECEDENCE[token.value]

            if power <= minimum:
                break

            if token.value in {"/", "\\"} and power in used_xfx:
                msg = (
                    f"xfx operator {token.value!r} at offset {token.offset} "
                    "requires parentheses"
                )
                raise MalformedPmbCcgError(msg)

            self._at += 1
            used_xfx.add(power)
            left = _Infix(token.value, left, self._value(power))

        return left

    def _primary(self) -> _Value:
        """Return one primary value.

        A punctuation character that stands as a whole argument reads as the
        atom it spells, which is how PMB writes a punctuation token's category.

        Raises
        ------
        MalformedPmbCcgError
            If no supported Prolog value begins at the current token.
        """
        token = self._peek()

        if token.kind == "integer":
            self._at += 1
            return int(token.value)

        if token.kind == "atom":
            self._at += 1

            if self._peek().kind == "open":
                return _Compound(token.value, self._sequence("open", "close"))

            return token.value

        if token.kind in {"dot", "comma"} and self._tokens[self._at + 1].kind in {
            "comma",
            "close",
        }:
            self._at += 1
            return token.value

        if token.kind == "open":
            self._at += 1
            value = self._value()
            self._take("close")
            return value

        if token.kind == "list_open":
            return _ListValue(self._sequence("list_open", "list_close"))

        msg = f"expected a Prolog value at offset {token.offset}, found {token.value!r}"
        raise MalformedPmbCcgError(msg)

    def _sequence(self, opening: str, closing: str) -> tuple[_Value, ...]:
        """Return a comma-separated sequence inside delimiters.

        Parameters
        ----------
        opening : str
            Opening token kind.
        closing : str
            Closing token kind.
        """
        self._take(opening)
        values: list[_Value] = []

        if self._peek().kind == closing:
            self._at += 1
            return ()

        while True:
            values.append(self._value())

            if self._peek().kind != "comma":
                break

            self._at += 1

        self._take(closing)
        return tuple(values)

    def _take(self, kind: str) -> _Token:
        """Return and consume a token of one kind.

        Parameters
        ----------
        kind : str
            Required token kind.

        Raises
        ------
        MalformedPmbCcgError
            If the next token has another kind.
        """
        token = self._peek()

        if token.kind != kind:
            msg = f"expected {kind} at offset {token.offset}, found {token.value!r}"
            raise MalformedPmbCcgError(msg)

        self._at += 1
        return token

    def _peek(self) -> _Token:
        """Return the next token."""
        return self._tokens[self._at]


def _parse(src: str | bytes) -> PmbFile:
    """Return parsed PMB CCG derivations.

    Parameters
    ----------
    src : str | bytes
        PMB CCG text or bytes.
    """
    return _Parser(
        _tokens(_body(_source_text(src))),
        frozenset(_PRECEDENCE),
    ).file()


def _stored_value(text: str) -> _Value:
    """Return a value recovered from canonical feature text.

    Parameters
    ----------
    text : str
        Canonical Prolog value text.
    """
    return _Parser(_tokens(text), frozenset(_PRECEDENCE)).one()


def _quote(value: str) -> str:
    """Return one atom in unambiguous Prolog syntax.

    Parameters
    ----------
    value : str
        Atom value.
    """
    if re.fullmatch(r"[a-z][A-Za-z0-9_]*", value):
        return value

    escaped = value.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def _write_value(value: _Value, parent: int = 0) -> str:
    """Return canonical Prolog text for one value.

    Parameters
    ----------
    value : _Value
        Parsed value to write.
    parent : int
        Binding power of the surrounding expression.
    """
    if isinstance(value, int):
        return str(value)

    if isinstance(value, str):
        return _quote(value)

    if isinstance(value, _Compound):
        arguments = ",".join(_write_value(argument) for argument in value.arguments)
        return f"{_quote(value.functor)}({arguments})"

    if isinstance(value, _ListValue):
        return f"[{','.join(_write_value(item) for item in value.values)}]"

    power = _PRECEDENCE[value.operator]
    written = (
        f"{_write_value(value.left, power)}{value.operator}"
        f"{_write_value(value.right, power)}"
    )

    return f"({written})" if power <= parent else written


def _feature_pairs(value: _Value) -> tuple[tuple[str, _Value], ...]:
    """Return the key-value pairs in a PMB feature list.

    Parameters
    ----------
    value : _Value
        Parsed feature-list value.

    Raises
    ------
    MalformedPmbCcgError
        If the value is not a list of uniquely named pairs.
    """
    if not isinstance(value, _ListValue):
        msg = "a lexical CCG node has no feature list"
        raise MalformedPmbCcgError(msg)

    pairs: list[tuple[str, _Value]] = []

    for entry in value.values:
        if (
            not isinstance(entry, _Infix)
            or entry.operator != ":"
            or not isinstance(entry.left, str)
        ):
            msg = f"a lexical feature is not a named pair: {_write_value(entry)!r}"
            raise MalformedPmbCcgError(msg)

        pairs.append((entry.left, entry.right))

    names = tuple(name for name, _ in pairs)

    if len(names) != len(set(names)):
        msg = "a lexical CCG node states one feature more than once"
        raise MalformedPmbCcgError(msg)

    return tuple(pairs)


def _offset(name: str, pairs: Mapping[str, _Value]) -> int:
    """Return one required character offset.

    Parameters
    ----------
    name : str
        Offset feature name.
    pairs : Mapping[str, _Value]
        Parsed lexical features.

    Raises
    ------
    MalformedPmbCcgError
        If the offset is absent or is not a nonnegative integer.
    """
    value = pairs.get(name)

    if not isinstance(value, int) or value < 0:
        msg = f"a lexical CCG node has no nonnegative integer {name!r} offset"
        raise MalformedPmbCcgError(msg)

    return value


def _feature_text(value: _Value) -> str:
    """Return the Layers text for one lexical feature.

    Parameters
    ----------
    value : _Value
        Parsed lexical feature value.
    """
    return value if isinstance(value, str) else _write_value(value)


def _feature_value(name: str, text: str) -> _Value:
    """Return a Prolog value from one Layers lexical feature.

    Parameters
    ----------
    name : str
        Feature name.
    text : str
        Layers feature text.
    """
    return _stored_value(text) if name in _LIST_FEATURES else text


def _node_of_value(value: _Value) -> PmbNode:
    """Return the derivation node represented by one Prolog value.

    Parameters
    ----------
    value : _Value
        Parsed Prolog value.

    Raises
    ------
    MalformedPmbCcgError
        If the value is not a lexical or nonterminal derivation node.
    """
    if not isinstance(value, _Compound) or not value.arguments:
        msg = f"a derivation node is not a compound term: {_write_value(value)!r}"
        raise MalformedPmbCcgError(msg)

    if value.functor != "t":
        return PmbTree(
            value.functor,
            _write_value(value.arguments[0]),
            tuple(
                _node_of_value(argument)
                if isinstance(argument, _Compound)
                else _write_value(argument)
                for argument in value.arguments[1:]
            ),
        )

    if len(value.arguments) != _LEAF_ARITY or not isinstance(value.arguments[1], str):
        msg = "a lexical CCG node does not have category, text, and features"
        raise MalformedPmbCcgError(msg)

    pairs = _feature_pairs(value.arguments[2])
    indexed = dict(pairs)
    start = _offset(_FROM_KEY, indexed)
    end = _offset(_TO_KEY, indexed)

    if end < start:
        msg = f"a lexical CCG node ends before it starts: [{start}, {end})"
        raise MalformedPmbCcgError(msg)

    text = value.arguments[1]
    features = tuple(
        (name, _feature_text(feature))
        for name, feature in pairs
        if name not in {_FROM_KEY, _TO_KEY}
    )

    return PmbLeaf(_write_value(value.arguments[0]), text, start, end, features)


def _value_of_node(node: PmbNode) -> _Compound:
    """Return the Prolog value for one derivation node.

    Parameters
    ----------
    node : PmbNode
        Parsed PMB derivation node.
    """
    if isinstance(node, PmbLeaf):
        pairs = (
            *(
                _Infix(":", name, _feature_value(name, value))
                for name, value in node.features
            ),
            _Infix(":", _FROM_KEY, node.start),
            _Infix(":", _TO_KEY, node.end),
        )
        return _Compound(
            "t",
            (
                _stored_value(node.category),
                node.text,
                _ListValue(pairs),
            ),
        )

    return _Compound(
        node.rule,
        (
            _stored_value(node.category),
            *(
                _stored_value(argument)
                if isinstance(argument, str)
                else _value_of_node(argument)
                for argument in node.arguments
            ),
        ),
    )


def _line(derivation: PmbDerivation) -> str:
    """Return one canonical top-level CCG term.

    Parameters
    ----------
    derivation : PmbDerivation
        Parsed PMB derivation.
    """
    value = _Compound(
        "ccg",
        (_stored_value(derivation.identifier), _value_of_node(derivation.tree)),
    )
    return f"{_write_value(value)}."


def _write(source: PmbFile) -> str:
    """Return canonical Prolog text for parsed derivations.

    Parameters
    ----------
    source : PmbFile
        Parsed PMB derivations.
    """
    lines = list(_HEADER)
    lines.extend(_line(item) for item in source)
    return "\n".join(lines) + "\n"


def _features(entries: Sequence[tuple[str, str]]) -> defs_records.FeatureMap | None:
    """Return a Layers feature map.

    Parameters
    ----------
    entries : Sequence[tuple[str, str]]
        Ordered feature entries.
    """
    if not entries:
        return None

    return defs_records.FeatureMap(
        entries=tuple(
            defs_records.Feature(key=key, value=value) for key, value in entries
        )
    )


def _entries(features: defs_records.FeatureMap | None) -> tuple[tuple[str, str], ...]:
    """Return ordered entries from a Layers feature map.

    Parameters
    ----------
    features : defs_records.FeatureMap | None
        Feature map or absent value.
    """
    if features is None:
        return ()

    return tuple((entry.key, entry.value) for entry in features.entries)


def _feature(features: defs_records.FeatureMap | None, name: str) -> str:
    """Return one required feature value.

    Parameters
    ----------
    features : defs_records.FeatureMap | None
        Feature map to search.
    name : str
        Required feature name.

    Raises
    ------
    MalformedPmbCcgError
        If the feature is absent.
    """
    for key, value in _entries(features):
        if key == name:
            return value

    msg = f"a PMB CCG record has no {name!r} feature"
    raise MalformedPmbCcgError(msg)


def _type_ref(category: str) -> str:
    """Return the ontology reference for one category.

    Parameters
    ----------
    category : str
        Canonical category text.
    """
    return f"{_ONTOLOGY}#{category}"


def _category(reference: str | None) -> str:
    """Return the category named by an ontology reference.

    Parameters
    ----------
    reference : str | None
        Annotation ontology reference.

    Raises
    ------
    MalformedPmbCcgError
        If the reference is not in the PMB CCG ontology.
    """
    prefix, marked, category = (reference or "").partition("#")

    if not marked or prefix != _ONTOLOGY or not category:
        msg = f"annotation type {reference!r} is not a PMB CCG category"
        raise MalformedPmbCcgError(msg)

    return category


def _key(index: int) -> str:
    """Return the record key for one derivation.

    Parameters
    ----------
    index : int
        Derivation position in the fragment.
    """
    return f"derivation-{index}"


def _uri(nsid: str, key: str) -> str:
    """Return the local URI for one record.

    Parameters
    ----------
    nsid : str
        Record collection.
    key : str
        Local record key.
    """
    return f"{_SCHEME}/{nsid}/{key}"


def _node_id(position: int) -> str:
    """Return the identifier for one derivation position.

    Parameters
    ----------
    position : int
        Preorder node position.
    """
    return f"node-{position}"


def _leaves(node: PmbNode) -> tuple[PmbLeaf, ...]:
    """Return lexical nodes in surface order.

    Parameters
    ----------
    node : PmbNode
        Derivation root.
    """
    if isinstance(node, PmbLeaf):
        return (node,)

    return tuple(leaf for child in node.children for leaf in _leaves(child))


def _surface(leaves: Sequence[PmbLeaf]) -> str:
    """Return the expression reconstructed from leaf character spans.

    Characters no leaf spans are written as spaces, as are the characters a
    token's spelling normalizes away.

    Parameters
    ----------
    leaves : Sequence[PmbLeaf]
        Lexical nodes in surface order.

    Raises
    ------
    MalformedPmbCcgError
        If spans overlap, arrive out of order, or are too narrow for their
        lexical text.
    """
    parts: list[str] = []
    cursor = 0

    for leaf in leaves:
        if leaf.start < cursor:
            msg = f"lexical span [{leaf.start}, {leaf.end}) overlaps its predecessor"
            raise MalformedPmbCcgError(msg)

        parts.append(" " * (leaf.start - cursor))
        parts.append(leaf.surface)
        cursor = leaf.end

    return "".join(parts)


def _tokens_of(
    leaves: Sequence[PmbLeaf], expression: str
) -> tuple[segmentation_records.Token, ...]:
    """Return Layers tokens with UTF-8 byte spans.

    A token carries the lexical spelling, so a multiword or normalized token's
    text differs from the expression text its byte span covers.

    Parameters
    ----------
    leaves : Sequence[PmbLeaf]
        Lexical nodes in surface order.
    expression : str
        Reconstructed expression text.
    """
    return tuple(
        segmentation_records.Token(
            tokenIndex=index,
            text=leaf.text,
            textSpan=defs_records.Span(
                byteStart=len(expression[: leaf.start].encode("utf-8")),
                byteEnd=len(expression[: leaf.end].encode("utf-8")),
            ),
        )
        for index, leaf in enumerate(leaves)
    )


def _placed(node: PmbNode) -> tuple[tuple[PmbNode, int | None], ...]:
    """Return nodes in preorder with their parent positions.

    Parameters
    ----------
    node : PmbNode
        Derivation root.
    """
    placed: list[tuple[PmbNode, int | None]] = []
    stack: list[tuple[PmbNode, int | None]] = [(node, None)]

    while stack:
        current, parent = stack.pop()
        position = len(placed)
        placed.append((current, parent))

        if isinstance(current, PmbTree):
            stack.extend((child, position) for child in reversed(current.children))

    return tuple(placed)


def _argument_features(node: PmbTree) -> tuple[tuple[str, str], ...]:
    """Return features that preserve non-child rule arguments.

    Parameters
    ----------
    node : PmbTree
        Nonterminal derivation node.
    """
    return (
        (_ARGUMENT_COUNT_KEY, str(len(node.arguments))),
        *(
            (f"{_ARGUMENT_PREFIX}{index}", argument)
            for index, argument in enumerate(node.arguments)
            if isinstance(argument, str)
        ),
    )


def _annotation(
    placed: _AnnotationInput,
    tokenization: str,
) -> annotation_records.Annotation:
    """Return the Layers annotation for one derivation node.

    Parameters
    ----------
    placed : _AnnotationInput
        Parsed node and its tree and token positions.
    tokenization : str
        Tokenization identifier.
    """
    anchor = (
        None
        if placed.token is None
        else defs_records.Anchor(
            tokenRef=defs_records.TokenRef(
                tokenIndex=placed.token,
                tokenizationId=defs_records.Uuid(value=tokenization),
            )
        )
    )
    return annotation_records.Annotation(
        uuid=defs_records.Uuid(value=_node_id(placed.position)),
        ontologyTypeRef=_type_ref(placed.node.category),
        label=None if isinstance(placed.node, PmbLeaf) else placed.node.rule,
        parentId=(
            None
            if placed.parent is None
            else defs_records.Uuid(value=_node_id(placed.parent))
        ),
        childIds=tuple(
            defs_records.Uuid(value=_node_id(child)) for child in placed.children
        ),
        anchor=anchor,
        features=_features(
            placed.node.features
            if isinstance(placed.node, PmbLeaf)
            else _argument_features(placed.node)
        ),
    )


def _annotations(
    tree: PmbNode,
    tokenization: str,
) -> tuple[annotation_records.Annotation, ...]:
    """Return the annotations for one derivation.

    Parameters
    ----------
    tree : PmbNode
        Derivation root.
    tokenization : str
        Tokenization identifier.
    """
    placed = _placed(tree)
    daughters: dict[int, list[int]] = {}

    for position, (_, parent) in enumerate(placed):
        if parent is not None:
            daughters.setdefault(parent, []).append(position)

    annotations: list[annotation_records.Annotation] = []
    token = 0

    for position, (node, parent) in enumerate(placed):
        token_position = token if isinstance(node, PmbLeaf) else None

        if token_position is not None:
            token += 1

        annotations.append(
            _annotation(
                _AnnotationInput(
                    node,
                    position,
                    parent,
                    tuple(daughters.get(position, ())),
                    token_position,
                ),
                tokenization,
            )
        )

    return tuple(annotations)


def _records_of(derivation: PmbDerivation, index: int) -> tuple[FragmentRecord, ...]:
    """Return the three Layers records for one derivation.

    Parameters
    ----------
    derivation : PmbDerivation
        Parsed PMB derivation.
    index : int
        Derivation position in the fragment.
    """
    key = _key(index)
    leaves = _leaves(derivation.tree)
    text = _surface(leaves)
    expression_uri = _uri(_EXPRESSION_NSID, key)
    tokenization = f"tokenization-{index}"
    records = (
        expression_records.Expression(
            id=key,
            kind="sentence",
            createdAt=_EPOCH,
            text=text,
            features=_features(((_CCG_ID_KEY, derivation.identifier),)),
        ),
        segmentation_records.Segmentation(
            createdAt=_EPOCH,
            expression=expression_uri,
            tokenizations=(
                segmentation_records.Tokenization(
                    uuid=defs_records.Uuid(value=tokenization),
                    kind="pmb",
                    tokens=_tokens_of(leaves, text),
                ),
            ),
        ),
        annotation_records.AnnotationLayer(
            createdAt=_EPOCH,
            expression=expression_uri,
            kind="tree",
            subkind="ccg",
            formalism="pmb",
            tokenizationId=defs_records.Uuid(value=tokenization),
            annotations=_annotations(derivation.tree, tokenization),
        ),
    )
    nsids = (_EXPRESSION_NSID, _SEGMENTATION_NSID, _ANNOTATION_NSID)
    return tuple(
        FragmentRecord(
            local_id=_uri(nsid, key),
            nsid=nsid,
            value_json=record.model_dump_json(),
        )
        for nsid, record in zip(nsids, records, strict=True)
    )


class _Group(NamedTuple):
    """One derivation's records grouped by collection."""

    key: str
    records: Mapping[str, str]


def _groups(records: Iterable[FragmentRecord]) -> tuple[_Group, ...]:
    """Return fragment records grouped by derivation.

    Parameters
    ----------
    records : Iterable[FragmentRecord]
        Layers fragment records.

    Raises
    ------
    MalformedPmbCcgError
        If a record URI disagrees with its collection or repeats one.
    """
    grouped: dict[str, dict[str, str]] = {}

    for record in records:
        collection, marked, key = record.local_id.removeprefix(
            f"{_SCHEME}/"
        ).rpartition("/")

        if not marked or collection != record.nsid:
            msg = f"record URI {record.local_id!r} disagrees with {record.nsid!r}"
            raise MalformedPmbCcgError(msg)

        held = grouped.setdefault(key, {})

        if record.nsid in held:
            msg = f"derivation {key!r} repeats a {record.nsid} record"
            raise MalformedPmbCcgError(msg)

        held[record.nsid] = record.value_json

    return tuple(_Group(key, held) for key, held in grouped.items())


def _record(group: _Group, nsid: str) -> str:
    """Return one required record from a group.

    Parameters
    ----------
    group : _Group
        Grouped derivation records.
    nsid : str
        Required record collection.

    Raises
    ------
    MalformedPmbCcgError
        If the collection is absent.
    """
    value = group.records.get(nsid)

    if value is None:
        msg = f"derivation {group.key!r} has no {nsid} record"
        raise MalformedPmbCcgError(msg)

    return value


def _character_span(text: str, token: segmentation_records.Token) -> tuple[int, int]:
    """Return character offsets from a token's UTF-8 byte span.

    Parameters
    ----------
    text : str
        Expression text.
    token : segmentation_records.Token
        Layers token with a byte span.

    Raises
    ------
    MalformedPmbCcgError
        If the span is absent, out of range, splits UTF-8, or disagrees with the token.
    """
    span = token.textSpan

    if span is None:
        msg = f"token {token.tokenIndex} has no text span"
        raise MalformedPmbCcgError(msg)

    encoded = text.encode("utf-8")

    if span.byteEnd > len(encoded):
        msg = f"token {token.tokenIndex} ends beyond its expression"
        raise MalformedPmbCcgError(msg)

    try:
        prefix = encoded[: span.byteStart].decode("utf-8")
        value = encoded[span.byteStart : span.byteEnd].decode("utf-8")
    except UnicodeDecodeError as error:
        msg = f"token {token.tokenIndex} splits a UTF-8 character"
        raise MalformedPmbCcgError(msg) from error

    if value != _spread(token.text or "", len(value)):
        msg = f"token {token.tokenIndex} text disagrees with its byte span"
        raise MalformedPmbCcgError(msg)

    return len(prefix), len(prefix) + len(value)


def _arguments(
    annotation: annotation_records.Annotation,
    children: Sequence[PmbNode],
) -> tuple[str | PmbNode, ...]:
    """Return source arguments for one nonterminal annotation.

    Parameters
    ----------
    annotation : annotation_records.Annotation
        Nonterminal Layers annotation.
    children : Sequence[PmbNode]
        Recovered child derivations.

    Raises
    ------
    MalformedPmbCcgError
        If argument metadata is absent or disagrees with the recovered children.
    ValueError
        If an indexed argument feature has a noninteger index.
    """
    count_text = _feature(annotation.features, _ARGUMENT_COUNT_KEY)

    try:
        count = int(count_text)
    except ValueError as error:
        msg = f"node {annotation.uuid.value!r} has noninteger argument count"
        raise MalformedPmbCcgError(msg) from error

    extras = {
        int(name.removeprefix(_ARGUMENT_PREFIX)): value
        for name, value in _entries(annotation.features)
        if name.startswith(_ARGUMENT_PREFIX)
    }
    recovered: list[str | PmbNode] = []
    child = iter(children)

    for position in range(count):
        if position in extras:
            recovered.append(extras[position])
        else:
            try:
                recovered.append(next(child))
            except StopIteration as error:
                msg = f"node {annotation.uuid.value!r} has too few child arguments"
                raise MalformedPmbCcgError(msg) from error

    try:
        next(child)
    except StopIteration:
        return tuple(recovered)

    msg = f"node {annotation.uuid.value!r} has too many child arguments"
    raise MalformedPmbCcgError(msg)


def _node_from_annotation(
    annotation: annotation_records.Annotation,
    children: tuple[PmbNode, ...],
    tokens: Sequence[segmentation_records.Token],
    text: str,
) -> PmbNode:
    """Return the PMB node represented by one Layers annotation.

    Parameters
    ----------
    annotation : annotation_records.Annotation
        Layers tree annotation.
    children : tuple[PmbNode, ...]
        Recovered child derivations.
    tokens : Sequence[segmentation_records.Token]
        Expression tokens.
    text : str
        Expression text.

    Raises
    ------
    MalformedPmbCcgError
        If the annotation has invalid category, rule, argument, or token data.
    """
    category = _category(annotation.ontologyTypeRef)

    if children:
        if not annotation.label:
            msg = f"nonterminal {annotation.uuid.value!r} has no rule label"
            raise MalformedPmbCcgError(msg)

        return PmbTree(
            annotation.label,
            category,
            _arguments(annotation, children),
        )

    anchor = annotation.anchor

    if anchor is None or anchor.tokenRef is None:
        msg = f"leaf {annotation.uuid.value!r} has no token anchor"
        raise MalformedPmbCcgError(msg)

    position = anchor.tokenRef.tokenIndex

    if not 0 <= position < len(tokens):
        msg = f"leaf {annotation.uuid.value!r} names absent token {position}"
        raise MalformedPmbCcgError(msg)

    token = tokens[position]
    start, end = _character_span(text, token)
    return PmbLeaf(
        category, token.text or "", start, end, _entries(annotation.features)
    )


def _tree(
    layer: annotation_records.AnnotationLayer,
    tokens: Sequence[segmentation_records.Token],
    text: str,
) -> PmbNode:
    """Return the PMB derivation in one Layers annotation layer.

    Parameters
    ----------
    layer : annotation_records.AnnotationLayer
        Layers tree annotation layer.
    tokens : Sequence[segmentation_records.Token]
        Expression tokens.
    text : str
        Expression text.

    Raises
    ------
    MalformedPmbCcgError
        If the layer does not encode one complete PMB derivation tree.
    """
    annotations = {item.uuid.value: item for item in layer.annotations}
    children = {
        item.uuid.value: tuple(child.value for child in item.childIds or ())
        for item in layer.annotations
    }
    roots = tuple(
        item.uuid.value for item in layer.annotations if item.parentId is None
    )

    if len(roots) != 1:
        msg = f"annotation layer has {len(roots)} roots rather than one"
        raise MalformedPmbCcgError(msg)

    for identifiers in children.values():
        if any(identifier not in annotations for identifier in identifiers):
            msg = "annotation layer names an absent child"
            raise MalformedPmbCcgError(msg)

    built: dict[str, PmbNode] = {}
    stack: list[tuple[str, bool]] = [(roots[0], False)]

    while stack:
        identifier, visited = stack.pop()

        if visited:
            built[identifier] = _node_from_annotation(
                annotations[identifier],
                tuple(built[child] for child in children[identifier]),
                tokens,
                text,
            )
            continue

        stack.append((identifier, True))
        stack.extend((child, False) for child in reversed(children[identifier]))

    return built[roots[0]]


def _derivation(group: _Group) -> PmbDerivation:
    """Return the derivation represented by one record group.

    Parameters
    ----------
    group : _Group
        Grouped Layers records.

    Raises
    ------
    MalformedPmbCcgError
        If the group is incomplete or its records do not encode one derivation.
    """
    expression = expression_records.Expression.model_validate_json(
        _record(group, _EXPRESSION_NSID)
    )
    segmentation = segmentation_records.Segmentation.model_validate_json(
        _record(group, _SEGMENTATION_NSID)
    )
    layer = annotation_records.AnnotationLayer.model_validate_json(
        _record(group, _ANNOTATION_NSID)
    )

    if expression.text is None:
        msg = f"derivation {group.key!r} has no expression text"
        raise MalformedPmbCcgError(msg)

    if len(segmentation.tokenizations) != 1:
        msg = f"derivation {group.key!r} does not have one tokenization"
        raise MalformedPmbCcgError(msg)

    tokens = segmentation.tokenizations[0].tokens
    return PmbDerivation(
        _feature(expression.features, _CCG_ID_KEY),
        _tree(layer, tokens, expression.text),
    )


class PmbCcgIso:
    """A translation from parsed PMB derivations to Layers fragments."""

    __slots__ = ()

    def forward(self, source: PmbFile, /) -> CorpusFragment:
        """Return the Layers fragment for parsed PMB derivations.

        Parameters
        ----------
        source : PmbFile
            Parsed PMB derivations.

        Raises
        ------
        MalformedPmbCcgError
            If a derivation has inconsistent lexical spans.
        """
        return CorpusFragment(
            records=tuple(
                record
                for index, derivation in enumerate(source)
                for record in _records_of(derivation, index)
            ),
            source=NAME,
        )

    def backward(self, target: CorpusFragment, /) -> PmbFile:
        """Return parsed PMB derivations from a Layers fragment.

        Parameters
        ----------
        target : CorpusFragment
            Layers fragment produced by this translation.

        Raises
        ------
        MalformedPmbCcgError
            If the fragment does not contain complete, consistent derivation records.
        """
        return tuple(_derivation(group) for group in _groups(target.records))


@registers_as(NAME)
class PmbCcgCodec:
    """A codec for PMB CCG Prolog terms."""

    __slots__ = ("_iso", "name")

    name: str

    def __init__(self) -> None:
        self.name = NAME
        self._iso = PmbCcgIso()

    @property
    def iso(self) -> Iso[PmbFile, CorpusFragment]:
        """Return the parsed-term-to-fragment translation."""
        return self._iso

    def decode(
        self,
        src: str | bytes,
        *,
        into: CorpusFragment | None = None,
    ) -> CorpusFragment:
        """Return a Layers fragment decoded from PMB CCG text.

        Parameters
        ----------
        src : str | bytes
            PMB CCG text or bytes.
        into : CorpusFragment | None
            Existing fragment whose records remain first.

        Raises
        ------
        MalformedPmbCcgError
            If the input is invalid PMB CCG text or has inconsistent lexical spans.
        """
        parsed = _parse(src)

        if into is None:
            return self._iso.forward(parsed)

        offset = sum(record.nsid == _EXPRESSION_NSID for record in into.records)
        records = tuple(
            record
            for index, derivation in enumerate(parsed, start=offset)
            for record in _records_of(derivation, index)
        )
        return CorpusFragment(records=into.records + records, source=NAME)

    def encode(self, records: Iterable[FragmentRecord]) -> str:
        """Return PMB CCG text for Layers records.

        Parameters
        ----------
        records : Iterable[FragmentRecord]
            Records produced by this codec.

        Raises
        ------
        MalformedPmbCcgError
            If the records do not encode complete, consistent PMB derivations.
        """
        fragment = CorpusFragment(records=tuple(records), source=NAME)
        return _write(self._iso.backward(fragment))
