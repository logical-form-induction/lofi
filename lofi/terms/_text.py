"""Signature-parameterized term text syntaxes and structured term records.

Three syntaxes read and write the terms of a declared signature. The
s-expression syntax spells every node as its quoted operator name, its payload
and its arguments. The constructor syntax writes a category inventory using
its declared prefix, infix, and mixfix forms. The angle bracket syntax writes
one declared binary constructor as an unlabeled
pair, so that under a type signature whose function constructor is that pair
``⟨⟨e,t⟩,t⟩`` is the type of a generalized quantifier; every other node is
written as its name with an optional ``{payload}`` and ``⟨argument, ...⟩``.

A term record is the structure of a term with its constants and free variables
named, independent of any store. It costs one Python object per node and is
meant for publishing a gold analysis or comparing two type systems, never for
an inner loop; a hot path holds interned terms.
"""

from json import JSONDecodeError, JSONDecoder, dumps
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Never,
    Protocol,
    Self,
    final,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi._native import (
    ForeignTermError,
    SortError,
    TermError,
    UnknownOperatorError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi._native import Term, TermStore
    from lofi._types import JsonValue
    from lofi.terms._ports import OpName, Path, Sort
    from lofi.terms._values import Operator, Signature

__all__ = [
    "AngleSyntax",
    "BoundRecord",
    "ConstantRecord",
    "ConstructorNotation",
    "ConstructorSyntax",
    "FreeRecord",
    "NotationOperands",
    "NotationReader",
    "NotationTable",
    "OperatorRecord",
    "SExpSyntax",
    "TermRecord",
    "TextCursor",
    "TextDepthError",
    "TextParseError",
    "from_record",
    "record_differences",
    "record_from_json",
    "record_to_json",
    "to_record",
]

_BINARY_ARITY: Final = 2


@final
class ConstructorNotation(FrozenValue):
    """One constructor's notation after declaration validation.

    The value belongs to the term layer rather than the declaration layer: it
    contains only the operator name and the tokens needed by the parser. A
    prefix constructor is written ``token(argument, ...)``; an infix
    constructor uses its declared precedence and associativity; and a mixfix
    constructor interleaves its arguments with exactly one more literal part.
    """

    _noun = "constructor notation"
    __slots__ = ("_associativity", "_name", "_parts", "_precedence", "_style")

    def __init__(
        self,
        name: OpName,
        style: Literal["prefix", "infix", "mixfix"],
        parts: tuple[str, ...],
        *,
        precedence: int,
        associativity: Literal["left", "right", "none"],
    ) -> None:
        held = ensure_name("constructor", name)
        if style not in {"prefix", "infix", "mixfix"}:
            msg = f"unknown constructor-notation style {style!r}"
            raise ValueError(msg)
        if precedence <= 0:
            msg = "constructor precedence must be positive"
            raise ValueError(msg)
        if associativity not in {"left", "right", "none"}:
            msg = f"unknown constructor associativity {associativity!r}"
            raise ValueError(msg)
        if not parts or not any(parts):
            msg = "constructor notation needs a surface token"
            raise ValueError(msg)
        if style in {"prefix", "infix"} and (len(parts) != 1 or not parts[0]):
            msg = f"{style} notation needs exactly one nonempty token"
            raise ValueError(msg)
        if style == "mixfix" and not parts[0]:
            msg = "mixfix notation needs a nonempty leading part"
            raise ValueError(msg)
        for part in parts:
            if any(character in "()[]," for character in part):
                msg = f"constructor token {part!r} contains reserved syntax"
                raise ValueError(msg)

        object.__setattr__(self, "_name", held)
        object.__setattr__(self, "_style", style)
        object.__setattr__(self, "_parts", parts)
        object.__setattr__(self, "_precedence", precedence)
        object.__setattr__(self, "_associativity", associativity)

    @classmethod
    def prefix(cls, name: OpName, spelling: str, /) -> Self:
        """Return a prefix notation."""
        return cls(name, "prefix", (spelling,), precedence=100, associativity="none")

    @classmethod
    def infix(
        cls,
        name: OpName,
        spelling: str,
        /,
        *,
        precedence: int,
        associativity: Literal["left", "right", "none"],
    ) -> Self:
        """Return an infix notation."""
        return cls(
            name,
            "infix",
            (spelling,),
            precedence=precedence,
            associativity=associativity,
        )

    @classmethod
    def mixfix(
        cls,
        name: OpName,
        parts: tuple[str, ...],
        /,
        *,
        precedence: int,
    ) -> Self:
        """Return a mixfix notation."""
        return cls(
            name,
            "mixfix",
            parts,
            precedence=precedence,
            associativity="none",
        )

    @property
    def name(self) -> OpName:
        """Return the declared operator name."""
        return self._name

    @property
    def style(self) -> Literal["prefix", "infix", "mixfix"]:
        """Return the notation form."""
        return self._style

    @property
    def parts(self) -> tuple[str, ...]:
        """Return the literal surface parts."""
        return self._parts

    @property
    def precedence(self) -> int:
        """Return the binding precedence."""
        return self._precedence

    @property
    def associativity(self) -> Literal["left", "right", "none"]:
        """Return the infix associativity."""
        return self._associativity


class _ParsedNode(NamedTuple):
    """An immutable parsed node awaiting store materialization."""

    op: Operator
    args: tuple[_ParsedNode, ...]
    index: int | None
    name: str | None
    offset: int
    depth: int


@final
class TextParseError(TermError):
    """A term-text error located at a character offset."""

    __slots__ = ("offset", "operator", "reason")

    def __init__(self, *, operator: str, offset: int, reason: str) -> None:
        """Initialize a located parse error."""
        self.operator = operator
        self.offset = ensure_non_negative("offset", offset)
        self.reason = ensure_name("reason", reason)
        super().__init__(
            f"cannot parse operator {operator!r} at character offset {offset}: {reason}"
        )


@final
class TextDepthError(TermError):
    """An error for a term too deep to render safely."""

    __slots__ = ("max_depth", "operator")

    def __init__(self, *, operator: str, max_depth: int) -> None:
        """Initialize a rendering-depth error."""
        self.operator = operator
        self.max_depth = max_depth
        super().__init__(
            f"cannot render operator {operator!r}: nesting exceeds maximum depth "
            f"{max_depth}"
        )


def _depth(value: int | None) -> int | None:
    """Validate an optional syntax nesting bound."""
    return None if value is None else ensure_positive("max_depth", value)


def _parsed(
    op: Operator,
    args: tuple[_ParsedNode, ...] = (),
    *,
    index: int | None = None,
    name: str | None = None,
    offset: int,
) -> _ParsedNode:
    """Build a validated intermediate node."""
    for position, (argument, expected) in enumerate(
        zip(args, op.arg_sorts, strict=True)
    ):
        if argument.op.sort != expected:
            raise SortError(
                operator=op.name,
                position=position,
                expected=expected,
                received=argument.op.sort,
            )

    return _ParsedNode(
        op,
        args,
        index,
        name,
        offset,
        1 + max((argument.depth for argument in args), default=0),
    )


def _materialize(
    store: TermStore,
    root: _ParsedNode,
) -> Term:
    """Intern a fully validated intermediate tree, children before parents."""
    built: list[Term] = []
    pending: list[tuple[_ParsedNode, bool]] = [(root, False)]

    while pending:
        node, expanded = pending.pop()

        if not expanded:
            pending.append((node, True))
            pending.extend((argument, False) for argument in reversed(node.args))
            continue

        arity = len(node.args)
        arguments = tuple(built[len(built) - arity :])
        del built[len(built) - arity :]
        payload = node.index
        op = node.op

        if node.name is not None and op.is_constant:
            payload = store.signature.vocabulary(op.sort).id(node.name)

        elif node.name is not None:
            payload = store.atom(op.sort, node.name)

        built.append(store.op(op, *arguments, payload=payload))

    return built[0]


def _fold[R](
    term: Term,
    write: Callable[[Term, tuple[R, ...]], R],
    max_depth: int | None,
) -> R:
    """Return what a writer builds from a term, children before parents.

    Parameters
    ----------
    term : Term
        The root of the term to write.
    write : Callable[[Term, tuple[R, ...]], R]
        Builds a node's value from the node and its arguments' values.
    max_depth : int | None
        The greatest node depth written; unbounded when None.

    Raises
    ------
    TextDepthError
        If a node, visited root first and left to right, lies deeper than
        the bound.
    """
    written: list[R] = []
    pending: list[tuple[Term, int, bool]] = [(term, 1, False)]

    while pending:
        node, depth, expanded = pending.pop()

        if expanded:
            arity = len(node.args)
            arguments = tuple(written[len(written) - arity :])
            del written[len(written) - arity :]
            written.append(write(node, arguments))
            continue

        if max_depth is not None and depth > max_depth:
            raise TextDepthError(operator=node.op.name, max_depth=max_depth)

        pending.append((node, depth, True))
        pending.extend((child, depth + 1, False) for child in reversed(node.args))

    return written[0]


class _Syntax:
    """Shared immutable state for a term syntax."""

    __slots__ = ("_max_depth", "_signature")

    def __init__(self, signature: Signature, *, max_depth: int | None) -> None:
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(self, "_max_depth", _depth(max_depth))

    @property
    def signature(self) -> Signature:
        """Return the signature read by this syntax."""
        return self._signature

    @property
    def max_depth(self) -> int | None:
        """Return the greatest accepted syntax-tree depth, or None when unbounded."""
        return self._max_depth

    def _require_store(self, store: TermStore) -> None:
        if store.signature is not self._signature:
            msg = (
                f"syntax for signature {self._signature.name!r} cannot use store for "
                f"{store.signature.name!r}"
            )
            raise ValueError(msg)

    def _require_term(
        self,
        store: TermStore,
        term: Term,
    ) -> None:
        self._require_store(store)

        if not store.contains(term):
            raise ForeignTermError(
                operator=term.op.name,
                position=0,
                expected=store.generation,
                received=term.generation,
            )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a term syntax is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a term syntax is immutable"
        raise AttributeError(msg)


class _Open:
    """A node whose head is read and whose arguments are still being read."""

    __slots__ = ("arguments", "depth", "index", "name", "offset", "op")

    def __init__(
        self,
        op: Operator,
        offset: int,
        depth: int,
        *,
        index: int | None = None,
        name: str | None = None,
    ) -> None:
        self.op = op
        self.offset = offset
        self.depth = depth
        self.index = index
        self.name = name
        self.arguments: list[_ParsedNode] = []


class TextCursor:
    """A position in one text with the token readers every syntax shares.

    Every reader of term text, and every reader built on declared constructor
    notation, advances one cursor of this kind, so token boundaries and
    located refusals mean the same thing in each of them.

    Parameters
    ----------
    text : str
        The text to read.
    word_boundaries : bool
        Whether a token ending in an identifier character matches only where an
        identifier ends. Text whose leaves may contain a token's characters,
        such as logical forms, needs the boundary; category text, whose atoms
        never contain a constructor token, lets a token delimit its atoms.
    """

    __slots__ = ("_word_boundaries", "offset", "text")

    def __init__(self, text: str, *, word_boundaries: bool) -> None:
        self.text = text
        self.offset = 0
        self._word_boundaries = word_boundaries

    @staticmethod
    def _word_character(character: str) -> bool:
        """Return whether a character can continue an identifier."""
        return character.isascii() and (character.isalnum() or character == "_")

    def space(self) -> None:
        """Advance past whitespace."""
        while self.offset < len(self.text) and self.text[self.offset].isspace():
            self.offset += 1

    def fail(self, context: str, offset: int, reason: str) -> Never:
        """Raise a located refusal naming the operator or form being read.

        Parameters
        ----------
        context : str
            The operator or form the refusal concerns.
        offset : int
            The character offset of the refusal.
        reason : str
            Why the text is refused.

        Raises
        ------
        TextParseError
            Always.
        """
        raise TextParseError(operator=context, offset=offset, reason=reason)

    def peek(self) -> str:
        """Return the next non-space character, or the empty string at the end."""
        self.space()

        return self.text[self.offset] if self.offset < len(self.text) else ""

    def starts(self, token: str) -> bool:
        """Return whether the next token is ``token``, without consuming it.

        Under word boundaries, a token whose last character can continue an
        identifier matches only where the identifier ends with it. Whitespace
        is skipped first unless the token itself begins with whitespace.
        """
        original = self.offset
        found = self.accept(token) is not None
        self.offset = original

        return found

    def accept(self, token: str) -> int | None:
        """Consume a token and return where it starts, or None when absent."""
        original = self.offset
        if not token or not token[0].isspace():
            self.space()
        start = self.offset

        if not self.text.startswith(token, start):
            self.offset = original
            return None

        end = start + len(token)

        if (
            self._word_boundaries
            and token
            and self._word_character(token[-1])
            and end < len(self.text)
            and self._word_character(self.text[end])
        ):
            self.offset = original
            return None

        self.offset = end
        return start

    def expect(self, token: str, context: str) -> int:
        """Consume a required token and return where it starts.

        Raises
        ------
        TextParseError
            If the next token is not ``token``.
        """
        start = self.accept(token)

        if start is None:
            self.space()
            self.fail(context, self.offset, f"expected {token!r}")

        return start

    def name(self) -> str | None:
        """Consume an ASCII identifier, or return None when none starts here."""
        self.space()
        start = self.offset

        if start == len(self.text):
            return None

        first = self.text[start]

        if not first.isascii() or not first.isalpha():
            return None

        self.offset += 1

        while self.offset < len(self.text) and self._word_character(
            self.text[self.offset]
        ):
            self.offset += 1

        return self.text[start : self.offset]

    def finish(self, context: str) -> None:
        """Refuse any input left after the last read.

        Raises
        ------
        TextParseError
            If non-space input remains.
        """
        self.space()

        if self.offset != len(self.text):
            self.fail(context, self.offset, "found trailing input")


class _Cursor(TextCursor):
    """A cursor with the payload readers the s-expression and angle syntaxes share."""

    __slots__ = ()

    def _quoted(self, operator: str) -> tuple[str, int]:
        self.space()
        start = self.offset

        try:
            value, self.offset = JSONDecoder().raw_decode(self.text, self.offset)
        except JSONDecodeError as error:
            self.fail(operator, error.pos, "expected a quoted name")

        if not isinstance(value, str):
            self.fail(operator, start, "expected a quoted name")

        return value, start

    def _index(self, operator: str) -> int:
        self.space()
        start = self.offset

        while self.offset < len(self.text) and self.text[self.offset] in "0123456789":
            self.offset += 1

        if start == self.offset:
            self.fail(operator, start, "expected a nonnegative integer payload")

        try:
            return int(self.text[start : self.offset])
        except ValueError:
            self.fail(operator, start, "integer payload is too large")

    def _built(
        self,
        op: Operator,
        args: tuple[_ParsedNode, ...],
        offset: int,
        *,
        index: int | None = None,
        name: str | None = None,
    ) -> _ParsedNode:
        try:
            return _parsed(op, args, index=index, name=name, offset=offset)
        except TermError as error:
            self.fail(op.name, offset, str(error))

    def _closed(self, node: _Open) -> _ParsedNode:
        return self._built(
            node.op,
            tuple(node.arguments),
            node.offset,
            index=node.index,
            name=node.name,
        )


class _SExpReader(_Cursor):
    """A cursor over one s-expression."""

    __slots__ = ("_syntax",)

    def __init__(
        self,
        syntax: SExpSyntax,
        text: str,
    ) -> None:
        super().__init__(text, word_boundaries=False)
        self._syntax = syntax

    def _payload(self, op: Operator) -> tuple[int | None, str | None]:
        if op.payload_sort == "none":
            return None, None

        if op.is_bound_variable:
            return self._index(op.name), None

        name, start = self._quoted(op.name)

        try:
            held = ensure_name("payload", name)
        except (TypeError, ValueError) as error:
            self.fail(op.name, start, str(error))

        if op.is_constant and held not in self._syntax.signature.vocabulary(op.sort):
            self.fail(op.name, start, f"{held!r} is not a constant of {op.sort!r}")

        return None, held

    def _open(self, context: str, depth: int) -> _Open:
        """Read a node's opening parenthesis, operator and payload."""
        bound = self._syntax.max_depth

        if bound is not None and depth > bound:
            self.fail(context, self.offset, f"nesting exceeds maximum depth {bound}")

        self.expect("(", context)
        name, head = self._quoted("<operator>")

        try:
            op = self._syntax.signature.by_name(name)
        except UnknownOperatorError:
            self.fail(
                name or "<empty operator>",
                head,
                "operator is not declared by the signature",
            )

        index, named = self._payload(op)

        return _Open(op, head, depth, index=index, name=named)

    def complete(self) -> _ParsedNode:
        """Read one complete s-expression."""
        opened = [self._open("<expression>", 1)]

        while True:
            top = opened[-1]

            if len(top.arguments) < len(top.op.arg_sorts):
                opened.append(self._open(top.op.name, top.depth + 1))
                continue

            self.expect(")", top.op.name)
            opened.pop()
            term = self._closed(top)

            if not opened:
                break

            opened[-1].arguments.append(term)

        self.space()

        if self.offset != len(self.text):
            self.fail(term.op.name, self.offset, "found trailing input")

        return term


@final
class SExpSyntax(_Syntax):
    """Canonical s-expression syntax for one signature."""

    __slots__ = ()

    def __init__(
        self,
        signature: Signature,
        *,
        max_depth: int | None = None,
    ) -> None:
        """Bind the syntax to a signature."""
        super().__init__(signature, max_depth=max_depth)

    def parse(self, store: TermStore, text: str, /) -> Term:
        """Parse a term into a store."""
        self._require_store(store)
        parsed = _SExpReader(self, text).complete()

        return _materialize(store, parsed)

    def render(self, store: TermStore, term: Term, /) -> str:
        """Render a term as a canonical s-expression."""
        self._require_term(store, term)

        def write(node: Term, arguments: tuple[str, ...]) -> str:
            op = self.signature.by_name(node.op.name)
            fields = [dumps(op.name, ensure_ascii=False)]

            if op.is_bound_variable:
                if node.payload is None:
                    msg = f"operator {op.name!r} requires an index payload"
                    raise ValueError(msg)

                fields.append(str(node.payload))
            elif op.is_constant or op.is_free_variable:
                if node.payload is None:
                    msg = f"operator {op.name!r} requires a named payload"
                    raise ValueError(msg)

                name = (
                    store.signature.vocabulary(op.sort).name(node.payload)
                    if op.is_constant
                    else store.atom_name(node.payload)
                )
                fields.append(dumps(name, ensure_ascii=False))
            elif node.payload is not None:
                msg = f"operator {op.name!r} takes no payload"
                raise ValueError(msg)

            fields.extend(arguments)

            return f"({' '.join(fields)})"

        return _fold(term, write, self.max_depth)


def _category_inventory(
    signature: Signature,
    category_sort: Sort,
    core_constructors: frozenset[OpName],
    feature_delimiter: Literal["brackets", "colon"],
) -> tuple[dict[OpName, Operator], dict[tuple[OpName, str], OpName]]:
    atoms: dict[OpName, Operator] = {}
    features: dict[tuple[OpName, str], OpName] = {}
    atom_operator: Operator | None = None

    try:
        vocabulary_names = signature.vocabulary(category_sort).names
    except UnknownOperatorError:
        vocabulary_names = ()
    else:
        atom_operator = signature.constant(category_sort)

    for name in vocabulary_names:
        core = signature.constant_core(category_sort, name)

        if core == name:
            if atom_operator is None:
                msg = f"sort {category_sort!r} has names but no constant operator"
                raise RuntimeError(msg)
            atoms[name] = atom_operator
            continue

        suffix = name.removeprefix(core)

        if not _valid_feature_suffix(suffix, feature_delimiter):
            syntax = (
                "bracketed suffix"
                if feature_delimiter == "brackets"
                else "colon-delimited suffix"
            )
            msg = f"featured constant {name!r} has no {syntax}"
            raise ValueError(msg)

        features[(core, suffix)] = name

    for op in signature.operators:
        if op.sort != category_sort or op.is_constant:
            continue

        core = signature.erase(op)

        if op.payload_sort != "none" or op.is_binder:
            msg = f"operator {op.name!r} is not representable in category syntax"
            raise ValueError(msg)

        if op.arity > 0 and core.name not in core_constructors:
            msg = f"constructor {core.name!r} has no surface token"
            raise ValueError(msg)

        if op.arity == 0 and op.erases_to is None:
            atoms[op.name] = op
            continue

        if op.erases_to is None:
            continue

        suffix = op.name.removeprefix(core.name)

        if not _valid_feature_suffix(suffix, feature_delimiter):
            syntax = (
                "bracketed suffix"
                if feature_delimiter == "brackets"
                else "colon-delimited suffix"
            )
            msg = f"featured operator {op.name!r} has no {syntax}"
            raise ValueError(msg)

        features[(core.name, suffix)] = op.name

    return atoms, features


def _valid_feature_suffix(suffix: str, delimiter: Literal["brackets", "colon"]) -> bool:
    if delimiter == "colon":
        return suffix.startswith(":") and len(suffix) > 1 and ":" not in suffix[1:]

    offset = 0

    while offset < len(suffix):
        if suffix[offset] != "[":
            return False

        close = suffix.find("]", offset + 1)

        if close <= offset + 1 or "[" in suffix[offset + 1 : close]:
            return False

        offset = close + 1

    return bool(suffix)


def _validate_category_atoms(
    atoms: Mapping[OpName, Operator],
    tokens: Mapping[str, OpName],
) -> None:
    reserved = "()[]"
    for atom in atoms:
        if any(character.isspace() or character in reserved for character in atom):
            msg = f"atomic operator {atom!r} contains reserved syntax"
            raise ValueError(msg)

        if any(token in atom for token in tokens):
            msg = f"atomic operator {atom!r} contains a constructor token"
            raise ValueError(msg)


class _ConstructorTable(NamedTuple):
    """Compiled tables for notation-directed category text."""

    atoms: tuple[OpName, ...]
    atom_operators: Mapping[OpName, Operator]
    notations: NotationTable
    by_name: Mapping[OpName, ConstructorNotation]
    features: Mapping[tuple[OpName, str], OpName]
    feature_delimiter: Literal["brackets", "colon"]
    tokens: tuple[str, ...]


def _constructor_table(
    signature: Signature,
    notations: tuple[ConstructorNotation, ...],
    feature_delimiter: Literal["brackets", "colon"],
) -> _ConstructorTable:
    """Compile validated constructor notations for one category sort."""
    if not notations:
        msg = "declared category syntax needs a constructor notation"
        raise ValueError(msg)

    names: set[OpName] = set()
    literal_parts: set[str] = set()
    by_name: dict[OpName, ConstructorNotation] = {}
    category_sort: Sort | None = None

    for notation in notations:
        if notation.name in names:
            msg = f"constructor {notation.name!r} has more than one notation"
            raise ValueError(msg)
        names.add(notation.name)
        op = signature.by_name(notation.name)
        if (
            not op.arg_sorts
            or op.payload_sort != "none"
            or op.is_binder
            or op.erases_to is not None
            or any(argument != op.sort for argument in op.arg_sorts)
        ):
            msg = f"operator {op.name!r} is not a payload-free category endofunctor"
            raise ValueError(msg)
        if category_sort is None:
            category_sort = op.sort
        elif op.sort != category_sort:
            msg = "declared constructors do not share one category sort"
            raise ValueError(msg)

        if notation.style == "infix" and op.arity != 2:
            msg = f"infix constructor {op.name!r} is not binary"
            raise ValueError(msg)
        if notation.style == "mixfix" and len(notation.parts) != op.arity + 1:
            msg = (
                f"mixfix constructor {op.name!r} needs {op.arity + 1} parts, "
                f"not {len(notation.parts)}"
            )
            raise ValueError(msg)

        for part in notation.parts:
            if not part:
                continue
            if part in literal_parts:
                msg = f"constructor token {part!r} is repeated"
                raise ValueError(msg)
            literal_parts.add(part)

        by_name[notation.name] = notation

    if category_sort is None:
        msg = "a nonempty notation inventory has no category sort"
        raise RuntimeError(msg)

    atoms, features = _category_inventory(
        signature, category_sort, frozenset(names), feature_delimiter
    )
    token_names = {
        part: notation.name for notation in notations for part in notation.parts if part
    }
    _validate_category_atoms(atoms, token_names)
    return _ConstructorTable(
        tuple(sorted(atoms, key=len, reverse=True)),
        MappingProxyType(atoms),
        NotationTable(notations),
        MappingProxyType(by_name),
        MappingProxyType(features),
        feature_delimiter,
        tuple(sorted(literal_parts, key=len, reverse=True)),
    )


@runtime_checkable
class NotationOperands[N](Protocol):
    """What a notation reader builds from its leaves and declared notations.

    A notation reader owns precedence, associativity, grouping and the order
    tokens are tried in; its operands own what a leaf is and what a parsed
    constructor becomes.
    """

    def atom(self, cursor: TextCursor, context: str, /) -> N:
        """Read one leaf at the cursor, refusing through the cursor otherwise."""
        ...

    def build(
        self,
        cursor: TextCursor,
        notation: ConstructorNotation,
        arguments: tuple[N, ...],
        start: int,
        /,
    ) -> N:
        """Return the node a notation forms over its parsed arguments."""
        ...

    def decorate(self, cursor: TextCursor, node: N, /) -> N:
        """Read any suffix a completed node carries, such as a feature."""
        ...

    def label(self, node: N, /) -> str:
        """Return the name a refusal about a node reports."""
        ...

    def arity(self, notation: ConstructorNotation, /) -> int:
        """Return how many arguments a prefix notation reads."""
        ...


@final
class NotationTable(FrozenValue):
    """Constructor notations ordered for reading.

    Prefix and mixfix notations lead an operand and infix notations follow one;
    within each group the longest token is tried first, so a token is never
    read as a shorter token it begins with.

    Parameters
    ----------
    notations : Sequence[ConstructorNotation]
        The declared notations.
    """

    _noun = "notation table"
    __slots__ = ("_infix", "_leading")

    def __init__(self, notations: Sequence[ConstructorNotation]) -> None:
        def longest_first(notation: ConstructorNotation) -> tuple[int, str]:
            return (-len(notation.parts[0]), notation.parts[0])

        object.__setattr__(
            self,
            "_leading",
            tuple(
                sorted(
                    (notation for notation in notations if notation.style != "infix"),
                    key=longest_first,
                )
            ),
        )
        object.__setattr__(
            self,
            "_infix",
            tuple(
                sorted(
                    (notation for notation in notations if notation.style == "infix"),
                    key=longest_first,
                )
            ),
        )

    @property
    def leading(self) -> tuple[ConstructorNotation, ...]:
        """Return the prefix and mixfix notations in reading order."""
        return self._leading

    @property
    def infix(self) -> tuple[ConstructorNotation, ...]:
        """Return the infix notations in reading order."""
        return self._infix


@final
class NotationReader[N]:
    """A precedence reader generated from a constructor-notation table.

    Parameters
    ----------
    cursor : TextCursor
        The cursor to read from, which an enclosing reader may share.
    table : NotationTable
        The notations to read.
    operands : NotationOperands[N]
        What leaves are and what notations build.
    max_depth : int | None
        Greatest parenthesis nesting read, or None for no bound.
    """

    __slots__ = ("_cursor", "_max_depth", "_nesting", "_operands", "_table")

    def __init__(
        self,
        cursor: TextCursor,
        table: NotationTable,
        operands: NotationOperands[N],
        *,
        max_depth: int | None = None,
    ) -> None:
        self._cursor = cursor
        self._table = table
        self._operands = operands
        self._max_depth = max_depth
        self._nesting = 0

    def _deeper(self, context: str) -> None:
        if self._max_depth is not None and self._nesting >= self._max_depth:
            self._cursor.fail(
                context,
                self._cursor.offset,
                f"nesting exceeds maximum depth {self._max_depth}",
            )

    def _leading(self, notation: ConstructorNotation, start: int) -> N:
        cursor = self._cursor
        operands = self._operands
        arguments: list[N] = []

        if notation.style == "prefix":
            cursor.expect("(", notation.name)
            arity = operands.arity(notation)
            for position in range(arity):
                arguments.append(self._expression(1))
                if position + 1 < arity:
                    cursor.expect(",", notation.name)
            cursor.expect(")", notation.name)
        else:
            for part in notation.parts[1:]:
                arguments.append(self._expression(notation.precedence))
                if part:
                    cursor.expect(part, notation.name)

        return operands.decorate(
            cursor, operands.build(cursor, notation, tuple(arguments), start)
        )

    def _primary(self, context: str) -> N:
        cursor = self._cursor
        if cursor.peek() == "(":
            self._deeper(context)
            cursor.offset += 1
            self._nesting += 1
            try:
                held = self._expression(1)
                cursor.expect(")", self._operands.label(held))
            finally:
                self._nesting -= 1
            return self._operands.decorate(cursor, held)

        for notation in self._table.leading:
            start = cursor.accept(notation.parts[0])
            if start is not None:
                return self._leading(notation, start)

        return self._operands.atom(cursor, context)

    def _infix(self) -> ConstructorNotation | None:
        for notation in self._table.infix:
            if self._cursor.starts(notation.parts[0]):
                return notation
        return None

    def _expression(self, minimum: int) -> N:
        cursor = self._cursor
        operands = self._operands
        left = self._primary("<expression>")
        last_nonassociative: int | None = None
        while notation := self._infix():
            precedence = notation.precedence
            if precedence < minimum:
                break
            if last_nonassociative == precedence:
                cursor.space()
                cursor.fail(
                    notation.name,
                    cursor.offset,
                    "nonassociative constructor is chained",
                )
            start = cursor.expect(notation.parts[0], notation.name)
            right = self._expression(
                precedence if notation.associativity == "right" else precedence + 1
            )
            left = operands.decorate(
                cursor, operands.build(cursor, notation, (left, right), start)
            )
            last_nonassociative = (
                precedence if notation.associativity == "none" else None
            )
        return left

    def expression(self) -> N:
        """Read one expression and leave any following input unread."""
        return self._expression(1)

    def _complete_infix(self) -> N:
        """Read an infix-only expression with explicit stacks."""
        cursor = self._cursor
        operands = self._operands
        values: list[N] = []
        operators: list[tuple[ConstructorNotation | None, int, int]] = []
        expecting_operand = True
        context = "<expression>"
        nesting = 0

        def reduce_one() -> None:
            notation, start, _ = operators.pop()
            if notation is None:
                msg = "cannot reduce an opening parenthesis"
                raise RuntimeError(msg)
            if len(values) < 2:
                cursor.fail(notation.name, cursor.offset, "expected a right operand")
            right = values.pop()
            left = values.pop()
            values.append(
                operands.decorate(
                    cursor, operands.build(cursor, notation, (left, right), start)
                )
            )

        while True:
            if expecting_operand:
                if cursor.peek() == "(":
                    if self._max_depth is not None and nesting >= self._max_depth:
                        cursor.fail(
                            context,
                            cursor.offset,
                            f"nesting exceeds maximum depth {self._max_depth}",
                        )
                    operators.append((None, cursor.offset, len(values)))
                    cursor.offset += 1
                    nesting += 1
                    continue
                values.append(operands.atom(cursor, context))
                expecting_operand = False
                continue

            if cursor.peek() == ")":
                while operators and operators[-1][0] is not None:
                    reduce_one()
                if not operators:
                    cursor.fail(
                        operands.label(values[-1]),
                        cursor.offset,
                        "found trailing input",
                    )
                _, _, baseline = operators.pop()
                if len(values) != baseline + 1:
                    cursor.fail(
                        operands.label(values[-1]), cursor.offset, "malformed group"
                    )
                cursor.offset += 1
                nesting -= 1
                values[-1] = operands.decorate(cursor, values[-1])
                continue

            notation = self._infix()
            if notation is not None:
                while operators and (top := operators[-1][0]) is not None:
                    if top.precedence < notation.precedence:
                        break
                    if (
                        top.precedence == notation.precedence
                        and notation.associativity == "right"
                    ):
                        break
                    if top.precedence == notation.precedence and (
                        top.associativity == "none" or notation.associativity == "none"
                    ):
                        cursor.space()
                        cursor.fail(
                            notation.name,
                            cursor.offset,
                            "nonassociative constructor is chained",
                        )
                    reduce_one()
                start = cursor.expect(notation.parts[0], notation.name)
                operators.append((notation, start, len(values)))
                expecting_operand = True
                context = notation.name
                continue

            if cursor.peek():
                cursor.fail(
                    operands.label(values[-1]), cursor.offset, "found trailing input"
                )
            if nesting:
                cursor.fail(operands.label(values[-1]), cursor.offset, "expected ')'")
            while operators:
                reduce_one()
            if len(values) != 1:
                msg = "infix reader did not produce one expression"
                raise RuntimeError(msg)
            return values[0]

    def complete(self) -> N:
        """Read one expression that runs to the end of the text.

        An infix-only table is read with explicit stacks, so nesting depth is
        bounded by ``max_depth`` rather than by the interpreter's recursion
        limit.

        Raises
        ------
        TextParseError
            If the text is not one expression of the table's notations.
        """
        if not self._table.leading:
            return self._complete_infix()
        held = self._expression(1)
        self._cursor.finish(self._operands.label(held))
        return held


@final
class _CategoryOperands:
    """Operands that read declared category atoms, features and constructors."""

    __slots__ = ("_max_depth", "_signature", "_table")

    def __init__(
        self,
        signature: Signature,
        table: _ConstructorTable,
        max_depth: int | None,
    ) -> None:
        self._signature = signature
        self._table = table
        self._max_depth = max_depth

    def _bounded(self, cursor: TextCursor, node: _ParsedNode) -> _ParsedNode:
        if self._max_depth is not None and node.depth > self._max_depth:
            cursor.fail(
                node.op.name,
                node.offset,
                f"nesting exceeds maximum depth {self._max_depth}",
            )
        return node

    def build(
        self,
        cursor: TextCursor,
        notation: ConstructorNotation,
        arguments: tuple[_ParsedNode, ...],
        start: int,
        /,
    ) -> _ParsedNode:
        """Build a declared constructor over parsed categories."""
        op = self._signature.by_name(notation.name)
        try:
            node = _parsed(op, arguments, offset=start)
        except TermError as error:
            cursor.fail(op.name, start, str(error))
        return self._bounded(cursor, node)

    def label(self, node: _ParsedNode, /) -> str:
        """Return a parsed category's operator name."""
        return node.op.name

    def arity(self, notation: ConstructorNotation, /) -> int:
        """Return the declared constructor's arity."""
        return self._signature.by_name(notation.name).arity

    def _marker(self) -> str:
        return "[" if self._table.feature_delimiter == "brackets" else ":"

    def decorate(self, cursor: TextCursor, node: _ParsedNode, /) -> _ParsedNode:
        """Read the declared features a category carries."""
        text = cursor.text
        original = cursor.offset
        cursor.space()
        marker = self._marker()
        if cursor.offset >= len(text) or text[cursor.offset] != marker:
            cursor.offset = original
            return node
        term = node
        while cursor.offset < len(text) and text[cursor.offset] == marker:
            start = cursor.offset
            core_name = term.name if term.op.is_constant else term.op.name
            candidates = sorted(
                (
                    (suffix, name)
                    for (core, suffix), name in self._table.features.items()
                    if core == core_name and text.startswith(suffix, start)
                ),
                key=lambda pair: len(pair[0]),
                reverse=True,
            )
            if not candidates:
                if self._table.feature_delimiter == "brackets":
                    close = text.find("]", start + 1)
                    if close < 0:
                        cursor.fail(
                            term.op.name, start, "feature has no closing bracket"
                        )
                    end = close + 1
                else:
                    end = start + 1
                    while end < len(text) and not self._boundary(text, end):
                        end += 1
                cursor.fail(
                    f"{term.op.name}{text[start:end]}",
                    start,
                    "featured operator is not declared by the signature",
                )
            suffix, name = candidates[0]
            cursor.offset += len(suffix)
            op = term.op if term.op.is_constant else self._signature.by_name(name)
            try:
                featured = _parsed(
                    op,
                    term.args,
                    index=term.index,
                    name=name if op.is_constant else term.name,
                    offset=term.offset,
                )
            except TermError as error:
                cursor.fail(op.name, start, str(error))
            term = self._bounded(cursor, featured)
        return term

    def _boundary(self, text: str, offset: int) -> bool:
        if offset == len(text):
            return True
        character = text[offset]
        return (
            character.isspace()
            or character in f"()[],{self._marker()}"
            or any(text.startswith(token, offset) for token in self._table.tokens)
        )

    def atom(self, cursor: TextCursor, context: str, /) -> _ParsedNode:
        """Read one declared category atom."""
        text = cursor.text
        cursor.space()
        start = cursor.offset
        for name in self._table.atoms:
            if text.startswith(name, start) and self._boundary(text, start + len(name)):
                cursor.offset += len(name)
                atom = self._table.atom_operators[name]
                return self.decorate(
                    cursor,
                    _parsed(
                        atom,
                        (),
                        name=name if atom.is_constant else None,
                        offset=start,
                    ),
                )
        end = start
        marker = self._marker()
        while end < len(text):
            if text[end].isspace() or text[end] in f"()[],{marker}":
                break
            if any(text.startswith(token, end) for token in self._table.tokens):
                break
            end += 1
        return cursor.fail(
            text[start:end] or context,
            start,
            "operator is not declared by the signature",
        )


@final
class ConstructorSyntax(_Syntax):
    """Category syntax generated from declared prefix, infix and mixfix forms."""

    __slots__ = ("_table",)

    def __init__(
        self,
        signature: Signature,
        notations: tuple[ConstructorNotation, ...],
        *,
        feature_delimiter: Literal["brackets", "colon"] = "brackets",
        max_depth: int | None = None,
    ) -> None:
        super().__init__(signature, max_depth=max_depth)
        object.__setattr__(
            self,
            "_table",
            _constructor_table(signature, notations, feature_delimiter),
        )

    @property
    def notations(self) -> Mapping[OpName, ConstructorNotation]:
        """Return notation keyed by constructor name."""
        return self._table.by_name

    def parse(self, store: TermStore, text: str, /) -> Term:
        """Parse one category into a store."""
        self._require_store(store)
        return _materialize(
            store,
            NotationReader(
                TextCursor(text, word_boundaries=False),
                self._table.notations,
                _CategoryOperands(self.signature, self._table, self.max_depth),
                max_depth=self.max_depth,
            ).complete(),
        )

    def render(self, store: TermStore, term: Term, /) -> str:
        """Render one category in the notation that generated this syntax."""
        self._require_term(store, term)
        table = self._table

        def embedded(written: tuple[str, int], minimum: int) -> str:
            text, precedence = written
            return f"({text})" if precedence < minimum else text

        def write(
            node: Term, arguments: tuple[tuple[str, int], ...]
        ) -> tuple[str, int]:
            declared = self.signature.by_name(node.op.name)
            if declared.is_constant:
                name = store.payload_name(node)
                core = self.signature.constant_core(declared.sort, name)
                return f"{core}{name.removeprefix(core)}", 10**9

            core = self.signature.erase(declared)
            suffix = declared.name.removeprefix(core.name)
            if not node.args:
                return f"{core.name}{suffix}", 10**9
            notation = table.by_name.get(core.name)
            if notation is None:
                msg = f"constructor {core.name!r} has no declared notation"
                raise ValueError(msg)
            if notation.style == "prefix":
                rendered = ", ".join(text for text, _ in arguments)
                text = f"{notation.parts[0]}({rendered})"
            elif notation.style == "mixfix":
                pieces = [notation.parts[0]]
                for argument, part in zip(arguments, notation.parts[1:], strict=True):
                    pieces.extend((embedded(argument, notation.precedence), part))
                text = "".join(pieces)
            else:
                left_minimum = notation.precedence + (
                    0 if notation.associativity == "left" else 1
                )
                right_minimum = notation.precedence + (
                    0 if notation.associativity == "right" else 1
                )
                left = embedded(arguments[0], left_minimum)
                right = embedded(arguments[1], right_minimum)
                text = f"{left}{notation.parts[0]}{right}"
            if suffix:
                text = f"({text}){suffix}"
                return text, 10**9
            return text, notation.precedence

        return _fold(term, write, self.max_depth)[0]


_OPEN: Final = "⟨"
_CLOSE: Final = "⟩"
_ANGLE_RESERVED: Final = frozenset(f'{_OPEN}{_CLOSE},{{}}"')


def _is_bare(name: str) -> bool:
    """Return whether a name can be written unquoted in angle bracket text."""
    return not any(
        character.isspace() or character in _ANGLE_RESERVED for character in name
    )


def _spelled(name: str) -> str:
    """Return a name as angle bracket text, quoted when it holds reserved syntax."""
    return name if _is_bare(name) else dumps(name, ensure_ascii=False)


class _AngleTable(NamedTuple):
    """Compiled lookup tables for one angle bracket syntax."""

    pair: Operator
    sort: Sort
    leaves: Mapping[tuple[Sort, OpName], Operator]


class _AngleReader(_Cursor):
    """A sort-directed cursor over one angle bracket expression."""

    __slots__ = ("_max_depth", "_signature", "_table")

    def __init__(
        self,
        signature: Signature,
        text: str,
        table: _AngleTable,
        max_depth: int | None,
    ) -> None:
        super().__init__(text, word_boundaries=False)
        self._signature = signature
        self._table = table
        self._max_depth = max_depth

    def _name(self, context: str) -> tuple[str, int]:
        if self.peek() == '"':
            name, start = self._quoted(context)

            if not name:
                self.fail(context, start, "expected a quoted name")

            return name, start

        start = self.offset
        end = start

        while end < len(self.text) and _is_bare(self.text[end]):
            end += 1

        if end == start:
            self.fail(context, start, "expected a name")

        self.offset = end

        return self.text[start:end], start

    def _require_sort(self, op: Operator, sort: Sort, offset: int) -> None:
        if op.sort != sort:
            self.fail(op.name, offset, f"has sort {op.sort!r} where {sort!r} is due")

    def _payload(self, op: Operator) -> tuple[int | None, str | None]:
        if op.payload_sort == "none":
            if self.peek() == "{":
                self.fail(op.name, self.offset, "takes no payload")

            return None, None

        self.expect("{", op.name)

        if op.is_bound_variable:
            index = self._index(op.name)
            self.expect("}", op.name)

            return index, None

        self.space()
        name, start = self._name(op.name)

        if op.is_constant and name not in self._signature.vocabulary(op.sort):
            self.fail(op.name, start, f"{name!r} is not a constant of {op.sort!r}")

        self.expect("}", op.name)

        return None, name

    def _leaf(self, sort: Sort, name: str, start: int) -> _ParsedNode:
        op = self._table.leaves.get((sort, name))

        if op is not None:
            return self._built(op, (), start)

        vocabulary = self._signature.vocabularies.get(sort)

        if vocabulary is not None and name in vocabulary:
            return self._built(self._signature.constant(sort), (), start, name=name)

        return self.fail(
            name, start, f"is neither a nullary operator nor a constant of {sort!r}"
        )

    def _open(self, sort: Sort, context: str, depth: int) -> _Open | _ParsedNode:
        """Read a node of a sort, up to its first argument if it has any."""
        if self._max_depth is not None and depth > self._max_depth:
            self.fail(
                context,
                self.offset,
                f"nesting exceeds maximum depth {self._max_depth}",
            )

        if self.peek() == _OPEN:
            pair = self._table.pair
            start = self.offset
            self._require_sort(pair, sort, start)
            self.offset += len(_OPEN)

            return _Open(pair, start, depth)

        name, start = self._name(context)

        if self.peek() not in {"{", _OPEN}:
            return self._leaf(sort, name, start)

        try:
            op = self._signature.by_name(name)
        except UnknownOperatorError:
            self.fail(name, start, "operator is not declared by the signature")

        self._require_sort(op, sort, start)
        index, named = self._payload(op)

        if not op.arg_sorts:
            if self.peek() == _OPEN:
                self.fail(op.name, self.offset, "takes no arguments")

            return self._built(op, (), start, index=index, name=named)

        self.expect(_OPEN, op.name)

        return _Open(op, start, depth, index=index, name=named)

    def complete(self) -> _ParsedNode:
        """Read one complete expression of the syntax's sort."""
        opened: list[_Open] = []
        read = self._open(self._table.sort, "<expression>", 1)

        while True:
            if isinstance(read, _Open):
                opened.append(read)
            elif opened:
                opened[-1].arguments.append(read)
            else:
                break

            top = opened[-1]
            position = len(top.arguments)

            if position < len(top.op.arg_sorts):
                if position:
                    self.expect(",", top.op.name)

                read = self._open(
                    top.op.arg_sorts[position], top.op.name, top.depth + 1
                )
                continue

            self.expect(_CLOSE, top.op.name)
            opened.pop()
            read = self._closed(top)

        if self.peek():
            self.fail(read.op.name, self.offset, "found trailing input")

        return read


def _angle_table(signature: Signature, pair: OpName, sort: Sort | None) -> _AngleTable:
    """Return the lookup tables of an angle bracket syntax.

    Raises
    ------
    UnknownOperatorError
        If the signature does not declare the pair constructor.
    ValueError
        If the pair constructor is not a payload-free, non-binding binary
        operator, or the root sort is undeclared.
    """
    op = signature.by_name(pair)

    if op.arity != _BINARY_ARITY or op.payload_sort != "none" or op.is_binder:
        msg = f"operator {op.name!r} is not a payload-free non-binding pair"
        raise ValueError(msg)

    root = op.sort if sort is None else sort

    if root not in signature.sorts:
        msg = f"sort {root!r} is not declared by signature {signature.name!r}"
        raise ValueError(msg)

    leaves = {
        (leaf.sort, leaf.name): leaf
        for leaf in signature.operators
        if leaf.arity == 0 and leaf.payload_sort == "none"
    }

    return _AngleTable(op, root, MappingProxyType(leaves))


@final
class AngleSyntax(_Syntax):
    """Angle bracket syntax over one declared pair constructor.

    The pair constructor applied to two arguments is written ``⟨a,b⟩``. A
    payload-free nullary operator, and a constant of the sort a position
    expects, is written as its name; a constant that shares its name with
    such an operator of its sort is written explicitly. Any other node is
    written ``name{payload}⟨argument,...⟩``, the payload present exactly when
    the operator carries one and the arguments exactly when it has any. A name
    holding whitespace, a comma, an angle bracket, a brace or a double quote is
    written as a JSON string. Reading is directed by sort: the text is read
    as a term of the root sort, and each argument as a term of the sort its
    position declares.

    Parameters
    ----------
    signature : Signature
        The signature whose terms the syntax reads and writes.
    pair : OpName
        The binary constructor written as an unlabeled pair.
    sort : Sort | None
        The sort a whole text denotes; the pair constructor's sort when None.
    max_depth : int | None
        The greatest accepted syntax-tree depth; unbounded when None.
    """

    __slots__ = ("_table",)

    def __init__(
        self,
        signature: Signature,
        pair: OpName,
        *,
        sort: Sort | None = None,
        max_depth: int | None = None,
    ) -> None:
        """Bind the pair notation to a constructor of a signature."""
        super().__init__(signature, max_depth=max_depth)
        object.__setattr__(self, "_table", _angle_table(signature, pair, sort))

    @property
    def pair(self) -> OpName:
        """Return the name of the constructor written as a pair."""
        return self._table.pair.name

    @property
    def sort(self) -> Sort:
        """Return the sort a whole text denotes."""
        return self._table.sort

    def parse(self, store: TermStore, text: str, /) -> Term:
        """Parse a term of the root sort into a store."""
        self._require_store(store)
        parsed = _AngleReader(
            self.signature, text, self._table, self.max_depth
        ).complete()

        return _materialize(store, parsed)

    def render(self, store: TermStore, term: Term, /) -> str:
        """Render a term of the root sort in canonical angle bracket syntax.

        Raises
        ------
        SortError
            If the term does not have the root sort.
        TextDepthError
            If the term nests deeper than the syntax's maximum depth.
        """
        self._require_term(store, term)
        table = self._table

        if term.op.sort != table.sort:
            raise SortError(
                operator=term.op.name,
                position=0,
                expected=table.sort,
                received=term.op.sort,
            )

        def write(node: Term, arguments: tuple[str, ...]) -> str:
            op = node.op

            if op.name == table.pair.name:
                left, right = arguments

                return f"{_OPEN}{left},{right}{_CLOSE}"

            head = _spelled(op.name)

            if op.is_constant:
                name = store.payload_name(node)

                if _is_bare(name) and (op.sort, name) not in table.leaves:
                    return name

                head = f"{head}{{{_spelled(name)}}}"
            elif op.is_free_variable:
                head = f"{head}{{{_spelled(store.payload_name(node))}}}"
            elif op.is_bound_variable:
                head = f"{head}{{{node.payload}}}"

            if not arguments:
                return head

            return f"{head}{_OPEN}{','.join(arguments)}{_CLOSE}"

        return _fold(term, write, self.max_depth)


class _Record(FrozenValue):
    """The hash slot every term record fills once at construction."""

    __slots__ = ("_hash",)

    _noun = "term record"


@final
class ConstantRecord(_Record):
    """A constant, named in its sort's vocabulary.

    Parameters
    ----------
    sort : Sort
        The constant's sort.
    name : str
        The constant's name.
    """

    __slots__ = ("_name", "_sort")

    def __init__(self, sort: Sort, name: str) -> None:
        object.__setattr__(self, "_sort", ensure_name("sort", sort))
        object.__setattr__(self, "_name", ensure_name("constant", name))
        object.__setattr__(self, "_hash", hash(("constant", sort, name)))

    @property
    def sort(self) -> Sort:
        """Return the constant's sort."""
        return self._sort

    @property
    def name(self) -> str:
        """Return the constant's name."""
        return self._name

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record is the same constant."""
        return (
            isinstance(other, ConstantRecord)
            and self._sort == other.sort
            and self._name == other.name
        )

    @override
    def __hash__(self) -> int:
        """Return the hash fixed at construction."""
        return self._hash

    @override
    def __repr__(self) -> str:
        """Return a constructor-shaped rendering."""
        return f"ConstantRecord({self._sort!r}, {self._name!r})"


@final
class FreeRecord(_Record):
    """A free variable, named.

    Parameters
    ----------
    sort : Sort
        The variable's sort.
    name : str
        The variable's name.
    """

    __slots__ = ("_name", "_sort")

    def __init__(self, sort: Sort, name: str) -> None:
        object.__setattr__(self, "_sort", ensure_name("sort", sort))
        object.__setattr__(self, "_name", ensure_name("variable", name))
        object.__setattr__(self, "_hash", hash(("free", sort, name)))

    @property
    def sort(self) -> Sort:
        """Return the variable's sort."""
        return self._sort

    @property
    def name(self) -> str:
        """Return the variable's name."""
        return self._name

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record is the same free variable."""
        return (
            isinstance(other, FreeRecord)
            and self._sort == other.sort
            and self._name == other.name
        )

    @override
    def __hash__(self) -> int:
        """Return the hash fixed at construction."""
        return self._hash

    @override
    def __repr__(self) -> str:
        """Return a constructor-shaped rendering."""
        return f"FreeRecord({self._sort!r}, {self._name!r})"


@final
class BoundRecord(_Record):
    """A bound variable at a de Bruijn index counted over its sort's sites.

    Parameters
    ----------
    sort : Sort
        The variable's sort.
    index : int
        The variable's index.
    """

    __slots__ = ("_index", "_sort")

    def __init__(self, sort: Sort, index: int) -> None:
        object.__setattr__(self, "_sort", ensure_name("sort", sort))
        object.__setattr__(self, "_index", ensure_non_negative("index", index))
        object.__setattr__(self, "_hash", hash(("bound", sort, index)))

    @property
    def sort(self) -> Sort:
        """Return the variable's sort."""
        return self._sort

    @property
    def index(self) -> int:
        """Return the variable's index."""
        return self._index

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record is the same bound variable."""
        return (
            isinstance(other, BoundRecord)
            and self._sort == other.sort
            and self._index == other.index
        )

    @override
    def __hash__(self) -> int:
        """Return the hash fixed at construction."""
        return self._hash

    @override
    def __repr__(self) -> str:
        """Return a constructor-shaped rendering."""
        return f"BoundRecord({self._sort!r}, {self._index!r})"


@final
class OperatorRecord(_Record):
    """A payload-free operator applied to argument records.

    Parameters
    ----------
    operator : OpName
        The operator's name.
    arguments : tuple[TermRecord, ...]
        The argument records in declaration order.
    """

    __slots__ = ("_arguments", "_operator")

    def __init__(
        self, operator: OpName, arguments: tuple[TermRecord, ...] = ()
    ) -> None:
        object.__setattr__(self, "_operator", ensure_name("operator", operator))
        object.__setattr__(self, "_arguments", tuple(arguments))
        object.__setattr__(
            self,
            "_hash",
            hash(("operator", operator, tuple(hash(arg) for arg in arguments))),
        )

    @property
    def operator(self) -> OpName:
        """Return the operator's name."""
        return self._operator

    @property
    def arguments(self) -> tuple[TermRecord, ...]:
        """Return the argument records."""
        return self._arguments

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record has the same structure."""
        if not isinstance(other, OperatorRecord):
            return False

        pending: list[tuple[TermRecord, TermRecord]] = [(self, other)]

        while pending:
            left, right = pending.pop()

            if left is right:
                continue

            if hash(left) != hash(right):
                return False

            if not isinstance(left, OperatorRecord):
                if left != right:
                    return False

                continue

            if (
                not isinstance(right, OperatorRecord)
                or left.operator != right.operator
                or len(left.arguments) != len(right.arguments)
            ):
                return False

            pending.extend(zip(left.arguments, right.arguments, strict=True))

        return True

    @override
    def __hash__(self) -> int:
        """Return the hash fixed at construction."""
        return self._hash

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the operator and the arity."""
        return f"OperatorRecord({self._operator!r}, arity={len(self._arguments)})"


type TermRecord = OperatorRecord | ConstantRecord | FreeRecord | BoundRecord
"""The store-independent structure of a term."""


def to_record(store: TermStore, term: Term, /) -> TermRecord:
    """Return the record of a term.

    Parameters
    ----------
    store : TermStore
        The store holding the term, which names its constants and free
        variables.
    term : Term
        The term.

    Returns
    -------
    TermRecord
        The term's structure with every constant and free variable named.
    """
    built: dict[int, TermRecord] = {}
    pending: list[tuple[Term, bool]] = [(term, False)]

    while pending:
        node, expanded = pending.pop()

        if id(node) in built:
            continue

        op = node.op
        payload = node.payload

        if op.is_constant:
            built[id(node)] = ConstantRecord(op.sort, store.payload_name(node))
        elif op.is_free_variable:
            built[id(node)] = FreeRecord(op.sort, store.payload_name(node))
        elif op.is_bound_variable and payload is not None:
            built[id(node)] = BoundRecord(op.sort, payload)
        elif expanded:
            built[id(node)] = OperatorRecord(
                op.name, tuple(built[id(arg)] for arg in node.args)
            )
        else:
            pending.append((node, True))
            pending.extend((arg, False) for arg in reversed(node.args))

    return built[id(term)]


def from_record(store: TermStore, record: TermRecord, /) -> Term:
    """Intern the term a record describes.

    Parameters
    ----------
    store : TermStore
        The store to intern into.
    record : TermRecord
        The record.

    Returns
    -------
    Term
        The interned term.

    Raises
    ------
    UnknownOperatorError
        If the store's signature declares no such operator, or no constant,
        bound-variable or free variable operator for a record's sort.
    UnknownConstantError
        If a constant is absent from its sort's vocabulary.
    TermError
        If an argument has the wrong sort or count, or a payload is refused.
    """
    signature = store.signature
    built: dict[int, Term] = {}
    pending: list[tuple[TermRecord, bool]] = [(record, False)]

    while pending:
        node, expanded = pending.pop()

        if id(node) in built:
            continue

        match node:
            case ConstantRecord():
                built[id(node)] = store.const(node.sort, node.name)
            case FreeRecord():
                built[id(node)] = store.fvar(node.sort, node.name)
            case BoundRecord():
                built[id(node)] = store.bvar(node.sort, node.index)
            case OperatorRecord() if expanded:
                op = signature.by_name(node.operator)
                built[id(node)] = store.op(
                    op, *(built[id(arg)] for arg in node.arguments)
                )
            case OperatorRecord():
                pending.append((node, True))
                pending.extend((arg, False) for arg in reversed(node.arguments))

    return built[id(record)]


def record_to_json(record: TermRecord, /) -> JsonValue:
    """Return a record's JSON form.

    An operator node is ``{"operator": name, "arguments": [...]}``; a constant
    ``{"constant": name, "sort": sort}``; a free variable ``{"free": name,
    "sort": sort}``; a bound variable ``{"bound": index, "sort": sort}``.

    Parameters
    ----------
    record : TermRecord
        The record.
    """
    built: dict[int, JsonValue] = {}
    pending: list[tuple[TermRecord, bool]] = [(record, False)]

    while pending:
        node, expanded = pending.pop()

        match node:
            case ConstantRecord():
                built[id(node)] = {"constant": node.name, "sort": node.sort}
            case FreeRecord():
                built[id(node)] = {"free": node.name, "sort": node.sort}
            case BoundRecord():
                built[id(node)] = {"bound": node.index, "sort": node.sort}
            case OperatorRecord() if expanded:
                built[id(node)] = {
                    "operator": node.operator,
                    "arguments": [built[id(arg)] for arg in node.arguments],
                }
            case OperatorRecord():
                pending.append((node, True))
                pending.extend((arg, False) for arg in reversed(node.arguments))

    return built[id(record)]


def _record_leaf(value: dict[str, JsonValue]) -> TermRecord:
    """Return the leaf record a JSON object describes.

    Raises
    ------
    ValueError
        If the object is not exactly one leaf with its sort.
    """
    sort = value.get("sort")
    kinds = set(value) - {"sort"}

    if not isinstance(sort, str) or len(kinds) != 1 or len(value) != len(kinds) + 1:
        msg = f"a term record leaf is not one kind with a sort: {value!r}"
        raise ValueError(msg)

    (kind,) = kinds
    held = value[kind]

    if kind == "bound" and isinstance(held, int) and not isinstance(held, bool):
        return BoundRecord(sort, held)

    if kind == "constant" and isinstance(held, str):
        return ConstantRecord(sort, held)

    if kind == "free" and isinstance(held, str):
        return FreeRecord(sort, held)

    msg = f"a term record leaf is malformed: {value!r}"
    raise ValueError(msg)


def record_from_json(value: JsonValue, /) -> TermRecord:
    """Return the record a JSON form describes.

    Parameters
    ----------
    value : JsonValue
        A form written by `record_to_json`.

    Raises
    ------
    TypeError
        If the value or a node within it is not a JSON object.
    ValueError
        If an object is not a record node's JSON form.
    """
    built: dict[int, TermRecord] = {}
    pending: list[tuple[JsonValue, bool]] = [(value, False)]

    while pending:
        node, expanded = pending.pop()

        if not isinstance(node, dict):
            msg = f"a term record is not a JSON object: {node!r}"
            raise TypeError(msg)

        if "operator" not in node:
            built[id(node)] = _record_leaf(node)
            continue

        operator = node["operator"]
        arguments = node.get("arguments", [])

        if (
            not isinstance(operator, str)
            or not isinstance(arguments, list)
            or set(node) - {"operator", "arguments"}
        ):
            msg = f"a term record operator node is malformed: {node!r}"
            raise ValueError(msg)

        if expanded:
            built[id(node)] = OperatorRecord(
                operator, tuple(built[id(arg)] for arg in arguments)
            )
        else:
            pending.append((node, True))
            pending.extend((arg, False) for arg in reversed(arguments))

    return built[id(value)]


def record_differences(left: TermRecord, right: TermRecord, /) -> tuple[Path, ...]:
    """Return the outermost paths at which two records differ.

    Two operator nodes with one operator and one arity are compared argument
    by argument; any other pair of unequal nodes differs at its own path.

    Parameters
    ----------
    left : TermRecord
        One record.
    right : TermRecord
        The other record.

    Returns
    -------
    tuple[Path, ...]
        The differing paths in root-first, left-to-right preorder; empty when
        the records are equal.
    """
    found: list[Path] = []
    pending: list[tuple[Path, TermRecord, TermRecord]] = [((), left, right)]

    while pending:
        path, here, there = pending.pop()

        if here == there:
            continue

        if (
            isinstance(here, OperatorRecord)
            and isinstance(there, OperatorRecord)
            and here.operator == there.operator
            and len(here.arguments) == len(there.arguments)
        ):
            pending.extend(
                ((*path, index), here.arguments[index], there.arguments[index])
                for index in range(len(here.arguments) - 1, -1, -1)
            )
            continue

        found.append(path)

    return tuple(found)
