"""Logical form parsing and printing generated from a declared signature."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Never, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.terms import NotationReader, NotationTable, TextCursor, TextParseError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.terms import ConstructorNotation, Path, Term, TermStore

__all__ = [
    "Application",
    "Atom",
    "Desugared",
    "Lambda",
    "LogicalFormParseError",
    "SurfaceSyntax",
    "TypeApplication",
    "TypeAtom",
    "TypeExpression",
]


class TypeAtom(NamedTuple):
    """A type constant or a quantified type variable."""

    name: str


class TypeApplication(NamedTuple):
    """A type constructor applied to its arguments."""

    constructor: str
    arguments: tuple[TypeExpression, ...]


type TypeExpression = TypeAtom | TypeApplication
"""A parsed semantic type."""


class Atom(NamedTuple):
    """A constant, free variable or bound name."""

    name: str


class Application(NamedTuple):
    """A curried term application."""

    function: Expression
    argument: Expression


class Lambda(NamedTuple):
    """A lambda abstraction with an optional surface annotation."""

    name: str
    annotation: TypeExpression | None
    body: Expression


type Expression = Atom | Application | Lambda
"""A parsed logical form after notation expansion."""


@final
class LogicalFormParseError(ValueError):
    """A surface syntax error at a character offset."""

    __slots__ = ("offset", "reason")

    def __init__(self, offset: int, reason: str) -> None:
        """Initialize a located parse refusal."""
        self.offset = ensure_non_negative("offset", offset)
        self.reason = reason
        super().__init__(
            f"cannot parse logical form at character offset {offset}: {reason}"
        )


class Desugared(FrozenValue):
    """A core term and the surface annotations keyed by binder path."""

    _noun = "logical-form value"
    __slots__ = ("_annotations", "_term")

    def __init__(self, term: Term, annotations: Mapping[Path, Term]) -> None:
        """Freeze a desugared term and its located annotations."""
        object.__setattr__(self, "_term", term)
        object.__setattr__(self, "_annotations", MappingProxyType(dict(annotations)))

    @property
    def term(self) -> Term:
        """Return the core term."""
        return self._term

    @property
    def annotations(self) -> Mapping[Path, Term]:
        """Return surface annotations keyed by binder path."""
        return self._annotations

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the term and annotations match."""
        return type(other) is Desugared and (
            self._term,
            self._annotations,
        ) == (other._term, other._annotations)

    @override
    def __hash__(self) -> int:
        """Return a hash over the term and ordered annotations."""
        return hash((Desugared, self._term, tuple(sorted(self._annotations.items()))))


_LOGICAL_FORM: Final = "<logical form>"
"""The context a refusal names while a logical form is read."""

_TYPE: Final = "<semantic type>"
"""The context a refusal names while a semantic type is read."""


@final
class _TypeOperands:
    """Operands that read type names and build declared type constructors."""

    __slots__ = ("_arities",)

    def __init__(self, arities: Mapping[str, int]) -> None:
        self._arities = arities

    def atom(self, cursor: TextCursor, context: str, /) -> TypeExpression:
        """Read a type constant or type variable name."""
        del context
        name = cursor.name()

        if name is None:
            cursor.space()
            cursor.fail(_TYPE, cursor.offset, "expected a semantic type")

        return TypeAtom(name)

    def build(
        self,
        cursor: TextCursor,
        notation: ConstructorNotation,
        arguments: tuple[TypeExpression, ...],
        start: int,
        /,
    ) -> TypeExpression:
        """Apply a declared type constructor to its arguments."""
        del cursor, start
        return TypeApplication(notation.name, arguments)

    def decorate(self, cursor: TextCursor, node: TypeExpression, /) -> TypeExpression:
        """Return a type unchanged; semantic types carry no suffix."""
        del cursor
        return node

    def label(self, node: TypeExpression, /) -> str:
        """Return the name a refusal about a type reports."""
        del node
        return _TYPE

    def arity(self, notation: ConstructorNotation, /) -> int:
        """Return a prefix type constructor's declared arity."""
        return self._arities[notation.name]


def _located(error: TextParseError) -> LogicalFormParseError:
    """Restate a cursor refusal as a logical form refusal at the same offset."""
    return LogicalFormParseError(error.offset, error.reason)


class _TermReader:
    """A Pratt parser compiled from constant notations."""

    __slots__ = ("_binders", "_cursor", "_infix", "_prefix", "_types")

    def __init__(
        self,
        text: str,
        *,
        types: SurfaceSyntax,
        prefix: Sequence[tuple[str, str, int]],
        infix: Sequence[tuple[str, str, int, Literal["left", "right"]]],
        binders: Sequence[tuple[str, str, int]],
    ) -> None:
        self._cursor = TextCursor(text, word_boundaries=True)
        self._types = types
        self._prefix = tuple(prefix)
        self._infix = tuple(infix)
        self._binders = tuple(binders)

    def _fail(self, reason: str) -> Never:
        """Refuse the logical form at the cursor."""
        self._cursor.space()
        self._cursor.fail(_LOGICAL_FORM, self._cursor.offset, reason)

    @staticmethod
    def _apply(function: Expression, arguments: Sequence[Expression]) -> Expression:
        """Return a left-associated curried application."""
        held = function

        for argument in arguments:
            held = Application(held, argument)

        return held

    def _annotation(self) -> TypeExpression | None:
        if self._cursor.accept(":") is None:
            return None

        return self._types.parse_type_cursor(self._cursor)

    def _lambda(self) -> Lambda | None:
        if self._cursor.accept("\\") is None and self._cursor.accept("λ") is None:
            return None

        name = self._cursor.name()

        if name is None:
            self._fail("expected a lambda-bound name")

        annotation = self._annotation()
        self._cursor.expect(".", _LOGICAL_FORM)
        return Lambda(name, annotation, self._expression(1))

    def _binder(self) -> Expression | None:
        for token, constant, bodies in self._binders:
            if self._cursor.accept(token) is None:
                continue

            name = self._cursor.name()

            if name is None:
                self._fail("expected a notation-bound name")

            annotation = self._annotation()
            self._cursor.expect(".", _LOGICAL_FORM)
            expressions: list[Expression] = []

            if bodies == 1:
                expressions.append(self._expression(1))
            else:
                self._cursor.expect("(", _LOGICAL_FORM)

                for position in range(bodies):
                    expressions.append(self._expression(1))

                    if position + 1 < bodies:
                        self._cursor.expect(",", _LOGICAL_FORM)

                self._cursor.expect(")", _LOGICAL_FORM)

            abstractions = tuple(
                Lambda(name, annotation, expression) for expression in expressions
            )
            return self._apply(Atom(constant), abstractions)

        return None

    def _atom(self) -> Expression:
        abstraction = self._lambda()

        if abstraction is not None:
            return abstraction

        binder = self._binder()

        if binder is not None:
            return binder

        if self._cursor.accept("(") is not None:
            held = self._expression(1)
            self._cursor.expect(")", _LOGICAL_FORM)
            return held

        name = self._cursor.name()

        if name is None:
            self._fail("expected a logical-form expression")

        return Atom(name)

    def _postfix(self) -> Expression:
        held = self._atom()

        while self._cursor.accept("(") is not None:
            arguments: list[Expression] = []

            if self._cursor.accept(")") is None:
                while True:
                    arguments.append(self._expression(1))

                    if self._cursor.accept(",") is None:
                        break

                self._cursor.expect(")", _LOGICAL_FORM)

            held = self._apply(held, arguments)

        return held

    def _leading(self) -> Expression:
        for token, constant, precedence in self._prefix:
            if self._cursor.accept(token) is not None:
                return Application(Atom(constant), self._expression(precedence))

        return self._postfix()

    def _expression(self, minimum: int) -> Expression:
        left = self._leading()

        while True:
            start = self._cursor.offset
            matched: tuple[str, str, int, Literal["left", "right"]] | None = None

            for declaration in self._infix:
                if self._cursor.accept(declaration[0]) is not None:
                    matched = declaration
                    break

            if matched is None:
                break

            _, constant, precedence, associativity = matched

            if precedence < minimum:
                self._cursor.offset = start
                break

            right = self._expression(
                precedence + 1 if associativity == "left" else precedence
            )
            left = self._apply(Atom(constant), (left, right))

        return left

    def complete(self) -> Expression:
        """Read one complete logical form."""
        held = self._expression(1)
        self._cursor.finish(_LOGICAL_FORM)
        return held


@final
class SurfaceSyntax(FrozenValue):
    """A parser, desugarer and printer compiled from one language declaration."""

    _noun = "language value"
    __slots__ = (
        "_application",
        "_binder_tokens",
        "_constants",
        "_infix",
        "_lambda",
        "_prefix",
        "_store",
        "_style",
        "_term_sort",
        "_type_notations",
        "_type_notations_by_name",
        "_type_operands",
        "_type_sort",
        "_variables",
    )

    def __init__(
        self,
        store: TermStore,
        constants: Sequence[str],
        *,
        prefix: Sequence[tuple[str, str, int]],
        infix: Sequence[tuple[str, str, int, Literal["left", "right"]]],
        binders: Sequence[tuple[str, str, int]],
        type_sort: str,
        term_sort: str,
        abstraction: str,
        application: str,
        style: Literal["church", "curry"],
        variables: Sequence[str],
        type_notations: Sequence[ConstructorNotation],
        type_arities: Mapping[str, int],
    ) -> None:
        """Compile lookup tables from declared constants and constructors."""

        def by_length(
            row: tuple[str, str, int] | tuple[str, str, int, str],
        ) -> tuple[int, str]:
            """Sort tokens longest first, then lexically."""
            return -len(row[0]), row[0]

        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_constants", frozenset(constants))
        object.__setattr__(self, "_prefix", tuple(sorted(prefix, key=by_length)))
        object.__setattr__(self, "_infix", tuple(sorted(infix, key=by_length)))
        object.__setattr__(
            self,
            "_binder_tokens",
            tuple(sorted(binders, key=by_length)),
        )
        object.__setattr__(self, "_type_notations", NotationTable(type_notations))
        object.__setattr__(
            self,
            "_type_notations_by_name",
            MappingProxyType({notation.name: notation for notation in type_notations}),
        )
        object.__setattr__(
            self,
            "_type_operands",
            _TypeOperands(MappingProxyType(dict(type_arities))),
        )
        object.__setattr__(self, "_type_sort", type_sort)
        object.__setattr__(self, "_term_sort", term_sort)
        object.__setattr__(self, "_lambda", abstraction)
        object.__setattr__(self, "_application", application)
        object.__setattr__(self, "_style", style)
        object.__setattr__(self, "_variables", tuple(variables))

    @property
    def style(self) -> Literal["church", "curry"]:
        """Return the declared representation of lambda annotations."""
        return self._style

    def parse_type(self, text: str, /) -> TypeExpression:
        """Parse one complete semantic type.

        Raises
        ------
        LogicalFormParseError
            If the text is not one type of the declared constructors.
        """
        try:
            return NotationReader(
                TextCursor(text, word_boundaries=True),
                self._type_notations,
                self._type_operands,
            ).complete()
        except TextParseError as error:
            raise _located(error) from error

    def parse_type_cursor(self, cursor: TextCursor) -> TypeExpression:
        """Parse a type annotation from a logical form reader's cursor."""
        return NotationReader(
            cursor, self._type_notations, self._type_operands
        ).expression()

    def parse(self, text: str, /) -> Expression:
        """Parse one logical form using only declared notation.

        Raises
        ------
        LogicalFormParseError
            If the text is not one logical form of the declared notation.
        """
        try:
            return _TermReader(
                text,
                types=self,
                prefix=self._prefix,
                infix=self._infix,
                binders=self._binder_tokens,
            ).complete()
        except TextParseError as error:
            raise _located(error) from error

    def type_term(
        self,
        expression: TypeExpression,
        /,
        *,
        quantified: frozenset[str] = frozenset(),
    ) -> Term:
        """Lower a parsed type into the language store."""
        if isinstance(expression, TypeAtom):
            if expression.name in quantified:
                return self._store.fvar(self._type_sort, expression.name)

            return self._store.const(self._type_sort, expression.name)

        op = self._store.signature.by_name(expression.constructor)
        arguments = tuple(
            self.type_term(argument, quantified=quantified)
            for argument in expression.arguments
        )
        return self._store.op(op, *arguments)

    def read_type(
        self,
        text: str,
        /,
        *,
        quantified: Sequence[str] = (),
    ) -> Term:
        """Parse and lower a semantic type or type scheme body."""
        return self.type_term(self.parse_type(text), quantified=frozenset(quantified))

    def print_type(self, term: Term, /) -> str:
        """Render one semantic type using the declared type notation."""
        if not self._store.contains(term) or term.op.sort != self._type_sort:
            msg = "a printed semantic type must belong to the language type sort"
            raise ValueError(msg)

        return self._write_type(term)

    def desugar(self, expression: Expression, /) -> Desugared:
        """Lower notation to application and lambda, retaining annotations."""
        annotations: dict[Path, Term] = {}

        def lower(
            node: Expression,
            bindings: tuple[tuple[str, str], ...],
            path: Path,
        ) -> Term:
            if isinstance(node, Atom):
                for name, private in reversed(bindings):
                    if node.name == name:
                        return self._store.fvar(self._term_sort, private)

                if node.name in self._constants:
                    return self._store.const(self._term_sort, node.name)

                return self._store.fvar(self._term_sort, node.name)

            if isinstance(node, Application):
                return self._store.op(
                    self._store.signature.by_name(self._application),
                    lower(node.function, bindings, (*path, 0)),
                    lower(node.argument, bindings, (*path, 1)),
                )

            private = f"\N{NULL}lofi.lf.binder:{'.'.join(map(str, path)) or 'root'}"
            body_position = 1 if self._style == "church" else 0
            body = lower(
                node.body,
                (*bindings, (node.name, private)),
                (*path, body_position),
            )
            annotation = (
                None if node.annotation is None else self.type_term(node.annotation)
            )

            if annotation is not None:
                annotations[path] = annotation

            if self._style == "curry":
                return self._store.bind(self._lambda, private, body)

            stored = annotation

            if stored is None:
                stored = self._store.fvar(
                    self._type_sort,
                    f"\N{NULL}lofi.lf.annotation:{'.'.join(map(str, path)) or 'root'}",
                )

            return self._store.bind(self._lambda, private, stored, body)

        return Desugared(lower(expression, (), ()), annotations)

    def spine(self, term: Term, /) -> tuple[Term, tuple[Term, ...]]:
        """Unwind a declared application spine."""
        head = term
        arguments: list[Term] = []

        while head.op.name == self._application and len(head.args) == 2:
            head, argument = head.args
            arguments.append(argument)

        arguments.reverse()
        return head, tuple(arguments)

    def _constant_name(self, term: Term) -> str | None:
        """Return a term constant's name, or None."""
        if not term.op.is_constant or term.op.sort != self._term_sort:
            return None

        return self._store.payload_name(term)

    def _notation(self, name: str) -> tuple[str, str, int, str] | None:
        """Return a constant's preferred notation fields."""
        for token, constant, precedence in self._prefix:
            if constant == name:
                return "prefix", token, precedence, "right"

        for token, constant, precedence, associativity in self._infix:
            if constant == name:
                return "infix", token, precedence, associativity

        for token, constant, bodies in self._binder_tokens:
            if constant == name:
                return "binder", token, bodies, "right"

        return None

    def _write_type(self, term: Term, *, minimum: int = 1) -> str:
        """Render one declared semantic type."""
        if term.op.is_constant:
            return self._store.payload_name(term)

        if term.op.is_free_variable:
            return self._store.payload_name(term)

        notation = self._type_notations_by_name.get(term.op.name)
        if notation is None:
            msg = f"type constructor {term.op.name!r} has no declared notation"
            raise ValueError(msg)
        if notation.style == "prefix":
            arguments = ", ".join(self._write_type(argument) for argument in term.args)
            text = f"{notation.parts[0]}({arguments})"
        elif notation.style == "mixfix":
            pieces = [notation.parts[0]]
            for argument, part in zip(term.args, notation.parts[1:], strict=True):
                pieces.extend(
                    (self._write_type(argument, minimum=notation.precedence), part)
                )
            text = "".join(pieces)
        else:
            left_minimum = notation.precedence + (
                1 if notation.associativity == "right" else 0
            )
            right_minimum = notation.precedence + (
                1 if notation.associativity != "right" else 0
            )
            left = self._write_type(term.args[0], minimum=left_minimum)
            right = self._write_type(term.args[1], minimum=right_minimum)
            text = f"{left} {notation.parts[0]} {right}"
        return f"({text})" if notation.precedence < minimum else text

    def _reserved(self, term: Term, /) -> frozenset[str]:
        """Return the names a printed bound variable must not take."""
        held = set(self._constants)
        held.update(row[0] for row in (*self._prefix, *self._infix))
        held.update(row[0] for row in self._binder_tokens)
        pending = [term]

        while pending:
            node = pending.pop()

            if node.op.is_constant or node.op.is_free_variable:
                held.add(self._store.payload_name(node))

            pending.extend(node.args)

        return frozenset(held)

    def print(self, term: Term, /) -> str:
        """Render a term in the notation that generated its parser.

        Bound variables are named apart from every constant, free variable
        and notation token. An abstraction or binder notation extends as far
        right as it can, so it is parenthesized only where text follows it.
        """
        names: list[str] = []
        reserved = self._reserved(term)

        def fresh() -> str:
            """Return a bound-variable name free in the printed scope."""
            for candidate in self._variables:
                if candidate not in reserved and candidate not in names:
                    return candidate

            msg = "the declared logical variable inventory is exhausted"
            raise ValueError(msg)

        def binder(node: Term) -> tuple[Term | None, Term]:
            site = node.op.binding[0]
            body = node.args[site.arguments[0]]
            annotation = None if site.annotation is None else node.args[site.annotation]
            return annotation, body

        def leaf(node: Term) -> str | None:
            """Render a variable or constant leaf, or decline."""
            if node.op.is_bound_variable:
                if node.payload is None or node.payload >= len(names):
                    msg = "bound variable lies outside its printed scope"
                    raise ValueError(msg)

                return names[-node.payload - 1]

            if node.op.is_constant or node.op.is_free_variable:
                return self._store.payload_name(node)

            return None

        def abstraction(node: Term, *, last: bool) -> str | None:
            """Render a lambda abstraction, or decline."""
            if node.op.name != self._lambda:
                return None

            annotation, body = binder(node)
            name = fresh()
            names.append(name)
            try:
                written = write(body)
            finally:
                names.pop()
            typed = "" if annotation is None else f":{self._write_type(annotation)}"
            text = f"\\{name}{typed}. {written}"
            return text if last else f"({text})"

        def notation_text(
            constant: str,
            arguments: tuple[Term, ...],
            minimum: int,
            *,
            last: bool,
        ) -> str | None:
            """Render an application through declared notation, or decline."""
            notation = self._notation(constant)

            if notation is None:
                return None

            kind, token, precedence, associativity = notation

            if kind == "prefix" and len(arguments) == 1:
                wrapped = precedence < minimum
                operand = write(arguments[0], precedence, last=wrapped or last)
                text = f"{token} {operand}"
                return f"({text})" if wrapped else text

            if kind == "infix" and len(arguments) == 2:
                left_min = precedence + (1 if associativity == "right" else 0)
                right_min = precedence + (1 if associativity == "left" else 0)
                wrapped = precedence < minimum
                left = write(arguments[0], left_min, last=False)
                right = write(arguments[1], right_min, last=wrapped or last)
                text = f"{left} {token} {right}"
                return f"({text})" if wrapped else text

            if kind != "binder" or len(arguments) != precedence:
                return None

            if any(
                argument.op.name != self._lambda or not argument.op.binding
                for argument in arguments
            ):
                return None

            unpacked = tuple(binder(argument) for argument in arguments)
            name = fresh()
            annotations = tuple(each[0] for each in unpacked)

            if len(set(annotations)) > 1:
                msg = f"binder notation {constant!r} has unequal annotations"
                raise ValueError(msg)

            names.append(name)
            try:
                bodies = tuple(write(each[1]) for each in unpacked)
            finally:
                names.pop()
            annotation = annotations[0]
            typed = "" if annotation is None else f":{self._write_type(annotation)}"
            body_text = bodies[0] if len(bodies) == 1 else f"({', '.join(bodies)})"
            text = f"{token} {name}{typed}. {body_text}"
            return text if last else f"({text})"

        def write(node: Term, minimum: int = 1, *, last: bool = True) -> str:
            """Render a term; ``last`` says nothing follows it in its scope."""
            written_leaf = leaf(node)

            if written_leaf is not None:
                return written_leaf

            written_abstraction = abstraction(node, last=last)

            if written_abstraction is not None:
                return written_abstraction

            head, arguments = self.spine(node)
            constant = self._constant_name(head)

            if constant is not None:
                written_notation = notation_text(
                    constant, arguments, minimum, last=last
                )

                if written_notation is not None:
                    return written_notation

            if head is not node:
                text = write(head, last=False)

                for argument in arguments:
                    text = f"{text}({write(argument)})"

                return text

            if node.op.arity == 0:
                return node.op.name

            msg = f"operator {node.op.name!r} has no declared surface notation"
            raise ValueError(msg)

        return write(term)

    @override
    def __eq__(self, other: object) -> bool:
        """Use identity: syntax retains a store."""
        return self is other

    @override
    def __hash__(self) -> int:
        """Return the identity hash."""
        return object.__hash__(self)
