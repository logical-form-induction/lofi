"""Interior values of a lowered higher-order language."""

from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, final, override

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue, Unpicklable
from lofi.spec.language._errors import AnnotationError, LanguageError
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    OperatorIndex,
    PreorderArityLayout,
    TermCodec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.formalism import TheoryArtifact
    from lofi.lf import LogicProfile, SurfaceSyntax
    from lofi.terms import Calculus, Operator, Signature, Term, TermStore
    from lofi.typecheck import LexicalSignature, Scheme, TypeChecker, TypingRule

__all__ = ["LanguageAssembly", "LanguageCodec", "SortBound"]


@final
class SortBound(FrozenValue):
    """The lowered finite representation bound of one sort."""

    _noun = "sort bound"

    __slots__ = ("_depth", "_free_ranks", "_layout", "_nodes", "_order")

    def __init__(
        self,
        layout: Literal["complete-tree", "preorder-arity"],
        *,
        depth: int,
        nodes: int | None,
        order: Literal["df", "bf"],
        free_ranks: int,
    ) -> None:
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_depth", ensure_positive("sort depth", depth))
        object.__setattr__(
            self, "_free_ranks", ensure_non_negative("free ranks", free_ranks)
        )
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_order", order)

    @property
    def layout(self) -> Literal["complete-tree", "preorder-arity"]:
        """Return the declared layout."""
        return self._layout

    @property
    def depth(self) -> int:
        """Return the maximum represented depth."""
        return self._depth

    @property
    def nodes(self) -> int | None:
        """Return the preorder node capacity, or None for a complete tree."""
        return self._nodes

    @property
    def order(self) -> Literal["df", "bf"]:
        """Return the complete-tree traversal order."""
        return self._order

    @property
    def free_ranks(self) -> int:
        """Return the admitted first-occurrence free-variable ranks."""
        return self._free_ranks


def _reachable(signature: Signature, root: str, /) -> tuple[Operator, ...]:
    """Return the operators reachable from one root sort."""
    pending = [root]
    sorts: set[str] = set()

    while pending:
        sort = pending.pop()

        if sort in sorts:
            continue

        sorts.add(sort)
        pending.extend(
            argument
            for op in signature.operators
            if op.sort == sort
            for argument in op.arg_sorts
        )

    return tuple(op for op in signature.operators if op.sort in sorts)


@final
class LanguageCodec(FrozenValue):
    """The one operator and choice index with one codec per root sort."""

    _noun = "language codec"

    __slots__ = ("_choices", "_index", "_name", "_sorts")

    def __init__(
        self,
        name: str,
        signature: Signature,
        bounds: Mapping[str, SortBound],
        /,
    ) -> None:
        index = OperatorIndex(signature)
        choices = ChoiceIndex(
            signature,
            depth_bounds={sort: bound.depth for sort, bound in bounds.items()},
            free_ranks={sort: bound.free_ranks for sort, bound in bounds.items()},
        )
        codecs: dict[str, TermCodec] = {}

        for sort in signature.sorts:
            bound = bounds[sort]
            layout = (
                CompleteTree(
                    signature,
                    _reachable(signature, sort),
                    order=bound.order,
                    depth=bound.depth,
                )
                if bound.layout == "complete-tree"
                else PreorderArityLayout(bound.nodes if bound.nodes is not None else 0)
            )
            codecs[sort] = TermCodec(index, layout)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_choices", choices)
        object.__setattr__(self, "_sorts", MappingProxyType(codecs))

    @property
    def index(self) -> OperatorIndex:
        """Return the language's one operator index."""
        return self._index

    @property
    def choices(self) -> ChoiceIndex:
        """Return the language's one payload-aware choice index."""
        return self._choices

    def at(self, sort: str, /) -> TermCodec:
        """Return one root sort's codec by stable identity."""
        try:
            return self._sorts[sort]

        except KeyError:
            msg = f"language {self._name!r} declares no sort {sort!r}"
            raise ValueError(msg) from None


@final
class LanguageAssembly(Unpicklable, FrozenValue):
    """One store together with syntax and typing for a declared language."""

    _noun = "language boundary value"
    _because = "retains interned term identities"
    __slots__ = (
        "_bounds",
        "_calculus",
        "_category_sort",
        "_checker",
        "_classes",
        "_codec",
        "_declaration",
        "_digest",
        "_function_type",
        "_lexicon",
        "_name",
        "_profile",
        "_rules",
        "_source_version",
        "_store",
        "_syntax",
        "_target_version",
        "_term_sort",
        "_theory",
        "_type_sort",
    )

    def __init__(
        self,
        name: str,
        declaration: bytes,
        store: TermStore,
        checker: TypeChecker,
        lexicon: LexicalSignature,
        rules: Mapping[str, TypingRule],
        calculus: Calculus,
        syntax: SurfaceSyntax,
        classes: Mapping[str, frozenset[str]],
        profile: LogicProfile,
        theory: TheoryArtifact,
        bounds: Mapping[str, SortBound],
        *,
        type_sort: str,
        term_sort: str,
        category_sort: str,
        function_type: str,
        source_version: str,
        target_version: str,
    ) -> None:
        """Freeze all parts built at the language boundary."""
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_digest", sha256(declaration).hexdigest())
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_checker", checker)
        object.__setattr__(self, "_lexicon", lexicon)
        object.__setattr__(self, "_rules", MappingProxyType(dict(rules)))
        object.__setattr__(self, "_calculus", calculus)
        object.__setattr__(self, "_syntax", syntax)
        object.__setattr__(self, "_classes", MappingProxyType(dict(classes)))
        object.__setattr__(self, "_profile", profile)
        object.__setattr__(self, "_theory", theory)
        held_bounds = dict(bounds)
        object.__setattr__(self, "_bounds", MappingProxyType(held_bounds))
        object.__setattr__(
            self, "_codec", LanguageCodec(name, store.signature, held_bounds)
        )
        object.__setattr__(self, "_type_sort", type_sort)
        object.__setattr__(self, "_term_sort", term_sort)
        object.__setattr__(self, "_category_sort", category_sort)
        object.__setattr__(self, "_function_type", function_type)
        object.__setattr__(self, "_source_version", source_version)
        object.__setattr__(self, "_target_version", target_version)

    @property
    def name(self) -> str:
        """Return the language name."""
        return self._name

    @property
    def declaration(self) -> bytes:
        """Return the canonical declaration bytes."""
        return self._declaration

    @property
    def declaration_digest(self) -> str:
        """Return the SHA-256 of the canonical declaration."""
        return self._digest

    @property
    def store(self) -> TermStore:
        """Return the one store containing types, logical forms and categories."""
        return self._store

    @property
    def bounds(self) -> Mapping[str, SortBound]:
        """Return each sort's lowered representation bound."""
        return self._bounds

    @property
    def codec(self) -> LanguageCodec:
        """Return the language-owned codec collection."""
        return self._codec

    @property
    def checker(self) -> TypeChecker:
        """Return the checker over the language store."""
        return self._checker

    @property
    def lexicon(self) -> LexicalSignature:
        """Return the typed term-constant signature."""
        return self._lexicon

    @property
    def rules(self) -> Mapping[str, TypingRule]:
        """Return structural typing rules keyed by operator."""
        return self._rules

    @property
    def calculus(self) -> Calculus:
        """Return the language's abstraction and application operators."""
        return self._calculus

    @property
    def syntax(self) -> SurfaceSyntax:
        """Return syntax compiled from the declared constants."""
        return self._syntax

    @property
    def constant_classes(self) -> Mapping[str, frozenset[str]]:
        """Return declared logical classes keyed by constant name."""
        return self._classes

    @property
    def profile(self) -> LogicProfile:
        """Return the logic profile derived from constant classes."""
        return self._profile

    @property
    def theory(self) -> TheoryArtifact:
        """Return the declaration artifact recorded with runs."""
        return self._theory

    @property
    def source_version(self) -> str:
        """Return the declaration version."""
        return self._source_version

    @property
    def type_sort(self) -> str:
        """Return the semantic-type sort."""
        return self._type_sort

    @property
    def term_sort(self) -> str:
        """Return the logical-form sort."""
        return self._term_sort

    @property
    def category_sort(self) -> str:
        """Return the syntactic-category sort."""
        return self._category_sort

    @property
    def function_type(self) -> str:
        """Return the declared binary function-type constructor."""
        return self._function_type

    @property
    def target_version(self) -> str:
        """Return the native assembly version."""
        return self._target_version

    def read_type(self, text: str, /) -> Term:
        """Read a closed semantic type into the language store."""
        return self._syntax.read_type(text)

    def print_type(self, term: Term, /) -> str:
        """Render one semantic type using the declared type notation."""
        return self._syntax.print_type(term)

    def constant(self, name: str, /) -> Term:
        """Return a declared term constant."""
        return self._store.const(self._term_sort, name)

    def apply(self, function: Term, *arguments: Term) -> Term:
        """Apply a term to zero or more arguments."""
        held = function
        operator = self._store.signature.by_name(self._calculus.app)

        for argument in arguments:
            held = self._store.op(operator, held, argument)

        return held

    def scheme(self, constant: str, /) -> Scheme:
        """Return a term constant's declared type scheme."""
        identifier = self._store.signature.vocabulary(self._term_sort).id(constant)
        return self._lexicon[identifier]

    def read(self, text: str, /) -> Term:
        """Read, type-check and elaborate one logical form."""
        desugared = self._syntax.desugar(self._syntax.parse(text))
        annotations = desugared.annotations if self._syntax.style == "curry" else None
        result = self._checker.check(
            desugared.term,
            self._lexicon,
            self._rules,
            annotations=annotations,
        )
        annotation_residual = tuple(
            constraint
            for constraint in result.residual
            if constraint.kind == "annotation"
        )

        if annotation_residual:
            raise AnnotationError(annotation_residual)

        if result.residual or result.scheme is None:
            locations = ", ".join(repr(item.path) for item in result.residual)
            msg = f"logical form is ill-typed at {locations}"
            raise LanguageError(msg)

        return self._elaborate(desugared.term, result.assignment)

    def _elaborate(
        self,
        term: Term,
        assignment: Mapping[tuple[int, ...], Term],
    ) -> Term:
        """Replace Church annotation placeholders with inferred principal types."""
        rebuilt: dict[tuple[Term, tuple[int, ...]], Term] = {}

        def visit(node: Term, path: tuple[int, ...]) -> Term:
            known = rebuilt.get((node, path))

            if known is not None:
                return known

            arguments = list(node.args)

            for site in node.op.binding:
                if site.annotation is not None:
                    arguments[site.annotation] = assignment[(*path, site.annotation)]

            arguments = [
                visit(argument, (*path, position))
                if all(site.annotation != position for site in node.op.binding)
                else argument
                for position, argument in enumerate(arguments)
            ]
            held = self._store.op(node.op, *arguments, payload=node.payload)
            rebuilt[(node, path)] = held
            return held

        return visit(term, ())

    def type_of(self, term: Term, /) -> Term:
        """Return the principal type of a well-typed logical form."""
        result = self._checker.check(term, self._lexicon, self._rules)

        if not result.is_well_typed or result.scheme is None:
            msg = "logical form has no principal type"
            raise LanguageError(msg)

        return result.scheme.body

    def print(self, term: Term, /) -> str:
        """Render one term using its declared notation."""
        return self._syntax.print(term)

    def spine(self, term: Term, /) -> tuple[Term, tuple[Term, ...]]:
        """Unwind the declared application spine."""
        return self._syntax.spine(term)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether declarations and theory artifacts match."""
        return type(other) is LanguageAssembly and (
            self._digest,
            self._theory,
        ) == (other._digest, other._theory)

    @override
    def __hash__(self) -> int:
        """Return a stable language hash."""
        return hash((LanguageAssembly, self._digest, self._theory.digest))
