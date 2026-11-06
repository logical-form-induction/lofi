"""Declared higher-order signatures for semantic types and logical forms."""

from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf import LogicalConstantSemantics, LogicalRole, LogicalRoleError
from lofi.registry import Registry
from lofi.spec._compile import (
    LANGUAGE_SOURCE_VERSION,
    LANGUAGE_TARGET_VERSION,
    LanguageCompileEnvelope,
    canonical_declaration_bytes,
)
from lofi.spec._constructor import (
    ConstructorSpec,
)
from lofi.spec.language._leaf_lexicon import LeafLexiconSpec
from lofi.spec.language._values import SortBound
from lofi.spec.provers import (
    LabelingProverSpec,
    ProverSpec,
    TruthTableProverSpec,
)
from lofi.terms import Calculus, ConstructorNotation, MVar, Node
from lofi.typecheck import TypingRule

if TYPE_CHECKING:
    from didactic.types._typing import FieldValue

    from lofi.typecheck import LeafLexicon

    class _LeafLexiconDeclaration(Protocol):
        """A boundary declaration that loads one interior leaf lexicon."""

        def load(self) -> LeafLexicon:
            """Load and lower the declared source."""
            ...


__all__ = [
    "AtomicTypeSpec",
    "BinderNotationSpec",
    "BinderSpec",
    "ConstantSpec",
    "InfixNotationSpec",
    "LanguageDecl",
    "LogicSpec",
    "NotationSpec",
    "PrefixNotationSpec",
    "PreparedLanguagePlugins",
    "SortBoundSpec",
]


class PreparedLanguagePlugins(FrozenValue):
    """Logical-role and constant-semantics selections fixed at validation."""

    _noun = "prepared language plugin selection"
    __slots__ = (
        "_atomic_types",
        "_binder_style",
        "_binders",
        "_bounds",
        "_calculus",
        "_category_sort",
        "_classes",
        "_constants",
        "_function_type",
        "_infix",
        "_logic",
        "_name",
        "_prefix",
        "_roles",
        "_rules",
        "_semantics",
        "_store_nodes",
        "_term_sort",
        "_type_arities",
        "_type_notations",
        "_type_sort",
    )

    def __init__(
        self,
        roles: dict[str, LogicalRole],
        semantics: dict[str, LogicalConstantSemantics],
        prefix: tuple[tuple[str, str, int], ...],
        infix: tuple[tuple[str, str, int, Literal["left", "right"]], ...],
        binders: tuple[tuple[str, str, int], ...],
        type_notations: tuple[ConstructorNotation, ...],
        type_arities: dict[str, int],
        classes: dict[str, frozenset[str]],
        rules: dict[str, TypingRule],
        calculus: Calculus,
        bounds: dict[str, SortBound],
        atomic_types: tuple[str, ...],
        constants: tuple[tuple[str, str, tuple[str, ...], str | None, bool], ...],
        binder_style: Literal["church", "curry"],
        logic: tuple[
            str,
            bool,
            bool,
            dict[str, int],
            tuple[str, ...],
            dict[str, str],
            tuple[str, ...],
            str,
            str | None,
            Literal["classical", "strict"],
        ],
        name: str,
        function_type: str,
        term_sort: str,
        type_sort: str,
        category_sort: str,
        store_nodes: int,
        /,
    ) -> None:
        object.__setattr__(self, "_roles", MappingProxyType(dict(roles)))
        object.__setattr__(self, "_semantics", MappingProxyType(dict(semantics)))
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_infix", infix)
        object.__setattr__(self, "_binders", binders)
        object.__setattr__(self, "_type_notations", type_notations)
        object.__setattr__(self, "_type_arities", MappingProxyType(dict(type_arities)))
        object.__setattr__(self, "_classes", MappingProxyType(dict(classes)))
        object.__setattr__(self, "_rules", MappingProxyType(dict(rules)))
        object.__setattr__(self, "_calculus", calculus)
        object.__setattr__(self, "_bounds", MappingProxyType(dict(bounds)))
        object.__setattr__(self, "_atomic_types", atomic_types)
        object.__setattr__(self, "_constants", constants)
        object.__setattr__(self, "_binder_style", binder_style)
        object.__setattr__(
            self,
            "_logic",
            (
                *logic[:3],
                MappingProxyType(dict(logic[3])),
                logic[4],
                MappingProxyType(dict(logic[5])),
                *logic[6:],
            ),
        )
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_function_type", function_type)
        object.__setattr__(self, "_term_sort", term_sort)
        object.__setattr__(self, "_type_sort", type_sort)
        object.__setattr__(self, "_category_sort", category_sort)
        object.__setattr__(self, "_store_nodes", store_nodes)

    @property
    def roles(self) -> MappingProxyType[str, LogicalRole]:
        """Return logical roles by declared constant name."""
        return self._roles

    @property
    def semantics(self) -> MappingProxyType[str, LogicalConstantSemantics]:
        """Return logical semantics by declared constant name."""
        return self._semantics

    @property
    def prefix(self) -> tuple[tuple[str, str, int], ...]:
        """Return prepared prefix-notation rows."""
        return self._prefix

    @property
    def infix(self) -> tuple[tuple[str, str, int, Literal["left", "right"]], ...]:
        """Return prepared infix-notation rows."""
        return self._infix

    @property
    def binders(self) -> tuple[tuple[str, str, int], ...]:
        """Return prepared binder-notation rows."""
        return self._binders

    @property
    def type_notations(self) -> tuple[ConstructorNotation, ...]:
        """Return prepared type-constructor notations."""
        return self._type_notations

    @property
    def type_arities(self) -> MappingProxyType[str, int]:
        """Return type-constructor arities by name."""
        return self._type_arities

    @property
    def classes(self) -> MappingProxyType[str, frozenset[str]]:
        """Return logical-constant classes by name."""
        return self._classes

    @property
    def rules(self) -> MappingProxyType[str, TypingRule]:
        """Return prepared structural typing rules."""
        return self._rules

    @property
    def calculus(self) -> Calculus:
        """Return the prepared binding calculus."""
        return self._calculus

    @property
    def bounds(self) -> MappingProxyType[str, SortBound]:
        """Return lowered finite representation bounds."""
        return self._bounds

    @property
    def atomic_types(self) -> tuple[str, ...]:
        """Return atomic type names in declaration order."""
        return self._atomic_types

    @property
    def constants(
        self,
    ) -> tuple[tuple[str, str, tuple[str, ...], str | None, bool], ...]:
        """Return runtime constant fields in declaration order."""
        return self._constants

    @property
    def binder_style(self) -> Literal["church", "curry"]:
        """Return the language-wide binder representation."""
        return self._binder_style

    @property
    def logic(
        self,
    ) -> tuple[
        str,
        bool,
        bool,
        MappingProxyType[str, int],
        tuple[str, ...],
        MappingProxyType[str, str],
        tuple[str, ...],
        str,
        str | None,
        Literal["classical", "strict"],
    ]:
        """Return prepared logical-fragment fields."""
        return self._logic

    @property
    def name(self) -> str:
        """Return the language name."""
        return self._name

    @property
    def function_type(self) -> str:
        """Return the distinguished function type constructor."""
        return self._function_type

    @property
    def term_sort(self) -> str:
        """Return the term sort."""
        return self._term_sort

    @property
    def type_sort(self) -> str:
        """Return the semantic type sort."""
        return self._type_sort

    @property
    def category_sort(self) -> str:
        """Return the categorial sort."""
        return self._category_sort

    @property
    def store_nodes(self) -> int:
        """Return the native term-store capacity."""
        return self._store_nodes


def _named(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring a nonempty text field."""
    return dx.axiom(f"len({field}) > 0", message=message)


def _all_named(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring every string in a tuple to be nonempty."""
    return dx.axiom(f"all({field})", message=message)


class AtomicTypeSpec(dx.Model, extra="forbid"):
    """An atomic semantic type with one open logical resource kind."""

    name: str
    kind: str = "individual"

    __axioms__ = (
        _named("name", "an atomic type needs a name"),
        _named("kind", "an atomic type needs a resource kind"),
    )


class NotationSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Surface notation generated for a typed term constant."""


class PrefixNotationSpec(NotationSpec, extra="forbid"):
    """Prefix notation for a one-place typed constant."""

    kind: Literal["prefix"] = "prefix"
    ascii: str
    unicode: str
    precedence: int

    __axioms__ = (
        _named("ascii", "prefix notation needs an ASCII token"),
        _named("unicode", "prefix notation needs a Unicode token"),
        dx.axiom("precedence > 0", message="prefix precedence is positive"),
    )


class InfixNotationSpec(NotationSpec, extra="forbid"):
    """Infix notation for a two-place typed constant."""

    kind: Literal["infix"] = "infix"
    ascii: str
    unicode: str
    precedence: int
    associativity: Literal["left", "right"]

    __axioms__ = (
        _named("ascii", "infix notation needs an ASCII token"),
        _named("unicode", "infix notation needs a Unicode token"),
        dx.axiom("precedence > 0", message="infix precedence is positive"),
    )


class BinderNotationSpec(NotationSpec, extra="forbid"):
    """Binder notation for a constant applied to one or more abstractions."""

    kind: Literal["binder"] = "binder"
    ascii: str
    unicode: str
    bodies: int = 1

    __axioms__ = (
        _named("ascii", "binder notation needs an ASCII token"),
        _named("unicode", "binder notation needs a Unicode token"),
        dx.axiom("bodies > 0", message="binder notation needs a body"),
    )


class ConstantSpec(dx.Model, extra="forbid"):
    """A typed constant with notation and optional logical semantics."""

    name: str
    type: str
    quantifies: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    notation: NotationSpec | None = None
    definition: str | None = None
    semantics: str | None = None

    __axioms__ = (
        _named("name", "a term constant needs a name"),
        _named("type", "a term constant needs a type or the value 'infer'"),
        dx.axiom(
            'type != "infer" or len(quantifies) == 0',
            message="an inferred constant cannot predeclare type variables",
        ),
        _all_named("quantifies", "a quantified type variable needs a name"),
        _all_named("classes", "a constant class needs a name"),
    )

    @dx.validates("quantifies", "classes")
    @classmethod
    def _distinct[T: str](cls, values: tuple[T, ...]) -> tuple[T, ...]:
        """Reject repeated quantified variables or classes."""
        if len(set(values)) != len(values):
            msg = "a constant declaration lists an entry twice"
            raise ValueError(msg)

        return values

    @dx.model_validator()
    def _logical_semantics_are_declared_once(self) -> Self:
        """Require either a definition or a postulated semantics plugin."""
        if self.definition is not None:
            if not self.definition:
                msg = "a constant definition needs a closed term"
                raise ValueError(msg)

            if self.semantics is not None:
                msg = "a defined constant cannot also declare a semantics plugin"
                raise ValueError(msg)

        if self.semantics == "":
            msg = "a logical-constant semantics plugin needs a name"
            raise ValueError(msg)

        return self


class BinderSpec(dx.Model, extra="forbid"):
    """The language-wide representation of lambda annotations."""

    style: Literal["church", "curry"] = "curry"


class LogicSpec(dx.Model, extra="forbid"):
    """The logic, prover and proof resources of one language."""

    order: str = "first"
    functions: bool = True
    equality: bool = True
    domains: dict[str, int] = dx.field(default_factory=dict)
    postulates: tuple[str, ...] = ()
    variables: tuple[str, ...] = (
        "x0",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "x8",
        "x9",
        "x10",
        "x11",
        "x12",
        "x13",
        "x14",
        "x15",
    )
    semantics: str = "henkin"
    negation: str | None = None
    label_map: Literal["classical", "strict"] = "classical"
    prover: ProverSpec = dx.field(default_factory=TruthTableProverSpec)
    labeling: LabelingProverSpec = dx.field(default_factory=LabelingProverSpec)

    __axioms__ = (
        _all_named("postulates", "a meaning postulate needs a closed formula"),
        _all_named("variables", "a logical variable needs a name"),
    )

    @dx.validates("order")
    @classmethod
    def _order_is_named(cls, order: str) -> str:
        """Return one open logical-order name."""
        return ensure_name("a logical order", order)

    @dx.validates("semantics")
    @classmethod
    def _logical_semantics_is_named(cls, value: str) -> str:
        """Return a named logical semantics."""
        return ensure_name("a logical setting", value)

    @dx.validates("negation")
    @classmethod
    def _negation_is_named(cls, value: str | None) -> str | None:
        """Return a named designated negation when one is declared."""
        return None if value is None else ensure_name("a designated negation", value)

    @dx.validates("variables")
    @classmethod
    def _variables_are_distinct(cls, variables: tuple[str, ...]) -> tuple[str, ...]:
        """Return a finite distinct nominal variable inventory."""
        if not variables:
            msg = "a logic needs a finite variable inventory"
            raise ValueError(msg)

        if len(set(variables)) != len(variables):
            msg = "a logic lists a variable twice"
            raise ValueError(msg)

        return variables


class SortBoundSpec(dx.Model, extra="forbid"):
    """Declares one sort's finite neural term representation."""

    layout: Literal["complete-tree", "preorder-arity"]
    depth: int
    nodes: int | None = None
    order: Literal["df", "bf"] = "df"
    free_ranks: int = 0

    __axioms__ = (
        dx.axiom("depth > 0", message="a sort depth bound is positive"),
        dx.axiom(
            "free_ranks >= 0", message="a free-variable rank bound is nonnegative"
        ),
    )

    @dx.model_validator()
    def _layout_owns_its_size(self) -> Self:
        """Require nodes exactly for a preorder-arity layout."""
        if self.layout == "preorder-arity":
            if self.nodes is None or self.nodes < 1:
                msg = "a preorder-arity sort bound declares positive nodes"
                raise ValueError(msg)
        elif self.nodes is not None:
            msg = "a complete-tree sort bound derives nodes from its depth"
            raise ValueError(msg)

        return self


def _default_bounds() -> dict[str, SortBoundSpec]:
    """Return the standard higher-order language bounds."""
    return {
        "Ty": SortBoundSpec(layout="complete-tree", depth=4),
        "Tm": SortBoundSpec(layout="preorder-arity", depth=8, nodes=24),
        "Cat": SortBoundSpec(layout="complete-tree", depth=4),
    }


class LanguageDecl(dx.Model, extra="forbid"):
    """One higher-order signature for semantic types and logical forms."""

    name: str
    atomic_types: tuple[AtomicTypeSpec, ...]
    constructors: tuple[ConstructorSpec, ...]
    constants: tuple[ConstantSpec, ...]
    leaf_lexicon: LeafLexiconSpec | None = None
    binders: BinderSpec
    logic: LogicSpec = dx.field(default_factory=LogicSpec)
    function_type: str = "->"
    term_sort: str = "Tm"
    type_sort: str = "Ty"
    category_sort: str = "Cat"
    abstraction: str = "lam"
    application: str = "eval"
    binder_family: str = "binder"
    constant_family: str = "constant"
    variable_family: str = "variable"
    store_nodes: int = 1_000_000
    bounds: dict[str, SortBoundSpec] = dx.field(default_factory=_default_bounds)

    __axioms__ = (
        _named("name", "a language needs a name"),
        dx.axiom("len(atomic_types) > 0", message="a language needs an atomic type"),
        _named("type_sort", "a language needs a type sort"),
        _named("term_sort", "a language needs a term sort"),
        _named("category_sort", "a language needs a category sort"),
        _named("abstraction", "a language needs an abstraction operator"),
        _named("application", "a language needs an application operator"),
        _named("binder_family", "a language needs a binder family"),
        _named("constant_family", "a language needs a constant family"),
        _named("variable_family", "a language needs a variable family"),
        dx.axiom("type_sort != term_sort", message="type and term sorts differ"),
        dx.axiom(
            "type_sort != category_sort and term_sort != category_sort",
            message="type, term and category sorts differ",
        ),
        dx.axiom(
            "abstraction != application",
            message="abstraction and application are distinct operators",
        ),
        dx.axiom("store_nodes > 0", message="a term-store capacity is positive"),
    )

    @dx.validates("atomic_types")
    @classmethod
    def _atomic_names_are_distinct(
        cls, values: tuple[AtomicTypeSpec, ...]
    ) -> tuple[AtomicTypeSpec, ...]:
        """Reject duplicate atomic type declarations."""
        names = tuple(value.name for value in values)

        if len(set(names)) != len(names):
            msg = "an atomic type is declared twice"
            raise ValueError(msg)

        return values

    @dx.validates("constructors")
    @classmethod
    def _constructor_names_are_distinct(
        cls, values: tuple[ConstructorSpec, ...]
    ) -> tuple[ConstructorSpec, ...]:
        """Reject duplicate type-constructor declarations."""
        names = tuple(value.name for value in values)

        if len(set(names)) != len(names):
            msg = "a type constructor is declared twice"
            raise ValueError(msg)

        return values

    @dx.model_validator()
    def _constructors_build_the_type_sort(self) -> Self:
        """Require every type constructor to be endomorphic over the type sort."""
        for constructor in self.constructors:
            if any(sort != self.type_sort for sort in constructor.kind):
                msg = (
                    f"type constructor {constructor.name!r} has kind "
                    f"{constructor.kind!r}, not a kind over {self.type_sort!r}"
                )
                raise ValueError(msg)

            constructor.refuse_notation_arity()

        return self

    @dx.validates("constants")
    @classmethod
    def _constant_names_are_distinct(
        cls, values: tuple[ConstantSpec, ...]
    ) -> tuple[ConstantSpec, ...]:
        """Reject duplicate term-constant declarations."""
        names = tuple(value.name for value in values)

        if len(set(names)) != len(names):
            msg = "a term constant is declared twice"
            raise ValueError(msg)

        token_owners = [
            (token, value.name)
            for value in values
            if value.notation is not None
            for token in {value.notation.ascii, value.notation.unicode}
        ]

        if len({token for token, _ in token_owners}) != len(token_owners):
            msg = "surface notation assigns one token twice"
            raise ValueError(msg)

        return values

    @dx.model_validator()
    def _function_type_is_binary(self) -> Self:
        """Require the function type to name a declared binary constructor."""
        matching = tuple(
            constructor
            for constructor in self.constructors
            if constructor.name == self.function_type
        )
        if not matching or matching[0].arity != 2:
            msg = (
                f"the function type {self.function_type!r} names no declared "
                "binary type constructor"
            )
            raise ValueError(msg)

        if matching[0].notation.style != "infix":
            msg = f"the function type {self.function_type!r} needs infix notation"
            raise ValueError(msg)

        return self

    @dx.model_validator()
    def _bounds_match_sorts(self) -> Self:
        """Require one and only one representation bound per language sort."""
        expected = {self.type_sort, self.term_sort, self.category_sort}

        if set(self.bounds) != expected:
            msg = (
                "sort bounds match the language sorts exactly: "
                f"expected={sorted(expected)!r}, found={sorted(self.bounds)!r}"
            )
            raise ValueError(msg)

        return self

    @dx.derived
    def compile_envelope(self) -> LanguageCompileEnvelope:
        """Return the eagerly prepared declaration and its lowering route."""
        return _compile_language_declaration(self)

    @property
    def prepared_plugins(self) -> PreparedLanguagePlugins:
        """Return the logical plugins resolved with this declaration."""
        return cast("PreparedLanguagePlugins", self.compile_envelope.prepared)

    @dx.model_validator()
    def _prepare_compile_envelope(self) -> Self:
        """Prepare canonical bytes once at the declaration boundary."""
        _ = self.compile_envelope
        return self

    @override
    def model_dump(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        by_alias: bool = False,
        exclude_none: bool = False,
        exclude_defaults: bool = False,
    ) -> dict[str, FieldValue]:
        """Render only fields supplied by the language declaration."""
        hidden = set() if exclude is None else set(exclude)
        hidden.add("compile_envelope")
        return super().model_dump(
            include=include,
            exclude=hidden,
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults,
        )

    @override
    def __repr__(self) -> str:
        """Return didactic's field rendering for diagnostics."""
        return super().__repr__()


def _prepare_language_plugins(source: LanguageDecl, /) -> PreparedLanguagePlugins:
    """Resolve every logical plugin selected by one language declaration."""
    classes = {
        constant.name: frozenset(constant.classes) for constant in source.constants
    }
    prefix: list[tuple[str, str, int]] = []
    infix: list[tuple[str, str, int, Literal["left", "right"]]] = []
    binders: list[tuple[str, str, int]] = []

    for constant in source.constants:
        notation = constant.notation

        if notation is None:
            continue

        tokens = tuple(
            dict.fromkeys((cast("str", notation.ascii), cast("str", notation.unicode)))
        )

        for token in tokens:
            if notation.kind == "prefix":
                prefix.append((token, constant.name, cast("int", notation.precedence)))
            elif notation.kind == "infix":
                infix.append(
                    (
                        token,
                        constant.name,
                        cast("int", notation.precedence),
                        cast('Literal["left", "right"]', notation.associativity),
                    )
                )
            else:
                binders.append((token, constant.name, cast("int", notation.bodies)))

    type_notations = [
        constructor.build_notation() for constructor in source.constructors
    ]

    type_sort = source.type_sort
    domain = MVar("A", type_sort)
    codomain = MVar("B", type_sort)
    function = Node(source.function_type, (domain, codomain))
    rules = {
        source.application: TypingRule(
            source.application,
            (None, None),
            (function, domain),
            codomain,
        ),
        source.abstraction: TypingRule(
            source.abstraction,
            (domain,),
            (codomain,),
            function,
        ),
    }
    bounds = {
        sort: SortBound(
            bound.layout,
            depth=bound.depth,
            nodes=bound.nodes,
            order=bound.order,
            free_ranks=bound.free_ranks,
        )
        for sort, bound in source.bounds.items()
    }
    discovered_roles: dict[str, LogicalRole] = {}

    for tag, member in Registry[object]("logical_role").members():
        if not isinstance(member, LogicalRole) or member.name != tag:
            msg = f"logical-role entry {tag!r} does not implement its contract"
            raise LogicalRoleError(msg)

        discovered_roles[tag] = member

    available_roles = frozenset(discovered_roles)
    roles: dict[str, LogicalRole] = {}

    for constant, declared in classes.items():
        selected = tuple(sorted(declared & available_roles))

        if len(selected) > 1:
            msg = (
                f"logical constant {constant!r} declares conflicting roles "
                f"{list(selected)!r}"
            )
            raise LogicalRoleError(msg)

        if selected:
            roles[constant] = discovered_roles[selected[0]]

    requested = {
        constant.name: constant.semantics
        for constant in source.constants
        if constant.semantics is not None
    }
    discovered_semantics: dict[str, LogicalConstantSemantics] = {}

    for tag, member in Registry[object]("logical_constant").members():
        if not isinstance(member, LogicalConstantSemantics) or member.name != tag:
            msg = (
                f"logical-constant entry {tag!r} does not implement "
                "its semantics contract"
            )
            raise TypeError(msg)

        discovered_semantics[tag] = member

    semantics: dict[str, LogicalConstantSemantics] = {}

    for constant, plugin in requested.items():
        if plugin not in discovered_semantics:
            available = ", ".join(sorted(discovered_semantics)) or "(none)"
            msg = (
                f"logical constant {constant!r} requests semantics {plugin!r}; "
                f"installed semantics are {available}"
            )
            raise ValueError(msg)

        semantics[constant] = discovered_semantics[plugin]

    return PreparedLanguagePlugins(
        roles,
        semantics,
        tuple(prefix),
        tuple(infix),
        tuple(binders),
        tuple(type_notations),
        {constructor.name: constructor.arity for constructor in source.constructors},
        classes,
        rules,
        Calculus(source.abstraction, source.application),
        bounds,
        tuple(atom.name for atom in source.atomic_types),
        tuple(
            (
                constant.name,
                constant.type,
                constant.quantifies,
                constant.definition,
                isinstance(constant.notation, BinderNotationSpec),
            )
            for constant in source.constants
        ),
        source.binders.style,
        (
            source.logic.order,
            source.logic.functions,
            source.logic.equality,
            source.logic.domains,
            source.logic.postulates,
            {atom.name: atom.kind for atom in source.atomic_types},
            source.logic.variables,
            source.logic.semantics,
            source.logic.negation,
            source.logic.label_map,
        ),
        source.name,
        source.function_type,
        source.term_sort,
        source.type_sort,
        source.category_sort,
        source.store_nodes,
    )


@cache
def _compile_language_declaration(source: LanguageDecl, /) -> LanguageCompileEnvelope:
    """Compile one immutable language declaration once per process."""
    return LanguageCompileEnvelope(
        source_version=LANGUAGE_SOURCE_VERSION,
        target_version=LANGUAGE_TARGET_VERSION,
        canonical_declaration=canonical_declaration_bytes(source),
        runtime=_prepare_language_plugins(source),
    )


def materialize_leaf_lexicon(source: LanguageDecl, /) -> LanguageDecl:
    """Replace a leaf source with its reproducible constant declarations."""
    declared = source.leaf_lexicon

    if declared is None:
        return source

    lexicon = cast("_LeafLexiconDeclaration", declared).load()
    constants = tuple(
        ConstantSpec(name=entry.name, type=entry.type, classes=entry.classes)
        for entry in lexicon
    )
    return source.with_(
        constants=(*source.constants, *constants),
        leaf_lexicon=None,
    )
