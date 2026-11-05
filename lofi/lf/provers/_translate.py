"""Prover translation over a declared one-store language.

Translation reads each constant's logical role from its declared classes and
never from its name. The truth type is the result type the language's
truth-functional and quantificational constants share, and the function type
is the one the language declares.
"""

from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, final

from lofi.lf._semantics import THF_SYNTAX
from lofi.lf.provers._capability import (
    FIRST_ORDER_CAPABILITY,
    HIGHER_ORDER_CAPABILITY,
    TranslatorCapability,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.lf import LogicProfile
    from lofi.lf._roles import LogicalRole
    from lofi.terms import Term, TermStore

__all__ = [
    "TPTP_DIALECT_CAPABILITIES",
    "FirstOrderTranslationError",
    "PolymorphismPolicy",
    "SmtQuery",
    "TptpDialect",
    "smt_query",
    "tptp_problem",
]

type TptpDialect = Literal["fof", "tff", "thf"]
"""The untyped first-order, typed first-order or typed higher-order dialect."""

TPTP_DIALECT_CAPABILITIES: Final[Mapping[TptpDialect, TranslatorCapability]] = (
    MappingProxyType(
        {
            "fof": FIRST_ORDER_CAPABILITY,
            "tff": FIRST_ORDER_CAPABILITY,
            "thf": HIGHER_ORDER_CAPABILITY,
        }
    )
)
"""The fragment TPTP translation serves in each dialect."""

type PolymorphismPolicy = Literal["refuse", "monomorphize", "erase"]
"""How typed prover translation handles a polymorphic occurrence."""


@final
class FirstOrderTranslationError(ValueError):
    """A construct with no translation in the declared prover dialect."""


class _Language(Protocol):
    """The language capabilities needed by prover translation."""

    @property
    def store(self) -> TermStore:
        """Return the language store."""
        ...

    @property
    def profile(self) -> LogicProfile:
        """Return the language logic profile."""
        ...

    @property
    def function_type(self) -> str:
        """Return the declared function-type constructor."""
        ...

    def scheme(self, constant: str, /) -> _Scheme:
        """Return a constant's declared type scheme."""
        ...

    def spine(self, term: Term, /) -> tuple[Term, tuple[Term, ...]]:
        """Return an application head and its arguments."""
        ...

    def type_of(self, term: Term, /) -> Term:
        """Return the principal type of a term."""
        ...


class _Scheme(Protocol):
    """The scheme capabilities needed by typed translation."""

    @property
    def quantified(self) -> tuple[int, ...]:
        """Return the variables quantified by the scheme."""
        ...

    @property
    def body(self) -> Term:
        """Return the scheme body."""
        ...


class SmtQuery(NamedTuple):
    """Declarations and assertions for one incremental SMT query."""

    declarations: tuple[str, ...]
    assertions: tuple[str, ...]


def _symbol(name: str, /) -> str:
    """Return a collision-resistant prover identifier."""
    stem = "".join(
        character.lower() if character.isalnum() else "_" for character in name
    )
    digest = sha256(name.encode()).hexdigest()[:8]
    return f"lofi_{stem}_{digest}"


def _constant_name(language: _Language, term: Term, /) -> str:
    """Return a term constant's declared name."""
    if not term.op.is_constant:
        msg = f"a prover head must be a declared constant: {term!r}"
        raise FirstOrderTranslationError(msg)

    return language.store.payload_name(term)


def _logical_role(language: _Language, name: str, /) -> LogicalRole | None:
    """Return the constant's resolved logical role, if it declares one."""
    return language.profile.logical_roles.get(name)


def _apply(language: _Language, function: Term, arguments: Sequence[Term], /) -> Term:
    """Apply a function through the language's declared application operator."""
    application = language.store.signature.by_name(language.profile.application)
    result = function

    for argument in arguments:
        result = language.store.op(application, result, argument)

    return result


def _binding(abstraction: Term, /) -> Term:
    """Return the body of one lambda abstraction."""
    if not abstraction.op.is_binder or len(abstraction.op.binding) != 1:
        msg = "a prover quantifier must apply to one lambda abstraction"
        raise FirstOrderTranslationError(msg)

    site = abstraction.op.binding[0]

    if len(site.arguments) != 1:
        msg = "a prover lambda must bind in one body"
        raise FirstOrderTranslationError(msg)

    return abstraction.args[site.arguments[0]]


def _binder_type(language: _Language, abstraction: Term, /) -> Term:
    """Return an abstraction's bound type, including below outer binders."""
    if not abstraction.op.is_binder or len(abstraction.op.binding) != 1:
        msg = "a quantified abstraction needs one binding site"
        raise FirstOrderTranslationError(msg)

    site = abstraction.op.binding[0]

    if site.annotation is not None:
        return abstraction.args[site.annotation]

    wrapped = abstraction
    wrappers = abstraction.maxidx

    for at in range(wrappers):
        wrapped = language.store.bind(
            abstraction.op.name,
            f"\N{NULL}lofi.prover.scope:{at}",
            wrapped,
        )

    if not wrapped.is_closed:
        msg = "a quantified abstraction lies outside its translated scope"
        raise FirstOrderTranslationError(msg)

    inferred = language.type_of(wrapped)

    for _ in range(wrappers):
        if inferred.op.name != language.function_type or len(inferred.args) != 2:
            msg = "a quantified abstraction has no enclosing function type"
            raise FirstOrderTranslationError(msg)

        inferred = inferred.args[1]

    if inferred.op.name != language.function_type or len(inferred.args) != 2:
        msg = "a quantified abstraction needs a function type"
        raise FirstOrderTranslationError(msg)

    return inferred.args[0]


def _bound(term: Term, environment: Sequence[str], /) -> str:
    """Return a bound variable's prover name."""
    payload = term.payload

    if payload is None or payload >= len(environment):
        msg = f"a bound variable lies outside its translated scope: {term!r}"
        raise FirstOrderTranslationError(msg)

    return environment[payload]


def _result(language: _Language, body: Term, /) -> Term:
    """Return the final result of a curried function type."""
    arrow = language.function_type

    while body.op.name == arrow and len(body.args) == 2:
        body = body.args[1]

    return body


def truth_type(language: _Language, /) -> str:
    """Return the truth type the language's logical constants produce.

    Raises
    ------
    FirstOrderTranslationError
        If no constant carries a truth-functional or quantificational role,
        or two such constants produce different types.
    """
    produced = set()

    for name in language.profile.logical_constants:
        scheme = language.scheme(name)
        result = _result(language, scheme.body)

        if not result.op.is_constant:
            if scheme.quantified:
                continue

            msg = f"logical constant {name!r} has no ground result type"
            raise FirstOrderTranslationError(msg)

        result_name = language.store.payload_name(result)

        semantics = language.profile.logical_semantics.get(name)

        if (
            semantics is not None
            and result_name not in language.profile.proposition_types
        ):
            continue

        produced.add(result_name)

    if len(produced) != 1:
        msg = (
            f"the logical constants must share one truth type, not {sorted(produced)!r}"
        )
        raise FirstOrderTranslationError(msg)

    return produced.pop()


@final
class _Typing:
    """Monomorphic first-order signatures of the constants a problem uses."""

    __slots__ = ("_language", "_polymorphism", "_symbols", "truth")

    def __init__(self, language: _Language, polymorphism: PolymorphismPolicy) -> None:
        self._language = language
        self._polymorphism = polymorphism
        self.truth = truth_type(language)
        self._symbols: dict[str, tuple[tuple[str, ...], str]] = {}

    @property
    def symbols(self) -> Mapping[str, tuple[tuple[str, ...], str]]:
        """Return each used constant's argument and result types."""
        return self._symbols

    def _type_name(self, term: Term, /) -> str:
        if not term.op.is_constant:
            msg = f"first-order typing requires a ground atomic type, not {term!r}"
            raise FirstOrderTranslationError(msg)

        return self._language.store.payload_name(term)

    def declare(
        self,
        name: str,
        occurrence: Term,
        arguments: tuple[Term, ...],
        /,
        *,
        formula: bool,
    ) -> str:
        """Record one monomorphic constant occurrence and return its symbol key."""
        scheme = self._language.scheme(name)

        if scheme.quantified:
            if self._polymorphism == "refuse":
                msg = f"typed prover translation refuses polymorphic {name!r}"
                raise FirstOrderTranslationError(msg)

            if self._polymorphism == "erase":
                erased = "lofi_erased"
                argument_types = (erased,) * len(arguments)
                produced = self.truth if formula else erased
            else:
                argument_types = tuple(
                    self._type_name(self._language.type_of(argument))
                    for argument in arguments
                )
                produced = (
                    self.truth
                    if formula
                    else self._type_name(self._language.type_of(occurrence))
                )

            key = f"{name}[{','.join((*argument_types, produced))}]"
            self._symbols[key] = (argument_types, produced)
            return key

        declared_arguments: list[str] = []
        result = scheme.body

        while result.op.name == self._language.function_type and len(result.args) == 2:
            declared_arguments.append(self._type_name(result.args[0]))
            result = result.args[1]

        if len(declared_arguments) != len(arguments):
            msg = (
                f"constant {name!r} is applied to {len(arguments)} arguments but its "
                f"scheme has {len(declared_arguments)}"
            )
            raise FirstOrderTranslationError(msg)

        produced = self._type_name(result)

        if (produced == self.truth) != formula:
            place = "a formula" if formula else "a term"
            msg = f"constant {name!r} of type {produced!r} occurs as {place}"
            raise FirstOrderTranslationError(msg)

        self._symbols[name] = (tuple(declared_arguments), produced)
        return name

    def binder(self, abstraction: Term, /) -> str:
        """Return the type a quantified abstraction binds."""
        return self._type_name(_binder_type(self._language, abstraction))

    def sorts(self) -> tuple[str, ...]:
        """Return the non-truth types the used constants mention, sorted."""
        return tuple(
            sorted(
                {
                    name
                    for arguments, result in self._symbols.values()
                    for name in (*arguments, result)
                    if name != self.truth
                }
            )
        )


class _TptpWriter:
    """Write first-order formulas in TPTP FOF, or in TFF when typed."""

    __slots__ = ("_language", "_typing")

    def __init__(self, language: _Language, typing: _Typing | None) -> None:
        self._language = language
        self._typing = typing

    def sort(self, name: str, /) -> str:
        """Return the TFF type a language type is written as."""
        if self._typing is not None and name == self._typing.truth:
            return "$o"

        return _symbol(f"sort:{name}")

    def _atom(
        self,
        occurrence: Term,
        name: str,
        arguments: tuple[Term, ...],
        environment: tuple[str, ...],
        *,
        formula: bool,
    ) -> str:
        key = name
        if self._typing is not None:
            key = self._typing.declare(
                name,
                occurrence,
                arguments,
                formula=formula,
            )

        written = tuple(self.term(argument, environment) for argument in arguments)
        symbol = _symbol(key)

        return symbol if not written else f"{symbol}({', '.join(written)})"

    def term(self, term: Term, environment: tuple[str, ...]) -> str:
        """Write an individual term."""
        if term.op.is_bound_variable:
            return _bound(term, environment)

        head, arguments = self._language.spine(term)
        name = _constant_name(self._language, head)

        if _logical_role(self._language, name):
            msg = f"logical constant {name!r} cannot occur as an individual term"
            raise FirstOrderTranslationError(msg)

        return self._atom(term, name, arguments, environment, formula=False)

    def formula(self, term: Term, environment: tuple[str, ...] = ()) -> str:
        """Write a truth-valued term."""
        head, arguments = self._language.spine(term)
        name = _constant_name(self._language, head)
        role = _logical_role(self._language, name)

        if name in self._language.profile.definitions:
            msg = f"defined logical constant {name!r} requires THF translation"
            raise FirstOrderTranslationError(msg)

        if name in self._language.profile.logical_semantics:
            msg = f"postulated logical constant {name!r} requires THF translation"
            raise FirstOrderTranslationError(msg)

        if role is not None and role.quantifier:
            if len(arguments) != role.arity:
                msg = f"quantifier {name!r} needs {role.arity} abstractions"
                raise FirstOrderTranslationError(msg)

            variable = f"X{len(environment)}"
            binder_sorts = tuple(
                self._typing.binder(argument)
                for argument in arguments
                if self._typing is not None
            )

            if len(set(binder_sorts)) > 1:
                msg = f"quantifier {name!r} binds incompatible sorts"
                raise FirstOrderTranslationError(msg)

            sort = None if not binder_sorts else self.sort(binder_sorts[0])
            bodies = tuple(
                f"({self.formula(_binding(argument), (variable, *environment))})"
                for argument in arguments
            )
            return role.tptp(bodies, variable=variable, sort=sort)

        if role is not None and role.truth_functional:
            return role.tptp(
                tuple(self.formula(argument, environment) for argument in arguments)
            )

        if role is not None:
            return role.tptp(
                tuple(self.term(argument, environment) for argument in arguments)
            )

        return self._atom(term, name, arguments, environment, formula=True)

    def declarations(self) -> tuple[str, ...]:
        """Return the TFF type and symbol declarations, empty for FOF."""
        typing = self._typing

        if typing is None:
            return ()

        sorts = tuple(
            f"tff(type_{at}, type, {self.sort(name)}: $tType)."
            for at, name in enumerate(typing.sorts())
        )
        symbols: list[str] = []

        for at, (name, (arguments, result)) in enumerate(
            sorted(typing.symbols.items())
        ):
            written = " * ".join(self.sort(argument) for argument in arguments)
            domain = (
                ""
                if not arguments
                else f"{written} > "
                if len(arguments) == 1
                else f"({written}) > "
            )
            symbols.append(
                f"tff(symbol_{at}, type, {_symbol(name)}: {domain}{self.sort(result)})."
            )

        return (*sorts, *symbols)


@final
class _ThfWriter:
    """Write monomorphic typed higher-order formulas in TPTP THF."""

    __slots__ = ("_language", "_polymorphism", "_sorts", "_symbols", "_truth")

    def __init__(self, language: _Language, polymorphism: PolymorphismPolicy) -> None:
        self._language = language
        self._polymorphism = polymorphism
        self._truth = truth_type(language)
        self._sorts: set[str] = set()
        self._symbols: dict[str, str] = {}

    def _type(self, term: Term, /) -> str:
        """Render one monomorphic semantic type."""
        if term.op.name == self._language.function_type and len(term.args) == 2:
            return f"({self._type(term.args[0])} > {self._type(term.args[1])})"

        if not term.op.is_constant:
            msg = f"THF requires a monomorphic type, not {term!r}"
            raise FirstOrderTranslationError(msg)

        name = self._language.store.payload_name(term)

        if name == self._truth:
            return "$o"

        self._sorts.add(name)
        return _symbol(f"sort:{name}")

    def _declare(
        self,
        name: str,
        occurrence: Term,
        arguments: tuple[Term, ...],
        /,
    ) -> str:
        """Record one THF constant instance and return its symbol key."""
        scheme = self._language.scheme(name)

        if scheme.quantified:
            if self._polymorphism == "refuse":
                msg = f"THF0 refuses polymorphic nonlogical constant {name!r}"
                raise FirstOrderTranslationError(msg)

            argument_types = tuple(
                self._type(self._language.type_of(argument)) for argument in arguments
            )
            result = self._type(self._language.type_of(occurrence))

            held_type = result
            for argument_type in reversed(argument_types):
                held_type = f"({argument_type} > {held_type})"

            discriminator = (
                str(len(arguments))
                if self._polymorphism == "erase"
                else ",".join((*argument_types, result))
            )
            key = f"{name}[{discriminator}]"
            self._symbols[key] = held_type
            return key

        if name in self._symbols:
            return name

        self._symbols[name] = self._type(scheme.body)
        return name

    def _binder_type(self, abstraction: Term, /) -> str:
        """Return one abstraction's bound-variable type."""
        return self._type(_binder_type(self._language, abstraction))

    def expression(self, term: Term, environment: tuple[str, ...] = ()) -> str:
        """Render a typed higher-order term or formula."""
        if term.op.is_bound_variable:
            return _bound(term, environment)

        if term.op.is_binder:
            variable = f"X{len(environment)}"
            bound_type = self._binder_type(term)
            body = self.expression(_binding(term), (variable, *environment))
            return f"(^ [{variable}: {bound_type}]: {body})"

        head, arguments = self._language.spine(term)

        if not head.op.is_constant:
            written_head = self.expression(head, environment)
            written_arguments = tuple(
                self.expression(argument, environment) for argument in arguments
            )
            return "(" + " @ ".join((written_head, *written_arguments)) + ")"

        return self._constant_expression(head, arguments, environment)

    def _constant_expression(
        self,
        head: Term,
        arguments: tuple[Term, ...],
        environment: tuple[str, ...],
    ) -> str:
        """Render one declared constant application."""
        name = _constant_name(self._language, head)
        role = _logical_role(self._language, name)
        definition = self._language.profile.definitions.get(name)

        if definition is not None:
            return self.expression(
                _apply(self._language, definition, arguments), environment
            )

        semantics = self._language.profile.logical_semantics.get(name)

        if semantics is not None:
            if len(arguments) != semantics.arity:
                msg = f"logical constant {name!r} needs {semantics.arity} arguments"
                raise FirstOrderTranslationError(msg)

            key = name

            if semantics.declares_symbol:
                occurrence = _apply(self._language, head, arguments)
                key = self._declare(name, occurrence, arguments)

            return semantics.encode(
                _symbol(key),
                tuple(self.expression(argument, environment) for argument in arguments),
                dialect=THF_SYNTAX,
            )

        if role is not None and role.quantifier:
            if len(arguments) != role.arity:
                msg = f"quantifier {name!r} needs {role.arity} abstractions"
                raise FirstOrderTranslationError(msg)

            variable = f"X{len(environment)}"
            bound_types = tuple(
                self._binder_type(abstraction) for abstraction in arguments
            )

            if len(set(bound_types)) > 1:
                msg = f"quantifier {name!r} binds incompatible types"
                raise FirstOrderTranslationError(msg)

            bodies = tuple(
                self.expression(_binding(abstraction), (variable, *environment))
                for abstraction in arguments
            )
            return role.thf(bodies, variable=variable, sort=bound_types[0])

        if role is not None:
            return role.thf(
                tuple(self.expression(argument, environment) for argument in arguments)
            )

        occurrence = _apply(self._language, head, arguments)
        key = self._declare(name, occurrence, arguments)
        written = tuple(
            self.expression(argument, environment) for argument in arguments
        )
        symbol = _symbol(key)

        return symbol if not written else f"({symbol} @ {' @ '.join(written)})"

    def declarations(self) -> tuple[str, ...]:
        """Return THF sort and symbol declarations in stable order."""
        sorts = tuple(
            f"thf(type_{at}, type, {_symbol(f'sort:{name}')}: $tType)."
            for at, name in enumerate(sorted(self._sorts))
        )
        symbols = tuple(
            f"thf(symbol_{at}, type, {_symbol(name)}: {held_type})."
            for at, (name, held_type) in enumerate(sorted(self._symbols.items()))
        )
        return (*sorts, *symbols)


def _require_closed(forms: Sequence[Term], /) -> None:
    """Refuse open prover formulas."""
    for at, term in enumerate(forms):
        if not term.is_closed or term.free:
            msg = f"prover formula {at} is open"
            raise FirstOrderTranslationError(msg)


def tptp_problem(
    language: _Language,
    premise: Term,
    hypothesis: Term,
    /,
    *,
    theory: Sequence[Term] = (),
    dialect: TptpDialect,
    polymorphism: PolymorphismPolicy = "refuse",
) -> str:
    """Render one entailment problem in TPTP FOF, TFF or THF."""
    assumptions = (*language.profile.theory, *tuple(theory), premise)
    forms = (*assumptions, hypothesis)
    _require_closed(forms)
    if dialect == "thf":
        higher = _ThfWriter(language, polymorphism)
        lines = tuple(
            f"thf(axiom_{at}, axiom, {higher.expression(form)})."
            for at, form in enumerate(assumptions)
        )
        conjecture = f"thf(conjecture, conjecture, {higher.expression(hypothesis)})."
        return "\n".join((*higher.declarations(), *lines, conjecture)) + "\n"

    writer = _TptpWriter(
        language,
        None if dialect == "fof" else _Typing(language, polymorphism),
    )
    lines = tuple(
        f"{dialect}(axiom_{at}, axiom, {writer.formula(form)})."
        for at, form in enumerate(assumptions)
    )
    conjecture = f"{dialect}(conjecture, conjecture, {writer.formula(hypothesis)})."
    return "\n".join((*writer.declarations(), *lines, conjecture)) + "\n"


class _SmtWriter:
    """Write sorted first-order formulas and collect their declarations."""

    __slots__ = ("_language", "_typing")

    def __init__(self, language: _Language, polymorphism: PolymorphismPolicy) -> None:
        self._language = language
        self._typing = _Typing(language, polymorphism)

    def _sort(self, name: str, /) -> str:
        return "Bool" if name == self._typing.truth else _symbol(f"sort:{name}")

    def _atom(
        self,
        occurrence: Term,
        name: str,
        arguments: tuple[Term, ...],
        environment: tuple[str, ...],
        *,
        formula: bool,
    ) -> str:
        key = self._typing.declare(
            name,
            occurrence,
            arguments,
            formula=formula,
        )
        written = tuple(self.term(argument, environment) for argument in arguments)
        symbol = _symbol(key)
        return symbol if not written else f"({symbol} {' '.join(written)})"

    def term(self, term: Term, environment: tuple[str, ...]) -> str:
        """Write an individual term."""
        if term.op.is_bound_variable:
            return _bound(term, environment)

        head, arguments = self._language.spine(term)
        name = _constant_name(self._language, head)

        if _logical_role(self._language, name):
            msg = f"logical constant {name!r} cannot occur as an SMT term"
            raise FirstOrderTranslationError(msg)

        return self._atom(term, name, arguments, environment, formula=False)

    def formula(self, term: Term, environment: tuple[str, ...] = ()) -> str:
        """Write a Boolean term."""
        head, arguments = self._language.spine(term)
        name = _constant_name(self._language, head)
        role = _logical_role(self._language, name)

        if name in self._language.profile.definitions:
            msg = f"defined logical constant {name!r} requires THF translation"
            raise FirstOrderTranslationError(msg)

        if name in self._language.profile.logical_semantics:
            msg = f"postulated logical constant {name!r} requires THF translation"
            raise FirstOrderTranslationError(msg)

        if role is not None and role.quantifier:
            if len(arguments) != role.arity:
                msg = f"quantifier {name!r} needs {role.arity} abstractions"
                raise FirstOrderTranslationError(msg)

            variable = f"x{len(environment)}"
            binder_sorts = tuple(
                self._typing.binder(argument) for argument in arguments
            )

            if len(set(binder_sorts)) > 1:
                msg = f"quantifier {name!r} binds incompatible sorts"
                raise FirstOrderTranslationError(msg)

            sort = self._sort(binder_sorts[0])
            bodies = tuple(
                self.formula(_binding(argument), (variable, *environment))
                for argument in arguments
            )
            return role.smt(bodies, variable=variable, sort=sort)

        if role is not None and role.truth_functional:
            return role.smt(
                tuple(self.formula(argument, environment) for argument in arguments)
            )

        if role is not None:
            return role.smt(
                tuple(self.term(argument, environment) for argument in arguments)
            )

        return self._atom(term, name, arguments, environment, formula=True)

    def declarations(self) -> tuple[str, ...]:
        """Return sort and symbol declarations in stable order."""
        sort_lines = tuple(
            f"(declare-sort {self._sort(name)} 0)" for name in self._typing.sorts()
        )
        symbol_lines = tuple(
            f"(declare-fun {_symbol(name)} "
            f"({' '.join(self._sort(argument) for argument in arguments)}) "
            f"{self._sort(result)})"
            for name, (arguments, result) in sorted(self._typing.symbols.items())
        )
        return (*sort_lines, *symbol_lines)


def smt_query(
    language: _Language,
    premise: Term,
    hypothesis: Term,
    /,
    *,
    theory: Sequence[Term] = (),
    polymorphism: PolymorphismPolicy = "refuse",
) -> SmtQuery:
    """Render declarations and assertions for one incremental SMT query."""
    assumptions = (*language.profile.theory, *tuple(theory), premise)
    forms = (*assumptions, hypothesis)
    _require_closed(forms)
    writer = _SmtWriter(language, polymorphism)
    assertions = tuple(f"(assert {writer.formula(form)})" for form in assumptions)
    negated = f"(assert (not {writer.formula(hypothesis)}))"
    return SmtQuery(writer.declarations(), (*assertions, negated))
