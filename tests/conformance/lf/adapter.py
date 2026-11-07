"""Conformance adapter for the declared one-store logical-form language."""

from math import inf, isinf
from math import log as log_weight
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, final, override

from tests.conformance.lf.corpus import (
    CONFORMANCE_CONSTANTS,
    CONFORMANCE_TYPES,
)
from tests.language import grammar_language
from tests.language import language as declared_language

from lofi.deduce import (
    Edge,
    ItemKey,
    LogInside,
    PackedForest,
    RootGoal,
    SpanIndex,
    outside_from_edges,
)
from lofi.interpret import MeaningSelection, compile_meaning_actions, run_action
from lofi.judge import ProofSearch
from lofi.lexicon import Clamp
from lofi.lf import (
    Expression,
    LogicalFormParseError,
    term_codec,
)
from lofi.spec import (
    CompositionSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec._rule_laws import LawGrammar
from lofi.spec.grammar import (
    FunctorSpec,
    SignatureSpec,
)
from lofi.spec.language import (
    BinderNotationSpec,
    ConstantSpec,
    LanguageIso,
    registered_language,
)
from lofi.terms import (
    Alpha,
    Calculus,
    Encoding,
    Identity,
    NonTerminatingError,
    Normalization,
    NotLocallyClosedError,
    OperatorIndex,
    PreorderArityLayout,
    TargetSet,
    Term,
    TermStore,
    beta_normal,
    beta_reducts,
    eta_contract,
    open_binder,
    substitute,
    weak_head,
)
from lofi.typecheck import (
    ClampedGoal,
    Erasure,
    InfiniteTypeError,
    LexicalSignature,
    PairProjection,
    TypingFailureKind,
    TypingResult,
    Unifier,
    UnknownSymbolError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.deduce import Cell
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Path
    from lofi.typecheck import Substitution

__all__ = ["Form", "Judgment", "LogicalFormAdapter"]

type Annotations = tuple[tuple[Path, Term], ...]
type Failure = tuple[TypingFailureKind, Term, Term, Path]
type Linearization = tuple[
    tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]
]
type Reading = tuple[Term, Term]
type Structure = Literal["abstraction", "application", "constant", "variable"]

_CATEGORY_DECLARATION: Final = SignatureSpec(
    name="adapter-categories",
    primitives=("NP", "S"),
    constructors=(
        ConstructorSpec(
            name="fwd",
            kind=("Cat", "Cat", "Cat"),
            notation=InfixConstructorNotationSpec(
                spelling="/", precedence=50, associativity="left"
            ),
        ),
        ConstructorSpec(
            name="bwd",
            kind=("Cat", "Cat", "Cat"),
            notation=InfixConstructorNotationSpec(
                spelling="\\", precedence=50, associativity="left"
            ),
        ),
    ),
    functors=(
        FunctorSpec(
            name="fwd",
            constructor="fwd",
            argument_side="right",
            opposite="bwd",
        ),
        FunctorSpec(
            name="bwd",
            constructor="bwd",
            argument_side="left",
            opposite="fwd",
        ),
    ),
)
_LANGUAGE: Final = grammar_language(
    _CATEGORY_DECLARATION,
    CONFORMANCE_CONSTANTS,
    types=CONFORMANCE_TYPES,
)
_FUEL: Final = 256
_LAW_GRAMMAR: Final = LawGrammar("direct")
_RESULT: Final = TermPatternSpec(name="X", sort="Cat")
_ARGUMENT: Final = TermPatternSpec(name="Y", sort="Cat")
_COMBINATORS: Final = (
    (
        "forward",
        PatternRuleSpec(
            name="forward",
            premises=(
                NodePatternSpec(
                    operator=FixedOperatorSpec(name="fwd"),
                    arguments=(_RESULT, _ARGUMENT),
                ),
                _ARGUMENT,
            ),
            conclusion=_RESULT,
        ),
    ),
    (
        "backward",
        PatternRuleSpec(
            name="backward",
            premises=(
                _ARGUMENT,
                NodePatternSpec(
                    operator=FixedOperatorSpec(name="bwd"),
                    arguments=(_RESULT, _ARGUMENT),
                ),
            ),
            conclusion=_RESULT,
        ),
    ),
    (
        "composition",
        CompositionSpec(view="fwd", order=1, direction="unrestricted"),
    ),
    ("type_raising", TypeRaisingSpec(view="fwd", arguments=None)),
)
"""The combinators whose meaning actions the adapter derives, compiled over
the law signature, whose ``fwd`` and ``bwd`` constructors and ``Cat`` sort
are the ones the category erasure reads."""
_ALPHA: Final = Alpha[Term]()

_FORMS: Final = (
    "sleep(john)",
    "love(john)(mary)",
    "\\x. sleep(x)",
    "\\y. sleep(y)",
    "\\x. \\y. love(x)(y)",
    "\\x. \\y. love(y)(x)",
    "forall x. sleep(x)",
    "exists x. love(x)(john)",
    "forall x. exists y. love(x)(y)",
    "~sleep(john)",
    "sleep(john) & sleep(mary)",
    "sleep(john) | ~sleep(mary)",
    "sleep(john) => sleep(mary)",
    "sleep(john) <=> sleep(mary)",
    "john = mary",
    "(\\x. sleep(x))(john)",
    "\\p. \\x. p(x)",
    "\\x. quietly(sleep(x))",
    "forall x. (sleep(x) => ~run(x))",
    "give(john)(book)(mary)",
)
_TYPED_FORMS: Final = (
    ("sleep(john)", "t"),
    ("love(john)(mary)", "t"),
    ("\\x:e. sleep(x)", "e -> t"),
    ("\\x:e. \\y:e. love(x)(y)", "e -> e -> t"),
    ("forall x:e. sleep(x)", "t"),
    ("exists x:e. love(x)(john)", "t"),
    ("~sleep(john)", "t"),
    ("sleep(john) & ~sleep(mary)", "t"),
    ("(\\x:e. sleep(x))(john)", "t"),
    ("\\p:(e -> t). \\x:e. p(x)", "(e -> t) -> e -> t"),
    ("\\x:e. quietly(sleep(x))", "e -> t"),
    ("give(john)(book)(mary)", "t"),
)
_ANNOTATED_FORMS: Final = (
    "\\x:e. sleep(x)",
    "forall x:e. love(x)(john)",
    "\\f:(e -> t). f(john)",
    "exists x:e. forall y:e. love(x)(y)",
)
_ANNOTATION_VARIANTS: Final = (
    ("\\x:e. sleep(x)", "\\y:e. sleep(y)"),
    ("forall x:e. love(x)(john)", "forall y:e. love(y)(john)"),
    ("\\f:(e -> t). f(john)", "\\g:(e -> t). g(john)"),
)
_REDEXES: Final = (
    "(\\x. sleep(x))(john)",
    "(\\x. \\y. love(x)(y))(john)(mary)",
    "(\\p. \\x. p(x))(\\y. run(y))(mary)",
    "forall x. (\\y. love(x)(y))(john)",
    "(\\x. x)(\\x. x)(john)",
    "sleep((\\x. x)(john))",
)
_BETA_EXAMPLES: Final = (
    ("(\\x. sleep(x))(john)", "sleep(john)"),
    ("(\\x. \\y. love(x)(y))(john)(mary)", "love(john)(mary)"),
    ("(\\p. p & p)(sleep(john))", "sleep(john) & sleep(john)"),
)


class Form(NamedTuple):
    """A core logical form and its requested surface annotations."""

    term: Term
    annotations: Annotations = ()


class Judgment(NamedTuple):
    """A checked form paired with its total typing result."""

    form: Form
    result: TypingResult


def _payload(term: Term) -> int:
    payload = term.payload
    if payload is None:
        msg = f"{term!r} holds no payload"
        raise ValueError(msg)
    return payload


def _atoms(mask: int) -> frozenset[int]:
    return frozenset(atom for atom in range(mask.bit_length()) if mask & (1 << atom))


class _Core:
    __slots__ = ("language", "store", "unifier")

    def __init__(self) -> None:
        self.language = _LANGUAGE
        self.store = _LANGUAGE.store
        self.unifier = Unifier(self.store)

    def form(self, term: Term, annotations: Annotations = ()) -> Form:
        if not self.store.contains(term):
            msg = "logical form is outside the declared language store"
            raise ValueError(msg)
        if not term.is_closed:
            raise NotLocallyClosedError(max_index=term.maxidx, depth=0)
        return Form(term, annotations)

    def parse(self, text: str) -> Expression:
        return self.language.syntax.parse(text)

    def desugar(self, expression: Expression) -> Form:
        lowered = self.language.syntax.desugar(expression)
        return self.form(lowered.term, tuple(sorted(lowered.annotations.items())))

    def read(self, text: str) -> Form:
        return self.desugar(self.parse(text))

    def print(self, form: Form) -> str:
        return self.language.print(form.term)

    def check(self, form: Form) -> Judgment:
        result = self.language.checker.check(
            form.term,
            self.language.lexicon,
            self.language.rules,
            annotations=dict(form.annotations),
        )
        return Judgment(form, result)

    def normalize(self, form: Form, fuel: int = _FUEL) -> Form:
        return self.form(
            beta_normal(self.store, form.term, self.language.calculus, fuel=fuel)
        )

    def type_of(self, form: Form) -> Term:
        result = self.check(form).result
        if not result.is_well_typed or result.scheme is None:
            msg = "logical form has no principal type"
            raise ValueError(msg)
        return result.scheme.body

    def alpha_equal(self, left: Form, right: Form) -> bool:
        return _ALPHA.equivalent(left.term, right.term)


class _TermOperations:
    __slots__ = ("_core", "scope_error")

    def __init__(self, core: _Core) -> None:
        self._core = core
        self.scope_error: type[Exception] = NotLocallyClosedError

    def forms(self, store: TermStore, /) -> tuple[Form, ...]:
        if store is not self._core.store:
            msg = "conformance forms require the declared language store"
            raise ValueError(msg)
        return tuple(self._core.read(text) for text in _FORMS)

    def typed_forms(self, store: TermStore, /) -> tuple[tuple[Form, Term], ...]:
        if store is not self._core.store:
            msg = "typed forms require the declared language store"
            raise ValueError(msg)
        return tuple(
            (self._core.read(text), self._core.language.read_type(expected))
            for text, expected in _TYPED_FORMS
        )

    def scope_depths(self, form: Form, /) -> tuple[tuple[int, int], ...]:
        found: list[tuple[int, int]] = []
        pending = [(form.term, 0)]
        while pending:
            node, depth = pending.pop()
            if node.op.payload_sort == "index":
                found.append((_payload(node), depth))
            pending.extend(
                (child, depth + len(node.op.scope(position)))
                for position, child in enumerate(node.args)
            )
        return tuple(found)

    def ill_scoped(self) -> Form:
        return self._core.form(self._core.store.bvar(_LANGUAGE.term_sort, 0))

    def type_of(self, form: Form, /) -> Term:
        return self._core.type_of(form)

    def truth_type(self) -> Term:
        return self._core.language.read_type("t")

    def alternate_languages(self) -> tuple[LanguageAssembly, LanguageAssembly]:
        return _LANGUAGE, declared_language(
            (*CONFORMANCE_CONSTANTS, "situation"),
            types={**CONFORMANCE_TYPES, "situation": "v"},
        )

    def term_for_language(self, language: LanguageAssembly, /) -> Term:
        return language.apply(language.constant("sleep"), language.constant("john"))

    def extended_binder(self) -> tuple[LanguageAssembly, str]:
        declaration = registered_language("neo-davidsonian")
        additions = tuple(
            ConstantSpec(name=name, type=CONFORMANCE_TYPES.get(name, "t"))
            for name in CONFORMANCE_CONSTANTS
            if name not in {constant.name for constant in declaration.constants}
        )
        extended = declaration.with_(
            name="conformance-iota",
            constants=(
                *declaration.constants,
                *additions,
                ConstantSpec(
                    name="iota",
                    type="(e -> t) -> e",
                    classes=("choice",),
                    notation=BinderNotationSpec(
                        ascii="iota", unicode="\N{GREEK SMALL LETTER IOTA}", bodies=1
                    ),
                ),
            ),
        )
        return LanguageIso().forward(extended), "iota x. sleep(x)"

    def vacuous(self) -> Form:
        return self._core.read("\\x. john")

    def free(self, name: str) -> Form:
        return self._core.read(name)


class _ParsingOperations:
    __slots__ = ()

    def application_atoms(self) -> tuple[str, ...]:
        return ("john", "mary", "TRUE", "x", "y")

    def state_snapshot(self) -> tuple[str, str]:
        return (_LANGUAGE.declaration_digest, _LANGUAGE.syntax.style)


class _PrintingOperations:
    __slots__ = ("_core",)

    def __init__(self, core: _Core) -> None:
        self._core = core

    def annotations(self, form: Form, /) -> Annotations:
        return form.annotations

    def annotated_forms(self) -> tuple[Form, ...]:
        return tuple(self._core.read(text) for text in _ANNOTATED_FORMS)

    def pipeline(self, text: str, /) -> Form:
        return self._core.normalize(self._core.read(text))


class _TypingOperations:
    __slots__ = ("_core", "_unifier", "infinite_type_error", "unknown_symbol_error")

    def __init__(self, core: _Core) -> None:
        self._core = core
        self._unifier = core.unifier
        self.infinite_type_error: type[Exception] = InfiniteTypeError
        self.unknown_symbol_error: type[Exception] = UnknownSymbolError

    def is_well_typed(self, judgment: Judgment, /) -> bool:
        return judgment.result.is_well_typed

    def judgment_type(self, judgment: Judgment, /) -> Term | None:
        return None if judgment.result.scheme is None else judgment.result.scheme.body

    def failure_path(self, judgment: Judgment, /) -> Path | None:
        paths = self.failure_paths(judgment)
        return paths[0] if paths else None

    def failure_paths(self, judgment: Judgment, /) -> tuple[Path, ...]:
        return tuple(constraint.path for constraint in judgment.result.residual)

    def failures(self, judgment: Judgment, /) -> tuple[Failure, ...]:
        return tuple(
            (constraint.kind, constraint.left, constraint.right, constraint.path)
            for constraint in judgment.result.residual
        )

    def checked_form(self, judgment: Judgment, /) -> Form:
        return judgment.form

    def independent_type(self, form: Form, /) -> Term:
        return self._core.type_of(form)

    def annotation_conflict(self) -> Form:
        return self._core.read("\\x:t. sleep(x)")

    def unification_conflict(self) -> Form:
        return self._core.read("sleep(sleep)")

    def _variable(self, name: str) -> Term:
        return self._core.store.fvar(_LANGUAGE.type_sort, name)

    def _arrow(self, domain: Term, codomain: Term) -> Term:
        return self._core.store.op(
            self._core.store.signature.by_name("->"), domain, codomain
        )

    def principal_cases(self) -> tuple[tuple[Form, Term, Term], ...]:
        entity = _LANGUAGE.read_type("e")
        truth = _LANGUAGE.read_type("t")
        first = self._variable("principal-a")
        second = self._variable("principal-b")
        return (
            (
                self._core.read("\\x. x"),
                self._arrow(first, first),
                self._arrow(entity, entity),
            ),
            (
                self._core.read("\\f. \\x. f(x)"),
                self._arrow(self._arrow(first, second), self._arrow(first, second)),
                self._arrow(self._arrow(entity, truth), self._arrow(entity, truth)),
            ),
        )

    def subsumes(self, general: Term, specific: Term, /) -> bool:
        return self._unifier.subsumes(general, specific) is not None

    def self_application(self) -> Form:
        return self._core.read("\\x. x(x)")

    def renumber(self, judgment: Judgment, /) -> Judgment:
        return judgment

    def unification_cases(
        self,
    ) -> tuple[tuple[Term, Term, Term, Substitution], ...]:
        entity = _LANGUAGE.read_type("e")
        truth = _LANGUAGE.read_type("t")
        alpha = self._variable("unify-a")
        beta = self._variable("unify-b")
        gamma = self._variable("unify-c")
        return (
            (
                self._arrow(alpha, truth),
                self._arrow(entity, beta),
                self._arrow(entity, truth),
                MappingProxyType({_payload(alpha): entity, _payload(beta): truth}),
            ),
            (
                self._arrow(alpha, alpha),
                self._arrow(beta, truth),
                self._arrow(truth, truth),
                MappingProxyType({_payload(alpha): truth, _payload(beta): truth}),
            ),
            (
                self._arrow(alpha, beta),
                self._arrow(self._arrow(entity, gamma), self._arrow(entity, gamma)),
                self._arrow(self._arrow(entity, gamma), self._arrow(entity, gamma)),
                MappingProxyType(
                    {
                        _payload(alpha): self._arrow(entity, truth),
                        _payload(beta): self._arrow(entity, truth),
                        _payload(gamma): truth,
                    }
                ),
            ),
        )

    def unify(self, left: Term, right: Term, /) -> Substitution:
        unified = self._unifier.unify(left, right)
        if unified is None:
            msg = f"{left!r} and {right!r} do not unify"
            raise ValueError(msg)
        return unified

    def is_substitution_instance(
        self, general: Substitution, specific: Substitution, /
    ) -> bool:
        probe = self._variable("instance-probe")
        free = self._core.store.signature.free(_LANGUAGE.type_sort)
        for atom in sorted({*general, *specific}, reverse=True):
            probe = self._arrow(
                self._core.store.op(free, payload=atom),
                probe,
            )
        return (
            self._unifier.subsumes(
                self._unifier.apply(general, probe),
                self._unifier.apply(specific, probe),
            )
            is not None
        )

    def apply_substitution(self, source: Term, substitution: Substitution, /) -> Term:
        return self._unifier.apply(substitution, source)

    def substitution_cases(
        self,
    ) -> tuple[tuple[Term, Substitution, Substitution, Term], ...]:
        entity = _LANGUAGE.read_type("e")
        truth = _LANGUAGE.read_type("t")
        alpha = self._variable("substitute-a")
        beta = self._variable("substitute-b")
        gamma = self._variable("substitute-c")
        return (
            (
                self._arrow(alpha, beta),
                MappingProxyType({_payload(alpha): self._arrow(gamma, truth)}),
                MappingProxyType({_payload(gamma): entity, _payload(beta): truth}),
                self._arrow(self._arrow(entity, truth), truth),
            ),
            (
                self._arrow(alpha, alpha),
                MappingProxyType({_payload(alpha): beta}),
                MappingProxyType({_payload(beta): self._arrow(entity, entity)}),
                self._arrow(self._arrow(entity, entity), self._arrow(entity, entity)),
            ),
        )

    def compose_substitutions(
        self, first: Substitution, second: Substitution, /
    ) -> Substitution:
        composed = {
            atom: self._unifier.apply(second, term) for atom, term in first.items()
        }
        for atom, term in second.items():
            composed.setdefault(atom, term)
        return MappingProxyType(composed)

    def identity_substitution(self) -> Substitution:
        return MappingProxyType({})

    def ground_types(self) -> tuple[Term, ...]:
        return tuple(_LANGUAGE.read_type(text) for text in ("e", "t", "e -> t"))

    def arity_form(self) -> Form:
        return self._core.read("give(john)(book)(mary)")

    def quantifier_forms(self) -> tuple[tuple[Form, Term], ...]:
        return (
            (self._core.read("forall x. sleep(x)"), _LANGUAGE.read_type("t")),
            (
                self._core.read("\\p. exists x. (p(x) & sleep(x))"),
                _LANGUAGE.read_type("(e -> t) -> t"),
            ),
            (
                self._core.read("\\x. forall y. love(x)(y)"),
                _LANGUAGE.read_type("e -> t"),
            ),
        )

    def concrete_rule_forms(self) -> tuple[Form, ...]:
        return tuple(
            self._core.read(text)
            for text in (
                "sleep(john)",
                "sleep(john) & sleep(mary)",
                "sleep(john) | sleep(mary)",
                "sleep(john) => sleep(mary)",
                "sleep(john) <=> sleep(mary)",
                "~sleep(john)",
                "john = mary",
                "forall x. sleep(x)",
                "exists x. sleep(x)",
                "\\x. sleep(x)",
            )
        )

    def negated_conjunct(self) -> Form:
        return self._core.read("~(sleep(john) & ~sleep(mary))")

    def infinite_unification_case(self) -> tuple[Term, Term]:
        alpha = self._variable("infinite-a")
        return alpha, self._arrow(alpha, _LANGUAGE.read_type("t"))

    def infinite_error_fields(self, error: Exception, /) -> tuple[Term, Term]:
        if not isinstance(error, InfiniteTypeError):
            msg = f"{error!r} is not an infinite-type error"
            raise TypeError(msg)
        return error.variable, error.term

    def unknown_symbol_case(self) -> tuple[Callable[[], Judgment], int, Form, Path]:
        form = self._core.read("sleep(john)")
        symbol = form.term.args[0]
        identifier = _payload(symbol)
        lexicon = LexicalSignature(
            {
                held: scheme
                for held, scheme in self._core.language.lexicon.schemes.items()
                if held != identifier
            }
        )

        def check() -> Judgment:
            result = self._core.language.checker.check(
                form.term, lexicon, self._core.language.rules
            )
            return Judgment(form, result)

        return check, identifier, Form(symbol), (0,)

    def unknown_symbol_fields(self, error: Exception, /) -> tuple[int, Form, Path]:
        if not isinstance(error, UnknownSymbolError):
            msg = f"{error!r} is not an unknown-symbol error"
            raise TypeError(msg)
        return error.symbol, Form(error.term), error.path

    def boundary_error_cases(
        self,
    ) -> tuple[tuple[Callable[[], Judgment], type[Exception]], ...]:
        foreign = declared_language(("john",)).constant("john")

        def open_term() -> Judgment:
            return self._core.check(
                self._core.form(self._core.store.bvar(_LANGUAGE.term_sort, 0))
            )

        def foreign_term() -> Judgment:
            return self._core.check(Form(foreign))

        def divergent() -> Judgment:
            omega = self._core.read("(\\x. x(x))(\\x. x(x))")
            return self._core.check(self._core.normalize(omega))

        return (
            (open_term, NotLocallyClosedError),
            (foreign_term, ValueError),
            (divergent, NonTerminatingError),
        )


class _ReductionOperations:
    __slots__ = ("_core", "fuel", "non_terminating")

    def __init__(self, core: _Core) -> None:
        self._core = core
        self.fuel = _FUEL
        self.non_terminating: type[Exception] = NonTerminatingError

    def redexes(self) -> tuple[Form, ...]:
        return tuple(self._core.read(text) for text in _REDEXES)

    def reduce_once(self, form: Form, /) -> Form:
        reducts = beta_reducts(self._core.store, form.term, _LANGUAGE.calculus)
        if not reducts:
            msg = "form has no beta reduct"
            raise ValueError(msg)
        return self._core.form(reducts[0])

    def substitute(self, body: Form, argument: Form, /) -> Form:
        atoms = _atoms(body.term.free)
        if len(atoms) != 1:
            msg = "substitution fixture body must have one free variable"
            raise ValueError(msg)
        return self._core.form(
            substitute(self._core.store, body.term, next(iter(atoms)), argument.term)
        )

    def substitution_cases(self) -> tuple[tuple[Form, Form, int], ...]:
        rows = (
            ("hole", "john"),
            ("forall x. hole", "\\y. y"),
            ("\\x. hole", "z"),
        )
        cases = []
        for body_text, argument_text in rows:
            body = self._core.read(body_text)
            hole = self._core.store.fvar(_LANGUAGE.term_sort, "hole")
            cases.append((body, self._core.read(argument_text), _payload(hole)))
        return tuple(cases)

    def free_variables(self, form: Form, /) -> frozenset[int]:
        return _atoms(form.term.free)

    def normalize_with_fuel(self, form: Form, fuel: int, /) -> Form:
        return self._core.normalize(form, fuel)

    def normalize_rightmost(self, form: Form, /) -> Form:
        current = form.term
        for _ in range(_FUEL):
            reducts = beta_reducts(self._core.store, current, _LANGUAGE.calculus)
            if not reducts:
                return self._core.form(current)
            current = reducts[-1]
        raise NonTerminatingError(operation="rightmost normalization", budget=_FUEL)

    def weak_head(self, form: Form, /) -> Form:
        return self._core.form(
            weak_head(self._core.store, form.term, _LANGUAGE.calculus, fuel=_FUEL)
        )

    def beta_examples(self) -> tuple[tuple[Form, Form], ...]:
        return tuple(
            (self._core.read(source), self._core.read(expected))
            for source, expected in _BETA_EXAMPLES
        )

    def eta_pair(self) -> tuple[Form, Form]:
        return self._core.read("\\x. sleep(x)"), self._core.read("sleep")

    def eta_contract(self, form: Form, /) -> Form:
        return self._core.form(
            eta_contract(self._core.store, form.term, _LANGUAGE.calculus)
        )

    def omega(self) -> Form:
        return self._core.read("(\\x. x(x))(\\x. x(x))")


class _BindingOperations:
    __slots__ = ("_core",)

    def __init__(self, core: _Core) -> None:
        self._core = core

    def annotation_variants(self) -> tuple[tuple[Form, Form], ...]:
        return tuple(
            (self._core.read(left), self._core.read(right))
            for left, right in _ANNOTATION_VARIANTS
        )

    def canonical(self, form: Form, /) -> Form:
        return self._core.form(_ALPHA.key(form.term))

    def binding_owners(self, form: Form, /) -> tuple[tuple[int, int], ...]:
        owners: list[tuple[int, int]] = []
        pending: list[tuple[Term, tuple[int, ...]]] = [(form.term, ())]
        position = 0
        while pending:
            node, binders = pending.pop()
            if node.op.payload_sort == "index":
                owners.append((position, binders[-1 - _payload(node)]))
            pending.extend(
                (child, (*binders, *((position,) * len(node.op.scope(at)))))
                for at, child in reversed(tuple(enumerate(node.args)))
            )
            position += 1
        return tuple(owners)

    def renamed_pairs(self) -> tuple[tuple[Form, Form], ...]:
        return (
            (self._core.read("\\x. x"), self._core.read("\\y. y")),
            (
                self._core.read("forall x. exists y. love(x)(y)"),
                self._core.read("forall a. exists b. love(a)(b)"),
            ),
        )


class _PairEquality:
    __slots__ = ()

    @property
    def name(self) -> str:
        return "reading-equality"

    @property
    def has_key(self) -> bool:
        return True

    def key(self, reading: Reading, /) -> Reading:
        return reading

    def equivalent(self, left: Reading, right: Reading, /) -> bool:
        return left == right


class _ProofCalculus:
    __slots__ = ("_store",)

    def __init__(self, store: TermStore) -> None:
        self._store = store

    @property
    def store(self) -> TermStore:
        return self._store

    @property
    def operators(self) -> Calculus:
        return _LANGUAGE.calculus

    def hypothesis(self, index: int, /) -> Term:
        return self._store.fvar(_LANGUAGE.term_sort, f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        return self._store.fvar(_LANGUAGE.term_sort, name)

    def bound_variable(self, index: int, /) -> Term:
        return self._store.bvar(_LANGUAGE.term_sort, index)

    def application(self, function: Term, argument: Term, /) -> Term:
        return _LANGUAGE.apply(function, argument)

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return self._store.bind(_LANGUAGE.calculus.lam, variable, body)


class _NeuralOperations:
    __slots__ = ("_actions", "_core", "_forest", "_roots")

    def __init__(self, core: _Core) -> None:
        self._core = core
        self._actions = self._derived_actions()
        self._forest, self._roots = self._packed()

    def _view(self, form: Form) -> tuple[Form, ...]:
        term = form.term
        if term.op.is_binder:
            opened, _ = open_binder(self._core.store, term)
            return tuple(Form(child) for child in opened.args)
        return tuple(Form(child) for child in term.args)

    def nodes(self, form: Form, /) -> tuple[Form, ...]:
        found: list[Form] = []
        pending = [form]
        while pending:
            node = pending.pop()
            found.append(node)
            pending.extend(reversed(self._view(node)))
        return tuple(found)

    def structure(self, node: Form, /) -> Structure:
        term = node.term
        if term.op.name == _LANGUAGE.calculus.lam:
            return "abstraction"
        if term.op.name == _LANGUAGE.calculus.app:
            return "application"
        if term.op.family == "variable":
            return "variable"
        return "constant"

    def children(self, node: Form, /) -> tuple[Form, ...]:
        return self._view(node)

    def linearize(self, form: Form, /) -> Linearization:
        codec = term_codec(_LANGUAGE, PreorderArityLayout(form.term.size))
        encoding = codec.encode(form.term)
        width = codec.index.width
        payloads = encoding.payloads or (0,) * encoding.width
        symbols = tuple(
            code + width * payload
            for code, payload in zip(encoding.symbols, payloads, strict=True)
        )
        children = [[Encoding.NO_PARENT, Encoding.NO_PARENT] for _ in symbols]
        for slot, parent in enumerate(encoding.parents):
            if parent != Encoding.NO_PARENT:
                children[parent][children[parent].index(Encoding.NO_PARENT)] = slot
        return (
            symbols,
            encoding.parents,
            tuple((left, right) for left, right in children),
        )

    def delinearize(self, encoding: Linearization, /) -> Form:
        symbols, parents, children = encoding
        index = OperatorIndex(_LANGUAGE.store.signature)
        depths: list[int] = []
        for parent in parents:
            depths.append(0 if parent == Encoding.NO_PARENT else depths[parent] + 1)
        for slot, pair in enumerate(children):
            declared = tuple(
                child for child, parent in enumerate(parents) if parent == slot
            )
            if (
                tuple(child for child in pair if child != Encoding.NO_PARENT)
                != declared
            ):
                msg = f"child channel at slot {slot} disagrees with parents"
                raise ValueError(msg)
        codec = term_codec(_LANGUAGE, PreorderArityLayout(len(symbols)))
        held = Encoding(
            tuple(symbol % index.width for symbol in symbols),
            parents,
            tuple(depths),
            (True,) * len(symbols),
            tuple(symbol // index.width for symbol in symbols),
        )
        return Form(codec.decode(held, self._core.store))

    def application_example(self) -> Form:
        return self._core.read("love(john)(mary)")

    def application_children(self, form: Form, /) -> tuple[Form, Form]:
        if form.term.op.name != _LANGUAGE.calculus.app:
            msg = f"{form.term!r} is not an application"
            raise ValueError(msg)
        function, argument = form.term.args
        return Form(function), Form(argument)

    def symbol_ids(self) -> tuple[tuple[str, int], ...]:
        index = OperatorIndex(_LANGUAGE.store.signature)
        return tuple(
            (operator.name, index.index(operator))
            for operator in _LANGUAGE.store.signature.operators
        )

    def _packed(
        self,
    ) -> tuple[PackedForest[Reading, float], tuple[tuple[ItemKey[Reading], Form], ...]]:
        categories = self._core.store
        noun = categories.const("Cat", "NP")
        sentence = categories.const("Cat", "S")
        verb = categories.op(categories.signature.by_name("bwd"), sentence, noun)
        adverb = categories.op(categories.signature.by_name("bwd"), sentence, sentence)
        read = self._core.read
        slept_form = read("quietly(sleep(john))")
        ran_form = read("quietly(run(john))")
        john = ItemKey[Reading]((0, 1), (noun, read("john").term), "lex")
        sleeps = ItemKey[Reading]((1, 2), (verb, read("\\x. sleep(x)").term), "lex")
        runs = ItemKey[Reading]((1, 2), (verb, read("\\x. run(x)").term), "lex")
        quietly = ItemKey[Reading](
            (2, 3), (adverb, read("\\p. quietly(p)").term), "lex"
        )
        sleeping = ItemKey[Reading](
            (0, 2), (sentence, read("sleep(john)").term), "step"
        )
        running = ItemKey[Reading]((0, 2), (sentence, read("run(john)").term), "step")
        slept = ItemKey[Reading]((0, 3), (sentence, slept_form.term), "step")
        ran = ItemKey[Reading]((0, 3), (sentence, ran_form.term), "step")
        forest = PackedForest[Reading, float](SpanIndex(3), LogInside())
        for key, name, children, weight, split in (
            (john, "john", (), 0.5, None),
            (sleeps, "sleeps", (), 0.6, None),
            (runs, "runs", (), 0.4, None),
            (quietly, "quietly", (), 0.7, None),
            (sleeping, "backward", (john, sleeps), 0.3, 1),
            (running, "backward", (john, runs), 0.2, 1),
            (slept, "backward", (sleeping, quietly), 0.9, 2),
            (ran, "backward", (running, quietly), 0.8, 2),
        ):
            forest.add(
                key,
                Edge(name, children, key.reading, log_weight(weight), split=split),
            )
        return forest, ((slept, slept_form), (ran, ran_form))

    def _goal(self, pin: Form, span: Cell | None) -> ClampedGoal[Term, Reading]:
        targets = TargetSet[Reading](
            tuple(root.reading for root, _ in self._roots), _PairEquality()
        )
        clamp = Clamp[Term](
            (0, self._forest.index.n) if span is None else span,
            meaning=pin.term,
        )
        return ClampedGoal(RootGoal(targets), (clamp,), PairProjection(), Identity())

    def clamp_cases(self) -> tuple[tuple[Form, Cell | None, Form, Form], ...]:
        slept = self._core.read("quietly(sleep(john))")
        ran = self._core.read("quietly(run(john))")
        return (
            (self._core.read("sleep(john)"), (0, 2), slept, ran),
            (ran, None, ran, slept),
        )

    def enumerate_with_pin(self, pin: Form, span: Cell | None, /) -> tuple[Form, ...]:
        goal = self._goal(pin, span)
        outside = outside_from_edges(self._forest, goal)
        admitted = frozenset(goal.roots(self._forest))
        return tuple(
            form
            for root, form in self._roots
            if root in admitted and not isinf(outside.inside(root))
        )

    def score_with_pin(self, form: Form, pin: Form, span: Cell | None, /) -> float:
        goal = self._goal(pin, span)
        outside = outside_from_edges(self._forest, goal)
        admitted = frozenset(goal.roots(self._forest))
        scores = [
            outside.inside(root)
            for root, meaning in self._roots
            if root in admitted and self._core.alpha_equal(meaning, form)
        ]
        return max(scores, default=-inf)

    def _derived_actions(self) -> Mapping[str, Term]:
        categories = self._core.store
        erasure = Erasure(
            categories,
            {"NP": _LANGUAGE.read_type("e"), "S": _LANGUAGE.read_type("t")},
            {
                name: self._core.store.op(
                    self._core.store.signature.by_name("->"),
                    self._core.store.fvar("Ty", "a1"),
                    self._core.store.fvar("Ty", "a0"),
                )
                for name in ("fwd", "bwd")
            },
            category_sort="Cat",
            function_type="->",
        )
        schemata = tuple(
            (name, _LAW_GRAMMAR.rule(declaration)) for name, declaration in _COMBINATORS
        )
        actions: dict[str, Term] = {}
        calculus = _ProofCalculus(self._core.store)
        for name, schema in schemata:
            compiled = compile_meaning_actions(
                (schema,),
                erasure=erasure,
                calculus=calculus,
                normalization=Normalization(
                    fuel=40,
                    order="leftmost-outermost",
                    eta="off",
                    associative_commutative=(),
                ),
                search=ProofSearch(
                    contraction=True, weakening=False, max_uses=1, max_inhabitants=64
                ),
                selection=MeaningSelection("unique"),
            )
            selected = compiled.resolve(schema.name, None)
            if len(selected) != 1:
                msg = f"no unique meaning action was derived for {name!r}"
                raise ValueError(msg)
            (action,) = selected
            actions[name] = action
        return MappingProxyType(actions)

    def combinator_cases(self) -> tuple[tuple[str, Form, Form, Form], ...]:
        read = self._core.read
        return (
            ("forward", read("\\x. sleep(x)"), read("john"), read("sleep(john)")),
            ("backward", read("john"), read("\\x. sleep(x)"), read("sleep(john)")),
            (
                "composition",
                read("\\p. quietly(p)"),
                read("\\x. sleep(x)"),
                read("\\x. quietly(sleep(x))"),
            ),
            (
                "type_raising",
                read("john"),
                read("\\x. sleep(x)"),
                read("sleep(john)"),
            ),
        )

    def apply_combinator(self, name: str, left: Form, right: Form, /) -> Form:
        calculus = _ProofCalculus(self._core.store)
        if name == "type_raising":
            raised = run_action(
                self._actions[name],
                (left.term,),
                calculus=calculus,
                normalization=Normalization(
                    fuel=_FUEL,
                    order="leftmost-outermost",
                    eta="off",
                    associative_commutative=(),
                ),
            )
            meaning = run_action(
                self._actions["forward"],
                (raised, right.term),
                calculus=calculus,
                normalization=Normalization(
                    fuel=_FUEL,
                    order="leftmost-outermost",
                    eta="off",
                    associative_commutative=(),
                ),
            )
        else:
            meaning = run_action(
                self._actions[name],
                (left.term, right.term),
                calculus=calculus,
                normalization=Normalization(
                    fuel=_FUEL,
                    order="leftmost-outermost",
                    eta="off",
                    associative_commutative=(),
                ),
            )
        return self._core.form(meaning)


@final
class LogicalFormAdapter:
    """The shipped declared language bound to the conformance laws."""

    __slots__ = (
        "_core",
        "binding",
        "language",
        "neural",
        "parse_error",
        "parsing",
        "printing",
        "reduction",
        "store",
        "term",
        "typing",
    )

    def __init__(self) -> None:
        core = _Core()
        self._core = core
        self.language = core.language
        self.store = core.store
        self.parse_error: type[Exception] = LogicalFormParseError
        self.term = _TermOperations(core)
        self.parsing = _ParsingOperations()
        self.printing = _PrintingOperations(core)
        self.typing = _TypingOperations(core)
        self.reduction = _ReductionOperations(core)
        self.binding = _BindingOperations(core)
        self.neural = _NeuralOperations(core)

    def parse(self, text: str, /) -> Expression:
        return self._core.parse(text)

    def print(self, form: Form, /) -> str:
        return self._core.print(form)

    def desugar(self, expression: Expression, /) -> Form:
        return self._core.desugar(expression)

    def check(self, form: Form, /) -> Judgment:
        return self._core.check(form)

    def normalize(self, form: Form, /) -> Form:
        return self._core.normalize(form)

    def alpha_equal(self, left: Form, right: Form, /) -> bool:
        return self._core.alpha_equal(left, right)

    @override
    def __repr__(self) -> str:
        return "LogicalFormAdapter()"
