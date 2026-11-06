"""Shared fixtures for logical-form conformance properties."""

from typing import TYPE_CHECKING, Literal, Never, Protocol, cast, override

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Path, Term, TermStore

__all__ = [
    "ACCEPTED",
    "ALPHABET_PAIRS",
    "PRECEDENCE_PAIRS",
    "REJECTED",
    "BindingOperations",
    "LogicalFormFixture",
    "NeuralOperations",
    "ParsingOperations",
    "PendingLogicalFormFixture",
    "PendingValue",
    "PrintingOperations",
    "ReductionOperations",
    "TermOperations",
    "TypingOperations",
    "logical_form_fixture",
]

ACCEPTED = (
    "λx. p(x) ∧ q(x)",
    "λx. λy. f(x)(y)",
    "∀x. (p(x) ⇒ q(x))",
    "∀x. ∃y. (p(x)(y) ∧ q(x)(y))",
    "λx. ∀y. (f(y)(x) ⇒ (p(y) ∧ ∃z. q(z)))",
    "f(TRUE)(FALSE)(x)",
    "(f(g(x))(h(y)) ∧ p(x)) ⇒ q(y)",
    "λx. λy. λz. (f(x)(y)(z) ∧ (g(y)(z) ⇒ h(x)))",
    "∀x. ∃y. ∀z. (p(x)(y)(z) ⇒ (q(y)(z) ∧ r(x)))",
    "λx. ∀y. (f(x)(y) ⇒ ∃z. (g(x)(y)(z) ∧ (h(TRUE)(y) \u2228 ¬p(z))))",
    "λv. λw. λx. λy. λz. f(v)(w)(x)(y)(z)",
    "((p(x) ∧ q(x)) \u2228 (r(x) ∧ s(x))) ⇒ (¬t(x) ⇔ u(x))",
    "∀w. ∃x. ∀y. ∃z. (p(w)(x)(y)(z) ⇔ (q(x)(y) ∧ r(w)(z)))",
    "λx. λy. ∀z. (¬p(x)(y)(z) \u2228 ∃w. (q(w)(y) ⇒ r(x)(z)))",
    "f(g(h(i(j(k(x))))))",
    "f(g(x))(h(i(y)))(j(k(z)(w)))",
    "λx. (∀y. p(x)(y) ⇔ ∃z. q(x)(z))",
    "p(x) ⇒ (q(x) ⇒ (r(x) ⇒ (s(x) ⇒ t(x))))",
    "∀x. ∃y. ((p(x)(y) ⇔ q(x)(y)) ⇒ (r(x) ⇔ s(y)))",
    "λv. ∀w. λx. ∃y. λz. ∀u. p(v)(w)(x)(y)(z)(u)",
    "∀x. ∀y. ((friend(x)(y) ∧ happy(x)) ⇒ happy(y))",
    "λx. λy. ∀z. (x(z) ⇒ ∃w. y(w)(z))",
    (
        "(((p(x) \u2228 q(x)) ∧ (r(x) \u2228 s(x))) ⇒ "
        "((t(x) ∧ u(x)) \u2228 (v(x) ∧ w(x))))"
    ),
    "λx. λy. λz. compose(x)(y)(z)",
    "∀x. ∃y. ((f(g(x)) ⇔ h(i(y))) ∧ (j(k(x)(y)) ⇒ l(m(n(y)))))",
    "∀x. ∀y. ∀z. (equal(x)(y) ⇒ (equal(y)(z) ⇒ equal(x)(z)))",
    "λx. λy. (((x ⇒ y) ⇒ x) ⇒ x)",
    "λf. λx. f(f(f(f(x))))",
    "λx. (λf. f(x(x)))(λf. f(x(x)))",
    "λx. ∀y. ∃z. (f(x)(g(y)(h(z))) ⇔ (¬j(k(x)(y)) \u2228 (l(z) ∧ m(y))))",
)

REJECTED = (
    "",
    "λ",
    "λx.",
    "∀",
    "∀x.",
    "(p(x)",
    "p(x))",
    "((p(x)",
    "p(x) $ q(x)",
    "p(x) @ q(x)",
    "p(x) extra_text",
    "λx. x ∧",
    "∀x. p(x) ⇒",
)

ALPHABET_PAIRS = (
    ("λx. x", r"\x. x"),
    ("p(x) ∧ q(x)", "p(x) & q(x)"),
    ("p(x) \u2228 q(x)", "p(x) | q(x)"),
    ("¬p(x)", "~p(x)"),
    ("p(x) ⇒ q(x)", "p(x) => q(x)"),
    ("p(x) ⇔ q(x)", "p(x) <=> q(x)"),
)

PRECEDENCE_PAIRS = (
    ("p(x) ∧ q(x) \u2228 r(x)", "((p(x) ∧ q(x)) \u2228 r(x))"),
    ("¬p(x) ∧ q(x)", "((¬p(x)) ∧ q(x))"),
    ("p(x) ⇒ q(x) ∧ r(x)", "(p(x) ⇒ (q(x) ∧ r(x)))"),
    ("p(x) ⇔ q(x) ⇒ r(x)", "(p(x) ⇔ (q(x) ⇒ r(x)))"),
    ("x = y ∧ TRUE", "((x = y) ∧ TRUE)"),
)

type PendingTypingFailureKind = Literal["unification", "annotation", "infinite_type"]
"""Failure classification exposed by a typing adapter."""


class TermOperations[F, T, Language, Store](Protocol):
    """Term construction and inspection used by algebra laws."""

    scope_error: type[Exception]

    def forms(self, store: Store, /) -> tuple[F, ...]: ...

    def typed_forms(self, store: Store, /) -> tuple[tuple[F, T], ...]: ...

    def scope_depths(self, form: F, /) -> tuple[tuple[int, int], ...]: ...

    def ill_scoped(self) -> F: ...

    def type_of(self, form: F, /) -> T: ...

    def truth_type(self) -> T: ...

    def alternate_languages(self) -> tuple[Language, Language]: ...

    def term_for_language(self, language: Language, /) -> T: ...

    def free(self, name: str, /) -> F: ...

    def extended_binder(self) -> tuple[Language, str]: ...

    def vacuous(self) -> F: ...


class ParsingOperations[State](Protocol):
    """Parser limits and generated surface atoms."""

    def application_atoms(self) -> tuple[str, ...]: ...

    def state_snapshot(self) -> State: ...


class PrintingOperations[F, T](Protocol):
    """Printer annotations and pipeline projections."""

    def annotations(self, form: F, /) -> tuple[tuple[tuple[int, ...], T], ...]: ...

    def annotated_forms(self) -> tuple[F, ...]: ...

    def pipeline(self, text: str, /) -> F: ...


class TypingOperations[F, T, J, S](Protocol):
    """Typing judgments, examples, and type algebra."""

    infinite_type_error: type[Exception]
    unknown_symbol_error: type[Exception]

    def is_well_typed(self, judgment: J, /) -> bool: ...

    def judgment_type(self, judgment: J, /) -> T | None: ...

    def failure_path(self, judgment: J, /) -> Path | None: ...

    def failure_paths(self, judgment: J, /) -> tuple[Path, ...]: ...

    def failures(
        self,
        judgment: J,
        /,
    ) -> tuple[tuple[PendingTypingFailureKind, T, T, Path], ...]: ...

    def checked_form(self, judgment: J, /) -> F: ...

    def independent_type(self, form: F, /) -> T: ...

    def annotation_conflict(self) -> F: ...

    def unification_conflict(self) -> F: ...

    def principal_cases(self) -> tuple[tuple[F, T, T], ...]: ...

    def subsumes(self, general: T, specific: T, /) -> bool: ...

    def self_application(self) -> F: ...

    def renumber(self, judgment: J, /) -> J: ...

    def unification_cases(self) -> tuple[tuple[T, T, T, S], ...]: ...

    def unify(self, left: T, right: T, /) -> S: ...

    def is_substitution_instance(self, general: S, specific: S, /) -> bool: ...

    def apply_substitution(self, source: T, substitution: S, /) -> T: ...

    def substitution_cases(self) -> tuple[tuple[T, S, S, T], ...]: ...

    def compose_substitutions(self, first: S, second: S, /) -> S: ...

    def identity_substitution(self) -> S: ...

    def ground_types(self) -> tuple[T, ...]: ...

    def arity_form(self) -> F: ...

    def quantifier_forms(self) -> tuple[tuple[F, T], ...]: ...

    def concrete_rule_forms(self) -> tuple[F, ...]: ...

    def negated_conjunct(self) -> F: ...

    def infinite_unification_case(self) -> tuple[T, T]: ...

    def infinite_error_fields(self, error: Exception, /) -> tuple[T, T]: ...

    def unknown_symbol_case(self) -> tuple[Callable[[], J], int, F, Path]: ...

    def unknown_symbol_fields(self, error: Exception, /) -> tuple[int, F, Path]: ...

    def boundary_error_cases(
        self,
    ) -> tuple[tuple[Callable[[], J], type[Exception]], ...]: ...


class ReductionOperations[F, T](Protocol):
    """Substitution and normalization operations."""

    non_terminating: type[Exception]
    fuel: int

    def redexes(self) -> tuple[F, ...]: ...

    def reduce_once(self, form: F, /) -> F: ...

    def substitute(self, body: F, argument: F, /) -> F: ...

    def substitution_cases(self) -> tuple[tuple[F, F, int], ...]: ...

    def free_variables(self, form: F, /) -> frozenset[int]: ...

    def normalize_with_fuel(self, form: F, fuel: int, /) -> F: ...

    def normalize_rightmost(self, form: F, /) -> F: ...

    def weak_head(self, form: F, /) -> F: ...

    def beta_examples(self) -> tuple[tuple[F, F], ...]: ...

    def eta_pair(self) -> tuple[F, F]: ...

    def eta_contract(self, form: F, /) -> F: ...

    def omega(self) -> F: ...


class BindingOperations[F](Protocol):
    """Canonicalization and index operations."""

    def annotation_variants(self) -> tuple[tuple[F, F], ...]: ...

    def canonical(self, form: F, /) -> F: ...

    def binding_owners(self, form: F, /) -> tuple[tuple[int, int], ...]: ...

    def renamed_pairs(self) -> tuple[tuple[F, F], ...]: ...


class NeuralOperations[F](Protocol):
    """Neural views, registries, clamps, and combinators."""

    def nodes(self, form: F, /) -> tuple[F, ...]: ...

    def structure(self, node: F, /) -> str: ...

    def children(self, node: F, /) -> tuple[F, ...]: ...

    def linearize(
        self,
        form: F,
        /,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]]: ...

    def delinearize(
        self,
        encoding: tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, int], ...],
        ],
        /,
    ) -> F: ...

    def application_example(self) -> F: ...

    def application_children(self, form: F, /) -> tuple[F, F]: ...

    def symbol_ids(self) -> tuple[tuple[str, int], ...]: ...

    def clamp_cases(self) -> tuple[tuple[F, tuple[int, int] | None, F, F], ...]: ...

    def enumerate_with_pin(
        self,
        pin: F,
        span: tuple[int, int] | None,
        /,
    ) -> tuple[F, ...]: ...

    def score_with_pin(
        self,
        form: F,
        pin: F,
        span: tuple[int, int] | None,
        /,
    ) -> float: ...

    def combinator_cases(self) -> tuple[tuple[str, F, F, F], ...]: ...

    def apply_combinator(self, name: str, left: F, right: F, /) -> F: ...


class LogicalFormFixture[
    StoreT,
    LanguageT,
    ParsedT,
    FormT,
    TypeT,
    JudgmentT,
    SubstitutionT,
    ParserStateT,
](Protocol):
    """Implementation adapter for logical-form conformance laws."""

    store: StoreT
    language: LanguageT
    parse: Callable[[str], ParsedT]
    print: Callable[[FormT], str]
    desugar: Callable[[ParsedT], FormT]
    check: Callable[[FormT], JudgmentT]
    normalize: Callable[[FormT], FormT]
    alpha_equal: Callable[[FormT, FormT], bool]
    parse_error: type[Exception]
    term: TermOperations[
        FormT,
        TypeT,
        LanguageT,
        StoreT,
    ]
    parsing: ParsingOperations[ParserStateT]
    printing: PrintingOperations[FormT, TypeT]
    typing: TypingOperations[FormT, TypeT, JudgmentT, SubstitutionT]
    reduction: ReductionOperations[FormT, TypeT]
    binding: BindingOperations[FormT]
    neural: NeuralOperations[FormT]


class _Missing:
    __slots__ = ()


type PendingValue = _Missing
type PendingLogicalFormFixture = LogicalFormFixture[
    TermStore,
    LanguageAssembly,
    _Missing,
    _Missing,
    Term,
    _Missing,
    _Missing,
    _Missing,
]


class _MissingFixture:
    __slots__ = ()

    @override
    def __getattribute__(self, name: str) -> Never:
        message = f"logical-form operation {name!r} is not available"

        raise NotImplementedError(message)


@pytest.fixture(name="lf_fixture")
def logical_form_fixture(request: pytest.FixtureRequest) -> PendingLogicalFormFixture:
    """Supply the nearest logical-form adapter."""
    try:
        adapter = request.getfixturevalue("lf_adapter")
    except pytest.FixtureLookupError:
        adapter = _MissingFixture()

    return cast("PendingLogicalFormFixture", adapter)
