from typing import Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.lf._fixtures import ACCEPTED
from tests.conformance.lf.corpus import CONFORMANCE_CONSTANTS, conformance_language
from tests.language import language

from lofi.terms import Term
from lofi.typecheck import (
    LexicalSignature,
    Scheme,
    TypingResult,
)

_GENERATED: Final = (
    "generated-argument0",
    "generated-argument1",
    "generated-argument2",
    "generated-argument3",
    "generated-argument4",
    "generated-argument5",
    "generated-head",
)
LANGUAGE: Final = language(_GENERATED)
STORE: Final = LANGUAGE.store
ENTITY: Final = LANGUAGE.read_type("e")
EVENT: Final = LANGUAGE.read_type("v")
TRUTH: Final = LANGUAGE.read_type("t")
RULES: Final = LANGUAGE.rules
HEAD: Final = LANGUAGE.constant("generated-head")
ARGUMENTS: Final = tuple(
    LANGUAGE.constant(f"generated-argument{position}") for position in range(6)
)
GROUND_TYPES: Final = (ENTITY, EVENT, TRUTH)
_CORPUS_RESIDUAL_PATHS: Final = {
    "λv. ∀w. λx. ∃y. λz. ∀u. p(v)(w)(x)(y)(z)(u)": (
        (0, 1, 0, 0),
        (0,),
    ),
    "λx. (λf. f(x(x)))(λf. f(x(x)))": (
        (0, 0, 0, 1),
        (0, 1, 0, 1),
        (0,),
    ),
}


def _payload(term: Term) -> int:
    """Get a symbol term's payload."""
    if not isinstance(term, Term):
        msg = f"a symbol must be a term, not {type(term).__name__}"
        raise TypeError(msg)

    return cast("int", term.payload)


def _arrow(
    domain: Term,
    codomain: Term,
) -> Term:
    """Build a semantic function type."""
    operator = STORE.signature.by_name(LANGUAGE.function_type)
    return STORE.op(operator, domain, codomain)


def _apply(
    function: Term,
    argument: Term,
) -> Term:
    """Build a logical-form application."""
    return LANGUAGE.apply(function, argument)


def test_signature_copies_and_freezes_its_declarations() -> None:
    declarations = {_payload(HEAD): Scheme(ENTITY)}
    signature = LexicalSignature(declarations)
    declarations[_payload(HEAD)] = Scheme(EVENT)

    assert signature.schemes == {_payload(HEAD): Scheme(ENTITY)}
    assert signature[_payload(HEAD)] == Scheme(ENTITY)
    assert tuple(signature) == (_payload(HEAD),)
    assert len(signature) == 1

    with pytest.raises(TypeError):
        cast("dict[int, Scheme]", signature.schemes)[_payload(HEAD)] = Scheme(TRUTH)

    with pytest.raises(AttributeError, match="immutable"):
        signature.__setattr__("_schemes", {})

    with pytest.raises(AttributeError, match="immutable"):
        signature.__delattr__("_schemes")


def test_signature_equality_and_hash_are_structural_and_order_independent() -> None:
    first = LexicalSignature(
        {
            _payload(HEAD): Scheme(ENTITY),
            _payload(ARGUMENTS[0]): Scheme(EVENT),
        }
    )
    second = LexicalSignature(
        {
            _payload(ARGUMENTS[0]): Scheme(EVENT),
            _payload(HEAD): Scheme(ENTITY),
        }
    )
    different = LexicalSignature({_payload(HEAD): Scheme(TRUTH)})

    assert first == second
    assert hash(first) == hash(second)
    assert first != different
    assert first != first.schemes
    assert repr(first).startswith("LexicalSignature(")


@pytest.mark.parametrize(
    ("atom", "error"),
    [(-1, ValueError), (cast("int", bool(1)), TypeError)],
)
def test_signature_rejects_invalid_symbol_atoms(
    atom: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        LexicalSignature({atom: Scheme(ENTITY)})


def test_a_declared_ternary_predicate_types_at_its_result() -> None:
    predicate_type = _arrow(ENTITY, _arrow(EVENT, _arrow(ENTITY, TRUTH)))
    term = HEAD

    for argument in ARGUMENTS[:3]:
        term = _apply(term, argument)

    signature = LexicalSignature(
        {
            _payload(HEAD): Scheme(predicate_type),
            _payload(ARGUMENTS[0]): Scheme(ENTITY),
            _payload(ARGUMENTS[1]): Scheme(EVENT),
            _payload(ARGUMENTS[2]): Scheme(ENTITY),
        }
    )
    result = LANGUAGE.checker.check(term, signature, RULES)

    assert result.is_well_typed
    assert result.assignment[()] is TRUTH


def test_two_environments_type_one_term_differently() -> None:
    atom = _payload(HEAD)
    checker = LANGUAGE.checker
    entity = checker.check(
        HEAD,
        LexicalSignature({atom: Scheme(ENTITY)}),
        RULES,
    )
    event = checker.check(
        HEAD,
        LexicalSignature({atom: Scheme(EVENT)}),
        RULES,
    )

    assert entity.is_well_typed
    assert event.is_well_typed
    assert entity.assignment[()] is ENTITY
    assert event.assignment[()] is EVENT
    assert entity != event


def test_the_full_surface_corpus_has_total_ordered_typing_results() -> None:
    corpus = conformance_language()
    variable = corpus.store.fvar(corpus.type_sort, "corpus-universal")
    atom = _payload(variable)
    polymorphic = Scheme(variable, (atom,))
    signature = LexicalSignature(
        {
            **corpus.lexicon.schemes,
            **{
                _payload(corpus.constant(name)): polymorphic
                for name in CONFORMANCE_CONSTANTS
            },
        }
    )
    observed: dict[str, tuple[tuple[int, ...], ...]] = {}

    for source in ACCEPTED:
        term = corpus.syntax.desugar(corpus.syntax.parse(source)).term
        result = corpus.checker.check(term, signature, corpus.rules)

        assert term.is_closed
        assert isinstance(result, TypingResult)
        assert () in result.assignment

        if result.residual:
            observed[source] = tuple(constraint.path for constraint in result.residual)

    assert observed == _CORPUS_RESIDUAL_PATHS


@given(
    domains=st.lists(st.sampled_from(GROUND_TYPES), min_size=0, max_size=6),
    codomain=st.sampled_from(GROUND_TYPES),
)
@settings(max_examples=120, deadline=None)
def test_generated_symbol_types_come_from_the_environment(
    domains: list[Term],
    codomain: Term,
) -> None:
    head_type = codomain

    for domain in reversed(domains):
        head_type = _arrow(domain, head_type)

    declarations = {_payload(HEAD): Scheme(head_type)}
    term = HEAD

    for position, domain in enumerate(domains):
        argument = ARGUMENTS[position]
        declarations[_payload(argument)] = Scheme(domain)
        term = _apply(term, argument)

    result = LANGUAGE.checker.check(
        term,
        LexicalSignature(declarations),
        RULES,
    )

    assert result.is_well_typed
    assert result.assignment[()] is codomain
