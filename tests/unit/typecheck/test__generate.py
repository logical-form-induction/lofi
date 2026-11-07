from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.language import (
    apply as language_apply,
)
from tests.language import (
    binary as language_binary,
)
from tests.language import (
    language,
    lexicon,
)
from tests.language import (
    unary as language_unary,
)

from lofi.terms import (
    Node,
    NotLocallyClosedError,
    Term,
    TermStore,
)
from lofi.typecheck import (
    Constraint,
    LexicalSignature,
    Scheme,
    TypingRule,
    Unifier,
    UnknownSymbolError,
    generate,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


CONSTANTS: Final = (
    "fresh-argument",
    "quantifier",
    "annotated-polymorphic-proposition",
    "first-predicate",
    "lambda-argument",
    "lambda-predicate",
    "negated-predicate",
    "polymorphic-proposition",
    "proposition",
    "quantified-argument",
    "second-predicate",
    "sleep",
    "tautology",
    "undeclared-symbol",
)

LANGUAGE: Final = language(CONSTANTS)
LF_STORE: Final = LANGUAGE.store
TYPE_STORE: Final = LANGUAGE.store
ARROW: Final = LANGUAGE.store.signature.by_name("->")
ENTITY: Final = LANGUAGE.read_type("e")
EVENT: Final = LANGUAGE.read_type("v")
TRUTH: Final = LANGUAGE.read_type("t")

T = Node("#const[Ty]", payload=TRUTH.payload)
RULES: Final[Mapping[str, TypingRule]] = LANGUAGE.rules
EMPTY_SIGNATURE: Final = LexicalSignature({})


def arrow(domain: Term, codomain: Term) -> Term:
    """Build a semantic function type."""
    return TYPE_STORE.op(ARROW, domain, codomain)


def apply(
    function: Term,
    argument: Term,
) -> Term:
    """Build a logical-form application."""
    return language_apply(LANGUAGE, function, argument)


def unary(name: str, body: Term) -> Term:
    """Build a unary logical-form node."""
    return language_unary(LANGUAGE, name, body)


def binary(
    name: str,
    left: Term,
    right: Term,
) -> Term:
    """Build a binary logical-form node."""
    return language_binary(LANGUAGE, name, left, right)


def symbol(
    name: str,
    type_term: Term,
    signature: dict[int, Scheme],
) -> Term:
    """Declare and build a logical-form symbol."""
    term = LANGUAGE.constant(name)
    payload = term.payload

    signature[cast("int", payload)] = Scheme(type_term)

    return term


def solve(
    assignment: Mapping[tuple[int, ...], Term],
    constraints: tuple[Constraint, ...],
) -> tuple[
    Mapping[tuple[int, ...], Term],
    tuple[Constraint, ...],
]:
    """Discharge constraints with the independent unifier."""
    unifier = Unifier(TYPE_STORE)
    substitution: dict[int, Term] = {}
    residual: list[Constraint] = []

    for constraint in constraints:
        left = unifier.apply(substitution, constraint.left)
        right = unifier.apply(substitution, constraint.right)
        update = unifier.unify(left, right)

        if update is None:
            residual.append(Constraint(constraint.kind, left, right, constraint.path))
            continue

        substitution = {
            atom: unifier.apply(update, replacement)
            for atom, replacement in substitution.items()
        }
        substitution.update(update)

    inferred = {
        path: unifier.apply(substitution, type_term)
        for path, type_term in assignment.items()
    }

    return inferred, tuple(residual)


def checked(
    term: Term,
    signature: Mapping[int, Scheme],
) -> tuple[
    Mapping[tuple[int, ...], Term],
    tuple[Constraint, ...],
]:
    """Generate and discharge one test judgment."""
    return solve(*generate(term, TYPE_STORE, lexicon(LANGUAGE, signature), RULES))


def test_a_quantified_negation_has_the_truth_type() -> None:
    signature: dict[int, Scheme] = {}
    predicate = symbol("negated-predicate", arrow(ENTITY, TRUTH), signature)
    body = unary("not", apply(predicate, LF_STORE.bvar(LANGUAGE.term_sort, 0)))
    term = unary("exists", body)
    assignment, residual = checked(term, signature)

    assert not residual
    assert assignment[()] is TRUTH


def test_a_negated_conjunct_has_the_truth_type() -> None:
    signature: dict[int, Scheme] = {}
    first = symbol("first-predicate", arrow(ENTITY, TRUTH), signature)
    second = symbol("second-predicate", arrow(ENTITY, TRUTH), signature)
    variable = LF_STORE.bvar(LANGUAGE.term_sort, 0)
    body = binary(
        "and",
        unary("not", apply(first, variable)),
        apply(second, variable),
    )
    term = unary("forall", body)
    assignment, residual = checked(term, signature)

    assert not residual
    assert assignment[()] is TRUTH


def test_a_lambda_application_does_not_reissue_a_child_variable() -> None:
    signature: dict[int, Scheme] = {}
    predicate = symbol("lambda-predicate", arrow(ENTITY, TRUTH), signature)
    argument = symbol("lambda-argument", ENTITY, signature)
    body = apply(predicate, LF_STORE.bvar(LANGUAGE.term_sort, 0))
    function = unary("lam", body)
    assignment, residual = checked(apply(function, argument), signature)

    assert not residual
    assert assignment[()] is TRUTH


def test_a_quantifier_can_bind_a_second_order_variable() -> None:
    signature: dict[int, Scheme] = {}
    argument = symbol("quantified-argument", ENTITY, signature)
    body = apply(LF_STORE.bvar(LANGUAGE.term_sort, 0), argument)
    term = unary("forall", body)
    generated, constraints = generate(
        term,
        TYPE_STORE,
        lexicon(LANGUAGE, signature),
        RULES,
    )
    assignment, residual = solve(generated, constraints)

    assert generated[(1, 0, 0)].op.name == "#fvar[Ty]"
    assert not residual
    assert assignment[()] is TRUTH
    assert assignment[(1, 0, 0)] is arrow(ENTITY, TRUTH)


def test_free_occurrences_share_one_atom_identified_assumption() -> None:
    free = LF_STORE.fvar(LANGUAGE.term_sort, "shared-free")
    term = binary("and", free, free)
    assignment, constraints = generate(term, TYPE_STORE, LANGUAGE.lexicon, RULES)

    assert assignment[(0, 1)] is assignment[(1,)]
    inferred, residual = solve(assignment, constraints)
    assert not residual
    assert inferred[(0, 1)] is inferred[(1,)] is TRUTH


def test_an_inconsistent_free_occurrence_leaves_a_located_constraint() -> None:
    signature: dict[int, Scheme] = {}
    sleeper = symbol("sleep", arrow(ENTITY, TRUTH), signature)
    tautology = symbol("tautology", arrow(TRUTH, TRUTH), signature)
    free = LF_STORE.fvar(LANGUAGE.term_sort, "inconsistent-free")
    term = binary("and", apply(sleeper, free), apply(tautology, free))
    _, residual = checked(term, signature)

    assert residual
    assert all(isinstance(constraint.path, tuple) for constraint in residual)
    assert (1,) in {constraint.path for constraint in residual}


def test_generation_uses_tuple_paths_and_returns_no_live_assignment() -> None:
    signature: dict[int, Scheme] = {}
    proposition = symbol("proposition", TRUTH, signature)
    term = unary("not", proposition)
    assignment, constraints = generate(
        term,
        TYPE_STORE,
        lexicon(LANGUAGE, signature),
        RULES,
    )

    inferred, residual = solve(assignment, constraints)
    assert inferred[()] is TRUTH
    assert inferred[(1,)] is TRUTH
    assert not residual
    assert tuple(constraint.kind for constraint in constraints) == (
        "unification",
        "unification",
    )

    with pytest.raises(TypeError):
        cast("dict[tuple[int, ...], Term]", assignment)[()] = ENTITY


def test_a_polymorphic_symbol_is_freshly_instantiated_at_each_occurrence() -> None:
    variable = TYPE_STORE.fvar("Ty", "polymorphic-symbol")
    atom = variable.payload

    assert atom is not None
    proposition = LANGUAGE.constant("polymorphic-proposition")
    payload = proposition.payload

    assert payload is not None
    term = binary("and", proposition, proposition)
    assignment, _ = generate(
        term,
        TYPE_STORE,
        lexicon(LANGUAGE, {payload: Scheme(variable, (atom,))}),
        RULES,
    )

    assert assignment[(0,)] is not assignment[(1,)]


def test_leaf_annotations_follow_ordinary_use_constraints() -> None:
    variable = TYPE_STORE.fvar("Ty", "annotated-polymorphic-symbol")
    atom = variable.payload

    assert atom is not None
    signature: dict[int, Scheme] = {}
    proposition = symbol(
        "annotated-polymorphic-proposition",
        variable,
        signature,
    )
    payload = proposition.payload

    assert payload is not None
    signature[payload] = Scheme(variable, (atom,))
    term = unary("not", proposition)
    assignment, constraints = generate(
        term,
        TYPE_STORE,
        lexicon(LANGUAGE, signature),
        RULES,
        annotations={(1,): ENTITY},
    )
    inferred, residual = solve(assignment, constraints)

    assert constraints[-1] == Constraint("annotation", assignment[(1,)], ENTITY, (1,))
    assert inferred[(1,)] is TRUTH
    assert residual == (Constraint("annotation", TRUTH, ENTITY, (1,)),)


def test_injected_constraints_target_any_generated_node() -> None:
    variable = TYPE_STORE.fvar("Ty", "injected-polymorphic-symbol")
    atom = variable.payload

    assert atom is not None
    signature: dict[int, Scheme] = {}
    proposition = symbol("polymorphic-proposition", variable, signature)
    payload = proposition.payload

    assert payload is not None
    signature[payload] = Scheme(variable, (atom,))
    term = unary("not", proposition)
    assignment, constraints = generate(
        term,
        TYPE_STORE,
        lexicon(LANGUAGE, signature),
        RULES,
        constraints={(1,): TRUTH},
    )
    inferred, residual = solve(assignment, constraints)

    assert constraints[-1] == Constraint("unification", assignment[(1,)], TRUTH, (1,))
    assert inferred[(1,)] is TRUTH
    assert not residual


def test_injected_constraint_atoms_are_not_reissued_as_fresh_variables() -> None:
    constraint = TYPE_STORE.fvar("Ty", "rigid-injected-constraint")
    term = LF_STORE.fvar(LANGUAGE.term_sort, "fresh-around-injected-constraint")
    assignment, _ = generate(
        term,
        TYPE_STORE,
        EMPTY_SIGNATURE,
        RULES,
        constraints={(): constraint},
    )

    assert assignment[()].payload != constraint.payload


def test_binder_and_bound_leaf_annotations_constrain_their_types() -> None:
    term = unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 0))
    assignment, constraints = generate(
        term,
        TYPE_STORE,
        EMPTY_SIGNATURE,
        RULES,
        annotations={(0,): ENTITY, (): ENTITY},
    )
    inferred, residual = solve(assignment, constraints)

    assert not residual
    assert inferred[()] is arrow(ENTITY, ENTITY)
    assert inferred[(0,)] is ENTITY
    assert tuple(constraint.path for constraint in constraints[-2:]) == ((), (0,))


def test_annotation_atoms_are_not_reissued_as_fresh_variables() -> None:
    annotation = TYPE_STORE.fvar("Ty", "rigid-annotation")
    term = LF_STORE.fvar(LANGUAGE.term_sort, "fresh-around-annotation")
    assignment, _ = generate(
        term,
        TYPE_STORE,
        EMPTY_SIGNATURE,
        RULES,
        annotations={(): annotation},
    )

    assert assignment[()].payload != annotation.payload


def test_generation_rejects_annotations_outside_the_supported_boundary() -> None:
    proposition = LF_STORE.fvar(LANGUAGE.term_sort, "unsupported-annotation")

    with pytest.raises(IndexError):
        generate(
            proposition,
            TYPE_STORE,
            EMPTY_SIGNATURE,
            RULES,
            annotations={(0,): ENTITY},
        )

    with pytest.raises(ValueError, match="does not address a binder or leaf"):
        generate(
            unary("not", proposition),
            TYPE_STORE,
            EMPTY_SIGNATURE,
            RULES,
            annotations={(): TRUTH},
        )

    foreign = TermStore(LANGUAGE.store.signature)

    with pytest.raises(ValueError, match="outside the type store lineage"):
        generate(
            proposition,
            TYPE_STORE,
            EMPTY_SIGNATURE,
            RULES,
            annotations={(): foreign.const(LANGUAGE.type_sort, "e")},
        )


def test_generation_rejects_injected_constraints_outside_the_term_or_store() -> None:
    proposition = LF_STORE.fvar(LANGUAGE.term_sort, "unsupported-injection")

    with pytest.raises(IndexError):
        generate(
            proposition,
            TYPE_STORE,
            EMPTY_SIGNATURE,
            RULES,
            constraints={(0,): ENTITY},
        )

    foreign = TermStore(LANGUAGE.store.signature)

    with pytest.raises(ValueError, match="outside the type store lineage"):
        generate(
            proposition,
            TYPE_STORE,
            EMPTY_SIGNATURE,
            RULES,
            constraints={(): foreign.const(LANGUAGE.type_sort, "e")},
        )


def test_generation_refuses_open_terms_and_inconsistent_rules() -> None:
    with pytest.raises(NotLocallyClosedError):
        generate(
            LF_STORE.bvar(LANGUAGE.term_sort, 0),
            TYPE_STORE,
            EMPTY_SIGNATURE,
            RULES,
        )

    free = LF_STORE.fvar(LANGUAGE.term_sort, "malformed-rule-free")
    malformed = dict(RULES)
    malformed[LANGUAGE.calculus.app] = TypingRule(
        LANGUAGE.calculus.lam,
        (None, None),
        (T, T),
        T,
    )

    with pytest.raises(ValueError, match="maps 'eval' to rule 'lam'"):
        generate(unary("not", free), TYPE_STORE, LANGUAGE.lexicon, malformed)


def test_generation_refuses_foreign_schemes_and_missing_declarations() -> None:
    undeclared = LANGUAGE.constant("undeclared-symbol")
    symbol = cast("int", undeclared.payload)

    with pytest.raises(UnknownSymbolError) as caught:
        generate(undeclared, TYPE_STORE, EMPTY_SIGNATURE, RULES)

    assert caught.value.symbol == symbol
    assert caught.value.term is undeclared
    assert caught.value.path == ()

    foreign_store = TermStore(LANGUAGE.store.signature)
    foreign_type = foreign_store.const(LANGUAGE.type_sort, "e")
    payload = cast("int", undeclared.payload)

    with pytest.raises(ValueError, match="outside the unifier's store lineage"):
        generate(
            undeclared,
            TYPE_STORE,
            LexicalSignature({payload: Scheme(foreign_type)}),
            RULES,
        )


def nested_conjunction(
    leaves: tuple[Term, ...],
) -> Term:
    """Build a right-branching conjunction."""
    head, *tail = leaves

    if not tail:
        return head

    return binary("and", head, nested_conjunction(tuple(tail)))


def free_paths(
    term: Term,
    path: tuple[int, ...] = (),
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return free-variable atoms and paths."""
    if term.op.payload_sort == "free":
        payload = term.payload

        assert payload is not None

        return ((payload, path),)

    return tuple(
        pair
        for position, child in enumerate(term.args)
        for pair in free_paths(child, (*path, position))
    )


def free_type_atoms(type_term: Term) -> frozenset[int]:
    """Collect the free atoms in a type term."""
    return frozenset(
        atom
        for atom in range(type_term.free.bit_length())
        if type_term.free & 1 << atom
    )


@given(
    count=st.integers(min_value=2, max_value=12),
    salt=st.integers(min_value=0, max_value=1_000_000),
)
@settings(max_examples=200, deadline=None)
def test_each_free_atom_receives_one_fresh_variable_per_run(
    count: int,
    salt: int,
) -> None:
    leaves = tuple(
        LF_STORE.fvar(LANGUAGE.term_sort, f"generated-{salt}-{index}")
        for index in range(count)
    )
    term = nested_conjunction((*leaves, leaves[0]))
    assignment, _ = generate(term, TYPE_STORE, LANGUAGE.lexicon, RULES)
    by_atom: dict[int, Term] = {}

    for atom, path in free_paths(term):
        inferred = assignment[path]
        known = by_atom.get(atom)

        if known is None:
            by_atom[atom] = inferred
        else:
            assert inferred is known

    assert len(by_atom) == count
    assert len(set(by_atom.values())) == count


@given(count=st.integers(min_value=2, max_value=12))
@settings(max_examples=200, deadline=None)
def test_rule_instances_share_one_fresh_supply(count: int) -> None:
    signature: dict[int, Scheme] = {}
    argument = symbol("fresh-argument", ENTITY, signature)
    leaves = tuple(
        apply(
            unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 0)),
            argument,
        )
        for _ in range(count)
    )
    assignment, constraints = generate(
        nested_conjunction(leaves),
        TYPE_STORE,
        lexicon(LANGUAGE, signature),
        RULES,
    )
    atoms = {
        atom
        for type_term in (
            *assignment.values(),
            *(constraint.left for constraint in constraints),
            *(constraint.right for constraint in constraints),
        )
        for atom in free_type_atoms(type_term)
    }

    assert len(atoms) == 8 * count - 4


@given(
    domain=st.sampled_from((ENTITY, EVENT, TRUTH, arrow(ENTITY, TRUTH))),
    quantifier=st.sampled_from(("exists", "forall")),
)
@settings(max_examples=200, deadline=None)
def test_quantifiers_mint_inferable_metavariables(
    domain: Term,
    quantifier: str,
) -> None:
    signature: dict[int, Scheme] = {}
    argument = symbol("quantifier", domain, signature)
    body = apply(LF_STORE.bvar(LANGUAGE.term_sort, 0), argument)
    term = unary(quantifier, body)
    generated, constraints = generate(
        term,
        TYPE_STORE,
        lexicon(LANGUAGE, signature),
        RULES,
    )
    assignment, residual = solve(generated, constraints)

    assert generated[(1, 0, 0)].op.name == "#fvar[Ty]"
    assert not residual
    assert assignment[(1, 0, 0)] is arrow(domain, TRUTH)
    assert assignment[()] is TRUTH
