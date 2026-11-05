from typing import Final, cast

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from tests.language import language
from tests.support.term_strategies import terms

from lofi.terms import RegionStateError, Term, TermStore
from lofi.typecheck import InfiniteTypeError, Substitution, Unifier

LANGUAGE: Final = language()
STORE: Final = LANGUAGE.store
UNIFIER: Final = Unifier(STORE)
ARROW: Final = STORE.signature.by_name(LANGUAGE.function_type)
ENTITY: Final = LANGUAGE.read_type("e")
EVENT: Final = LANGUAGE.read_type("v")
TRUTH: Final = LANGUAGE.read_type("t")
GROUND: Final = (
    ENTITY,
    EVENT,
    TRUTH,
    STORE.op(ARROW, ENTITY, TRUTH),
    STORE.op(ARROW, EVENT, TRUTH),
)
TYPE_TERMS = terms(STORE, sort=LANGUAGE.type_sort, max_size=9, closed=True)


def _atoms(mask: int) -> tuple[int, ...]:
    """Return the atom indices set in a free-variable mask."""
    return tuple(index for index in range(mask.bit_length()) if mask & (1 << index))


def _ground_instance(
    data: st.DataObject,
    term: Term,
) -> tuple[Term, Substitution]:
    """Build a ground instance of a generated type term."""
    replacement = {
        atom: data.draw(st.sampled_from(GROUND), label=f"atom-{atom}")
        for atom in _atoms(term.free)
    }

    return UNIFIER.apply(replacement, term), replacement


def test_unification_matches_primitives_and_arrows() -> None:
    variable = STORE.fvar(LANGUAGE.type_sort, "A")
    result = STORE.fvar(LANGUAGE.type_sort, "B")
    left = STORE.op(ARROW, variable, TRUTH)
    right = STORE.op(ARROW, ENTITY, result)
    substitution = UNIFIER.unify(left, right)

    assert substitution == {variable.payload: ENTITY, result.payload: TRUTH}
    assert UNIFIER.apply(substitution, left) is UNIFIER.apply(substitution, right)


def test_arrow_argument_order_is_the_declared_term_order() -> None:
    forward = STORE.op(ARROW, ENTITY, TRUTH)
    backward = STORE.op(ARROW, TRUTH, ENTITY)

    assert ARROW.arg_sorts == (LANGUAGE.type_sort, LANGUAGE.type_sort)
    assert forward.args == (ENTITY, TRUTH)
    assert backward.args == (TRUTH, ENTITY)
    assert UNIFIER.unify(forward, backward) is None


def test_unification_returns_none_for_incompatible_constructors() -> None:
    assert UNIFIER.unify(ENTITY, EVENT) is None
    assert UNIFIER.unify(ENTITY, TRUTH) is None


def test_occurs_check_reports_the_recursive_variable_and_type() -> None:
    variable = STORE.fvar(LANGUAGE.type_sort, "recursive")
    recursive = STORE.op(ARROW, variable, TRUTH)

    with pytest.raises(InfiniteTypeError) as refusal:
        UNIFIER.unify(variable, recursive)

    assert refusal.value.variable is variable
    assert refusal.value.term is recursive


def test_occurs_check_follows_existing_variable_bindings() -> None:
    first = STORE.fvar(LANGUAGE.type_sort, "first")
    second = STORE.fvar(LANGUAGE.type_sort, "second")
    first_replacement = STORE.op(ARROW, second, ENTITY)
    second_replacement = STORE.op(ARROW, first, TRUTH)
    left = STORE.op(ARROW, first, second)
    right = STORE.op(ARROW, first_replacement, second_replacement)

    with pytest.raises(InfiniteTypeError) as refusal:
        UNIFIER.unify(left, right)

    assert refusal.value.variable is second
    assert refusal.value.term is second_replacement


def test_unification_normalizes_variable_chains() -> None:
    first = STORE.fvar(LANGUAGE.type_sort, "chain-first")
    second = STORE.fvar(LANGUAGE.type_sort, "chain-second")
    left = STORE.op(ARROW, first, second)
    right = STORE.op(ARROW, second, ENTITY)
    substitution = UNIFIER.unify(left, right)

    assert substitution == {first.payload: ENTITY, second.payload: ENTITY}
    assert substitution is not None
    assert all(
        UNIFIER.apply(substitution, term) is term for term in substitution.values()
    )


def test_subsumption_binds_only_metavariables_in_the_general_type() -> None:
    argument = STORE.fvar(LANGUAGE.type_sort, "argument")
    result = STORE.fvar(LANGUAGE.type_sort, "result")
    general = STORE.op(ARROW, argument, result)
    specific = STORE.op(ARROW, ENTITY, TRUTH)
    substitution = UNIFIER.subsumes(general, specific)

    assert substitution == {argument.payload: ENTITY, result.payload: TRUTH}
    assert UNIFIER.apply(substitution, general) is specific
    assert UNIFIER.subsumes(specific, general) is None
    assert UNIFIER.unify(general, specific) is not None
    assert UNIFIER.unify(specific, general) is not None


def test_subsumption_requires_repeated_metavariables_to_agree() -> None:
    variable = STORE.fvar(LANGUAGE.type_sort, "repeated")
    general = STORE.op(ARROW, variable, variable)

    assert UNIFIER.subsumes(general, STORE.op(ARROW, ENTITY, ENTITY)) is not None
    assert UNIFIER.subsumes(general, STORE.op(ARROW, ENTITY, TRUTH)) is None


def test_specific_metavariables_remain_rigid() -> None:
    variable = STORE.fvar(LANGUAGE.type_sort, "rigid")

    assert UNIFIER.subsumes(ENTITY, variable) is None
    assert UNIFIER.subsumes(variable, ENTITY) == {variable.payload: ENTITY}


def test_specific_metavariables_remain_rigid_when_shared_with_the_pattern() -> None:
    shared = STORE.fvar(LANGUAGE.type_sort, "shared")
    general = STORE.op(ARROW, shared, shared)

    assert UNIFIER.subsumes(general, STORE.op(ARROW, shared, ENTITY)) is None
    assert UNIFIER.subsumes(general, STORE.op(ARROW, ENTITY, shared)) is None

    identity = UNIFIER.subsumes(general, general)

    assert identity == {shared.payload: shared}
    assert UNIFIER.apply(identity, general) is general


def test_specific_metavariables_remain_rigid_inside_shared_compounds() -> None:
    variable = STORE.fvar(LANGUAGE.type_sort, "shared-compound")
    shared = STORE.op(ARROW, variable, ENTITY)
    general = STORE.op(ARROW, shared, variable)
    specific = STORE.op(ARROW, shared, TRUTH)

    assert UNIFIER.subsumes(general, specific) is None


def test_returned_substitutions_are_immutable() -> None:
    variable = STORE.fvar(LANGUAGE.type_sort, "immutable")
    substitution = UNIFIER.unify(variable, ENTITY)
    atom = variable.payload

    assert substitution is not None
    assert atom is not None

    with pytest.raises(TypeError):
        cast("dict[int, Term]", substitution)[atom] = TRUTH


def test_foreign_terms_and_substitution_values_are_refused() -> None:
    foreign_store = TermStore(STORE.signature)
    foreign = foreign_store.const(LANGUAGE.type_sort, "e")

    with pytest.raises(ValueError, match="left"):
        UNIFIER.unify(foreign, ENTITY)

    with pytest.raises(ValueError, match="specific"):
        UNIFIER.subsumes(ENTITY, foreign)

    with pytest.raises(ValueError, match="substitution"):
        UNIFIER.apply({0: foreign}, ENTITY)


def test_ancestor_terms_are_valid_in_a_child_unifier() -> None:
    root = TermStore(STORE.signature)
    entity = root.const(LANGUAGE.type_sort, "e")

    with root.region("types", max_nodes=16) as child:
        variable = child.fvar(LANGUAGE.type_sort, "child")
        unifier = Unifier(child)

        assert unifier.unify(variable, entity) == {variable.payload: entity}


def test_released_stores_refuse_unification() -> None:
    root = TermStore(STORE.signature)
    variable: Term | None = None

    with root.region("types", max_nodes=16) as child:
        variable = child.fvar(LANGUAGE.type_sort, "released")

    assert variable is not None

    with pytest.raises(RegionStateError, match="released"):
        Unifier(child).unify(variable, variable)


def test_malformed_substitution_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="substitution atom"):
        UNIFIER.apply({-1: TRUTH}, ENTITY)

    with pytest.raises(TypeError, match="substitution atom"):
        UNIFIER.apply(
            cast("dict[int, Term]", {True: TRUTH}),
            ENTITY,
        )


@given(data=st.data(), source=TYPE_TERMS)
@settings(max_examples=200, deadline=None)
def test_unification_substitutions_are_idempotent(
    data: st.DataObject,
    source: Term,
) -> None:
    assume(source.free)
    instance, _ = _ground_instance(data, source)
    substitution = UNIFIER.unify(source, instance)

    assert substitution is not None
    once = UNIFIER.apply(substitution, source)
    twice = UNIFIER.apply(substitution, once)

    assert twice is once

    for replacement in substitution.values():
        assert UNIFIER.apply(substitution, replacement) is replacement


@given(data=st.data(), source=TYPE_TERMS)
@settings(max_examples=200, deadline=None)
def test_unification_substitution_equalizes_both_terms(
    data: st.DataObject,
    source: Term,
) -> None:
    assume(source.free)
    instance, _ = _ground_instance(data, source)
    substitution = UNIFIER.unify(source, instance)

    assert substitution is not None
    assert UNIFIER.apply(substitution, source) is UNIFIER.apply(
        substitution,
        instance,
    )


@given(
    atom=st.integers(min_value=0, max_value=100_000),
    argument=st.sampled_from(GROUND),
    result=st.sampled_from(GROUND),
)
@settings(max_examples=200, deadline=None)
def test_subsumption_is_asymmetric_where_unification_is_symmetric(
    atom: int,
    argument: Term,
    result: Term,
) -> None:
    left = STORE.fvar(LANGUAGE.type_sort, f"left-{atom}")
    right = STORE.fvar(LANGUAGE.type_sort, f"right-{atom}")
    general = STORE.op(ARROW, left, right)
    specific = STORE.op(ARROW, argument, result)
    substitution = UNIFIER.subsumes(general, specific)

    assert substitution is not None
    assert UNIFIER.apply(substitution, general) is specific
    assert UNIFIER.subsumes(specific, general) is None
    assert UNIFIER.unify(general, specific) is not None
    assert UNIFIER.unify(specific, general) is not None


@given(
    atom=st.integers(min_value=0, max_value=100_000),
    ground=st.sampled_from(GROUND),
)
@settings(max_examples=200, deadline=None)
def test_variable_mgu_factors_more_specific_unifiers(
    atom: int,
    ground: Term,
) -> None:
    first = STORE.fvar(LANGUAGE.type_sort, f"factor-first-{atom}")
    second = STORE.fvar(LANGUAGE.type_sort, f"factor-second-{atom}")
    mgu = UNIFIER.unify(first, second)

    assert mgu is not None
    assert first.payload is not None
    assert second.payload is not None

    representative = UNIFIER.apply(mgu, first)

    assert representative is UNIFIER.apply(mgu, second)
    assert representative.payload is not None

    factor = {representative.payload: ground}
    specific = {first.payload: ground, second.payload: ground}

    assert UNIFIER.apply(factor, UNIFIER.apply(mgu, first)) is UNIFIER.apply(
        specific, first
    )
    assert UNIFIER.apply(factor, UNIFIER.apply(mgu, second)) is UNIFIER.apply(
        specific, second
    )
