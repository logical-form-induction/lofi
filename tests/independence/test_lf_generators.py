from ast import Import, ImportFrom, parse, walk
from pathlib import Path
from typing import Final

import pytest
from hypothesis import find, given, seed, settings
from hypothesis import strategies as st
from tests.conformance.lf._naive_check import (
    And,
    Apply,
    Bound,
    Constant,
    ConstantDeclaration,
    Exists,
    Forall,
    FunctionType,
    GroundType,
    IllTypedError,
    Lambda,
    Not,
    Or,
    SignatureDeclaration,
    TermDeclaration,
    UnknownConstantError,
    type_of,
)
from tests.conformance.lf.strategies import (
    ENTITY,
    ENTITY_PREDICATE,
    SIGNATURE,
    TRUTH,
    TYPE_POOL,
    arbitrary_term,
    well_typed,
)

EXAMPLES: Final = 1_000
SEED: Final = 20260912

TYPED_DECLARATIONS: Final = st.sampled_from(TYPE_POOL).flatmap(
    lambda goal: st.tuples(st.just(goal), well_typed(SIGNATURE, goal, depth=3))
)


@seed(SEED)
@settings(max_examples=EXAMPLES, deadline=None, database=None)
@given(case=TYPED_DECLARATIONS)
def test_a_thousand_type_directed_draws_pass_the_independent_checker(
    case: tuple[GroundType, TermDeclaration],
) -> None:
    goal, term = case

    assert type_of(SIGNATURE, term) == goal


def test_arbitrary_draws_reject_at_least_one_fifth() -> None:
    checked = 0
    rejected = 0

    @seed(SEED)
    @settings(max_examples=EXAMPLES, deadline=None, database=None)
    @given(term=arbitrary_term(SIGNATURE, depth=3))
    def check(term: TermDeclaration) -> None:
        nonlocal checked, rejected
        checked += 1

        try:
            type_of(SIGNATURE, term)
        except IllTypedError:
            rejected += 1

    check()

    assert checked == EXAMPLES
    assert rejected / checked >= 0.2


def test_the_closed_pool_and_environment_are_honored() -> None:
    assert len(TYPE_POOL) == 6
    assert len(set(TYPE_POOL)) == 6

    @seed(SEED)
    @settings(max_examples=100, deadline=None, database=None)
    @given(
        term=well_typed(
            sig=SIGNATURE,
            goal=ENTITY,
            env=(TRUTH, ENTITY),
            depth=2,
        )
    )
    def check(term: TermDeclaration) -> None:
        assert type_of(SIGNATURE, term, (TRUTH, ENTITY)) == ENTITY

    check()


def test_depth_zero_uses_the_requested_environment_and_signature() -> None:
    empty = SignatureDeclaration(())
    bound = find(
        well_typed(empty, ENTITY, env=(TRUTH, ENTITY), depth=0),
        lambda _: True,
    )
    constant = find(
        well_typed(SIGNATURE, ENTITY, env=(), depth=0),
        lambda _: True,
    )

    assert bound == Bound(1)
    assert constant == Constant("a")


def test_a_sparse_signature_generates_a_vacuous_typed_lambda() -> None:
    sparse = SignatureDeclaration((ConstantDeclaration("true", TRUTH),))
    goal = FunctionType(ENTITY, TRUTH)
    term = find(
        well_typed(sparse, goal, env=(), depth=1),
        lambda candidate: isinstance(candidate, Lambda),
    )

    assert term == Lambda(ENTITY, Constant("true"))
    assert type_of(sparse, term) == goal


def test_the_generators_reject_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        well_typed(SIGNATURE, TRUTH, depth=-1)

    with pytest.raises(ValueError, match="outside the closed type pool"):
        well_typed(SIGNATURE, FunctionType(TRUTH, TRUTH), depth=1)


def test_the_naive_checker_handles_each_typing_rule_recursively() -> None:
    predicate_call = Apply(Constant("p"), Constant("a"))
    body = And(predicate_call, Not(Constant("true")))
    quantified = Exists(ENTITY, body)
    universal = Forall(ENTITY, Or(predicate_call, Constant("true")))
    identity = Lambda(ENTITY, Bound(0))

    assert type_of(SIGNATURE, predicate_call) == TRUTH
    assert type_of(SIGNATURE, quantified) == TRUTH
    assert type_of(SIGNATURE, universal) == TRUTH
    assert type_of(SIGNATURE, identity) == FunctionType(ENTITY, ENTITY)
    assert type_of(SIGNATURE, Lambda(ENTITY, predicate_call)) == FunctionType(
        ENTITY, TRUTH
    )


def test_the_naive_checker_rejects_each_ground_type_mismatch() -> None:
    with pytest.raises(IllTypedError, match="nonfunction"):
        type_of(SIGNATURE, Apply(Constant("a"), Constant("a")))

    with pytest.raises(IllTypedError, match="expected"):
        type_of(SIGNATURE, Apply(Constant("p"), Constant("true")))

    with pytest.raises(IllTypedError, match="expected"):
        type_of(SIGNATURE, Not(Constant("a")))

    with pytest.raises(IllTypedError, match="expected"):
        type_of(SIGNATURE, Exists(ENTITY_PREDICATE, Constant("p")))


def test_the_naive_checker_rejects_unknown_constants_and_open_indices() -> None:
    with pytest.raises(UnknownConstantError, match="missing"):
        type_of(SIGNATURE, Constant("missing"))

    with pytest.raises(IllTypedError, match="outside"):
        type_of(SIGNATURE, Bound(0))

    with pytest.raises(IllTypedError, match="outside"):
        type_of(SIGNATURE, Bound(-1), (ENTITY,))


def test_the_naive_checker_has_no_runtime_or_substrate_dependency() -> None:
    source = (
        Path(__file__).parents[1] / "conformance" / "lf" / "_naive_check.py"
    ).read_text(encoding="utf-8")
    imports: set[str] = set()

    for node in walk(parse(source)):
        if isinstance(node, ImportFrom) and node.module is not None:
            imports.add(node.module.partition(".")[0])
        elif isinstance(node, Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)

    assert imports == {"__future__", "typing"}
