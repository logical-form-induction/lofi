"""Hypothesis strategies for ground-typed logical-form declarations."""

from functools import cache
from typing import Final

from hypothesis import strategies as st
from tests.conformance.lf._naive_check import (
    And,
    Apply,
    AtomicType,
    Bound,
    Constant,
    ConstantDeclaration,
    Exists,
    Forall,
    FunctionType,
    GroundType,
    Lambda,
    Not,
    Or,
    SignatureDeclaration,
    TermDeclaration,
)

__all__ = [
    "BINARY_RELATION",
    "ENTITY",
    "ENTITY_FUNCTION",
    "ENTITY_PREDICATE",
    "GENERALIZED_QUANTIFIER",
    "SIGNATURE",
    "TRUTH",
    "TYPE_POOL",
    "arbitrary_term",
    "well_typed",
]

ENTITY: Final = AtomicType("e")
TRUTH: Final = AtomicType("t")
ENTITY_PREDICATE: Final = FunctionType(ENTITY, TRUTH)
BINARY_RELATION: Final = FunctionType(ENTITY, ENTITY_PREDICATE)
GENERALIZED_QUANTIFIER: Final = FunctionType(ENTITY_PREDICATE, TRUTH)
ENTITY_FUNCTION: Final = FunctionType(ENTITY, ENTITY)

TYPE_POOL: Final = (
    ENTITY,
    TRUTH,
    ENTITY_PREDICATE,
    BINARY_RELATION,
    GENERALIZED_QUANTIFIER,
    ENTITY_FUNCTION,
)
"""Closed ground-type pool used by LF conformance generators."""

SIGNATURE: Final = SignatureDeclaration(
    (
        ConstantDeclaration("a", ENTITY),
        ConstantDeclaration("true", TRUTH),
        ConstantDeclaration("p", ENTITY_PREDICATE),
        ConstantDeclaration("r", BINARY_RELATION),
        ConstantDeclaration("q", GENERALIZED_QUANTIFIER),
        ConstantDeclaration("f", ENTITY_FUNCTION),
    )
)
"""Small declaration signature covering every type in ``TYPE_POOL``."""


def _validate_depth(depth: int) -> None:
    """Reject a negative generation depth."""
    if depth < 0:
        msg = f"depth must be nonnegative, received {depth}"
        raise ValueError(msg)


@cache
def well_typed(
    sig: SignatureDeclaration,
    goal: GroundType,
    env: tuple[GroundType, ...] = (),
    depth: int = 3,
) -> st.SearchStrategy[TermDeclaration]:
    """Generate declarations with a requested ground type."""
    _validate_depth(depth)

    if goal not in TYPE_POOL:
        msg = f"goal is outside the closed type pool: {goal!r}"
        raise ValueError(msg)

    leaves: list[st.SearchStrategy[TermDeclaration]] = [
        st.just(Bound(index)) for index, held in enumerate(env) if held == goal
    ]
    leaves.extend(
        st.just(Constant(declaration.name))
        for declaration in sig.constants
        if declaration.type == goal
    )

    if depth == 0:
        return st.one_of(*leaves)

    branches = list(leaves)
    shallower = depth - 1

    if isinstance(goal, FunctionType):
        branches.append(
            well_typed(
                sig,
                goal.codomain,
                (goal.domain, *env),
                shallower,
            ).map(lambda body: Lambda(goal.domain, body))
        )

    if goal == TRUTH:
        truths = well_typed(sig, TRUTH, env, shallower)
        branches.extend(
            (
                truths.map(Not),
                st.tuples(truths, truths).map(lambda terms: And(*terms)),
                st.tuples(truths, truths).map(lambda terms: Or(*terms)),
            )
        )
        branches.extend(
            well_typed(
                sig,
                TRUTH,
                (parameter, *env),
                shallower,
            ).map(lambda body, parameter=parameter: Exists(parameter, body))
            for parameter in TYPE_POOL
        )
        branches.extend(
            well_typed(
                sig,
                TRUTH,
                (parameter, *env),
                shallower,
            ).map(lambda body, parameter=parameter: Forall(parameter, body))
            for parameter in TYPE_POOL
        )

    branches.extend(
        st.tuples(
            well_typed(
                sig,
                function_type,
                env,
                shallower,
            ),
            well_typed(sig, function_type.domain, env, shallower),
        ).map(lambda terms: Apply(*terms))
        for function_type in TYPE_POOL
        if isinstance(function_type, FunctionType) and function_type.codomain == goal
    )

    return st.one_of(*branches)


@cache
def arbitrary_term(
    sig: SignatureDeclaration = SIGNATURE,
    env: tuple[GroundType, ...] = (),
    depth: int = 3,
) -> st.SearchStrategy[TermDeclaration]:
    """Generate well-scoped declarations without enforcing their types."""
    _validate_depth(depth)
    constants = tuple(Constant(each.name) for each in sig.constants)
    variables = tuple(Bound(index) for index in range(len(env)))
    leaves = (*constants, *variables)

    if not leaves:
        msg = "an arbitrary-term signature or environment must provide a leaf"
        raise ValueError(msg)

    leaf_strategy: st.SearchStrategy[TermDeclaration] = st.sampled_from(leaves)

    if depth == 0:
        return leaf_strategy

    shallower = depth - 1
    inner = arbitrary_term(sig, env, shallower)
    binary = st.tuples(inner, inner)
    branches: list[st.SearchStrategy[TermDeclaration]] = [
        binary.map(lambda terms: Apply(*terms)),
        leaf_strategy,
        inner.map(Not),
        binary.map(lambda terms: And(*terms)),
        binary.map(lambda terms: Or(*terms)),
    ]
    branches.extend(
        arbitrary_term(
            sig,
            (parameter, *env),
            shallower,
        ).map(lambda body, parameter=parameter: Lambda(parameter, body))
        for parameter in TYPE_POOL
    )
    branches.extend(
        arbitrary_term(
            sig,
            (parameter, *env),
            shallower,
        ).map(lambda body, parameter=parameter: Exists(parameter, body))
        for parameter in TYPE_POOL
    )
    branches.extend(
        arbitrary_term(
            sig,
            (parameter, *env),
            shallower,
        ).map(lambda body, parameter=parameter: Forall(parameter, body))
        for parameter in TYPE_POOL
    )

    return st.one_of(*branches)
