from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from lofi.judge import HOLE, GeneralizedSubstitution, Rule
from lofi.spec import SubstitutionSpec
from lofi.spec._rule_laws import LawGrammar, LawView
from lofi.terms import (
    MVar,
    Node,
    Spine,
    Term,
    at_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.terms import Functors, OpName, Pattern

type SlashRole = Literal["forward", "backward"]
"""Distinguished slash role."""

type Mismatch = Literal["slash", "argument"]
"""Parallel-frame mismatch kind."""

GRAMMARS: Final = {
    view: LawGrammar(view) for view in ("direct", "swapped", "undirected")
}
VIEWS: Final[tuple[LawView, ...]] = ("direct", "swapped", "undirected")
DIRECTED: Final[tuple[LawView, ...]] = ("direct", "swapped")
DIRECT_GRAMMAR: Final = GRAMMARS["direct"]


def _substitute(
    grammar: LawGrammar,
    role: SlashRole,
    order: int | None,
    *,
    crossed: bool = False,
) -> GeneralizedSubstitution:
    """Compile one substitution rule over a law grammar."""
    rule = grammar.rule(
        SubstitutionSpec(view=grammar.view_name(role), order=order, crossed=crossed)
    )

    assert type(rule) is GeneralizedSubstitution

    return rule


class SubstitutionCase(NamedTuple):
    """Terms for one substitution equation."""

    functor_term: Term
    container: Term
    output: Term
    inputs: tuple[Term, Term]


def _frame_slashes(
    functors: Functors[Term],
    slash: OpName,
    order: int,
    *,
    crossed: bool,
    flip_outer: bool,
) -> tuple[OpName, ...]:
    """Construct shared slashes with the required deepest frame."""
    deepest = functors.opposite(slash) if crossed else slash
    outer = functors.opposite(deepest) if flip_outer else deepest

    return (*(outer for _ in range(order - 1)), deepest)


def _spine(
    grammar: LawGrammar,
    foot: Term,
    slashes: Sequence[OpName],
    arguments: Sequence[str] = ("z0", "z1"),
) -> Term:
    """Construct a result spine around a foot."""
    functors, _ = grammar.functor("forward")
    current = foot

    for slash, argument in reversed(tuple(zip(slashes, arguments, strict=True))):
        current = functors.make(slash, current, grammar.atom(argument))

    return current


def _case(
    grammar: LawGrammar,
    role: SlashRole,
    order: int,
    *,
    crossed: bool,
    flip_outer: bool,
) -> SubstitutionCase:
    """Construct one valid substitution equation."""
    functors, slash = grammar.functor(role)
    slashes = _frame_slashes(
        functors, slash, order, crossed=crossed, flip_outer=flip_outer
    )
    principal = functors.make(slash, grammar.atom("x"), grammar.atom("y"))
    arguments = tuple(("z0", "z1")[at % 2] for at in range(order))
    functor_term = _spine(grammar, principal, slashes, arguments)
    container = _spine(grammar, grammar.atom("y"), slashes, arguments)
    output = _spine(grammar, grammar.atom("x"), slashes, arguments)
    inputs = (
        (functor_term, container) if role == "forward" else (container, functor_term)
    )

    return SubstitutionCase(functor_term, container, output, inputs)


def _assert_off_spine_identity(
    functors: Functors[Term],
    source: Term,
    rebuilt: Term,
    order: int,
) -> None:
    """Check off-spine argument identity."""
    source_frame = source
    rebuilt_frame = rebuilt

    for _ in range(order):
        assert at_path(source_frame, functors.argument_path) == at_path(
            rebuilt_frame, functors.argument_path
        )
        source_frame = at_path(source_frame, functors.result_path)
        rebuilt_frame = at_path(rebuilt_frame, functors.result_path)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("crossed", [False, True], ids=("harmonic", "crossed"))
@seed(20260912)
@settings(max_examples=2, deadline=None, database=None)
@given(flip_outer=st.booleans())
def test_the_substitution_equations_cover_all_twenty_four_variants(
    view: LawView,
    role: SlashRole,
    order: int,
    *,
    crossed: bool,
    flip_outer: bool,
) -> None:
    grammar = GRAMMARS[view]
    functors, _ = grammar.functor(role)
    case = _case(grammar, role, order, crossed=crossed, flip_outer=flip_outer)
    rule = _substitute(grammar, role, order, crossed=crossed)
    outputs = tuple(rule.apply(case.inputs))
    expected_slots = (
        frozenset()
        if view == "undirected"
        else frozenset({1 if role == "forward" else 0})
    )

    assert isinstance(rule, Rule)
    assert rule.probe.alternatives == ((functors.slashes, functors.slashes),)
    assert outputs == (case.output,)
    assert rule.admits(case.inputs, case.output)
    assert not rule.admits(case.inputs, grammar.atom("other"))
    assert rule.residual_slots == expected_slots
    assert ("residual" in rule.directions) == bool(expected_slots)
    _assert_off_spine_identity(functors, case.container, outputs[0], order)

    for slot in range(rule.arity):
        known = tuple(
            HOLE if at == slot else term for at, term in enumerate(case.inputs)
        )
        residuals = tuple(rule.residuate(case.output, known))

        if slot in expected_slots:
            assert residuals == (case.inputs[slot],)
            _assert_off_spine_identity(functors, case.output, residuals[0], order)

        else:
            assert residuals == ()


@pytest.mark.parametrize("view", DIRECTED)
@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("crossed", [False, True], ids=("harmonic", "crossed"))
def test_the_deepest_frame_controls_crossing(
    view: LawView,
    role: SlashRole,
    *,
    crossed: bool,
) -> None:
    grammar = GRAMMARS[view]
    functors, slash = grammar.functor(role)
    required = functors.opposite(slash) if crossed else slash
    wrong = functors.opposite(required)
    principal = functors.make(slash, grammar.atom("x"), grammar.atom("y"))
    functor_term = _spine(grammar, principal, (required, wrong))
    container = _spine(grammar, grammar.atom("y"), (required, wrong))
    inputs = (
        (functor_term, container) if role == "forward" else (container, functor_term)
    )
    rule = _substitute(grammar, role, 2, crossed=crossed)

    assert tuple(rule.apply(inputs)) == ()


@pytest.mark.parametrize(
    ("view", "mismatch"),
    [
        *((view, mismatch) for view in DIRECTED for mismatch in ("slash", "argument")),
        ("undirected", "argument"),
    ],
)
@pytest.mark.parametrize("role", ["forward", "backward"])
def test_parallel_frame_mismatches_are_rejected(
    view: LawView,
    role: SlashRole,
    mismatch: Mismatch,
) -> None:
    grammar = GRAMMARS[view]
    functors, slash = grammar.functor(role)
    slashes = _frame_slashes(functors, slash, 2, crossed=False, flip_outer=False)
    principal = functors.make(slash, grammar.atom("x"), grammar.atom("y"))
    functor_term = _spine(grammar, principal, slashes)
    container_slashes = (
        (functors.opposite(slashes[0]), slashes[1]) if mismatch == "slash" else slashes
    )
    container_arguments = ("other", "z1") if mismatch == "argument" else ("z0", "z1")
    container = _spine(
        grammar, grammar.atom("y"), container_slashes, container_arguments
    )
    inputs = (
        (functor_term, container) if role == "forward" else (container, functor_term)
    )
    rule = _substitute(grammar, role, 2)

    assert tuple(rule.apply(inputs)) == ()


def test_malformed_and_unsupported_queries_return_no_values() -> None:
    rule = _substitute(DIRECT_GRAMMAR, "forward", 1)
    case = _case(DIRECT_GRAMMAR, "forward", 1, crossed=False, flip_outer=False)

    assert tuple(rule.apply(case.inputs[:1])) == ()
    assert not rule.admits(case.inputs[:1], case.output)
    assert tuple(rule.residuate(case.output, (HOLE,))) == ()
    assert tuple(rule.residuate(case.output, (HOLE, HOLE))) == ()
    assert tuple(rule.residuate(case.output, case.inputs)) == ()
    assert tuple(rule.residuate(case.output, (HOLE, case.container))) == ()


def test_an_unbounded_substitution_rule_admits_orders_above_three() -> None:
    case = _case(DIRECT_GRAMMAR, "forward", 4, crossed=False, flip_outer=True)
    rule = _substitute(DIRECT_GRAMMAR, "forward", None)

    assert rule.name == "substitution:fwd:any:harmonic"
    assert tuple(rule.apply(case.inputs)) == (case.output,)
    assert tuple(rule.residuate(case.output, (case.functor_term, HOLE))) == (
        case.container,
    )


def test_the_rule_is_frozen_slotted_and_declarative() -> None:
    functors, slash = DIRECT_GRAMMAR.functor("forward")
    rule = _substitute(DIRECT_GRAMMAR, "forward", 1)

    assert rule.functors.slashes == functors.slashes
    assert rule.matching.name == "identity"
    assert rule.order == 1
    assert rule.slash == slash
    assert not rule.crossed
    assert rule.name == "substitution:fwd:1:harmonic"
    assert rule.to_spec() == {
        "rule": "substitution",
        "matching": "identity",
        "order": 1,
        "slash": "fwd",
        "crossed": False,
        "restriction": None,
    }

    with pytest.raises(TypeError):
        vars(rule)

    with pytest.raises(AttributeError, match="immutable"):
        rule.__setattr__("_order", 2)


def _categories(grammar: LawGrammar, depth: int) -> st.SearchStrategy[Term]:
    functors, _ = grammar.functor("forward")
    atoms = st.sampled_from(
        tuple(grammar.atom(name) for name in ("x", "y", "z0", "z1"))
    )

    if depth == 0:
        return atoms

    smaller = _categories(grammar, depth - 1)

    return st.one_of(
        atoms,
        st.tuples(st.sampled_from(sorted(functors.slashes)), smaller, smaller).map(
            lambda parts: functors.make(*parts)
        ),
    )


def _functor_premises(grammar: LawGrammar) -> st.SearchStrategy[Term]:
    """Draw premises whose foot is a functor over ``x``."""
    functors, _ = grammar.functor("forward")
    slashes = st.sampled_from(sorted(functors.slashes))
    return st.tuples(
        slashes,
        st.sampled_from((grammar.atom("x"), grammar.atom("y"))),
        slashes,
        st.sampled_from((grammar.atom("z0"), grammar.atom("z1"))),
    ).map(
        lambda parts: functors.make(
            parts[2], functors.make(parts[0], grammar.atom("x"), parts[1]), parts[3]
        )
    )


def _container_premises(grammar: LawGrammar) -> st.SearchStrategy[Term]:
    """Draw premises with one spine frame over an atomic foot."""
    functors, _ = grammar.functor("forward")
    return st.tuples(
        st.sampled_from(sorted(functors.slashes)),
        st.sampled_from((grammar.atom("x"), grammar.atom("y"))),
        st.sampled_from((grammar.atom("z0"), grammar.atom("z1"))),
    ).map(lambda parts: functors.make(*parts))


@settings(max_examples=300, deadline=None)
@given(
    data=st.data(),
    view=st.sampled_from(VIEWS),
    role=st.sampled_from(("forward", "backward")),
    order=st.sampled_from((1, 2, None)),
    crossed=st.booleans(),
)
def test_every_licensed_pair_shares_a_join_key(
    *,
    data: st.DataObject,
    view: LawView,
    role: SlashRole,
    order: int | None,
    crossed: bool,
) -> None:
    grammar = GRAMMARS[view]
    left = data.draw(st.one_of(_categories(grammar, 3), _functor_premises(grammar)))
    right = data.draw(st.one_of(_categories(grammar, 3), _container_premises(grammar)))
    rule = _substitute(grammar, role, order, crossed=crossed)
    shared = set(rule.join_keys(left, 0)) & set(rule.join_keys(right, 1))

    assert not tuple(rule.apply((left, right))) or shared


def test_join_keys_pair_the_foot_argument_with_the_container_foot() -> None:
    functors, _ = DIRECT_GRAMMAR.functor("forward")
    x = DIRECT_GRAMMAR.atom("x")
    y = DIRECT_GRAMMAR.atom("y")
    z0 = DIRECT_GRAMMAR.atom("z0")
    functor_term = functors.make("fwd", functors.make("fwd", x, y), z0)
    container = functors.make("fwd", y, z0)
    rule = _substitute(DIRECT_GRAMMAR, "forward", 1)
    (key,) = rule.join_keys(functor_term, 0)

    assert tuple(rule.apply((functor_term, container)))
    assert rule.join_keys(container, 1) == (key,)
    assert isinstance(key, tuple)
    assert y in key
    assert rule.join_keys(x, 1) == ()


def _metavariable_sorts(pattern: Pattern) -> frozenset[str]:
    """Return the sorts of every metavariable in one pattern."""
    match pattern:
        case MVar():
            return frozenset({pattern.sort})
        case Node():
            return frozenset().union(*map(_metavariable_sorts, pattern.args))
        case Spine():
            return _metavariable_sorts(pattern.core)


def test_schema_metavariables_carry_the_functor_category_sort() -> None:
    rule = _substitute(DIRECT_GRAMMAR, "forward", 1)
    functors, _ = DIRECT_GRAMMAR.functor("forward")
    sorts = {
        sort
        for premises, conclusion in rule.reachability_schemata
        for pattern in (*premises, conclusion)
        for sort in _metavariable_sorts(pattern)
    }

    assert sorts == {functors.category_sort}
