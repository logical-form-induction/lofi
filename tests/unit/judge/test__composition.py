from itertools import product
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from lofi.judge import (
    COMPOSITION_RESTRICTIONS,
    HOLE,
    GeneralizedComposition,
    NoNormalForm,
    Rule,
    RuleSet,
)
from lofi.spec import ApplicationSpec, CompositionSpec, RuleRestrictionSpec
from lofi.spec._rule_laws import LawGrammar, LawView
from lofi.spec.grammar import FeatureErasureMatchingSpec
from lofi.terms import (
    Identity,
    MVar,
    Node,
    Spine,
    Term,
    at_path,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge import CompositionRestriction
    from lofi.spec.grammar import MatchingSpec
    from lofi.terms import Functors, OpName, Pattern

type SlashRole = Literal["forward", "backward"]
"""Distinguished slash role."""

GRAMMARS: Final = {
    view: LawGrammar(view) for view in ("direct", "swapped", "undirected")
}
VIEWS: Final[tuple[LawView, ...]] = ("direct", "swapped", "undirected")
DIRECT_GRAMMAR: Final = GRAMMARS["direct"]
DIRECT: Final = DIRECT_GRAMMAR.functor("forward")[0]
X: Final = DIRECT_GRAMMAR.atom("x")
Y: Final = DIRECT_GRAMMAR.atom("y")
ARGUMENTS: Final = (DIRECT_GRAMMAR.atom("z0"), DIRECT_GRAMMAR.atom("z1"))
OTHER: Final = DIRECT_GRAMMAR.atom("other")


def _compose(
    grammar: LawGrammar,
    role: SlashRole,
    order: int | None,
    direction: CompositionRestriction = "kuhlmann",
    *,
    crossed: bool = False,
    matching: MatchingSpec | None = None,
    restriction: RuleRestrictionSpec | None = None,
) -> GeneralizedComposition:
    """Compile one application or composition rule over a law grammar."""
    view = grammar.view_name(role)
    declaration = (
        ApplicationSpec(view=view, restriction=restriction)
        if order == 0
        else CompositionSpec(
            view=view,
            order=order,
            direction=direction,
            crossed=crossed,
            restriction=restriction,
        )
    )
    (rule,) = grammar.rules((declaration,), matching=matching)

    assert type(rule) is GeneralizedComposition

    return rule


def _rebuild(grammar: LawGrammar, spec: Mapping[str, JsonValue]) -> Rule[Term]:
    """Compile the rule a composition declaration names."""
    order = spec["order"]
    slash = spec["slash"]
    direction = next(
        known for known in COMPOSITION_RESTRICTIONS if known == spec["direction"]
    )

    assert isinstance(order, int | None)
    assert isinstance(slash, str)
    assert isinstance(spec["crossed"], bool)

    return grammar.rule(
        CompositionSpec(
            view=grammar.view_of(slash),
            order=order,
            direction=direction,
            crossed=spec["crossed"],
        )
    )


class CompositionCase(NamedTuple):
    """Terms for one composition equation."""

    principal: Term
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
    """Construct a spine whose deepest frame satisfies the rule."""
    if order == 0:
        return ()

    deepest = functors.opposite(slash) if crossed else slash
    outer = functors.opposite(deepest) if flip_outer else deepest

    return (*(outer for _ in range(order - 1)), deepest)


def _spine(
    grammar: LawGrammar,
    foot: Term,
    slashes: Sequence[OpName],
) -> Term:
    """Construct a result spine around a foot."""
    functors, _ = grammar.functor("forward")
    arguments = tuple(grammar.atom(name) for name in ("z0", "z1", "t"))
    current = foot

    for slash, argument in reversed(tuple(zip(slashes, arguments, strict=False))):
        current = functors.make(slash, current, argument)

    return current


def _case(
    grammar: LawGrammar,
    role: SlashRole,
    order: int,
    *,
    crossed: bool,
    flip_outer: bool,
) -> CompositionCase:
    """Construct one valid composition equation."""
    functors, slash = grammar.functor(role)
    slashes = _frame_slashes(
        functors,
        slash,
        order,
        crossed=crossed,
        flip_outer=flip_outer,
    )
    x = grammar.atom("x")
    y = grammar.atom("y")
    principal = functors.make(slash, x, y)
    container = _spine(grammar, y, slashes)
    output = _spine(grammar, x, slashes)
    inputs = (principal, container) if role == "forward" else (container, principal)

    return CompositionCase(principal, container, output, inputs)


def _completed(
    inputs: tuple[Term, ...],
    slot: int,
    candidate: Term,
) -> tuple[Term, ...]:
    """Return the premises with one slot filled by a residual candidate."""
    return tuple(candidate if at == slot else term for at, term in enumerate(inputs))


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
@pytest.mark.parametrize(
    ("order", "crossed"),
    [
        (order, crossed)
        for order in range(4)
        for crossed in (False, True)
        if not (crossed and order == 0)
    ],
    ids=str,
)
@seed(20260912)
@settings(max_examples=2, deadline=None, database=None)
@given(flip_outer=st.booleans())
def test_the_composition_equations_cover_every_declared_variant(
    view: LawView,
    role: SlashRole,
    order: int,
    *,
    crossed: bool,
    flip_outer: bool,
) -> None:
    grammar = GRAMMARS[view]
    functors, slash = grammar.functor(role)
    case = _case(grammar, role, order, crossed=crossed, flip_outer=flip_outer)
    rule = _compose(grammar, role, order, crossed=crossed)
    outputs = tuple(rule.apply(case.inputs))

    assert isinstance(rule, Rule)
    alternatives = rule.probe.alternatives
    slashes = functors.slashes
    expected_probe = (
        ((frozenset({slash}), slashes),)
        if order > 0 and role == "forward"
        else ((slashes, frozenset({slash})),)
        if order > 0
        else (
            (frozenset({slash}), None),
            (None, frozenset({slash})),
        )
        if view == "undirected"
        else ((frozenset({slash}), None),)
        if role == "forward"
        else ((None, frozenset({slash})),)
    )

    assert alternatives == expected_probe
    assert outputs == (case.output,)
    assert rule.admits(case.inputs, case.output)
    assert not rule.admits(case.inputs, grammar.atom("other"))

    if order == 0:
        assert outputs[0] == grammar.atom("x")

    else:
        _assert_off_spine_identity(functors, case.container, outputs[0], order)

    assert rule.residual_slots == frozenset(range(rule.arity))
    assert "residual" in rule.directions
    assert rule.applied_order(case.inputs, case.output) == order

    for slot in range(rule.arity):
        known = tuple(
            HOLE if at == slot else term for at, term in enumerate(case.inputs)
        )
        residuals = tuple(rule.residuate(case.output, known))

        assert any(candidate == case.inputs[slot] for candidate in residuals)
        assert all(
            rule.admits(_completed(case.inputs, slot, candidate), case.output)
            for candidate in residuals
        )


@pytest.mark.parametrize("view", ["direct", "swapped"])
@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("crossed", [False, True], ids=("harmonic", "crossed"))
def test_only_the_deepest_frame_controls_crossing(
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
    container = _spine(grammar, grammar.atom("y"), (required, wrong))
    inputs = (principal, container) if role == "forward" else (container, principal)
    rule = _compose(grammar, role, 2, crossed=crossed)

    assert tuple(rule.apply(inputs)) == ()


def test_malformed_and_unsupported_queries_return_no_values() -> None:
    rule = _compose(DIRECT_GRAMMAR, "forward", 2)
    case = _case(DIRECT_GRAMMAR, "forward", 2, crossed=False, flip_outer=False)

    assert tuple(rule.apply(case.inputs[:1])) == ()
    assert not rule.admits(case.inputs[:1], case.output)
    assert tuple(rule.residuate(case.output, (HOLE,))) == ()
    assert tuple(rule.residuate(case.output, (HOLE, HOLE))) == ()
    assert tuple(rule.residuate(case.output, case.inputs)) == ()
    assert case.principal in tuple(rule.residuate(case.output, (HOLE, case.container)))


def test_the_rule_is_frozen_and_slotted() -> None:
    rule = _compose(DIRECT_GRAMMAR, "forward", 1)

    assert rule.functors.slashes == DIRECT.slashes
    assert rule.matching.name == "identity"
    assert rule.order == 1
    assert rule.slash == DIRECT.forward_slash
    assert not rule.crossed
    assert rule.name == "composition:fwd:1:harmonic:kuhlmann"
    assert (
        _compose(DIRECT_GRAMMAR, "forward", None).name
        == "composition:fwd:any:harmonic:kuhlmann"
    )
    assert _compose(DIRECT_GRAMMAR, "forward", 0).name == "application:fwd"

    with pytest.raises(TypeError):
        vars(rule)

    with pytest.raises(AttributeError, match="immutable"):
        rule.__setattr__("_order", 2)


Z0: Final = ARGUMENTS[0]
Z1: Final = ARGUMENTS[1]


class SpineCase(NamedTuple):
    """A hand-listed premise pair with the restrictions that license it."""

    label: str
    role: SlashRole
    inputs: tuple[Term, Term]
    output: Term
    licensed: frozenset[str]


UNIFORM: Final = frozenset({"steedman", "kuhlmann", "unrestricted"})
DEEPEST_ONLY: Final = frozenset({"kuhlmann", "unrestricted"})
NEITHER: Final = frozenset({"unrestricted"})

SPINE_CASES: Final = (
    SpineCase(
        label="forward-uniform",
        role="forward",
        inputs=(
            DIRECT.make("fwd", X, Y),
            DIRECT.make("fwd", DIRECT.make("fwd", Y, Z1), Z0),
        ),
        output=DIRECT.make("fwd", DIRECT.make("fwd", X, Z1), Z0),
        licensed=UNIFORM,
    ),
    SpineCase(
        label="forward-mixed",
        role="forward",
        inputs=(
            DIRECT.make("fwd", X, Y),
            DIRECT.make("bwd", DIRECT.make("fwd", Y, Z1), Z0),
        ),
        output=DIRECT.make("bwd", DIRECT.make("fwd", X, Z1), Z0),
        licensed=DEEPEST_ONLY,
    ),
    SpineCase(
        label="forward-deepest-flipped",
        role="forward",
        inputs=(
            DIRECT.make("fwd", X, Y),
            DIRECT.make("fwd", DIRECT.make("bwd", Y, Z1), Z0),
        ),
        output=DIRECT.make("fwd", DIRECT.make("bwd", X, Z1), Z0),
        licensed=NEITHER,
    ),
    SpineCase(
        label="backward-uniform",
        role="backward",
        inputs=(
            DIRECT.make("bwd", DIRECT.make("bwd", Y, Z1), Z0),
            DIRECT.make("bwd", X, Y),
        ),
        output=DIRECT.make("bwd", DIRECT.make("bwd", X, Z1), Z0),
        licensed=UNIFORM,
    ),
    SpineCase(
        label="backward-mixed",
        role="backward",
        inputs=(
            DIRECT.make("fwd", DIRECT.make("bwd", Y, Z1), Z0),
            DIRECT.make("bwd", X, Y),
        ),
        output=DIRECT.make("fwd", DIRECT.make("bwd", X, Z1), Z0),
        licensed=DEEPEST_ONLY,
    ),
    SpineCase(
        label="backward-deepest-flipped",
        role="backward",
        inputs=(
            DIRECT.make("bwd", DIRECT.make("fwd", Y, Z1), Z0),
            DIRECT.make("bwd", X, Y),
        ),
        output=DIRECT.make("bwd", DIRECT.make("fwd", X, Z1), Z0),
        licensed=NEITHER,
    ),
)


def _second_order(
    direction: CompositionRestriction, role: SlashRole
) -> GeneralizedComposition:
    """Build one harmonic second-order rule under a direction restriction."""
    return _compose(DIRECT_GRAMMAR, role, 2, direction)


@pytest.mark.parametrize("case", SPINE_CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("direction", COMPOSITION_RESTRICTIONS)
def test_the_direction_restriction_decides_each_hand_listed_triple(
    case: SpineCase,
    direction: CompositionRestriction,
) -> None:
    rule = _second_order(direction, case.role)
    expected = (case.output,) if direction in case.licensed else ()

    assert tuple(rule.apply(case.inputs)) == expected
    assert rule.admits(case.inputs, case.output) == (direction in case.licensed)


@pytest.mark.parametrize("role", ["forward", "backward"])
def test_the_restrictions_are_strictly_nested_on_the_hand_listed_triples(
    role: SlashRole,
) -> None:
    admitted = {
        direction: frozenset(
            case.label
            for case in SPINE_CASES
            if case.role == role
            and _second_order(direction, role).admits(case.inputs, case.output)
        )
        for direction in COMPOSITION_RESTRICTIONS
    }

    assert admitted["steedman"] < admitted["kuhlmann"]
    assert admitted["kuhlmann"] < admitted["unrestricted"]


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("crossed", [False, True], ids=("harmonic", "crossed"))
def test_the_restriction_narrows_the_oriented_proof_schemata(
    role: SlashRole,
    *,
    crossed: bool,
) -> None:
    schemata = {
        direction: frozenset(
            _compose(DIRECT_GRAMMAR, role, 2, direction, crossed=crossed).proof_schemata
        )
        for direction in COMPOSITION_RESTRICTIONS
    }

    assert schemata["steedman"] < schemata["kuhlmann"]
    assert schemata["kuhlmann"] < schemata["unrestricted"]


def _committed_inventory(
    direction: CompositionRestriction,
) -> RuleSet[Term]:
    """Build the fixed application and composition inventory under a restriction."""
    rules = tuple(
        _compose(DIRECT_GRAMMAR, role, order, direction)
        for order in (0, 1, 2)
        for role in ("forward", "backward")
    )

    return RuleSet(f"composition/{direction}", rules, Identity(), NoNormalForm())


CLOSURE_SEEDS: Final = (
    Y,
    DIRECT.make("fwd", X, Y),
    DIRECT.make("bwd", OTHER, Y),
    DIRECT.make("fwd", Y, Z0),
    DIRECT.make("bwd", Y, Z0),
    DIRECT.make("fwd", DIRECT.make("fwd", Y, Z1), Z0),
    DIRECT.make("bwd", DIRECT.make("fwd", Y, Z1), Z0),
    DIRECT.make("fwd", DIRECT.make("bwd", Y, Z1), Z0),
    DIRECT.make("bwd", DIRECT.make("bwd", Y, Z1), Z0),
)


def test_the_closure_over_the_committed_inventory_differs_by_restriction() -> None:
    reports = {
        direction: _committed_inventory(direction).closure(
            CLOSURE_SEEDS, max_depth=6, budget=500
        )
        for direction in COMPOSITION_RESTRICTIONS
    }
    reached = {
        direction: frozenset(report.reached) for direction, report in reports.items()
    }

    assert all(report.saturated for report in reports.values())
    assert all(report.never_applicable == () for report in reports.values())
    assert reached["steedman"] < reached["kuhlmann"]
    assert reached["kuhlmann"] < reached["unrestricted"]
    assert all(report.category_set == "finite" for report in reports.values())


def test_the_declaration_carries_the_direction_restriction() -> None:
    for direction in COMPOSITION_RESTRICTIONS:
        rule = _second_order(direction, "forward")
        spec = rule.to_spec()

        assert rule.direction == direction
        assert spec["direction"] == direction
        assert _rebuild(DIRECT_GRAMMAR, spec).to_spec() == spec


def _unbounded(role: SlashRole) -> GeneralizedComposition:
    """Build the schema admitting every finite order above zero."""
    return _compose(DIRECT_GRAMMAR, role, None, "unrestricted")


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("order", range(4))
def test_one_schema_composes_at_every_order_its_container_supplies(
    role: SlashRole,
    order: int,
) -> None:
    rule = _unbounded(role)
    case = _case(DIRECT_GRAMMAR, role, order, crossed=False, flip_outer=False)
    expected = () if order == 0 else (case.output,)
    applied = order if order > 0 else None
    _, slash = DIRECT_GRAMMAR.functor(role)
    constraint = (
        (frozenset({slash}), DIRECT.slashes)
        if role == "forward"
        else (DIRECT.slashes, frozenset({slash}))
    )

    assert rule.order is None
    assert rule.probe.alternatives == (constraint,)
    assert tuple(rule.apply(case.inputs)) == expected
    assert rule.admits(case.inputs, case.output) == (order > 0)
    assert rule.applied_order(case.inputs, case.output) == applied


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("order", range(1, 4))
def test_the_schema_residuates_both_premises_at_every_order(
    role: SlashRole,
    order: int,
) -> None:
    rule = _unbounded(role)
    case = _case(DIRECT_GRAMMAR, role, order, crossed=False, flip_outer=False)

    for slot in range(rule.arity):
        known = tuple(
            HOLE if at == slot else term for at, term in enumerate(case.inputs)
        )
        residuals = tuple(rule.residuate(case.output, known))

        assert case.inputs[slot] in residuals
        assert all(
            rule.admits(_completed(case.inputs, slot, candidate), case.output)
            for candidate in residuals
        )


def test_the_schema_declares_no_order_and_presents_no_finite_schema() -> None:
    rule = _unbounded("forward")
    spec = rule.to_spec()

    assert spec["order"] is None
    assert rule.proof_schemata == ()
    assert _rebuild(DIRECT_GRAMMAR, spec).to_spec() == spec


@pytest.mark.parametrize("case", SPINE_CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("direction", COMPOSITION_RESTRICTIONS)
def test_the_direction_restriction_still_decides_the_unbounded_schema(
    case: SpineCase,
    direction: CompositionRestriction,
) -> None:
    rule = _compose(DIRECT_GRAMMAR, case.role, None, direction)
    expected = (case.output,) if direction in case.licensed else ()

    assert tuple(rule.apply(case.inputs)) == expected


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


@settings(max_examples=300, deadline=None)
@given(
    data=st.data(),
    view=st.sampled_from(VIEWS),
    role=st.sampled_from(("forward", "backward")),
    order=st.sampled_from((0, 1, 2, None)),
    crossed=st.booleans(),
    restriction=st.sampled_from(range(len(COMPOSITION_RESTRICTIONS))),
)
def test_every_licensed_pair_shares_a_join_key(
    *,
    data: st.DataObject,
    view: LawView,
    role: SlashRole,
    order: int | None,
    crossed: bool,
    restriction: int,
) -> None:
    grammar = GRAMMARS[view]
    left = data.draw(_categories(grammar, 3))
    right = data.draw(_categories(grammar, 3))
    rule = _compose(
        grammar,
        role,
        order,
        COMPOSITION_RESTRICTIONS[restriction],
        crossed=crossed and order != 0,
    )
    shared = set(rule.join_keys(left, 0)) & set(rule.join_keys(right, 1))

    assert not tuple(rule.apply((left, right))) or shared


def test_join_keys_name_the_argument_and_each_admitted_foot() -> None:
    principal = DIRECT.make("fwd", X, Y)
    container = DIRECT.make("fwd", DIRECT.make("bwd", Y, Z0), Z1)
    second = _compose(DIRECT_GRAMMAR, "forward", 2, "unrestricted")
    application = _compose(DIRECT_GRAMMAR, "forward", 0)
    (key,) = second.join_keys(principal, 0)
    (applied,) = application.join_keys(container, 1)

    assert second.join_keys(container, 1) == (key,)
    assert isinstance(key, tuple)
    assert Y in key
    assert second.join_keys(X, 0) == ()
    assert isinstance(applied, tuple)
    assert container in applied
    assert application.join_keys(principal, 0) != (applied,)


def _relation(
    grammar: LawGrammar,
    order: int | None,
    direction: CompositionRestriction,
    *,
    crossed: bool,
) -> Mapping[str, JsonValue]:
    return _compose(grammar, "forward", order, direction, crossed=crossed).relation


@pytest.mark.parametrize("order", [1, 2, None])
def test_crossing_is_inert_exactly_when_no_spine_frame_is_fixed(
    order: int | None,
) -> None:
    assert _relation(DIRECT_GRAMMAR, order, "unrestricted", crossed=True) == _relation(
        DIRECT_GRAMMAR, order, "unrestricted", crossed=False
    )

    for direction in ("steedman", "kuhlmann"):
        assert _relation(DIRECT_GRAMMAR, order, direction, crossed=True) != _relation(
            DIRECT_GRAMMAR, order, direction, crossed=False
        )


def test_steedman_and_kuhlmann_fix_the_same_frame_only_at_order_one() -> None:
    for crossed in (False, True):
        assert _relation(DIRECT_GRAMMAR, 1, "steedman", crossed=crossed) == _relation(
            DIRECT_GRAMMAR, 1, "kuhlmann", crossed=crossed
        )

        for order in (2, None):
            assert _relation(
                DIRECT_GRAMMAR, order, "steedman", crossed=crossed
            ) != _relation(DIRECT_GRAMMAR, order, "kuhlmann", crossed=crossed)


@pytest.mark.parametrize("order", [1, 2, None])
def test_over_one_undirected_constructor_no_declaration_changes_the_relation(
    order: int | None,
) -> None:
    relations = [
        _relation(GRAMMARS["undirected"], order, direction, crossed=crossed)
        for direction in ("unrestricted", "steedman", "kuhlmann")
        for crossed in (False, True)
    ]

    assert all(relation == relations[0] for relation in relations)


def _universe(grammar: LawGrammar) -> tuple[Term, ...]:
    """Return categories whose result spines reach depth three over the view."""
    functors, _ = grammar.functor("forward")
    slashes = sorted(functors.slashes)
    atoms = (grammar.atom("x"), grammar.atom("y"))
    simple = tuple(
        functors.make(slash, result, argument)
        for slash in slashes
        for result in atoms
        for argument in atoms
    )
    deeper = tuple(
        functors.make(slash, inner, argument)
        for slash in slashes
        for inner in simple
        for argument in atoms
    )
    deepest = tuple(
        functors.make(slash, inner, atoms[0]) for slash in slashes for inner in deeper
    )

    return (*atoms, *simple, *deeper, *deepest)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("order", [1, 2, None])
def test_equal_relations_are_exactly_equal_conclusions(
    view: LawView, order: int | None
) -> None:
    grammar = GRAMMARS[view]
    universe = _universe(grammar)
    declarations = [
        (direction, crossed)
        for direction in COMPOSITION_RESTRICTIONS
        for crossed in (False, True)
    ]
    rules = [
        _compose(grammar, "forward", order, direction, crossed=crossed)
        for direction, crossed in declarations
    ]
    relations = [rule.relation for rule in rules]
    conclusions = [
        tuple(
            frozenset(rule.apply((left, right)))
            for left in universe
            for right in universe
        )
        for rule in rules
    ]

    for left, right in product(range(len(declarations)), repeat=2):
        assert (relations[left] == relations[right]) == (
            conclusions[left] == conclusions[right]
        )


def test_the_premise_matching_and_the_restriction_each_fix_the_relation() -> None:
    def relation(
        matching: MatchingSpec | None = None,
        restriction: RuleRestrictionSpec | None = None,
    ) -> Mapping[str, JsonValue]:
        return _compose(
            DIRECT_GRAMMAR,
            "forward",
            1,
            "unrestricted",
            matching=matching,
            restriction=restriction,
        ).relation

    exact = relation()

    assert relation() == exact
    assert relation(FeatureErasureMatchingSpec()) != exact
    assert relation(restriction=RuleRestrictionSpec(primary_targets=("t",))) != exact
    assert relation(
        restriction=RuleRestrictionSpec(primary_targets=("t",))
    ) != relation(restriction=RuleRestrictionSpec(primary_targets=("u",)))


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
    rule = _compose(DIRECT_GRAMMAR, "forward", 2, "unrestricted")
    sorts = {
        sort
        for premises, conclusion in rule.reachability_schemata
        for pattern in (*premises, conclusion)
        for sort in _metavariable_sorts(pattern)
    }

    assert sorts == {DIRECT.category_sort}
