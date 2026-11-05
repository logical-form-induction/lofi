from typing import TYPE_CHECKING, Final, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.terms import (
    Constructors,
    DeclaredConstructors,
    NeighborhoodError,
    Operator,
    Signature,
    Term,
    TermStore,
    TreeEditDistance,
    ball,
    constituents,
    truncate,
    typeless,
    types,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SIGNATURE: Final = Signature(
    "neighborhood-categories",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("np", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("wrap", "Cat", arg_sorts=("Cat",), family="wrapper"),
        Operator(
            "ternary",
            "Cat",
            arg_sorts=("Cat", "Cat", "Cat"),
            family="slash",
        ),
    ),
)
STORE: Final = TermStore(SIGNATURE)
N: Final = STORE.op(SIGNATURE.by_name("n"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
S: Final = STORE.op(SIGNATURE.by_name("s"))
FWD: Final = SIGNATURE.by_name("fwd")
BWD: Final = SIGNATURE.by_name("bwd")
WRAP: Final = SIGNATURE.by_name("wrap")
TERNARY: Final = SIGNATURE.by_name("ternary")
CONSTRUCTORS: Final = DeclaredConstructors(
    STORE,
    {
        "fwd": ("result", "argument"),
        "bwd": ("result", "argument"),
        "wrap": ("scope",),
        "ternary": ("result", "left", "right"),
    },
)


class Rejecting:
    """A constructor view that rejects every construction."""

    __slots__ = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Return the one declared constructor."""
        return ("fwd",)

    def roles(self, constructor: str, /) -> tuple[str, ...]:
        """Return the constructor's child roles."""
        if constructor != "fwd":
            raise KeyError(constructor)
        return ("result", "argument")

    def is_constructor(self, term: Term, /) -> bool:
        """Return whether a term uses the declared constructor."""
        return term.op.name == "fwd"

    def constructor(self, term: Term, /) -> str:
        """Return the term's constructor."""
        return term.op.name

    def make(self, constructor: str, children: Sequence[Term], /) -> Term:
        """Reject a constructor application."""
        del constructor, children
        message = "rejected"

        raise ValueError(message)


class Misbuilding(Rejecting):
    """A constructor view that returns a primitive construction."""

    __slots__ = ()

    @override
    def make(self, constructor: str, children: Sequence[Term], /) -> Term:
        """Return the result instead of a functor."""
        del constructor

        return children[0]


VIEW: Final = DeclaredConstructors(
    STORE,
    {"fwd": ("result", "argument"), "bwd": ("result", "argument")},
)
ONE_SLASH: Final = DeclaredConstructors(STORE, {"fwd": ("result", "argument")})
INTRANSITIVE: Final = VIEW.make("bwd", (S, NP))
TRANSITIVE: Final = VIEW.make("fwd", (INTRANSITIVE, NP))
DITRANSITIVE: Final = VIEW.make("fwd", (TRANSITIVE, NP))


type Shape = tuple[str | None, tuple[Shape, ...]]


def test_declared_constructor_views_implement_the_constructor_protocol() -> None:
    assert isinstance(CONSTRUCTORS, Constructors)


def depth_of(term: Term) -> int:
    """Return a term's one-based depth."""
    return 1 + max((depth_of(child) for child in term.args), default=0)


def shape_of(term: Term) -> Shape:
    """Return a term's functor topology and slashes."""
    slash = VIEW.constructor(term) if VIEW.is_constructor(term) else None

    return slash, tuple(shape_of(child) for child in term.args)


def bounded_occurrences(term: Term, max_depth: int) -> tuple[tuple[int, Term], ...]:
    """Return bounded occurrences by recursive preorder."""
    found = ((1, term),)

    if max_depth == 1:
        return found

    return found + tuple(
        (depth + 1, child)
        for argument in term.args
        for depth, child in bounded_occurrences(argument, max_depth - 1)
    )


def test_types_returns_the_exact_one_slash_set_through_five_nodes() -> None:
    pair = ONE_SLASH.make("fwd", (N, N))
    expected = frozenset(
        {
            N,
            pair,
            ONE_SLASH.make("fwd", (pair, N)),
            ONE_SLASH.make("fwd", (N, pair)),
        }
    )

    assert types((N,), ONE_SLASH, max_nodes=5) == expected


def test_types_has_the_hand_computed_size_distribution_for_two_slashes() -> None:
    generated = types((N, NP, N), VIEW, max_nodes=5)
    by_size = {
        size: frozenset(term for term in generated if term.size == size)
        for size in (1, 3, 5)
    }

    assert {size: len(terms) for size, terms in by_size.items()} == {
        1: 2,
        3: 8,
        5: 64,
    }
    assert len(generated) == 74
    assert all(
        term in generated
        for slash in VIEW.names
        for term in (
            VIEW.make(slash, (N, N)),
            VIEW.make(slash, (N, NP)),
            VIEW.make(slash, (NP, N)),
            VIEW.make(slash, (NP, NP)),
        )
    )


def test_types_derives_depth_from_the_node_bound() -> None:
    through_three = types((N, NP), VIEW, max_nodes=3)

    assert types((N, NP), VIEW, max_nodes=4) == through_three
    assert all(depth_of(term) <= 2 for term in through_three)
    assert any(depth_of(term) == 3 for term in types((N, NP), VIEW, max_nodes=5))


def test_ball_is_the_stable_filter_of_the_explicit_candidate_universe() -> None:
    candidates = (N, NP, INTRANSITIVE, TRANSITIVE, N)
    distance = TreeEditDistance()

    assert ball(N, candidates, distance, radius=0) == (N, N)
    assert ball(N, candidates, distance, radius=1) == (N, NP, N)
    assert ball(N, candidates, distance, radius=2) == (N, NP, N)
    assert ball(N, candidates, distance, radius=3) == (N, NP, INTRANSITIVE, N)
    assert ball(N, candidates, distance, radius=10) == candidates


def test_ball_uses_the_supplied_distance_once_per_candidate() -> None:
    calls: list[Term] = []

    def select_np(_center: Term, candidate: Term) -> int:
        calls.append(candidate)

        return 0 if candidate is NP else 1

    candidates = (N, NP, N)

    assert ball(S, candidates, select_np, radius=0) == (NP,)
    assert calls == list(candidates)


def test_typeless_preserves_topology_and_slashes_and_interns_equal_shapes() -> None:
    left = VIEW.make("fwd", (VIEW.make("bwd", (S, NP)), N))
    right = VIEW.make("fwd", (VIEW.make("bwd", (N, S)), NP))
    expected = VIEW.make("fwd", (VIEW.make("bwd", (N, N)), N))
    left_projection = typeless(left, N, VIEW)
    right_projection = typeless(right, N, VIEW)

    assert shape_of(left_projection) == shape_of(left)
    assert left_projection is expected
    assert right_projection is expected
    assert typeless(left_projection, N, VIEW) is left_projection
    assert typeless(NP, N, VIEW) is N


def test_projections_preserve_unary_product_and_ternary_roles() -> None:
    product = CONSTRUCTORS.make("fwd", (S, NP))
    wrapped = CONSTRUCTORS.make("wrap", (product,))
    term = CONSTRUCTORS.make("ternary", (wrapped, NP, S))
    projected = CONSTRUCTORS.make(
        "ternary",
        (
            CONSTRUCTORS.make("wrap", (CONSTRUCTORS.make("fwd", (N, N)),)),
            N,
            N,
        ),
    )

    assert typeless(term, N, CONSTRUCTORS) is projected
    assert truncate(term, N, CONSTRUCTORS, depth=2) is CONSTRUCTORS.make(
        "ternary", (N, NP, S)
    )
    assert CONSTRUCTORS.roles("wrap") == ("scope",)
    assert CONSTRUCTORS.roles("ternary") == ("result", "left", "right")


def test_declared_constructor_enumeration_includes_every_arity_within_the_bound() -> (
    None
):
    generated = types((N,), CONSTRUCTORS, max_nodes=4)

    assert CONSTRUCTORS.make("wrap", (N,)) in generated
    assert CONSTRUCTORS.make("fwd", (N, N)) in generated
    assert CONSTRUCTORS.make("ternary", (N, N, N)) in generated


def test_truncate_uses_one_based_depth_and_keeps_shallow_identity() -> None:
    depth_two = VIEW.make("fwd", (N, NP))
    depth_three = VIEW.make("fwd", (VIEW.make("fwd", (N, NP)), NP))

    assert truncate(DITRANSITIVE, N, VIEW, depth=1) is N
    assert truncate(DITRANSITIVE, N, VIEW, depth=2) is depth_two
    assert truncate(DITRANSITIVE, N, VIEW, depth=3) is depth_three
    assert truncate(DITRANSITIVE, N, VIEW, depth=4) is DITRANSITIVE
    assert truncate(DITRANSITIVE, N, VIEW, depth=5) is DITRANSITIVE
    assert truncate(NP, N, VIEW, depth=1) is NP
    assert truncate(depth_three, N, VIEW, depth=3) is depth_three


def test_constituents_preserves_occurrences_and_both_documented_orders() -> None:
    inner = VIEW.make("bwd", (S, N))
    term = VIEW.make("fwd", (inner, NP))

    assert constituents(term, max_depth=1) == (term,)
    assert constituents(term, max_depth=2) == (term, inner, NP)
    assert constituents(term, max_depth=3) == (term, inner, S, N, NP)
    assert constituents(term, max_depth=3, sort_by_depth=True) == (
        term,
        inner,
        NP,
        S,
        N,
    )
    assert constituents(TRANSITIVE, max_depth=3).count(NP) == 2


@pytest.mark.parametrize("max_nodes", [0, -1])
def test_types_rejects_a_nonpositive_node_bound(max_nodes: int) -> None:
    with pytest.raises(NeighborhoodError, match="max_nodes"):
        types((N,), VIEW, max_nodes=max_nodes)


def test_ball_rejects_a_negative_radius() -> None:
    with pytest.raises(NeighborhoodError, match="radius"):
        ball(N, (N,), TreeEditDistance(), radius=-1)


@pytest.mark.parametrize("depth", [0, -1])
def test_truncate_rejects_a_nonpositive_depth(depth: int) -> None:
    with pytest.raises(NeighborhoodError, match="depth"):
        truncate(N, N, VIEW, depth=depth)


@pytest.mark.parametrize("max_depth", [0, -1])
def test_constituents_rejects_a_nonpositive_depth(max_depth: int) -> None:
    with pytest.raises(NeighborhoodError, match="max_depth"):
        constituents(N, max_depth=max_depth)


def test_primitives_must_be_nonfunctor_leaves() -> None:
    functor = VIEW.make("fwd", (N, NP))
    wrapped = STORE.op(WRAP, N)

    for rejected in (functor, wrapped):
        with pytest.raises(NeighborhoodError, match="primitive"):
            types((rejected,), VIEW, max_nodes=3)

        with pytest.raises(NeighborhoodError, match="primitive"):
            typeless(N, rejected, VIEW)

        with pytest.raises(NeighborhoodError, match="primitive"):
            truncate(N, rejected, VIEW, depth=1)


@pytest.mark.parametrize("view", [Rejecting(), Misbuilding()])
def test_rejected_constructor_constructions_raise_the_named_value_error(
    view: Rejecting,
) -> None:
    with pytest.raises(NeighborhoodError, match="constructor construction"):
        types((N,), view, max_nodes=3)


def test_a_nondeclared_constructor_with_three_children_is_rejected() -> None:
    term = STORE.op(TERNARY, N, NP, S)

    with pytest.raises(NeighborhoodError, match="primitive"):
        typeless(term, N, VIEW)


def test_truncate_rejects_invalid_structure_below_its_cutoff() -> None:
    wrapped = STORE.op(WRAP, N)
    term = VIEW.make("fwd", (wrapped, NP))

    with pytest.raises(NeighborhoodError, match="primitive"):
        truncate(term, N, VIEW, depth=1)


GENERATED_POOL: Final = tuple(types((N, NP), VIEW, max_nodes=7))
GENERATED: Final = st.sampled_from(GENERATED_POOL)


@settings(max_examples=2_000, deadline=None)
@given(term=GENERATED, depth=st.integers(min_value=1, max_value=7))
def test_generated_projections_obey_shape_depth_and_occurrence_laws(
    term: Term,
    depth: int,
) -> None:
    projection = typeless(term, N, VIEW)
    shortened = truncate(term, N, VIEW, depth=depth)
    occurrences = bounded_occurrences(term, depth)
    preorder = tuple(found for _, found in occurrences)
    depth_order = tuple(found for _, found in sorted(occurrences, key=lambda x: x[0]))

    assert shape_of(projection) == shape_of(term)
    assert typeless(projection, N, VIEW) is projection
    assert depth_of(shortened) <= depth
    assert truncate(shortened, N, VIEW, depth=depth) is shortened
    assert shortened is term if depth >= depth_of(term) else shortened is not term
    assert constituents(term, max_depth=depth) == preorder
    assert constituents(term, max_depth=depth, sort_by_depth=True) == depth_order


@settings(max_examples=1_000, deadline=None)
@given(
    center=GENERATED,
    candidates=st.lists(GENERATED, max_size=12),
    radius=st.integers(min_value=0, max_value=8),
)
def test_generated_balls_equal_their_explicit_distance_filter(
    center: Term,
    candidates: list[Term],
    radius: int,
) -> None:
    distance = TreeEditDistance()
    expected = tuple(
        candidate for candidate in candidates if distance(center, candidate) <= radius
    )

    assert ball(center, candidates, distance, radius=radius) == expected
