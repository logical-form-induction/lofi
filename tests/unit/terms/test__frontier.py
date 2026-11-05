from collections.abc import Mapping, MutableMapping
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Never, override

import pytest
from tests.support.term_reference import categories

from lofi.terms import Term, TermStore
from lofi.terms._frontier import Frontier
from lofi.terms._values import BindingSite, Operator, Signature, Vocabulary

if TYPE_CHECKING:
    from lofi.terms._ports import Path


class NonBooleanState:
    """Decoder state without Boolean equality."""

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> Never:
        del other
        msg = "decoder-state equality is undefined"

        raise TypeError(msg)

    __hash__ = None


def partial() -> tuple[Term, dict[Path, str]]:
    signature = categories().with_holes()
    store = TermStore(signature)
    hole = store.hole("Cat")
    noun_phrase = store.op(signature.by_name("np"))
    left = store.op(signature.by_name("fwd"), hole, noun_phrase)
    skeleton = store.op(signature.by_name("fwd"), left, hole)

    return skeleton, {(1,): "right", (0, 0): "deep-left"}


def test_frontier_holds_one_read_only_state_per_skeleton_hole() -> None:
    skeleton, live = partial()
    frontier = Frontier(skeleton, live, -0.25)
    live[(0,)] = "sneaked in"

    assert frontier.holes == ((1,), (0, 0))
    assert set(frontier.states) == {(1,), (0, 0)}
    assert not frontier.is_complete
    assert isinstance(frontier.states, Mapping)
    assert not isinstance(frontier.states, MutableMapping)


def test_frontier_refuses_a_missing_hole_state() -> None:
    skeleton, _states = partial()

    with pytest.raises(ValueError, match=r"missing=\(\(0, 0\),\)"):
        Frontier(skeleton, {(1,): "right"}, 0.0)


def test_frontier_refuses_an_extraneous_state_path() -> None:
    skeleton, states = partial()
    states[(0,)] = "not a hole"

    with pytest.raises(ValueError, match=r"extra=\(\(0,\),\)"):
        Frontier(skeleton, states, 0.0)


def test_a_complete_skeleton_requires_an_empty_state_map() -> None:
    signature = categories()
    store = TermStore(signature)
    complete = store.op(
        signature.by_name("fwd"),
        store.op(signature.by_name("s")),
        store.op(signature.by_name("np")),
    )
    frontier = Frontier(complete, {}, 0.0)

    assert frontier.holes == ()
    assert frontier.is_complete

    with pytest.raises(ValueError, match=r"extra=\(\(\),\)"):
        Frontier(complete, {(): "not a hole"}, 0.0)


def test_a_declared_operator_named_like_a_hole_is_no_hole() -> None:
    leaf = Operator("gap", "Cat", family="gap")
    signature = Signature("gapped", ("Cat",), (leaf,))
    store = TermStore(signature)
    frontier = Frontier(store.op(leaf), {}, 0.0)

    assert frontier.is_complete


def test_frontier_reweights_without_touching_the_skeleton() -> None:
    skeleton, states = partial()
    frontier = Frontier(skeleton, states, -0.25)
    heavier = frontier.with_weight(-3.5)

    assert heavier.log_weight == -3.5
    assert heavier.skeleton is frontier.skeleton
    assert dict(heavier.states) == dict(frontier.states)
    assert frontier.log_weight == -0.25


def test_frontier_is_identified_with_itself_and_not_its_twin() -> None:
    skeleton, states = partial()
    frontier = Frontier(skeleton, states, 0.0)
    twin = Frontier(skeleton, states, 0.0)

    assert frontier != twin
    assert len({frontier, twin}) == 2

    for field, value in (("log_weight", 1.0), ("skeleton", skeleton), ("states", {})):
        with pytest.raises(AttributeError):
            setattr(frontier, field, value)


def test_frontiers_keep_decoder_state_outside_the_partial_term() -> None:
    signature = categories().with_holes()
    store = TermStore(signature)
    hole = store.hole("Cat")
    sentence = store.op(signature.by_name("s"))
    skeleton = store.op(signature.by_name("fwd"), sentence, hole)
    first = Frontier(skeleton, {(1,): NonBooleanState()}, -0.5)
    second = Frontier(
        store.op(signature.by_name("fwd"), sentence, store.hole("Cat")),
        {(1,): NonBooleanState()},
        -0.75,
    )

    assert first.skeleton is second.skeleton
    assert hash(first.skeleton) == hash(second.skeleton)
    assert first.skeleton == second.skeleton
    assert first != second
    assert first.holes == second.holes == ((1,),)


def test_frontier_private_fields_are_frozen() -> None:
    signature = categories().with_holes()
    store = TermStore(signature)
    frontier = Frontier(store.hole("Cat"), {(): "state"}, 0.0)

    for field in Frontier.__slots__:
        with pytest.raises(AttributeError, match="immutable"):
            frontier.__setattr__(field, "changed")

        with pytest.raises(AttributeError, match="immutable"):
            frontier.__delattr__(field)


def test_copying_an_immutable_frontier_preserves_its_identity() -> None:
    signature = categories().with_holes()
    store = TermStore(signature)
    frontier = Frontier(store.hole("Cat"), {(): "state"}, -0.25)

    assert copy(frontier) is frontier
    assert deepcopy(frontier) is frontier


def church() -> Signature:
    """Return a two-sorted Church-style signature with holes."""
    return Signature(
        "church",
        ("Ty", "Tm"),
        (
            Operator("arrow", "Ty", arg_sorts=("Ty", "Ty")),
            Operator(
                "lam",
                "Tm",
                arg_sorts=("Ty", "Tm"),
                binding=(BindingSite((1,), "Tm", 0),),
            ),
            Operator(
                "tlam",
                "Tm",
                arg_sorts=("Tm",),
                binding=(BindingSite((0,), "Ty"),),
            ),
            Operator("app", "Tm", arg_sorts=("Tm", "Tm")),
        ),
        vocabularies=(Vocabulary("Ty", ("e",)), Vocabulary("Tm", ("jo",))),
        classifiers={"Tm": "Ty"},
        holes=True,
    )


def test_the_next_hole_has_its_path_sort_and_the_sites_in_scope_per_sort() -> None:
    signature = church()
    store = TermStore(signature)
    lam, tlam = signature.by_name("lam"), signature.by_name("tlam")
    inner = store.op(lam, store.hole("Ty"), store.hole("Tm"))
    skeleton = store.op(lam, store.const("Ty", "e"), store.op(tlam, inner))
    frontier = Frontier(skeleton, {(1, 0, 0): "type", (1, 0, 1): "form"}, 0.0)
    context = frontier.context(12)

    assert frontier.holes == ((1, 0, 0), (1, 0, 1))
    assert context.path == (1, 0, 0)
    assert context.sort == "Ty"
    # the annotation lies outside the inner abstraction's scope.
    assert dict(context.in_scope) == {"Tm": 1, "Ty": 1}
    assert context.remaining == 12 - skeleton.size

    body = Frontier(
        store.op(
            lam,
            store.const("Ty", "e"),
            store.op(tlam, store.op(lam, store.const("Ty", "e"), store.hole("Tm"))),
        ),
        {(1, 0, 1): "form"},
        0.0,
    )

    assert dict(body.context(12).in_scope) == {"Tm": 2, "Ty": 1}
    assert body.context(12).sort == "Tm"


def test_the_context_counts_the_distinct_free_variables_of_each_sort() -> None:
    signature = church()
    store = TermStore(signature)
    app = signature.by_name("app")
    x, y = store.fvar("Tm", "x"), store.fvar("Tm", "y")
    skeleton = store.op(app, store.op(app, x, x), store.op(app, y, store.hole("Tm")))
    context = Frontier(skeleton, {(1, 1): "state"}, 0.0).context(20)

    assert dict(context.free_used) == {"Tm": 2}
    assert dict(context.in_scope) == {}
    assert isinstance(context.in_scope, Mapping)
    assert not isinstance(context.in_scope, MutableMapping)


def test_a_complete_frontier_has_no_context_and_the_bound_is_whole() -> None:
    store = TermStore(church())
    complete = Frontier(store.const("Tm", "jo"), {}, 0.0)
    partial_term = Frontier(store.hole("Tm"), {(): "state"}, 0.0)

    with pytest.raises(ValueError, match="no next hole"):
        complete.context(4)

    with pytest.raises(ValueError, match="max_size is negative"):
        partial_term.context(-1)

    assert partial_term.context(0).remaining == -1
