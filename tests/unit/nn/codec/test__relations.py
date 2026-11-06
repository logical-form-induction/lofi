from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch

from lofi.nn.codec import (
    CodecMetadata,
    DeclaredStructure,
    NeuralCodecView,
    reflexive_closure,
)
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import Consumption, StructuralRelation
from lofi.terms import Encoding, Operator, OperatorIndex, Signature

PAD: Final = OperatorIndex.PAD

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.nn.ports import StructuralDeclaration

N_SYMBOLS: Final = 3
CAPACITY: Final = 7
POLICY: Final = DevicePolicy(torch.device("cpu"), torch.float32)
LEGALITY: Final = ((True,) * N_SYMBOLS,) * CAPACITY
INDEX: Final = OperatorIndex(
    Signature(
        "relations-fixture",
        ("T",),
        (Operator("a", "T"), Operator("f", "T", arg_sorts=("T", "T"))),
    )
)

# the same complete binary tree of depth three under two node orders.
DEPTH_FIRST_PARENTS: Final = (Encoding.NO_PARENT, 0, 1, 1, 0, 4, 4)
DEPTH_FIRST_DEPTHS: Final = (0, 1, 2, 2, 1, 2, 2)
BREADTH_FIRST_PARENTS: Final = (Encoding.NO_PARENT, 0, 0, 1, 1, 2, 2)
BREADTH_FIRST_DEPTHS: Final = (0, 1, 1, 2, 2, 2, 2)

EVERYTHING: Final = MappingProxyType(
    dict.fromkeys(StructuralRelation, Consumption.BIAS)
)
POSITION_ONLY: Final = MappingProxyType(
    {StructuralRelation.POSITION: Consumption.EMBEDDING}
)


class Codec(NamedTuple):
    n_symbols: int
    capacity: int
    skeleton: Encoding | None
    legal_symbols: tuple[tuple[bool, ...], ...]
    index: OperatorIndex


def view(
    parents: tuple[int, ...], depths: tuple[int, ...]
) -> NeuralCodecView[CodecMetadata]:
    skeleton = Encoding((0,) * CAPACITY, parents, depths, (False,) * CAPACITY)

    return NeuralCodecView(Codec(N_SYMBOLS, CAPACITY, skeleton, LEGALITY, INDEX))


DEPTH_FIRST: Final = view(DEPTH_FIRST_PARENTS, DEPTH_FIRST_DEPTHS)
BREADTH_FIRST: Final = view(BREADTH_FIRST_PARENTS, BREADTH_FIRST_DEPTHS)
UNSTRUCTURED: Final = NeuralCodecView(Codec(N_SYMBOLS, CAPACITY, None, LEGALITY, INDEX))


def ancestors(parents: tuple[int, ...], slot: int) -> list[int]:
    """Return a slot and the slots above it, nearest first."""
    held = [slot]

    while parents[held[-1]] != Encoding.NO_PARENT:
        held.append(parents[held[-1]])

    return held


def distance(parents: tuple[int, ...], one: int, other: int) -> int:
    """Return the number of edges between two slots, walking up to their meeting."""
    above_one = ancestors(parents, one)
    above_other = ancestors(parents, other)
    meeting = next(at for at in above_one if at in above_other)

    return above_one.index(meeting) + above_other.index(meeting)


def subtree(parents: tuple[int, ...], slot: int) -> list[int]:
    return [at for at in range(len(parents)) if slot in ancestors(parents, at)]


class Layout(NamedTuple):
    held: NeuralCodecView[CodecMetadata]
    parents: tuple[int, ...]
    depths: tuple[int, ...]


LAYOUTS: Final = (
    Layout(DEPTH_FIRST, DEPTH_FIRST_PARENTS, DEPTH_FIRST_DEPTHS),
    Layout(BREADTH_FIRST, BREADTH_FIRST_PARENTS, BREADTH_FIRST_DEPTHS),
)


@pytest.fixture(params=LAYOUTS, ids=["depth-first", "breadth-first"])
def layout(request: pytest.FixtureRequest) -> Layout:
    return cast("Layout", request.param)


def everything(
    held: NeuralCodecView[CodecMetadata],
) -> DeclaredStructure[CodecMetadata]:
    return held.serve(EVERYTHING)


def test_position_is_the_slot_index(layout: Layout) -> None:
    position = everything(layout.held).position(policy=POLICY)

    assert position.dtype is torch.int64
    assert position.tolist() == list(range(CAPACITY))


def test_depth_is_the_skeletons_depth(layout: Layout) -> None:
    depth = everything(layout.held).depth(policy=POLICY)

    assert depth.dtype is torch.int64
    assert depth.tolist() == list(layout.depths)


def test_arity_counts_the_child_slots(layout: Layout) -> None:
    arity = everything(layout.held).arity(policy=POLICY)

    assert arity.dtype is torch.int64
    assert arity.tolist() == [layout.parents.count(at) for at in range(CAPACITY)]


def test_parent_holds_from_a_slot_to_the_one_it_descends_from(layout: Layout) -> None:
    parent = everything(layout.held).parent(policy=POLICY)

    assert parent.dtype is torch.bool
    assert parent.tolist() == [
        [layout.parents[one] == other for other in range(CAPACITY)]
        for one in range(CAPACITY)
    ]
    assert not any(parent[0].tolist())


def side_of(parents: tuple[int, ...], slot: int) -> int:
    """Return the argument position a slot occupies under its parent."""
    return [at for at in range(slot + 1) if parents[at] == parents[slot]].index(slot)


def test_legal_beneath_admits_pad_alone_where_the_parent_symbol_has_no_argument(
    layout: Layout,
) -> None:
    beneath = everything(layout.held).legal_beneath(policy=POLICY)
    leaf = INDEX.index("a")
    constructor = INDEX.index("f")

    assert beneath.dtype is torch.bool
    assert tuple(beneath.shape) == (CAPACITY, N_SYMBOLS, N_SYMBOLS)
    assert INDEX.arity(constructor) == BREADTH_FIRST.branching

    for slot in range(1, CAPACITY):
        for given in (PAD, leaf):
            assert beneath[slot, given].tolist() == [True, False, False]

        assert beneath[slot, constructor].tolist() == [False, True, True]
        assert side_of(layout.parents, slot) < INDEX.arity(constructor)


def test_legal_beneath_at_the_root_is_the_layout_s_own_row(layout: Layout) -> None:
    beneath = everything(layout.held).legal_beneath(policy=POLICY)
    static = layout.held.legality(policy=POLICY)

    assert bool(beneath[0].eq(static[0].unsqueeze(0)).all())
    assert bool(beneath[0].all())


def test_legal_beneath_is_within_the_layout_s_own_row_everywhere(
    layout: Layout,
) -> None:
    beneath = everything(layout.held).legal_beneath(policy=POLICY)
    static = layout.held.legality(policy=POLICY)

    assert bool((beneath & ~static.unsqueeze(1)).logical_not().all())
    assert bool(beneath.any(dim=2).all())


def test_legal_beneath_follows_a_wider_constructor_side_by_side() -> None:
    wide = OperatorIndex(
        Signature(
            "wide-fixture",
            ("T",),
            (
                Operator("a", "T"),
                Operator("g", "T", arg_sorts=("T",)),
                Operator("f", "T", arg_sorts=("T", "T")),
            ),
        )
    )
    legality = ((True,) * wide.width,) * CAPACITY
    skeleton = Encoding(
        (0,) * CAPACITY,
        BREADTH_FIRST_PARENTS,
        BREADTH_FIRST_DEPTHS,
        (False,) * CAPACITY,
    )
    held = NeuralCodecView(Codec(wide.width, CAPACITY, skeleton, legality, wide))
    beneath = held.serve(EVERYTHING).legal_beneath(policy=POLICY)
    unary = wide.index("g")
    left, right = held.child_index(0, 0), held.child_index(0, 1)

    assert beneath[left, unary].tolist() == [False, True, True, True]
    assert beneath[right, unary].tolist() == [True, False, False, False]
    assert beneath[right, wide.index("f")].tolist() == [False, True, True, True]


def test_legal_beneath_is_read_under_the_parent_relation() -> None:
    served = DEPTH_FIRST.serve({StructuralRelation.PARENT: Consumption.RECURRENCE})

    assert tuple(served.legal_beneath(policy=POLICY).shape) == (
        CAPACITY,
        N_SYMBOLS,
        N_SYMBOLS,
    )

    with pytest.raises(ValueError, match="does not name parent"):
        DEPTH_FIRST.serve(POSITION_ONLY).legal_beneath(policy=POLICY)

    with pytest.raises(ValueError, match="does not name parent"):
        UNSTRUCTURED.serve(POSITION_ONLY).legal_beneath(policy=POLICY)


def test_ancestor_holds_from_a_slot_to_itself_and_the_chain_above(
    layout: Layout,
) -> None:
    ancestor = everything(layout.held).ancestor(policy=POLICY)

    assert ancestor.dtype is torch.bool
    assert ancestor.tolist() == [
        [other in ancestors(layout.parents, one) for other in range(CAPACITY)]
        for one in range(CAPACITY)
    ]
    assert bool(ancestor.diagonal().all())


def test_sibling_holds_between_distinct_slots_sharing_a_parent(
    layout: Layout,
) -> None:
    sibling = everything(layout.held).sibling(policy=POLICY)

    assert sibling.dtype is torch.bool
    assert sibling.tolist() == [
        [
            one != other
            and layout.parents[one] == layout.parents[other]
            and layout.parents[one] != Encoding.NO_PARENT
            for other in range(CAPACITY)
        ]
        for one in range(CAPACITY)
    ]
    assert not any(sibling[0].tolist())
    assert torch.equal(sibling, sibling.T)


def test_tree_distance_counts_the_edges_between_two_slots(layout: Layout) -> None:
    apart = everything(layout.held).tree_distance(policy=POLICY)

    assert apart.dtype is torch.int64
    assert apart.tolist() == [
        [distance(layout.parents, one, other) for other in range(CAPACITY)]
        for one in range(CAPACITY)
    ]
    assert not any(apart.diagonal().tolist())
    assert torch.equal(apart, apart.T)


def test_same_constituent_holds_within_one_application(layout: Layout) -> None:
    together = everything(layout.held).same_constituent(policy=POLICY)
    parents = layout.parents

    assert together.dtype is torch.bool
    assert together.tolist() == [
        [
            one == other
            or parents[one] == other
            or parents[other] == one
            or (parents[one] == parents[other] and parents[one] != Encoding.NO_PARENT)
            for other in range(CAPACITY)
        ]
        for one in range(CAPACITY)
    ]
    assert torch.equal(together, together.T)


def test_span_contains_holds_over_the_slots_a_subtree_spans(layout: Layout) -> None:
    contains = everything(layout.held).span_contains(policy=POLICY)

    assert contains.dtype is torch.bool

    for one in range(CAPACITY):
        below = subtree(layout.parents, one)

        assert contains[one].tolist() == [
            min(below) <= other <= max(below) for other in range(CAPACITY)
        ]


def test_span_contains_is_ancestor_reversed_only_where_subtrees_are_contiguous() -> (
    None
):
    depth_first = everything(DEPTH_FIRST)
    breadth_first = everything(BREADTH_FIRST)

    assert torch.equal(
        depth_first.span_contains(policy=POLICY),
        depth_first.ancestor(policy=POLICY).T,
    )
    assert not torch.equal(
        breadth_first.span_contains(policy=POLICY),
        breadth_first.ancestor(policy=POLICY).T,
    )
    assert bool(breadth_first.span_contains(policy=POLICY)[1, 2])
    assert not bool(breadth_first.ancestor(policy=POLICY)[2, 1])


def test_one_tree_under_two_orders_has_two_relation_tensors() -> None:
    for relation in (
        DeclaredStructure[CodecMetadata].parent,
        DeclaredStructure[CodecMetadata].ancestor,
        DeclaredStructure[CodecMetadata].sibling,
        DeclaredStructure[CodecMetadata].tree_distance,
    ):
        assert not torch.equal(
            relation(everything(DEPTH_FIRST), policy=POLICY),
            relation(everything(BREADTH_FIRST), policy=POLICY),
        )


def test_a_pairwise_accessor_returns_one_tensor_whatever_the_consumption() -> None:
    for consumption in Consumption:
        served = DEPTH_FIRST.serve({StructuralRelation.ANCESTOR: consumption})

        assert torch.equal(
            served.ancestor(policy=POLICY),
            everything(DEPTH_FIRST).ancestor(policy=POLICY),
        )


def test_the_relation_tensors_land_where_the_policy_says() -> None:
    served = everything(DEPTH_FIRST)

    assert served.ancestor(policy=POLICY).device == POLICY.device
    assert served.depth(policy=POLICY).device == POLICY.device


class PositionReader:
    """A component that declares the position of a slot and nothing else."""

    def __init__(self, held: NeuralCodecView[CodecMetadata]) -> None:
        self._served = held.serve(POSITION_ONLY)

    @property
    def structure(self) -> StructuralDeclaration:
        return self._served.declaration

    def positions(self) -> torch.Tensor:
        return self._served.position(policy=POLICY)

    def parents(self) -> torch.Tensor:
        return self._served.parent(policy=POLICY)


def test_a_component_declaring_position_alone_cannot_obtain_parent() -> None:
    reader = PositionReader(DEPTH_FIRST)

    assert reader.positions().tolist() == list(range(CAPACITY))

    with pytest.raises(ValueError, match="does not name parent") as caught:
        reader.parents()

    assert "{position: embedding}" in str(caught.value)


ACCESSORS: Final[tuple[Callable[..., torch.Tensor], ...]] = (
    DeclaredStructure[CodecMetadata].position,
    DeclaredStructure[CodecMetadata].depth,
    DeclaredStructure[CodecMetadata].arity,
    DeclaredStructure[CodecMetadata].parent,
    DeclaredStructure[CodecMetadata].ancestor,
    DeclaredStructure[CodecMetadata].sibling,
    DeclaredStructure[CodecMetadata].tree_distance,
    DeclaredStructure[CodecMetadata].same_constituent,
    DeclaredStructure[CodecMetadata].span_contains,
)


def test_every_accessor_refuses_a_relation_the_declaration_leaves_out() -> None:
    assert len(ACCESSORS) == len(StructuralRelation)

    for relation, accessor in zip(StructuralRelation, ACCESSORS, strict=True):
        others = {
            other: Consumption.BIAS for other in StructuralRelation if other != relation
        }
        served = DEPTH_FIRST.serve(others)

        with pytest.raises(ValueError, match=f"does not name {relation.value}"):
            accessor(served, policy=POLICY)

        assert (
            accessor(
                DEPTH_FIRST.serve({relation: Consumption.BIAS}), policy=POLICY
            ).shape[0]
            == CAPACITY
        )


def test_a_layout_the_term_settles_serves_a_declaration_of_position_alone() -> None:
    served = UNSTRUCTURED.serve(POSITION_ONLY)

    assert served.served == frozenset({StructuralRelation.POSITION})
    assert served.position(policy=POLICY).tolist() == list(range(CAPACITY))

    for relation in StructuralRelation:
        if relation is StructuralRelation.POSITION:
            continue

        with pytest.raises(ValueError, match="does not serve") as caught:
            UNSTRUCTURED.serve({**POSITION_ONLY, relation: Consumption.MASK})

        assert relation.value in str(caught.value)
        assert "serves position alone" in str(caught.value)
        assert "{position: embedding, " in str(caught.value)


def test_a_declaration_over_the_wrong_types_is_refused_before_it_is_served() -> None:
    with pytest.raises(TypeError, match="keyed by StructuralRelation"):
        DEPTH_FIRST.serve(cast("StructuralDeclaration", {"parent": Consumption.MASK}))

    with pytest.raises(TypeError, match="maps parent to a Consumption"):
        DEPTH_FIRST.serve(
            cast("StructuralDeclaration", {StructuralRelation.PARENT: "mask"})
        )


def test_an_empty_declaration_is_served_and_reads_nothing() -> None:
    served = UNSTRUCTURED.serve({})

    assert dict(served.declaration) == {}

    with pytest.raises(ValueError, match="does not name position"):
        served.position(policy=POLICY)


def test_the_reflexive_closure_reaches_along_chains_and_from_a_slot_to_itself() -> None:
    step = torch.zeros(4, 4, dtype=torch.bool)
    step[1, 0] = True
    step[2, 1] = True
    closed = reflexive_closure(step)

    assert closed.dtype is torch.bool
    assert closed.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [False, False, False, True],
    ]
    assert torch.equal(reflexive_closure(closed), closed)
    assert torch.equal(
        reflexive_closure(torch.zeros(3, 3, dtype=torch.bool)),
        torch.eye(3, dtype=torch.bool),
    )
