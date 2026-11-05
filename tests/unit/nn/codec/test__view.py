import math
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest
import torch
from hypothesis import given, reject, settings
from hypothesis import strategies as st
from tests.support.term_strategies import terms

from lofi.nn._values import LogDistribution
from lofi.nn.codec import CodecMetadata, DeclaredStructure, NeuralCodecView, _view
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import Consumption, StructuralRelation
from lofi.terms import (
    BatchEncoding,
    CompleteTree,
    Encoding,
    EncodingOverflowError,
    Operator,
    OperatorIndex,
    Signature,
    Term,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable

PAD = 0
N_SYMBOLS = 5
CAPACITY = 7
WIDTH = 3

POLICY = DevicePolicy(torch.device("cpu"), torch.float32)

# the root's second child is slot 4 depth-first and slot 2 breadth-first.
DEPTH_FIRST_PARENTS = (Encoding.NO_PARENT, 0, 1, 1, 0, 4, 4)
DEPTH_FIRST_DEPTHS = (0, 1, 2, 2, 1, 2, 2)
BREADTH_FIRST_PARENTS = (Encoding.NO_PARENT, 0, 0, 1, 1, 2, 2)
BREADTH_FIRST_DEPTHS = (0, 1, 1, 2, 2, 2, 2)

# the alphabet has padding, two atomic symbols, and two binary symbols.
INDEX = OperatorIndex(
    Signature(
        "view-fixture",
        ("T",),
        (
            Operator("a", "T"),
            Operator("b", "T"),
            Operator("f", "T", arg_sorts=("T", "T")),
            Operator("g", "T", arg_sorts=("T", "T")),
        ),
    )
)
ANY_SYMBOL = (True,) * N_SYMBOLS
NOT_PADDING = (False, True, True, True, True)
LEAF = (True, True, True, False, False)

DEPTH_FIRST_LEGALITY = (NOT_PADDING, ANY_SYMBOL, LEAF, LEAF, ANY_SYMBOL, LEAF, LEAF)
BREADTH_FIRST_LEGALITY = (NOT_PADDING, ANY_SYMBOL, ANY_SYMBOL, LEAF, LEAF, LEAF, LEAF)

# one backward-slash functor under each node order.
DEPTH_FIRST_SYMBOLS = (4, 1, PAD, PAD, 2, PAD, PAD)
BREADTH_FIRST_SYMBOLS = (4, 1, 2, PAD, PAD, PAD, PAD)

TABLE = torch.arange(N_SYMBOLS * WIDTH, dtype=torch.float32).reshape(N_SYMBOLS, WIDTH)
DISTRIBUTION_AXES = ("term", "slot", "symbol")


class Codec(NamedTuple):
    n_symbols: int
    capacity: int
    skeleton: Encoding | None
    legal_symbols: tuple[tuple[bool, ...], ...]
    index: OperatorIndex


def structure(parents: tuple[int, ...], depths: tuple[int, ...]) -> Encoding:
    return Encoding((PAD,) * CAPACITY, parents, depths, (False,) * CAPACITY, None)


DEPTH_FIRST = NeuralCodecView(
    Codec(
        N_SYMBOLS,
        CAPACITY,
        structure(DEPTH_FIRST_PARENTS, DEPTH_FIRST_DEPTHS),
        DEPTH_FIRST_LEGALITY,
        INDEX,
    )
)
BREADTH_FIRST = NeuralCodecView(
    Codec(
        N_SYMBOLS,
        CAPACITY,
        structure(BREADTH_FIRST_PARENTS, BREADTH_FIRST_DEPTHS),
        BREADTH_FIRST_LEGALITY,
        INDEX,
    )
)
UNSTRUCTURED = NeuralCodecView(
    Codec(N_SYMBOLS, CAPACITY, None, DEPTH_FIRST_LEGALITY, INDEX)
)


def row(
    symbols: tuple[int, ...],
    parents: tuple[int, ...] = DEPTH_FIRST_PARENTS,
    depths: tuple[int, ...] = DEPTH_FIRST_DEPTHS,
    payloads: tuple[int, ...] | None = None,
) -> Encoding:
    return Encoding(
        symbols, parents, depths, tuple(code != PAD for code in symbols), payloads
    )


def heap_children(parent: int) -> tuple[int, int]:
    return (2 * parent + 1, 2 * parent + 2)


def children_of(view: NeuralCodecView[CodecMetadata], parent: int) -> tuple[int, int]:
    return (view.child_index(parent, 0), view.child_index(parent, 1))


def one_hot(codes: torch.Tensor) -> LogDistribution:
    hot = torch.nn.functional.one_hot(codes, N_SYMBOLS).to(torch.float32)

    return LogDistribution(hot.log(), DISTRIBUTION_AXES, normalized_over="symbol")


def encodings(
    view: NeuralCodecView[CodecMetadata],
) -> st.SearchStrategy[Encoding]:
    legality = view.legality(policy=POLICY).tolist()

    return st.builds(
        row,
        st.tuples(
            *(
                st.sampled_from([code for code, legal in enumerate(slot) if legal])
                for slot in legality
            )
        ),
        st.tuples(*(st.integers(-1, CAPACITY - 1) for _ in range(CAPACITY))),
        st.tuples(*(st.integers(0, 3) for _ in range(CAPACITY))),
    )


def test_the_child_index_of_a_depth_first_layout_is_not_the_heap_formula() -> None:
    inner = (0, 1, 2)

    assert [children_of(BREADTH_FIRST, at) for at in inner] == [
        heap_children(at) for at in inner
    ]
    assert children_of(DEPTH_FIRST, 0) == (1, 4)
    assert children_of(DEPTH_FIRST, 1) == (2, 3)
    assert children_of(DEPTH_FIRST, 4) == (5, 6)
    assert children_of(DEPTH_FIRST, 0) != heap_children(0)
    assert DEPTH_FIRST.branching == 2


def test_a_layout_of_one_slot_branches_nowhere() -> None:
    root_only = NeuralCodecView(
        Codec(
            N_SYMBOLS,
            1,
            Encoding((PAD,), (Encoding.NO_PARENT,), (0,), (False,), None),
            (NOT_PADDING,),
            INDEX,
        )
    )

    assert root_only.branching == 0

    with pytest.raises(IndexError, match="slot 0 has 0 children"):
        root_only.child_index(0, 0)


def test_a_leaf_has_no_child_to_index() -> None:
    with pytest.raises(IndexError, match="slot 2 has 0 children"):
        DEPTH_FIRST.child_index(2, 0)


def test_a_slot_outside_the_layout_has_no_child_to_index() -> None:
    with pytest.raises(IndexError, match="holds 7 slots, so slot 9 is not one"):
        DEPTH_FIRST.child_index(9, 0)


def test_a_negative_slot_is_not_read_from_the_far_end() -> None:
    with pytest.raises(ValueError, match="parent is negative"):
        DEPTH_FIRST.child_index(-1, 0)


def test_a_layout_the_term_settles_publishes_no_child_index() -> None:
    assert not UNSTRUCTURED.path_addressed

    with pytest.raises(ValueError, match="not path addressed"):
        UNSTRUCTURED.child_index(0, 0)

    with pytest.raises(ValueError, match="not path addressed"):
        _ = UNSTRUCTURED.branching


def test_a_path_addressed_layout_serves_every_relation() -> None:
    assert DEPTH_FIRST.served == frozenset(StructuralRelation)
    assert BREADTH_FIRST.served == frozenset(StructuralRelation)


def test_a_layout_the_term_settles_serves_the_position_alone() -> None:
    assert UNSTRUCTURED.served == frozenset({StructuralRelation.POSITION})


def test_serving_a_declaration_binds_it_to_the_layout() -> None:
    declaration = {
        StructuralRelation.POSITION: Consumption.EMBEDDING,
        StructuralRelation.PARENT: Consumption.RECURRENCE,
    }
    served = DEPTH_FIRST.serve(declaration)

    assert isinstance(served, DeclaredStructure)
    assert served.declaration == declaration
    assert served.served == DEPTH_FIRST.served
    assert served.parent(policy=POLICY).tolist() == [
        [other == parent for other in range(CAPACITY)] for parent in DEPTH_FIRST_PARENTS
    ]


def test_a_served_declaration_is_fixed_when_it_is_served() -> None:
    declaration = {StructuralRelation.POSITION: Consumption.EMBEDDING}
    served = DEPTH_FIRST.serve(declaration)
    declaration[StructuralRelation.PARENT] = Consumption.RECURRENCE

    assert StructuralRelation.PARENT not in served.declaration

    with pytest.raises(ValueError, match="does not name parent"):
        served.parent(policy=POLICY)

    with pytest.raises(TypeError, match="does not support item assignment"):
        cast("dict[StructuralRelation, Consumption]", served.declaration)[
            StructuralRelation.PARENT
        ] = Consumption.RECURRENCE


def test_a_layout_the_term_settles_serves_no_declaration_beyond_position() -> None:
    assert UNSTRUCTURED.serve({StructuralRelation.POSITION: Consumption.MASK}).position(
        policy=POLICY
    ).tolist() == list(range(CAPACITY))

    with pytest.raises(ValueError, match="names depth, which this codec does not"):
        UNSTRUCTURED.serve({StructuralRelation.DEPTH: Consumption.EMBEDDING})


def test_a_layout_the_term_settles_still_places_a_batch() -> None:
    batch = BatchEncoding((row(DEPTH_FIRST_SYMBOLS),))

    assert UNSTRUCTURED.place_batch(batch, policy=POLICY).lengths.tolist() == [3]


def test_the_legality_mask_is_the_codecs_table_on_the_runs_device() -> None:
    legality = DEPTH_FIRST.legality(policy=POLICY)

    assert legality.dtype is torch.bool
    assert legality.device == POLICY.device
    assert legality.tolist() == [
        [False, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, True, False, False],
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, True, False, False],
    ]


def test_a_batch_crosses_channel_by_channel_with_nothing_summarized() -> None:
    encoding = row(DEPTH_FIRST_SYMBOLS)
    placed = DEPTH_FIRST.place_batch(BatchEncoding((encoding,)), policy=POLICY)

    assert placed.term_count == 1
    assert placed.width == CAPACITY
    assert placed.device == POLICY.device
    assert placed.symbols.tolist() == [list(DEPTH_FIRST_SYMBOLS)]
    assert placed.parents.tolist() == [list(DEPTH_FIRST_PARENTS)]
    assert placed.depths.tolist() == [list(DEPTH_FIRST_DEPTHS)]
    assert placed.mask.tolist() == [list(encoding.occupied)]
    assert placed.lengths.tolist() == [encoding.length]
    assert placed.payloads is None


def test_one_term_under_two_orders_places_to_two_different_tensors() -> None:
    first = DEPTH_FIRST.place_batch(
        BatchEncoding((row(DEPTH_FIRST_SYMBOLS),)), policy=POLICY
    )
    second = BREADTH_FIRST.place_batch(
        BatchEncoding(
            (row(BREADTH_FIRST_SYMBOLS, BREADTH_FIRST_PARENTS, BREADTH_FIRST_DEPTHS),)
        ),
        policy=POLICY,
    )

    assert first.symbols.tolist() != second.symbols.tolist()
    assert first.lengths.tolist() == second.lengths.tolist()


def test_a_payload_channel_crosses_with_the_rest() -> None:
    payloads = (0, 7, 0, 0, 9, 0, 0)
    batch = BatchEncoding((row(DEPTH_FIRST_SYMBOLS, payloads=payloads),))
    placed = DEPTH_FIRST.place_batch(batch, policy=POLICY)

    assert placed.payloads is not None
    assert placed.payloads.tolist() == [list(payloads)]


def test_a_batch_with_a_payload_channel_for_some_terms_alone_is_refused() -> None:
    batch = BatchEncoding(
        (
            row(DEPTH_FIRST_SYMBOLS, payloads=(0,) * CAPACITY),
            row(DEPTH_FIRST_SYMBOLS),
        )
    )

    with pytest.raises(ValueError, match="1 of 2 terms have a payload channel"):
        DEPTH_FIRST.place_batch(batch, policy=POLICY)


def test_a_row_of_another_width_than_the_capacity_is_refused() -> None:
    narrow = Encoding((1, 2), (Encoding.NO_PARENT, 0), (0, 1), (True, True))

    with pytest.raises(ValueError, match="term 0 is 2 slots wide"):
        DEPTH_FIRST.place_batch(BatchEncoding((narrow,)), policy=POLICY)


def test_a_symbol_outside_the_alphabet_is_refused(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_view, on=True)
    batch = BatchEncoding((row((4, 1, PAD, PAD, 9, PAD, PAD)),))

    with pytest.raises(ValueError, match="slot 4 holds symbol 9 beside an alphabet"):
        DEPTH_FIRST.place_batch(batch, policy=POLICY)


def test_a_symbol_the_codec_forbids_at_a_slot_is_refused(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_view, on=True)
    batch = BatchEncoding((row((PAD, 1, PAD, PAD, 2, PAD, PAD)),))

    with pytest.raises(ValueError, match="slot 0 holds symbol 0, which this codec"):
        DEPTH_FIRST.place_batch(batch, policy=POLICY)


def test_a_parent_naming_no_slot_is_refused(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_view, on=True)
    batch = BatchEncoding(
        (row(DEPTH_FIRST_SYMBOLS, (Encoding.NO_PARENT, 0, 1, 1, 9, 4, 4)),)
    )

    with pytest.raises(ValueError, match="slot 4 has parent 9, which names no slot"):
        DEPTH_FIRST.place_batch(batch, policy=POLICY)


def test_a_negative_depth_is_refused(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(_view, on=True)
    batch = BatchEncoding((row(DEPTH_FIRST_SYMBOLS, depths=(0, 1, 2, 2, -1, 2, 2)),))

    with pytest.raises(ValueError, match="the depth of term 0 slot 4 is negative"):
        DEPTH_FIRST.place_batch(batch, policy=POLICY)


def test_a_batch_of_no_terms_keeps_the_codecs_width() -> None:
    placed = DEPTH_FIRST.place_batch(BatchEncoding(()), policy=POLICY)

    assert placed.term_count == 0
    assert placed.width == CAPACITY
    assert placed.lengths.tolist() == []


@given(encoding=encodings(DEPTH_FIRST))
def test_an_encoding_reaches_the_tensors_unchanged(encoding: Encoding) -> None:
    placed = DEPTH_FIRST.place_batch(BatchEncoding((encoding,)), policy=POLICY)

    assert placed.symbols.tolist() == [list(encoding.symbols)]
    assert placed.parents.tolist() == [list(encoding.parents)]
    assert placed.depths.tolist() == [list(encoding.depths)]
    assert placed.mask.tolist() == [list(encoding.occupied)]


@given(first=encodings(DEPTH_FIRST), second=encodings(DEPTH_FIRST))
def test_two_encodings_reach_two_tensors_exactly_when_they_differ(
    first: Encoding, second: Encoding
) -> None:
    placed = DEPTH_FIRST.place_batch(BatchEncoding((first, second)), policy=POLICY)
    channels = (placed.symbols, placed.parents, placed.depths, placed.mask)
    alike = all(bool(channel[0].equal(channel[1])) for channel in channels)

    assert alike == (first == second)


def test_the_arities_are_read_from_the_operator_index_with_zero_at_padding() -> None:
    assert DEPTH_FIRST.arities == (0, 0, 0, 2, 2)
    assert UNSTRUCTURED.arities == (0, 0, 0, 2, 2)
    assert len(DEPTH_FIRST.arities) == N_SYMBOLS


SIGNATURE = INDEX.signature
LAYOUT = CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3)
CODEC = TermCodec(INDEX, LAYOUT)
SHIPPED = NeuralCodecView(CODEC)
STORE = TermStore(SIGNATURE)


@settings(max_examples=60, deadline=None)
@given(term=terms(STORE, sort="T", max_size=CAPACITY, closed=True))
def test_assembling_the_symbols_of_a_placed_term_returns_its_encoding(
    term: Term,
) -> None:
    try:
        encoding = CODEC.encode(term)

    except EncodingOverflowError:
        reject()

    batch = BatchEncoding((encoding, CODEC.encode(STORE.op(INDEX.operator(1)))))
    placed = SHIPPED.place_batch(batch, policy=POLICY)

    assembled = SHIPPED.assemble_batch(placed.symbols)

    assert assembled == batch
    assert CODEC.decode(assembled.rows[0], STORE) is term


def test_assembly_reaches_the_slots_a_term_s_arguments_fill_and_no_other() -> None:
    symbols = torch.tensor([[3, 4, 1, 2, 1, PAD, PAD]], dtype=torch.int64)

    assembled = DEPTH_FIRST.assemble_batch(symbols)

    assert assembled.rows[0].occupied == (True, True, True, True, True, False, False)
    assert assembled.rows[0].parents == DEPTH_FIRST_PARENTS
    assert assembled.rows[0].depths == DEPTH_FIRST_DEPTHS
    assert assembled.rows[0].payloads is None


def test_assembly_of_no_terms_holds_no_row() -> None:
    assembled = DEPTH_FIRST.assemble_batch(
        torch.zeros((0, CAPACITY), dtype=torch.int64)
    )

    assert assembled.rows == ()


@pytest.mark.parametrize(
    ("symbols", "message"),
    [
        ((3, PAD, 1, 2, 1, PAD, PAD), "slot 1 holds pad, and a node fills it"),
        (
            (1, 2, PAD, PAD, PAD, PAD, PAD),
            "slot 1 holds symbol 2, and no node fills it",
        ),
        ((3, 1, 2, PAD, 1, PAD, PAD), "slot 2 holds symbol 2, and no node fills it"),
        ((PAD, PAD, PAD, PAD, PAD, PAD, PAD), "slot 0 holds pad, and a node fills it"),
        ((3, 1, PAD, PAD, 9, PAD, PAD), "slot 4 holds symbol 9 beside an alphabet"),
    ],
)
def test_an_assignment_that_denotes_no_term_is_refused(
    symbols: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DEPTH_FIRST.assemble_batch(torch.tensor([symbols], dtype=torch.int64))


def test_assembly_needs_one_int64_symbol_per_slot_of_a_path_addressed_layout() -> None:
    with pytest.raises(TypeError, match="int64"):
        DEPTH_FIRST.assemble_batch(torch.zeros((1, CAPACITY), dtype=torch.int32))

    with pytest.raises(ValueError, match=rf"over \(term, {CAPACITY}\), not \(1, 3\)"):
        DEPTH_FIRST.assemble_batch(torch.zeros((1, 3), dtype=torch.int64))

    with pytest.raises(ValueError, match="not path addressed"):
        UNSTRUCTURED.assemble_batch(torch.zeros((1, CAPACITY), dtype=torch.int64))


def test_the_soft_lookup_of_a_one_hot_is_the_hard_lookup() -> None:
    batch = BatchEncoding((row(DEPTH_FIRST_SYMBOLS), row(BREADTH_FIRST_SYMBOLS)))
    placed = DEPTH_FIRST.place_batch(batch, policy=POLICY)
    hard = DEPTH_FIRST.symbol_rows(placed, TABLE)
    soft = DEPTH_FIRST.soft_symbols(one_hot(placed.symbols), TABLE)

    assert hard.axes == ("term", "slot", "width")
    assert soft.axes == hard.axes
    assert torch.allclose(soft.tensor, hard.tensor, atol=1e-6)
    assert hard.mask is placed.mask


def test_the_soft_lookup_of_a_split_decision_is_the_mean_of_two_rows() -> None:
    scores = torch.full((1, 1, N_SYMBOLS), -math.inf)
    scores[0, 0, 1] = math.log(0.5)
    scores[0, 0, 2] = math.log(0.5)
    split = LogDistribution(scores, DISTRIBUTION_AXES, normalized_over="symbol")

    mixed = DEPTH_FIRST.soft_symbols(split, TABLE)

    assert torch.allclose(mixed.tensor[0, 0], (TABLE[1] + TABLE[2]) / 2, atol=1e-6)


def test_an_unnormalized_score_is_not_a_distribution_over_symbols() -> None:
    scores = LogDistribution(torch.zeros(1, 1, N_SYMBOLS), DISTRIBUTION_AXES)

    with pytest.raises(ValueError, match="names no axis its mass sums over"):
        DEPTH_FIRST.soft_symbols(scores, TABLE)


def test_a_symbol_axis_that_is_not_innermost_is_refused() -> None:
    hot = torch.nn.functional.one_hot(torch.zeros(1, 2, dtype=torch.int64), N_SYMBOLS)
    turned = LogDistribution(
        hot.to(torch.float32).log().transpose(1, 2),
        ("term", "symbol", "slot"),
        normalized_over="symbol",
    )

    with pytest.raises(ValueError, match="'symbol' is at 1 rather than innermost"):
        DEPTH_FIRST.soft_symbols(turned, TABLE)


def test_a_distribution_over_another_alphabet_is_refused() -> None:
    hot = torch.nn.functional.one_hot(torch.zeros(1, 1, dtype=torch.int64), 4)
    narrow = LogDistribution(
        hot.to(torch.float32).log(), DISTRIBUTION_AXES, normalized_over="symbol"
    )

    with pytest.raises(ValueError, match="holds 4 symbols beside an alphabet of 5"):
        DEPTH_FIRST.soft_symbols(narrow, TABLE)


def test_a_mask_that_covers_the_symbol_axis_is_refused() -> None:
    hot = torch.nn.functional.one_hot(torch.zeros(1, 1, dtype=torch.int64), N_SYMBOLS)
    masked = LogDistribution(
        hot.to(torch.float32).log(),
        DISTRIBUTION_AXES,
        torch.ones(1, 1, N_SYMBOLS, dtype=torch.bool),
        normalized_over="symbol",
    )

    with pytest.raises(ValueError, match="mask covers the symbol axis"):
        DEPTH_FIRST.soft_symbols(masked, TABLE)


def test_a_table_holding_another_number_of_rows_than_symbols_is_refused() -> None:
    placed = DEPTH_FIRST.place_batch(
        BatchEncoding((row(DEPTH_FIRST_SYMBOLS),)), policy=POLICY
    )

    with pytest.raises(ValueError, match="one row for each of 5 symbols, not 4"):
        DEPTH_FIRST.symbol_rows(placed, TABLE[:4])


def test_a_table_that_is_not_a_matrix_is_refused() -> None:
    placed = DEPTH_FIRST.place_batch(
        BatchEncoding((row(DEPTH_FIRST_SYMBOLS),)), policy=POLICY
    )

    with pytest.raises(ValueError, match="rank is 2, not 3"):
        DEPTH_FIRST.symbol_rows(placed, TABLE.unsqueeze(0))
