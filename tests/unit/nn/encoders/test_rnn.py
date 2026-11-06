from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Never, cast, override

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance._sites import SiteObservation
from tests.conformance.observable import ObservableConformance, ObservableSuite
from tests.conformance.term_encoder import (
    SoftTermEncoderSuite,
    TermEncoderConformance,
    TermEncoderSuite,
)

from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme, ArityUnsupportedError, RoleFillerScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.encoders import rnn as rnn_module
from lofi.nn.ports import Consumption, Reading, StructuralRelation
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.probe import ensure_publishes
from lofi.registry import RegistryError
from lofi.spec.layers import BUILTIN_RECURRENT_CELLS, RecurrentCell
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    Term,
    TermCodec,
    TermStore,
    Vocabulary,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.nn.codec import TermCoding
    from lofi.nn.ports import ArityScheme


WIDTH: Final = 8
LAYERS: Final = 2
CAP: Final = 0.75

COMPONENT: Final = ComponentId("tree-rnn")
TERM_SITE: Final = Site(COMPONENT, Role.TERM_ENCODING)
NODE_SITE: Final = Site(COMPONENT, Role.NODE_ENCODING)


def hidden(layer: int) -> Site:
    return Site(COMPONENT, Role.RECURRENT_HIDDEN, (layer,))


def gate(layer: int, name: str) -> Site:
    return Site(COMPONENT, Role.GATE, (layer, name))


ATOM: Final = Operator("a", "T", family="primitive")
OTHER_ATOM: Final = Operator("b", "T", family="primitive")
UNARY: Final = Operator("raise", "T", arg_sorts=("T",), family="constructor")
BINARY: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
OTHER_BINARY: Final = Operator("join", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature(
    "rnn-fixture", ("T",), (ATOM, OTHER_ATOM, UNARY, BINARY, OTHER_BINARY)
)
INDEX: Final = OperatorIndex(SIGNATURE)
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=3)
CODEC: Final = TermCodec(INDEX, LAYOUT)
VIEW: Final = NeuralCodecView(CODEC)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": LAYOUT.depth})
CAPACITY: Final = VIEW.capacity
N_SYMBOLS: Final = VIEW.n_symbols
REACHED: Final = (ATOM, UNARY, BINARY)

PARENTS: Final = LAYOUT.skeleton.parents
DEPTHS: Final = LAYOUT.skeleton.depths
LIVE: Final = (True,) * CAPACITY

A: Final = INDEX.index(ATOM)
B: Final = INDEX.index(OTHER_ATOM)
RAISE: Final = INDEX.index(UNARY)
PAIR: Final = INDEX.index(BINARY)
JOIN: Final = INDEX.index(OTHER_BINARY)
SHAPE: Final = (JOIN, PAIR, RAISE, A, B, A, B)


class Codec(NamedTuple):
    n_symbols: int
    capacity: int
    skeleton: Encoding | None
    legal_symbols: tuple[tuple[bool, ...], ...]
    index: OperatorIndex


def fixed(parents: tuple[int, ...], depths: tuple[int, ...]) -> NeuralCodecView:
    skeleton = Encoding(
        (OperatorIndex.PAD,) * len(parents), parents, depths, (False,) * len(parents)
    )

    metadata = Codec(
        N_SYMBOLS,
        len(parents),
        skeleton,
        ((True,) * N_SYMBOLS,) * len(parents),
        INDEX,
    )

    return NeuralCodecView(cast("TermCoding", metadata))


LOOSE: Final = NeuralCodecView(TermCodec(INDEX, PreorderArityLayout(CAPACITY)))
CHAIN: Final = fixed((Encoding.NO_PARENT, 0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6))
LOPSIDED: Final = fixed((Encoding.NO_PARENT, 0, 0, 1, 1, 3, 3), (0, 1, 1, 2, 2, 3, 3))
TERNARY: Final = fixed((Encoding.NO_PARENT, 0, 0, 0, 1, 1, 1), (0, 1, 1, 1, 2, 2, 2))


def row(
    symbols: tuple[int, ...],
    *,
    parents: tuple[int, ...] = PARENTS,
    depths: tuple[int, ...] = DEPTHS,
    occupied: tuple[bool, ...] = LIVE,
) -> Encoding:
    return Encoding(symbols, parents, depths, occupied)


def under(held: NeuralCodecView, symbols: tuple[int, ...]) -> BatchEncoding:
    skeleton = held.codec.skeleton

    if skeleton is None:
        pytest.fail("the fixture codec fixes no slot structure")

    return BatchEncoding(
        (row(symbols, parents=skeleton.parents, depths=skeleton.depths),)
    )


def terms() -> BatchEncoding:
    store = TermStore(SIGNATURE)
    left = store.op(ATOM)
    right = store.op(OTHER_ATOM)

    return CODEC.encode_batch(
        (
            store.op(
                OTHER_BINARY,
                store.op(BINARY, left, right),
                store.op(OTHER_BINARY, right, left),
            ),
            store.op(BINARY, right, left),
        )
    )


TERMS: Final = terms()


def scheme(*, width: int = WIDTH, reached: Sequence[Operator] = REACHED) -> ArityScheme:
    return ArityHeadScheme(width=width, reached=reached)


def build(
    *,
    cell: RecurrentCell = "gru",
    layers: int = LAYERS,
    width: int = WIDTH,
    codec: NeuralCodecView = VIEW,
    norm_cap: float | None = None,
    inventory: Literal["uncached", "shared-dag"] = "uncached",
) -> TreeRnnEncoder:
    torch.manual_seed(17)

    return TreeRnnEncoder(
        codec=codec,
        choices=CHOICES,
        component=COMPONENT,
        scheme=scheme(width=width),
        cell=cell,
        layers=layers,
        norm_cap=norm_cap,
        inventory=inventory,
    )


def test_shared_dag_and_uncached_hard_embeddings_and_gradients_agree() -> None:
    uncached = build(inventory="uncached")
    shared = build(inventory="shared-dag")
    shared.load_state_dict(uncached.state_dict())

    uncached_values = uncached.encode(TERMS).tensor
    shared_values = shared.encode(TERMS).tensor
    uncached_values.square().sum().backward()
    shared_values.square().sum().backward()

    assert shared.inventory == "shared-dag"
    assert torch.allclose(shared_values, uncached_values, atol=1e-6, rtol=1e-6)

    for (uncached_name, uncached_parameter), (shared_name, shared_parameter) in zip(
        uncached.named_parameters(), shared.named_parameters(), strict=True
    ):
        assert uncached_name == shared_name
        assert (uncached_parameter.grad is None) is (shared_parameter.grad is None)

        if uncached_parameter.grad is None or shared_parameter.grad is None:
            continue

        assert torch.allclose(
            shared_parameter.grad,
            uncached_parameter.grad,
            atol=1e-6,
            rtol=1e-6,
        )


def test_shared_dag_restores_node_embeddings_to_the_codec_slots() -> None:
    uncached = build(inventory="uncached")
    shared = build(inventory="shared-dag")
    shared.load_state_dict(uncached.state_dict())

    expected = uncached.encode_nodes(TERMS)
    found = shared.encode_nodes(TERMS)

    assert found.mask is not None
    assert expected.mask is not None
    assert torch.equal(found.mask, expected.mask)
    assert torch.allclose(found.tensor, expected.tensor, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("cell", "gates"),
    [
        ("elman-tanh", {"candidate"}),
        ("elman-relu", {"candidate"}),
        ("gru", {"candidate", "reset", "update"}),
        ("lstm", {"candidate", "forget", "input", "output"}),
    ],
)
def test_each_registered_cell_encodes_and_publishes_its_activations(
    cell: RecurrentCell,
    gates: set[str],
) -> None:
    subject = build(cell=cell, layers=1)

    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        encoded = subject.encode(TERMS)

    assert subject.cell == cell
    assert bool(encoded.tensor.isfinite().all())
    assert {
        site.key[1] for site in subject.site_map() if site.role is Role.GATE
    } == gates
    assert {site.key[1] for site in scope.readings if site.role is Role.GATE} == gates


def test_the_registered_cell_list_is_the_tree_encoder_cell_list() -> None:
    assert {build(cell=cell, layers=1).cell for cell in BUILTIN_RECURRENT_CELLS} == (
        set(BUILTIN_RECURRENT_CELLS)
    )


def test_an_unregistered_cell_is_refused_when_the_encoder_is_built() -> None:
    with pytest.raises(RegistryError, match="not-a-cell"):
        build(cell="not-a-cell")


def one_hot(terms: BatchEncoding) -> LogDistribution:
    codes = torch.tensor(
        tuple(CHOICES.codes(row) for row in terms.rows), dtype=torch.int64
    )
    mass = torch.nn.functional.one_hot(codes, CHOICES.width).double()

    return LogDistribution(
        mass.log(), ("term", "slot", "choice"), normalized_over="choice"
    )


def observations(subject: TreeRnnEncoder) -> Mapping[Site, Reading]:
    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        subject.encode(TERMS)

    return scope.readings


def conformance() -> TermEncoderConformance[TreeRnnEncoder]:
    return TermEncoderConformance(
        build=build,
        terms=TERMS,
        choices=lambda terms: tuple(CHOICES.codes(row) for row in terms.rows),
        choice_width=CHOICES.width,
        sites=SiteObservation(
            build=build,
            observe=observations,
            read=lambda subject: (
                subject.encode(TERMS).tensor,
                subject.encode_nodes(TERMS).tensor,
            ),
        ),
        submodules=lambda subject: tuple(
            name for name, _ in subject.named_modules() if name
        ),
        place=lambda subject, placement: subject.to(torch.device(placement)),
    )


class TestTreeRnnEncoder(TermEncoderSuite, SoftTermEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TermEncoderConformance[TreeRnnEncoder]:
        return conformance()


def observable_conformance() -> ObservableConformance[TreeRnnEncoder, Reading]:
    return ObservableConformance(
        build=build,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        run=lambda subject: (subject.encode(TERMS).tensor,),
        configuration=lambda subject: {
            "layers": subject.layers,
            "output_width": subject.output_width,
        },
        submodules=lambda subject: frozenset(
            name for name, _ in subject.named_modules()
        ),
        placements=lambda subject: tuple(
            str(tensor.device) for tensor in (*subject.parameters(), *subject.buffers())
        ),
    )


class TestRnnObservations(ObservableSuite):
    @pytest.fixture
    def conformance(self) -> ObservableConformance[TreeRnnEncoder, Reading]:
        return observable_conformance()


class RecordingScheme(ArityHeadScheme):
    def __init__(self, *, width: int, reached: Sequence[Operator]) -> None:
        super().__init__(width=width, reached=reached)
        self.asked: list[int] = []
        self.handed: list[tuple[torch.Tensor, ...]] = []

    @override
    def combine(self, children: Sequence[Encoded], /) -> Encoded:
        self.asked.append(len(children))
        self.handed.append(tuple(child.tensor.detach().clone() for child in children))

        return super().combine(children)


def leaves() -> st.SearchStrategy[Term]:
    return st.sampled_from((ATOM, OTHER_ATOM)).map(STORE.op)


def branches(
    below: st.SearchStrategy[Term],
) -> st.SearchStrategy[Term]:
    unary = below.map(lambda child: STORE.op(UNARY, child))
    binary = st.tuples(st.sampled_from((BINARY, OTHER_BINARY)), below, below).map(
        lambda drawn: STORE.op(drawn[0], drawn[1], drawn[2])
    )

    return st.one_of(unary, binary)


STORE: Final = TermStore(SIGNATURE)
TERM: Final = st.one_of(leaves(), branches(st.one_of(leaves(), branches(leaves()))))
BATCH: Final = st.lists(TERM, min_size=1, max_size=4).map(
    lambda drawn: CODEC.encode_batch(tuple(drawn))
)


@given(batch=BATCH)
@settings(max_examples=30, deadline=None)
def test_a_one_hot_distribution_encodes_as_the_term_over_generated_terms(
    batch: BatchEncoding,
) -> None:
    subject = build(norm_cap=CAP)

    with torch.no_grad():
        subject.get_parameter("_embedding._table.weight").mul_(4.0)

    hard = subject.encode(batch)
    soft = subject.encode_soft(one_hot(batch), structure=batch)

    assert hard.axes == soft.axes
    assert torch.allclose(hard.tensor, soft.tensor, atol=1e-6)


def test_constant_payload_identity_changes_the_term_encoding() -> None:
    signature = Signature(
        "payload-aware-rnn",
        ("T",),
        (),
        vocabularies=(Vocabulary("T", ("sleep", "snore")),),
    )
    constant = signature.constant("T")
    store = TermStore(signature)
    codec = TermCodec(
        OperatorIndex(signature), CompleteTree(signature, (constant,), depth=1)
    )
    index = ChoiceIndex(signature, depth_bounds={"T": 1})
    subject = TreeRnnEncoder(
        codec=NeuralCodecView(codec),
        choices=index,
        component=COMPONENT,
        scheme=ArityHeadScheme(width=WIDTH, reached=(constant,)),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )
    terms = codec.encode_batch(
        (store.op(constant, payload=0), store.op(constant, payload=1))
    )

    encoded = subject.encode(terms).tensor

    assert not torch.allclose(encoded[0], encoded[1])


def test_the_encoder_declares_the_parent_as_a_recurrence_and_nothing_else() -> None:
    subject = build()

    assert dict(subject.structure) == {
        StructuralRelation.PARENT: Consumption.RECURRENCE
    }

    with pytest.raises(TypeError):
        cast("dict[StructuralRelation, Consumption]", subject.structure)[
            StructuralRelation.POSITION
        ] = Consumption.EMBEDDING


def test_a_codec_serving_position_alone_is_refused_by_name() -> None:
    assert LOOSE.served == {StructuralRelation.POSITION}

    with pytest.raises(ValueError, match="does not serve") as caught:
        build(codec=LOOSE)

    assert "names parent" in str(caught.value)


def test_the_norm_cap_projects_the_rows_both_paths_read() -> None:
    subject = build(norm_cap=CAP)

    with torch.no_grad():
        subject.get_parameter("_embedding._table.weight").mul_(10.0)

    once = subject.encode(TERMS).tensor.clone()
    soft = subject.encode_soft(one_hot(TERMS), structure=TERMS).tensor

    with torch.no_grad():
        subject.get_parameter("_embedding._table.weight").mul_(10.0)

    assert torch.allclose(once, soft, atol=1e-6)
    assert torch.allclose(once, subject.encode(TERMS).tensor, atol=1e-6)
    assert not torch.allclose(once, build().encode(TERMS).tensor, atol=1e-6)


def test_the_fixture_is_what_the_codec_encodes() -> None:
    assert TERMS.rows[0].symbols == (JOIN, PAIR, JOIN, A, B, B, A)
    assert TERMS.rows[0].occupied == LIVE
    assert TERMS.rows[1].symbols == (PAIR, B, A, *((OperatorIndex.PAD,) * 4))
    assert TERMS.rows[1].occupied == (True, True, True, False, False, False, False)
    assert TERMS.rows[0].parents == TERMS.rows[1].parents == PARENTS


def test_equal_leaves_at_different_positions_have_equal_states() -> None:
    same = BatchEncoding((row((A,) * CAPACITY),))
    subject = build()
    nodes = subject.encode_nodes(same).tensor[0]

    assert torch.allclose(nodes[3], nodes[4], atol=1e-6)
    assert torch.allclose(nodes[3], nodes[5], atol=1e-6)


def test_ordinary_terms_still_distinguish_operators_and_tree_structure() -> None:
    changed_root = row((PAIR, PAIR, JOIN, A, B, B, A))
    pruned_right_branch = row(
        (JOIN, PAIR, A, A, B, OperatorIndex.PAD, OperatorIndex.PAD)
    )
    encoded = (
        build()
        .encode(BatchEncoding((TERMS.rows[0], changed_root, pruned_right_branch)))
        .tensor
    )

    assert not torch.allclose(encoded[0], encoded[1], atol=1e-6)
    assert not torch.allclose(encoded[0], encoded[2], atol=1e-6)


def test_one_address_starts_from_one_state_across_terms() -> None:
    pair = BatchEncoding((row((A,) * CAPACITY), row((A, A, A, B, A, A, A))))
    subject = build()
    nodes = subject.encode_nodes(pair).tensor

    assert torch.allclose(nodes[0, 5], nodes[1, 5], atol=1e-6)
    assert not torch.allclose(nodes[0, 3], nodes[1, 3], atol=1e-6)


def test_one_learned_start_is_shared_by_every_leaf_position() -> None:
    subject = build()
    leaves = subject.encode_nodes(BatchEncoding((row((A,) * CAPACITY),))).tensor[0]
    addresses = (3, 4, 5, 6)

    for first in addresses:
        for second in addresses:
            if first < second:
                assert torch.allclose(leaves[first], leaves[second], atol=1e-6)


def test_combination_is_asked_for_a_rank_and_never_for_an_operator() -> None:
    torch.manual_seed(17)
    recorder = RecordingScheme(width=WIDTH, reached=REACHED)
    subject = TreeRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        scheme=recorder,
        cell="gru",
        layers=LAYERS,
        norm_cap=None,
        inventory="uncached",
    )
    subject.encode(TERMS)

    assert recorder.asked == [2, 2]

    recorder.asked.clear()
    chained = TreeRnnEncoder(
        codec=CHAIN,
        choices=CHOICES,
        component=COMPONENT,
        scheme=recorder,
        cell="gru",
        layers=LAYERS,
        norm_cap=None,
        inventory="uncached",
    )
    chained.encode(under(CHAIN, SHAPE))

    assert recorder.asked == [1] * (CAPACITY - 1)


def test_the_family_is_handed_children_in_argument_order() -> None:
    torch.manual_seed(17)
    recorder = RecordingScheme(width=WIDTH, reached=REACHED)
    subject = TreeRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        scheme=recorder,
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )
    nodes = subject.encode_nodes(TERMS).tensor
    lower, root = recorder.handed

    for parent, (first, second) in ((1, (3, 4)), (2, (5, 6))):
        assert torch.allclose(lower[0][0, parent, 0], nodes[0, first], atol=1e-6)
        assert torch.allclose(lower[1][0, parent, 0], nodes[0, second], atol=1e-6)

    assert torch.allclose(root[0][0, 0, 0], nodes[0, 1], atol=1e-6)
    assert torch.allclose(root[1][0, 0, 0], nodes[0, 2], atol=1e-6)
    assert not torch.allclose(nodes[0, 1], nodes[0, 2], atol=1e-6)


def test_a_leaf_encodes_alike_beside_a_branching_sibling() -> None:
    subject = build(codec=LOPSIDED)
    skeleton = LOPSIDED.codec.skeleton

    if skeleton is None:
        pytest.fail("the fixture codec fixes no slot structure")

    mixed = subject.encode_nodes(under(LOPSIDED, SHAPE)).tensor[0]
    pruned = subject.encode_nodes(
        BatchEncoding(
            (
                row(
                    SHAPE,
                    parents=skeleton.parents,
                    depths=skeleton.depths,
                    occupied=(True, True, True, True, True, False, False),
                ),
            )
        )
    ).tensor[0]
    leaves = subject.encode_nodes(
        BatchEncoding(
            (
                row(
                    SHAPE,
                    parents=skeleton.parents,
                    depths=skeleton.depths,
                    occupied=(True, True, True, False, False, False, False),
                ),
            )
        )
    ).tensor[0]

    assert torch.allclose(mixed[2], leaves[2], atol=1e-6)
    assert torch.allclose(mixed[4], pruned[4], atol=1e-6)
    assert not torch.allclose(mixed[3], pruned[3], atol=1e-6)
    assert not torch.allclose(mixed[1], leaves[1], atol=1e-6)


def test_an_arity_the_family_serves_no_module_at_is_refused() -> None:
    torch.manual_seed(17)
    subject = TreeRnnEncoder(
        codec=TERNARY,
        choices=CHOICES,
        component=COMPONENT,
        scheme=scheme(reached=(ATOM, BINARY)),
        cell="gru",
        layers=LAYERS,
        norm_cap=None,
        inventory="uncached",
    )

    with pytest.raises(ArityUnsupportedError, match="not at 3"):
        subject.encode(under(TERNARY, SHAPE))


def test_the_layer_axis_never_reaches_a_consumer() -> None:
    for layers in (1, 3):
        subject = build(layers=layers)
        terms = subject.encode(TERMS)
        nodes = subject.encode_nodes(TERMS)

        assert subject.layers == layers
        assert terms.tensor.ndim == 2
        assert nodes.tensor.ndim == 3
        assert terms.axes == ("term", "width")
        assert nodes.axes == ("term", "slot", "width")
        assert terms.size("width") == subject.output_width
        assert nodes.size("width") == subject.output_width


def test_stacking_a_layer_changes_what_a_term_encodes_to() -> None:
    shallow = build(layers=1).encode(TERMS).tensor
    deep = build(layers=3).encode(TERMS).tensor

    assert shallow.shape == deep.shape
    assert not torch.allclose(shallow, deep, atol=1e-6)


def test_the_output_width_is_the_one_the_scheme_declares() -> None:
    narrow = build(width=4)
    wide = build(width=16)

    assert narrow.output_width == narrow.scheme.width == 4
    assert wide.output_width == wide.scheme.width == 16
    assert narrow.encode(TERMS).size("width") == 4
    assert wide.encode_nodes(TERMS).size("width") == 16


def test_the_parent_relation_is_read_from_the_codec_and_not_from_the_batch() -> None:
    reparented = BatchEncoding(
        (row(SHAPE, parents=(Encoding.NO_PARENT, 0, 0, 1, 2, 1, 2)),)
    )
    rerooted = BatchEncoding(
        (row(SHAPE, parents=(1, Encoding.NO_PARENT, 0, 1, 1, 2, 2)),)
    )
    subject = build()
    kept = subject.encode_nodes(BatchEncoding((row(SHAPE),))).tensor
    lopsided = build(codec=LOPSIDED).encode_nodes(under(LOPSIDED, SHAPE)).tensor

    assert torch.allclose(kept, subject.encode_nodes(reparented).tensor, atol=1e-6)
    assert torch.allclose(kept, subject.encode_nodes(rerooted).tensor, atol=1e-6)
    assert torch.allclose(
        subject.encode(BatchEncoding((row(SHAPE),))).tensor,
        subject.encode(rerooted).tensor,
        atol=1e-6,
    )
    assert not torch.allclose(kept[0], lopsided[0], atol=1e-6)
    assert not torch.allclose(kept[0, 0], lopsided[0, 0], atol=1e-6)


def test_a_child_reaches_its_parent_and_a_sibling_does_not() -> None:
    subject = build()
    original = subject.encode_nodes(TERMS).tensor[0]
    changed = subject.encode_nodes(
        BatchEncoding((row((JOIN, PAIR, JOIN, B, B, B, A)),))
    ).tensor[0]

    assert not torch.allclose(original[3], changed[3], atol=1e-6)
    assert not torch.allclose(original[1], changed[1], atol=1e-6)
    assert not torch.allclose(original[0], changed[0], atol=1e-6)
    assert torch.allclose(original[2], changed[2], atol=1e-6)
    assert torch.allclose(original[5], changed[5], atol=1e-6)


def test_a_slot_with_children_reads_its_own_constructor_code() -> None:
    subject = build()
    original = subject.encode_nodes(TERMS).tensor[0]
    inner = subject.encode_nodes(
        BatchEncoding((row((JOIN, JOIN, JOIN, A, B, B, A)),))
    ).tensor[0]
    root = subject.encode_nodes(
        BatchEncoding((row((PAIR, PAIR, JOIN, A, B, B, A)),))
    ).tensor[0]

    assert not torch.allclose(original[1], inner[1], atol=1e-6)
    assert not torch.allclose(original[0], inner[0], atol=1e-6)
    assert not torch.allclose(original[0], root[0], atol=1e-6)

    for slot in (2, 3, 4, 5, 6):
        assert torch.allclose(original[slot], inner[slot], atol=1e-6)

    for slot in (1, 2, 3, 4, 5, 6):
        assert torch.allclose(original[slot], root[slot], atol=1e-6)


SINGLE: Final = BatchEncoding(
    (
        row(
            (A, *((OperatorIndex.PAD,) * (CAPACITY - 1))),
            occupied=(True, *((False,) * (CAPACITY - 1))),
        ),
    )
)


def test_a_term_encodes_to_the_vector_at_its_root() -> None:
    subject = build()
    terms = subject.encode(TERMS).tensor
    nodes = subject.encode_nodes(TERMS).tensor

    assert torch.allclose(terms, nodes[:, 0, :], atol=1e-6)


def test_an_unoccupied_slot_is_the_child_of_nothing() -> None:
    occupied = (True, True, True, False, False, False, False)
    held = BatchEncoding((row(SHAPE, occupied=occupied),))
    torch.manual_seed(17)
    recorder = RecordingScheme(width=WIDTH, reached=REACHED)
    subject = TreeRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        scheme=recorder,
        cell="gru",
        layers=LAYERS,
        norm_cap=None,
        inventory="uncached",
    )
    nodes = subject.encode_nodes(held).tensor[0]

    assert recorder.asked == [2]
    assert torch.equal(nodes[3:], torch.zeros_like(nodes[3:]))
    assert not torch.allclose(nodes[1], nodes[2], atol=1e-6)


def test_each_layer_reads_the_one_below_and_the_topmost_is_published() -> None:
    subject = build(layers=2)
    baseline = subject.encode(SINGLE).tensor.clone()

    with torch.no_grad():
        subject.get_parameter("_cells.0.bias_hh").add_(1.0)

    lower = subject.encode(SINGLE).tensor.clone()

    with torch.no_grad():
        subject.get_parameter("_cells.1.bias_hh").add_(1.0)

    upper = subject.encode(SINGLE).tensor

    assert not torch.allclose(baseline, lower, atol=1e-6)
    assert not torch.allclose(lower, upper, atol=1e-6)


def test_every_layer_has_one_learned_leaf_start() -> None:
    subject = build(layers=3)
    subject.encode(SINGLE).tensor.square().sum().backward()
    gradient = subject.get_parameter("_start").grad

    assert gradient is not None
    assert bool(gradient.ne(0.0).all())


def test_padded_symbol_codes_cannot_change_live_outputs() -> None:
    occupied = (True, True, True, False, False, False, False)
    original = BatchEncoding(
        (
            row(
                (
                    5,
                    4,
                    3,
                    OperatorIndex.PAD,
                    OperatorIndex.PAD,
                    OperatorIndex.PAD,
                    OperatorIndex.PAD,
                ),
                occupied=occupied,
            ),
        )
    )
    changed = BatchEncoding((row((JOIN, PAIR, RAISE, B, A, B, A), occupied=occupied),))
    subject = build()
    first = subject.encode_nodes(original)
    second = subject.encode_nodes(changed)

    assert torch.allclose(first.tensor[:, :3], second.tensor[:, :3], atol=1e-6)
    assert torch.equal(first.tensor[:, 3:], torch.zeros_like(first.tensor[:, 3:]))
    assert torch.allclose(
        subject.encode(original).tensor, subject.encode(changed).tensor, atol=1e-6
    )


def test_a_row_holding_no_term_comes_back_masked() -> None:
    empty = BatchEncoding(
        (
            row(
                (A, *((OperatorIndex.PAD,) * (CAPACITY - 1))),
                occupied=(False,) * CAPACITY,
            ),
        )
    )
    subject = build()
    terms = subject.encode(empty)
    soft = subject.encode_soft(one_hot(empty), structure=empty)

    assert terms.mask is not None
    assert terms.mask.tolist() == [False]
    assert torch.equal(terms.tensor, torch.zeros_like(terms.tensor))
    assert soft.mask is not None
    assert torch.equal(soft.mask, terms.mask)
    assert torch.equal(soft.tensor, terms.tensor)


def test_a_batch_of_no_term_encodes_to_no_row() -> None:
    subject = build()
    terms = subject.encode(BatchEncoding(()))

    assert terms.tensor.shape == (0, WIDTH)
    assert subject.encode_nodes(BatchEncoding(())).tensor.shape == (
        0,
        CAPACITY,
        WIDTH,
    )


def test_soft_encoding_uses_the_same_structure_as_hard_encoding() -> None:
    subject = build()
    hard = subject.encode(TERMS)
    soft = subject.encode_soft(one_hot(TERMS), structure=TERMS)

    assert hard.axes == soft.axes
    assert torch.allclose(hard.tensor, soft.tensor, atol=1e-6)


def test_soft_encoding_takes_symbols_only_from_the_distribution() -> None:
    swapped = {A: B, B: A, PAIR: JOIN, JOIN: PAIR}
    changed = BatchEncoding(
        tuple(
            Encoding(
                tuple(swapped.get(code, code) for code in term.symbols),
                term.parents,
                term.depths,
                term.occupied,
            )
            for term in TERMS.rows
        )
    )
    symbols = one_hot(TERMS)
    subject = build()

    assert torch.allclose(
        subject.encode_soft(symbols, structure=TERMS).tensor,
        subject.encode_soft(symbols, structure=changed).tensor,
        atol=1e-6,
    )


def test_soft_encoding_uses_each_choices_marginal_mass() -> None:
    structure = BatchEncoding((TERMS.rows[0],))
    logits = one_hot(structure).tensor.clone()
    first = CHOICES.index(ATOM)
    second = CHOICES.index(OTHER_ATOM)
    logits[0, 0] = -torch.inf
    logits[0, 0, first] = torch.tensor(0.4).log()
    logits[0, 0, second] = torch.tensor(0.6).log()
    logits.requires_grad_()
    choices = LogDistribution(
        logits, ("term", "slot", "choice"), normalized_over="choice"
    )

    build().encode_soft(choices, structure=structure).tensor.square().sum().backward()

    assert logits.grad is not None
    assert float(logits.grad[0, 0, first].abs()) > 0.0
    assert float(logits.grad[0, 0, second].abs()) > 0.0


@pytest.mark.parametrize("mixed", [False, True], ids=("one-hot", "mixed"))
def test_soft_encoding_refuses_positive_mass_on_an_unsupported_arity(
    *, mixed: bool
) -> None:
    atom = Operator("atom", "T")
    binary = Operator("binary", "T", arg_sorts=("T", "T"))
    ternary = Operator("ternary", "T", arg_sorts=("T", "T", "T"))
    signature = Signature("unsupported-soft-arity", ("T",), (atom, binary, ternary))
    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, order="bf", depth=2)
    codec = TermCodec(index, layout)
    view = NeuralCodecView(codec)
    choices = ChoiceIndex(signature, depth_bounds={"T": layout.depth})
    store = TermStore(signature)
    term = store.op(ternary, *(store.op(atom) for _ in range(3)))
    structure = codec.encode_batch((term,))
    codes = torch.tensor((choices.codes(structure.rows[0]),), dtype=torch.int64)
    probabilities = torch.nn.functional.one_hot(codes, choices.width).double()

    if mixed:
        probabilities[0, 0] = 0.0
        probabilities[0, 0, choices.index(atom)] = 0.5
        probabilities[0, 0, choices.index(ternary)] = 0.5

    rows = LogDistribution(
        probabilities.log(),
        ("term", "slot", "choice"),
        normalized_over="choice",
    )
    subject = TreeRnnEncoder(
        codec=view,
        choices=choices,
        component=ComponentId("unsupported-soft-arity"),
        scheme=ArityHeadScheme(width=4, reached=(atom,)),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )

    with pytest.raises(ArityUnsupportedError, match="operator arity 3"):
        subject.encode_soft(rows, structure=structure)


def test_soft_encoding_requires_explicit_structure() -> None:
    method = cast("Callable[..., Encoded]", build().encode_soft)

    with pytest.raises(TypeError, match="structure"):
        method(one_hot(TERMS))


def test_soft_encoding_requires_choice_normalization() -> None:
    held = one_hot(TERMS)
    unnormalized = LogDistribution(held.tensor, held.axes, held.mask)

    with pytest.raises(ValueError, match="normalized over 'choice'"):
        build().encode_soft(unnormalized, structure=TERMS)


def test_input_placement_follows_model_parameters() -> None:
    for dtype in (torch.float32, torch.float64):
        subject = build().to(dtype=dtype)
        parameter = next(subject.parameters())
        hard = subject.encode(TERMS)
        soft = subject.encode_soft(one_hot(TERMS), structure=TERMS)

        assert hard.tensor.dtype is parameter.dtype
        assert soft.tensor.dtype is parameter.dtype
        assert hard.tensor.device == parameter.device


def test_gradients_reach_the_start_the_choices_the_cells_and_the_family() -> None:
    subject = build()
    subject.encode(TERMS).tensor.square().sum().backward()
    gradients = {name: parameter.grad for name, parameter in subject.named_parameters()}

    for name in (
        "_embedding._table.weight",
        "_start",
        "_cells.0.weight_ih",
        "_cells.1.weight_hh",
        "_scheme._up.1.weight",
    ):
        assert gradients[name] is not None
        assert bool(cast("torch.Tensor", gradients[name]).isfinite().all())


def test_the_encoder_publishes_outputs_then_each_layer_s_hidden_and_gates() -> None:
    sites = build().site_map()
    expected = [
        TERM_SITE,
        NODE_SITE,
        Site(COMPONENT, Role.PARAMETER, ("choice-embedding",)),
    ]

    for layer in range(LAYERS):
        expected.extend(
            (
                hidden(layer),
                gate(layer, "reset"),
                gate(layer, "update"),
                gate(layer, "candidate"),
            )
        )

    assert sites.kind is ComponentKind.TERM_ENCODER
    assert ensure_publishes(sites) is sites
    assert list(sites) == expected
    assert sites[TERM_SITE].axes == ("term", "width")
    assert sites[NODE_SITE].axes == ("term", "slot", "width")
    assert sites[TERM_SITE].dtype_class is DTypeClass.REAL
    assert sites[NODE_SITE].access is Access.OBSERVE

    for layer in range(LAYERS):
        assert sites[hidden(layer)].axes == ("term", "slot", "width")
        assert sites[hidden(layer)].dtype_class is DTypeClass.REAL
        assert sites[gate(layer, "reset")].dtype_class is DTypeClass.PROBABILITY
        assert sites[gate(layer, "update")].dtype_class is DTypeClass.PROBABILITY
        assert sites[gate(layer, "candidate")].dtype_class is DTypeClass.REAL
        assert all(
            sites[site].access is Access.OBSERVE
            for site in (
                hidden(layer),
                gate(layer, "reset"),
                gate(layer, "update"),
                gate(layer, "candidate"),
            )
        )


def test_the_observed_gates_are_the_ones_used_by_the_recurrent_equation() -> None:
    subject = build(layers=1)
    watched = frozenset(subject.site_map())

    with subject.instrument(watched, patches={}) as scope:
        encoded = subject.encode(SINGLE)

    inputs = subject.get_parameter("_embedding._table.weight")[A].unsqueeze(0)
    previous = subject.get_parameter("_start")[0].reshape(1, WIDTH)
    weight_ih = subject.get_parameter("_cells.0.weight_ih")
    weight_hh = subject.get_parameter("_cells.0.weight_hh")
    bias_ih = subject.get_parameter("_cells.0.bias_ih")
    bias_hh = subject.get_parameter("_cells.0.bias_hh")
    input_reset, input_update, input_candidate = torch.nn.functional.linear(
        inputs, weight_ih, bias_ih
    ).chunk(3, dim=1)
    hidden_reset, hidden_update, hidden_candidate = torch.nn.functional.linear(
        previous, weight_hh, bias_hh
    ).chunk(3, dim=1)
    expected_reset = torch.sigmoid(input_reset + hidden_reset)
    expected_update = torch.sigmoid(input_update + hidden_update)
    expected_candidate = torch.tanh(input_candidate + expected_reset * hidden_candidate)
    expected_hidden = (
        1.0 - expected_update
    ) * expected_candidate + expected_update * previous

    assert torch.allclose(
        scope.readings[gate(0, "reset")].tensor[:, 0], expected_reset, atol=1e-7
    )
    assert torch.allclose(
        scope.readings[gate(0, "update")].tensor[:, 0], expected_update, atol=1e-7
    )
    assert torch.allclose(
        scope.readings[gate(0, "candidate")].tensor[:, 0],
        expected_candidate,
        atol=1e-7,
    )
    assert torch.allclose(
        scope.readings[hidden(0)].tensor[:, 0], expected_hidden, atol=1e-7
    )
    assert torch.allclose(encoded.tensor, expected_hidden, atol=1e-7)


def test_recurrent_readings_use_the_term_slot_width_axes_and_occupancy_mask() -> None:
    subject = build()

    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        subject.encode(TERMS)

    expected_mask = torch.tensor(
        tuple(row.occupied for row in TERMS.rows), dtype=torch.bool
    )

    for layer in range(LAYERS):
        for site in (
            hidden(layer),
            gate(layer, "reset"),
            gate(layer, "update"),
            gate(layer, "candidate"),
        ):
            reading = scope.readings[site]

            assert reading.axes == ("term", "slot", "width")
            assert reading.tensor.shape == (len(TERMS.rows), CAPACITY, WIDTH)
            assert reading.mask is not None
            assert torch.equal(reading.mask, expected_mask)
            assert torch.equal(
                reading.tensor[~expected_mask],
                torch.zeros_like(reading.tensor[~expected_mask]),
            )


def test_recurrent_gate_ranges_match_each_activation() -> None:
    subject = build()

    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        subject.encode(TERMS)

    for layer in range(LAYERS):
        reset = scope.readings[gate(layer, "reset")].tensor
        update = scope.readings[gate(layer, "update")].tensor
        candidate = scope.readings[gate(layer, "candidate")].tensor

        assert bool(reset.ge(0.0).all() and reset.le(1.0).all())
        assert bool(update.ge(0.0).all() and update.le(1.0).all())
        assert bool(candidate.ge(-1.0).all() and candidate.le(1.0).all())


def test_nested_scopes_receive_only_the_recurrent_readings_they_select() -> None:
    subject = build()
    outer_site = hidden(0)
    inner_site = gate(1, "candidate")

    with (
        subject.instrument(frozenset({outer_site}), patches={}) as outer,
        subject.instrument(frozenset({inner_site}), patches={}) as inner,
    ):
        subject.encode(TERMS)

    assert set(outer.readings) == {outer_site}
    assert set(inner.readings) == {inner_site}


class _IdentityPatch:
    def stands_in_for(self, reading: Reading, /) -> torch.Tensor:
        return reading.tensor


def test_unknown_sites_and_every_patch_are_refused_by_site_name() -> None:
    subject = build()
    unknown = gate(LAYERS, "reset")
    published = gate(0, "reset")

    with pytest.raises(ValueError, match="GATE") as absent:
        subject.instrument(frozenset({unknown}), patches={})

    assert str(LAYERS) in str(absent.value)

    with pytest.raises(ValueError, match="GATE") as patch:
        subject.instrument(frozenset(), patches={published: _IdentityPatch()})

    assert "observation-only" in str(patch.value)

    with pytest.raises(ValueError, match="GATE") as absent_patch:
        subject.instrument(frozenset(), patches={unknown: _IdentityPatch()})

    assert str(LAYERS) in str(absent_patch.value)


def test_an_uninstrumented_call_allocates_no_site_or_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = build()

    def forbidden_site(
        component: ComponentId,
        role: Role,
        key: tuple[int | str, ...] = (),
    ) -> Never:
        del component, role, key
        pytest.fail("an uninstrumented call allocated a site")

    def forbidden_reading(
        site: Site,
        tensor: torch.Tensor,
        axes: tuple[str, ...],
        mask: torch.Tensor | None = None,
    ) -> Never:
        del site, tensor, axes, mask
        pytest.fail("an uninstrumented call allocated a reading")

    def forbidden_observation(
        role: Role,
        key: tuple[int | str, ...],
        tensor: torch.Tensor,
        mask: torch.Tensor | None,
        /,
    ) -> Never:
        del role, key, tensor, mask
        pytest.fail("an uninstrumented call entered the observation path")

    monkeypatch.setattr(rnn_module, "Site", forbidden_site)
    monkeypatch.setattr(rnn_module, "_Reading", forbidden_reading)
    monkeypatch.setattr(subject, "_observe", forbidden_observation)

    subject.encode(TERMS)


def test_a_recurrent_encoder_map_refuses_permuted_node_axes() -> None:
    sites = build().site_map()
    changed = dict(sites)
    changed[NODE_SITE] = SiteSpec(
        ("slot", "term", "width"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE,
    )

    with pytest.raises(ValueError, match=r"NODE_ENCODING.*runs over"):
        ensure_publishes(SiteMap(COMPONENT, ComponentKind.TERM_ENCODER, changed))


def test_the_encoder_retains_its_codec_and_family() -> None:
    torch.manual_seed(17)
    family = scheme()
    subject = TreeRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        scheme=family,
        cell="gru",
        layers=LAYERS,
        norm_cap=None,
        inventory="uncached",
    )

    assert subject.codec is VIEW
    assert subject.scheme is family


def test_a_role_filler_family_serves_the_encoder_too() -> None:
    torch.manual_seed(17)
    subject = TreeRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        scheme=RoleFillerScheme(width=WIDTH, reached=REACHED),
        cell="gru",
        layers=LAYERS,
        norm_cap=None,
        inventory="uncached",
    )
    terms = subject.encode(TERMS)

    assert terms.size("width") == WIDTH
    assert bool(terms.tensor.isfinite().all())
