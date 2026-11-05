from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Never, cast

import pytest
import torch
from hypothesis import strategies as st
from tests.conformance._sites import SiteObservation
from tests.conformance.observable import ObservableConformance, ObservableSuite
from tests.conformance.term_encoder import TermEncoderConformance, TermEncoderSuite

from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders import TransformerEncoderSettings, TransformerTermEncoder
from lofi.nn.encoders import transformer as transformer_module
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import (
    Consumption,
    Reading,
    StandardDeclaration,
    StructuralDeclaration,
    StructuralRelation,
)
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
from lofi.terms import (
    BatchEncoding,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    Term,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


WIDTH: Final = 8
HEADS: Final = 2
LAYERS: Final = 2
FEEDFORWARD_WIDTH: Final = 16
CAP: Final = 0.75

SETTINGS: Final = TransformerEncoderSettings(
    WIDTH, HEADS, LAYERS, FEEDFORWARD_WIDTH, norm_cap=None, activation="gelu"
)

COMPONENT: Final = ComponentId("tree-transformer")
TERM_SITE: Final = Site(COMPONENT, Role.TERM_ENCODING)
NODE_SITE: Final = Site(COMPONENT, Role.NODE_ENCODING)


def attention_logits(layer: int, head: int) -> Site:
    return Site(COMPONENT, Role.ATTENTION_LOGITS, (layer, head))


def attention_weights(layer: int, head: int) -> Site:
    return Site(COMPONENT, Role.ATTENTION_WEIGHTS, (layer, head))


def residual(layer: int, stage: str) -> Site:
    return Site(COMPONENT, Role.RESIDUAL_STREAM, (layer, stage))


def preactivation(layer: int) -> Site:
    return Site(COMPONENT, Role.FFN_PREACTIVATION, (layer,))


ATOM: Final = Operator("a", "T", family="primitive")
OTHER_ATOM: Final = Operator("b", "T", family="primitive")
UNARY: Final = Operator("raise", "T", arg_sorts=("T",), family="constructor")
BINARY: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
OTHER_BINARY: Final = Operator("join", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature(
    "transformer-fixture", ("T",), (ATOM, OTHER_ATOM, UNARY, BINARY, OTHER_BINARY)
)
INDEX: Final = OperatorIndex(SIGNATURE)
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=3)
CODEC: Final = TermCodec(INDEX, LAYOUT)
VIEW: Final = NeuralCodecView(CODEC)
CAPACITY: Final = VIEW.capacity
N_SYMBOLS: Final = VIEW.n_symbols

PARENTS: Final = LAYOUT.skeleton.parents
DEPTHS: Final = LAYOUT.skeleton.depths
LIVE: Final = (True,) * CAPACITY

A: Final = INDEX.index(ATOM)
B: Final = INDEX.index(OTHER_ATOM)
RAISE: Final = INDEX.index(UNARY)
PAIR: Final = INDEX.index(BINARY)
JOIN: Final = INDEX.index(OTHER_BINARY)

POSITIONS: Final = StandardDeclaration.POSITIONS.declaration
WITH_DEPTH: Final = StandardDeclaration.POSITIONS_WITH_DEPTH.declaration
ANCESTOR_MASK: Final = StandardDeclaration.ANCESTOR_MASK.declaration
BIASED: Final = StandardDeclaration.ANCESTOR_AND_SIBLING_BIAS.declaration
PATHS: Final = MappingProxyType({StructuralRelation.ANCESTOR: Consumption.EMBEDDING})
NOTHING: Final = MappingProxyType({})
CONSUMABLE: Final = (POSITIONS, WITH_DEPTH, ANCESTOR_MASK, BIASED, PATHS)


LOOSE: Final = NeuralCodecView(TermCodec(INDEX, PreorderArityLayout(CAPACITY)))


def row(
    symbols: tuple[int, ...],
    *,
    parents: tuple[int, ...] = PARENTS,
    occupied: tuple[bool, ...] = LIVE,
) -> Encoding:
    return Encoding(symbols, parents, DEPTHS, occupied)


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
ALIKE: Final = BatchEncoding((row((A,) * CAPACITY),))


def build(
    structure: StructuralDeclaration = ANCESTOR_MASK,
    *,
    codec: NeuralCodecView = VIEW,
    settings: TransformerEncoderSettings = SETTINGS,
) -> TransformerTermEncoder:
    torch.manual_seed(17)

    return TransformerTermEncoder(
        codec=codec,
        component=COMPONENT,
        settings=settings,
        structure=structure,
    )


def policy() -> DevicePolicy:
    return DevicePolicy(torch.device("cpu"), torch.float32)


def observations(
    subject: TransformerTermEncoder,
) -> Mapping[Site, Reading]:
    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        subject.encode(TERMS)

    return scope.readings


def conformance() -> TermEncoderConformance[TransformerTermEncoder]:
    return TermEncoderConformance(
        build=build,
        terms=TERMS,
        choices=lambda terms: terms.symbols,
        choice_width=N_SYMBOLS,
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


class TestTransformerTermEncoder(TermEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TermEncoderConformance[TransformerTermEncoder]:
        return conformance()


def observable_conformance() -> ObservableConformance[TransformerTermEncoder, Reading]:
    return ObservableConformance(
        build=build,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        run=lambda subject: (subject.encode(TERMS).tensor,),
        configuration=lambda subject: {"output_width": subject.output_width},
        submodules=lambda subject: frozenset(
            name for name, _ in subject.named_modules()
        ),
        placements=lambda subject: tuple(
            str(tensor.device) for tensor in (*subject.parameters(), *subject.buffers())
        ),
    )


class TestTransformerObservations(ObservableSuite):
    @pytest.fixture
    def conformance(
        self,
    ) -> ObservableConformance[TransformerTermEncoder, Reading]:
        return observable_conformance()


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


@pytest.mark.parametrize("structure", CONSUMABLE)
def test_each_consumable_declaration_is_what_the_encoder_reports(
    structure: StructuralDeclaration,
) -> None:
    subject = build(structure)

    assert subject.structure == structure
    assert subject.codec is VIEW

    with pytest.raises(TypeError):
        cast("dict[StructuralRelation, Consumption]", subject.structure)[
            StructuralRelation.PARENT
        ] = Consumption.MASK


def test_each_standard_declaration_encodes_a_fixed_input_differently() -> None:
    encoded = {
        standard: build(standard.declaration).encode(TERMS).tensor
        for standard in StandardDeclaration
        if standard is not StandardDeclaration.PARENT_RECURRENCE
    }

    assert len(encoded) == len(StandardDeclaration) - 1

    for one, first in encoded.items():
        for other, second in encoded.items():
            if one is not other:
                assert not torch.allclose(first, second, atol=1e-6)


def test_a_recurrence_is_refused() -> None:
    with pytest.raises(ValueError, match="no recurrence") as caught:
        build(StandardDeclaration.PARENT_RECURRENCE.declaration)

    assert "parent" in str(caught.value)


def test_a_declaration_naming_only_position_reads_no_parents() -> None:
    subject = build(POSITIONS, codec=LOOSE)
    reparented = BatchEncoding(
        (row((JOIN, PAIR, JOIN, A, B, B, A), parents=(-1, 0, 0, 1, 2, 1, 2)),)
    )
    kept = BatchEncoding((row((JOIN, PAIR, JOIN, A, B, B, A)),))

    assert LOOSE.served == {StructuralRelation.POSITION}

    with pytest.raises(ValueError, match="does not name parent"):
        LOOSE.serve(subject.structure).parent(policy=policy())

    assert torch.allclose(
        subject.encode_nodes(kept).tensor,
        subject.encode_nodes(reparented).tensor,
        atol=1e-6,
    )


def test_a_term_is_read_at_the_root_its_own_parent_channel_marks() -> None:
    subject = build(POSITIONS, codec=LOOSE)
    rooted_first = BatchEncoding((row((JOIN, PAIR, JOIN, A, B, B, A)),))
    rooted_second = BatchEncoding(
        (row((PAIR, JOIN, JOIN, A, B, B, A), parents=(1, -1, 0, 1, 1, 2, 2)),)
    )
    nodes = subject.encode_nodes(rooted_second).tensor[0]

    assert torch.allclose(
        subject.encode(rooted_first).tensor[0],
        subject.encode_nodes(rooted_first).tensor[0, 0],
        atol=1e-6,
    )
    assert torch.allclose(subject.encode(rooted_second).tensor[0], nodes[1], atol=1e-6)
    assert not torch.allclose(nodes[0], nodes[1], atol=1e-6)


@pytest.mark.parametrize("structure", [WITH_DEPTH, ANCESTOR_MASK, BIASED, PATHS])
def test_a_tree_declaration_is_refused_by_a_codec_serving_position_alone(
    structure: StructuralDeclaration,
) -> None:
    with pytest.raises(ValueError, match="does not serve") as caught:
        build(structure, codec=LOOSE)

    relations = set(structure) - {StructuralRelation.POSITION}
    assert any(relation.value in str(caught.value) for relation in relations)


def test_the_ancestor_mask_admits_a_slot_its_ancestors_and_its_descendants() -> None:
    admitted = build(ANCESTOR_MASK).get_buffer("_admitted")
    ancestor = VIEW.serve(ANCESTOR_MASK).ancestor(policy=policy())

    assert torch.equal(admitted, ancestor | ancestor.T)
    assert admitted[0].tolist() == [True] * CAPACITY
    assert admitted[3].tolist() == [True, True, False, True, False, False, False]
    assert admitted[1].tolist() == [True, True, False, True, True, False, False]


def test_a_masked_slot_reads_its_own_subtree_and_root_path_only() -> None:
    original = BatchEncoding((row((JOIN, PAIR, JOIN, A, B, B, A)),))
    cousin = BatchEncoding((row((JOIN, PAIR, JOIN, A, B, A, A)),))
    parent = BatchEncoding((row((JOIN, JOIN, JOIN, A, B, B, A)),))
    masked = build(ANCESTOR_MASK, settings=SETTINGS._replace(layers=1))
    open_to_all = build(POSITIONS, settings=SETTINGS._replace(layers=1))
    before = masked.encode_nodes(original).tensor[0]
    after_cousin = masked.encode_nodes(cousin).tensor[0]
    after_parent = masked.encode_nodes(parent).tensor[0]

    assert torch.allclose(before[3], after_cousin[3], atol=1e-6)
    assert torch.allclose(before[1], after_cousin[1], atol=1e-6)
    assert not torch.allclose(before[0], after_cousin[0], atol=1e-6)
    assert not torch.allclose(before[2], after_cousin[2], atol=1e-6)
    assert not torch.allclose(before[3], after_parent[3], atol=1e-6)
    assert not torch.allclose(
        open_to_all.encode_nodes(original).tensor[0, 3],
        open_to_all.encode_nodes(cousin).tensor[0, 3],
        atol=1e-6,
    )


def test_without_a_mask_every_slot_attends_to_every_other() -> None:
    for structure in (POSITIONS, WITH_DEPTH, BIASED, PATHS, NOTHING):
        assert bool(build(structure).get_buffer("_admitted").all())


def test_a_mask_is_the_bias_that_is_minus_infinity_off_the_relation() -> None:
    ancestor_bias = {**ANCESTOR_MASK, StructuralRelation.ANCESTOR: Consumption.BIAS}
    masked = build(ANCESTOR_MASK)
    biased = build(ancestor_bias)
    shared = dict(masked.named_parameters())

    with torch.no_grad():
        for name, parameter in biased.named_parameters():
            if name in shared:
                parameter.copy_(shared[name])

        table = biased.get_parameter("_bias_tables.0")
        table[:, 0] = -torch.inf
        table[:, 1] = 0.0

    assert torch.allclose(
        masked.encode_nodes(TERMS).tensor, biased.encode_nodes(TERMS).tensor, atol=1e-6
    )
    assert not torch.allclose(
        masked.encode_nodes(TERMS).tensor,
        build(ancestor_bias).encode_nodes(TERMS).tensor,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "structure",
    [
        *CONSUMABLE,
        {StructuralRelation.DEPTH: Consumption.BIAS},
        {StructuralRelation.TREE_DISTANCE: Consumption.BIAS},
        {StructuralRelation.SIBLING: Consumption.BIAS},
    ],
)
def test_inference_encodes_as_training_does(structure: StructuralDeclaration) -> None:
    subject = build(structure)
    training = subject.encode(TERMS).tensor.detach()
    training_nodes = subject.encode_nodes(TERMS).tensor.detach()
    subject.eval()

    with torch.no_grad():
        inference = subject.encode(TERMS).tensor
        inference_nodes = subject.encode_nodes(TERMS).tensor

    assert bool(inference.isfinite().all())
    assert torch.allclose(training, inference, atol=1e-6)
    assert torch.allclose(training_nodes, inference_nodes, atol=1e-6)


def test_a_biased_relation_moves_attention_where_it_holds() -> None:
    subject = build(BIASED, settings=SETTINGS._replace(layers=1))
    before = subject.encode_nodes(ALIKE).tensor[0]

    with torch.no_grad():
        subject.get_parameter("_bias_tables.1")[:, 1] += 8.0

    after = subject.encode_nodes(ALIKE).tensor[0]
    moved = (after - before).abs().amax(dim=1).gt(1e-6)
    has_sibling = torch.tensor(
        [
            any(other != one and PARENTS[other] == PARENTS[one] for other in range(7))
            for one in range(CAPACITY)
        ]
    )

    assert torch.equal(moved, has_sibling)
    assert not bool(has_sibling[0])


def test_an_embedded_relation_tells_slots_apart_by_its_value() -> None:
    subject = build(WITH_DEPTH)

    with torch.no_grad():
        subject.get_parameter("_embeddings.0.weight").zero_()

    nodes = subject.encode_nodes(ALIKE).tensor[0]

    for one in range(CAPACITY):
        for other in range(CAPACITY):
            alike = torch.allclose(nodes[one], nodes[other], atol=1e-6)

            assert alike is (DEPTHS[one] == DEPTHS[other])


def test_the_ancestor_embedding_tells_slots_apart_by_their_path() -> None:
    nodes = build(PATHS).encode_nodes(ALIKE).tensor[0]

    for one in range(CAPACITY):
        for other in range(CAPACITY):
            assert torch.allclose(nodes[one], nodes[other], atol=1e-6) is (one == other)


def test_the_ancestor_embedding_follows_the_path_and_not_the_index() -> None:
    depth_first = NeuralCodecView(
        TermCodec(
            INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3)
        )
    )
    breadth_first = build(PATHS).encode_nodes(ALIKE).tensor[0]
    reordered = build(PATHS, codec=depth_first).encode_nodes(ALIKE).tensor[0]

    assert torch.allclose(breadth_first[0], reordered[0], atol=1e-6)
    assert torch.allclose(breadth_first[1], reordered[1], atol=1e-6)
    assert torch.allclose(breadth_first[2], reordered[4], atol=1e-6)
    assert torch.allclose(breadth_first[3], reordered[2], atol=1e-6)
    assert torch.allclose(breadth_first[6], reordered[6], atol=1e-6)
    assert not torch.allclose(breadth_first[2], reordered[2], atol=1e-6)


def test_with_no_structure_declared_the_root_reads_a_bag_of_symbols() -> None:
    ordered = BatchEncoding((row((JOIN, PAIR, RAISE, A, B, B, A)),))
    swapped = BatchEncoding((row((JOIN, RAISE, PAIR, A, B, B, A)),))
    bag = build(NOTHING)
    positioned = build(POSITIONS)

    assert torch.allclose(
        bag.encode(ordered).tensor, bag.encode(swapped).tensor, atol=1e-6
    )
    assert not torch.allclose(
        positioned.encode(ordered).tensor, positioned.encode(swapped).tensor, atol=1e-6
    )


def test_a_pairwise_relation_other_than_the_ancestor_is_refused_as_an_embedding() -> (
    None
):
    with pytest.raises(ValueError, match="one value per pair of slots"):
        build({**POSITIONS, StructuralRelation.SIBLING: Consumption.EMBEDDING})


@pytest.mark.parametrize(
    ("structure", "message"),
    [
        ({StructuralRelation.DEPTH: Consumption.MASK}, "one value per slot"),
        ({StructuralRelation.TREE_DISTANCE: Consumption.MASK}, "counts edges"),
    ],
)
def test_a_relation_a_mask_cannot_be_built_from_is_refused(
    structure: StructuralDeclaration, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build(structure)


def test_a_counted_relation_is_biased_by_its_count() -> None:
    subject = build({StructuralRelation.TREE_DISTANCE: Consumption.BIAS})
    rows = subject.get_buffer("_bias_rows")[0]
    distance = VIEW.serve(subject.structure).tree_distance(policy=policy())

    assert torch.equal(rows, distance)
    assert tuple(subject.get_parameter("_bias_tables.0").shape) == (HEADS, CAPACITY)
    assert bool(subject.encode(TERMS).tensor.isfinite().all())


def test_a_unary_bias_is_read_by_the_difference_of_the_two_slots_values() -> None:
    subject = build({StructuralRelation.DEPTH: Consumption.BIAS})
    rows = subject.get_buffer("_bias_rows")[0]
    depths = torch.tensor(DEPTHS)

    assert torch.equal(rows, depths.unsqueeze(1) - depths.unsqueeze(0) + CAPACITY - 1)
    assert tuple(subject.get_parameter("_bias_tables.0").shape) == (
        HEADS,
        2 * CAPACITY - 1,
    )
    assert bool(subject.encode(TERMS).tensor.isfinite().all())


def test_a_declaration_over_the_wrong_types_is_refused() -> None:
    with pytest.raises(TypeError, match="StructuralRelation"):
        build(cast("StructuralDeclaration", {"parent": Consumption.MASK}))


def test_the_norm_cap_projects_the_rows_both_paths_read() -> None:
    subject = build(settings=SETTINGS._replace(norm_cap=CAP))

    with torch.no_grad():
        subject.get_parameter("_symbols.weight").mul_(10.0)

    once = subject.encode(TERMS).tensor.clone()
    with torch.no_grad():
        subject.get_parameter("_symbols.weight").mul_(10.0)

    assert torch.allclose(once, subject.encode(TERMS).tensor, atol=1e-6)
    assert not torch.allclose(once, build().encode(TERMS).tensor, atol=1e-6)


def test_the_norm_cap_must_be_finite_and_above_zero() -> None:
    with pytest.raises(ValueError, match="norm cap"):
        build(settings=SETTINGS._replace(norm_cap=0.0))


def test_padded_symbol_codes_cannot_change_live_outputs() -> None:
    occupied = (True, True, True, False, False, False, False)
    original = BatchEncoding(
        (row((JOIN, A, B, *((OperatorIndex.PAD,) * 4)), occupied=occupied),)
    )
    changed = BatchEncoding((row((JOIN, A, B, A, B, B, A), occupied=occupied),))
    subject = build()
    first = subject.encode_nodes(original)
    second = subject.encode_nodes(changed)

    assert torch.allclose(first.tensor[:, :3], second.tensor[:, :3], atol=1e-6)
    assert torch.equal(first.tensor[:, 3:], torch.zeros_like(first.tensor[:, 3:]))
    assert torch.equal(second.tensor[:, 3:], torch.zeros_like(second.tensor[:, 3:]))
    assert torch.allclose(
        subject.encode(original).tensor, subject.encode(changed).tensor, atol=1e-6
    )


def test_a_slot_no_node_occupies_is_not_attended_to() -> None:
    shallow = TermCodec(
        INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=2)
    )
    narrow = NeuralCodecView(shallow)
    roomy = build(POSITIONS)
    snug = build(POSITIONS, codec=narrow)

    with torch.no_grad():
        roomy.get_parameter("_embeddings.0.weight")[: narrow.capacity] = (
            snug.get_parameter("_embeddings.0.weight")
        )

    store = TermStore(SIGNATURE)
    term = store.op(BINARY, store.op(OTHER_ATOM), store.op(ATOM))
    padded = roomy.encode_nodes(CODEC.encode_batch((term,))).tensor[0]
    exact = snug.encode_nodes(shallow.encode_batch((term,))).tensor[0]

    assert torch.allclose(padded[: narrow.capacity], exact, atol=1e-6)
    assert torch.allclose(
        roomy.encode(CODEC.encode_batch((term,))).tensor,
        snug.encode(shallow.encode_batch((term,))).tensor,
        atol=1e-6,
    )


def test_a_row_holding_no_term_comes_back_masked() -> None:
    empty = BatchEncoding(
        (row((A, *((OperatorIndex.PAD,) * 6)), occupied=(False,) * CAPACITY),)
    )
    encoded = build().encode(empty)

    assert encoded.mask is not None
    assert encoded.mask.tolist() == [False]
    assert torch.equal(encoded.tensor, torch.zeros_like(encoded.tensor))


def test_a_batch_of_no_term_encodes_to_no_row() -> None:
    subject = build()

    assert subject.encode(BatchEncoding(())).tensor.shape == (0, WIDTH)
    assert subject.encode_nodes(BatchEncoding(())).tensor.shape == (
        0,
        CAPACITY,
        WIDTH,
    )


def test_input_placement_follows_model_parameters() -> None:
    for dtype in (torch.float32, torch.float64):
        subject = build().to(dtype=dtype)
        parameter = next(subject.parameters())
        hard = subject.encode(TERMS)

        assert hard.tensor.dtype is parameter.dtype
        assert hard.tensor.device == parameter.device


def test_gradients_reach_symbols_structure_and_attention() -> None:
    subject = build(
        {**WITH_DEPTH, StructuralRelation.SIBLING: Consumption.BIAS},
    )
    subject.encode(TERMS).tensor.square().sum().backward()
    gradients = {name: parameter.grad for name, parameter in subject.named_parameters()}

    for name in (
        "_symbols.weight",
        "_embeddings.0.weight",
        "_embeddings.1.weight",
        "_bias_tables.0",
        "_blocks.0._queries_keys_values.weight",
        "_blocks.0._contract.weight",
        "_blocks.1._merge.weight",
        "_norm.weight",
    ):
        assert gradients[name] is not None
        assert bool(cast("torch.Tensor", gradients[name]).isfinite().all())
        assert bool(cast("torch.Tensor", gradients[name]).ne(0.0).any())

    paths = build(PATHS)
    paths.encode(TERMS).tensor.square().sum().backward()

    for name in ("_paths._root", "_paths._child.weight", "_paths._inherit.weight"):
        gradient = paths.get_parameter(name).grad

        assert gradient is not None
        assert bool(gradient.isfinite().all())
        assert bool(gradient.ne(0.0).any())


def test_the_encoder_publishes_every_output_and_internal_site_in_execution_order() -> (
    None
):
    sites = build().site_map()
    expected = [TERM_SITE, NODE_SITE]

    for layer in range(LAYERS):
        expected.append(residual(layer, "input"))

        for head in range(HEADS):
            expected.extend(
                (attention_logits(layer, head), attention_weights(layer, head))
            )

        expected.extend(
            (
                residual(layer, "post-attention"),
                preactivation(layer),
                residual(layer, "output"),
            )
        )

    assert sites.kind is ComponentKind.TERM_ENCODER
    assert ensure_publishes(sites) is sites
    assert list(sites) == expected
    assert sites[TERM_SITE].axes == ("term", "width")
    assert sites[NODE_SITE].axes == ("term", "slot", "width")
    assert sites[TERM_SITE].dtype_class is DTypeClass.REAL
    assert sites[NODE_SITE].dtype_class is DTypeClass.REAL

    for site in expected:
        spec = sites[site]

        assert spec.access is Access.OBSERVE
        assert spec.differentiable

        if site.role in (Role.ATTENTION_LOGITS, Role.ATTENTION_WEIGHTS):
            assert spec.axes == ("batch", "position", "attended-position")
        elif site.role in (Role.RESIDUAL_STREAM, Role.FFN_PREACTIVATION):
            assert spec.axes == ("batch", "position", "feature")

        if site.role is Role.ATTENTION_WEIGHTS:
            assert spec.dtype_class is DTypeClass.PROBABILITY
        else:
            assert spec.dtype_class is DTypeClass.REAL


def test_attention_readings_are_the_masked_logits_and_normalized_weights() -> None:
    subject = build()
    watched = frozenset(
        site
        for site in subject.site_map()
        if site.role in (Role.ATTENTION_LOGITS, Role.ATTENTION_WEIGHTS)
    )

    with subject.instrument(watched, patches={}) as scope:
        encoded = subject.encode_nodes(TERMS)

    assert encoded.mask is not None
    occupied = encoded.mask
    slots = occupied.shape[1]
    eye = torch.eye(slots, dtype=torch.bool)
    admitted = subject.get_buffer("_admitted")
    readable = admitted.unsqueeze(0) & (occupied.unsqueeze(1) | eye)
    refused = occupied.unsqueeze(-1) & ~readable

    for layer in range(LAYERS):
        for head in range(HEADS):
            logits = scope.readings[attention_logits(layer, head)]
            weights = scope.readings[attention_weights(layer, head)]

            assert logits.mask is not None
            assert weights.mask is not None
            assert torch.equal(logits.mask, occupied)
            assert torch.equal(weights.mask, occupied)
            assert tuple(logits.tensor.shape) == (
                len(TERMS.rows),
                CAPACITY,
                CAPACITY,
            )
            assert torch.equal(
                logits.tensor[~occupied], torch.zeros_like(logits.tensor[~occupied])
            )
            assert torch.equal(
                weights.tensor[~occupied], torch.zeros_like(weights.tensor[~occupied])
            )
            assert bool(torch.isneginf(logits.tensor[refused]).all())
            assert torch.equal(
                weights.tensor[refused], torch.zeros_like(weights.tensor[refused])
            )
            assert torch.equal(
                weights.tensor[occupied],
                torch.softmax(logits.tensor[occupied], dim=-1),
            )
            assert torch.allclose(
                weights.tensor.sum(dim=-1)[occupied],
                torch.ones_like(weights.tensor.sum(dim=-1)[occupied]),
            )


def test_attention_logits_are_the_exact_scaled_query_key_scores_with_bias() -> None:
    subject = build({**ANCESTOR_MASK, StructuralRelation.SIBLING: Consumption.BIAS})
    watched = frozenset(
        {
            *(
                attention_logits(layer, head)
                for layer in range(LAYERS)
                for head in range(HEADS)
            ),
            *(residual(layer, "input") for layer in range(LAYERS)),
        }
    )

    with subject.instrument(watched, patches={}) as scope:
        encoded = subject.encode_nodes(TERMS)

    assert encoded.mask is not None
    occupied = encoded.mask
    eye = torch.eye(CAPACITY, dtype=torch.bool)
    admitted = subject.get_buffer("_admitted")
    readable = admitted.unsqueeze(0) & (occupied.unsqueeze(1) | eye)
    structural_bias = torch.where(
        readable,
        torch.zeros((), dtype=encoded.tensor.dtype),
        torch.full((), -torch.inf, dtype=encoded.tensor.dtype),
    ).reshape(len(TERMS.rows), 1, CAPACITY, CAPACITY)

    for relation, rows in enumerate(subject.get_buffer("_bias_rows")):
        table = subject.get_parameter(f"_bias_tables.{relation}")
        structural_bias = structural_bias + table[:, rows].unsqueeze(0)

    for layer in range(LAYERS):
        stream = scope.readings[residual(layer, "input")].tensor
        normalized = subject.get_submodule(f"_blocks.{layer}._attention_norm")(stream)
        projected = subject.get_submodule(f"_blocks.{layer}._queries_keys_values")(
            normalized
        )
        queries, keys, _ = projected.reshape(
            len(TERMS.rows), CAPACITY, 3, HEADS, WIDTH // HEADS
        ).permute(2, 0, 3, 1, 4)
        expected = (
            torch.matmul(queries, keys.transpose(-2, -1)) / (WIDTH // HEADS) ** 0.5
        )
        expected = expected + structural_bias
        expected = torch.where(
            occupied.unsqueeze(1).unsqueeze(-1),
            expected,
            torch.zeros_like(expected),
        )
        observed = torch.stack(
            tuple(
                scope.readings[attention_logits(layer, head)].tensor
                for head in range(HEADS)
            ),
            dim=1,
        )

        assert torch.equal(observed, expected)


def test_observed_attention_weights_are_the_ones_that_mix_values() -> None:
    subject = build(settings=SETTINGS._replace(layers=1))
    watched = frozenset(
        {
            residual(0, "input"),
            *(attention_weights(0, head) for head in range(HEADS)),
        }
    )
    merged_inputs: list[torch.Tensor] = []

    def capture_merge_input(
        module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        del module
        merged_inputs.append(arguments[0].detach())

    merge = subject.get_submodule("_blocks.0._merge")
    handle = merge.register_forward_pre_hook(capture_merge_input)

    try:
        with subject.instrument(watched, patches={}) as scope:
            subject.encode_nodes(TERMS)
    finally:
        handle.remove()

    stream = scope.readings[residual(0, "input")].tensor
    normalized = subject.get_submodule("_blocks.0._attention_norm")(stream)
    projected = subject.get_submodule("_blocks.0._queries_keys_values")(normalized)
    values = projected.reshape(
        len(TERMS.rows), CAPACITY, 3, HEADS, WIDTH // HEADS
    ).permute(2, 0, 3, 1, 4)[2]
    weights = torch.stack(
        tuple(
            scope.readings[attention_weights(0, head)].tensor for head in range(HEADS)
        ),
        dim=1,
    )
    reproduced = (
        torch.matmul(weights, values)
        .transpose(1, 2)
        .reshape(len(TERMS.rows), CAPACITY, WIDTH)
    )

    assert len(merged_inputs) == 1
    assert torch.equal(reproduced, merged_inputs[0])


def test_feedforward_and_residual_readings_are_the_values_used_by_the_layer() -> None:
    subject = build()
    watched = frozenset(
        site
        for site in subject.site_map()
        if site.role in (Role.RESIDUAL_STREAM, Role.FFN_PREACTIVATION)
    )
    merged: list[torch.Tensor] = []
    expanded: list[torch.Tensor] = []
    activated: list[torch.Tensor] = []
    contracted: list[torch.Tensor] = []

    def keep(
        destination: list[torch.Tensor],
    ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def capture(
            module: torch.nn.Module,
            arguments: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, arguments
            destination.append(output.detach())

        return capture

    def keep_input(
        destination: list[torch.Tensor],
    ) -> Callable[[torch.nn.Module, tuple[torch.Tensor, ...]], None]:
        def capture(
            module: torch.nn.Module,
            arguments: tuple[torch.Tensor, ...],
        ) -> None:
            del module
            destination.append(arguments[0].detach())

        return capture

    handles = [
        subject.get_submodule(f"_blocks.{layer}.{name}").register_forward_hook(
            keep(destination)
        )
        for layer in range(LAYERS)
        for name, destination in (
            ("_merge", merged),
            ("_expand", expanded),
            ("_contract", contracted),
        )
    ]
    handles.extend(
        subject.get_submodule(f"_blocks.{layer}._contract").register_forward_pre_hook(
            keep_input(activated)
        )
        for layer in range(LAYERS)
    )

    try:
        with subject.instrument(watched, patches={}) as scope:
            encoded = subject.encode_nodes(TERMS)
    finally:
        for handle in handles:
            handle.remove()

    assert encoded.mask is not None
    live = encoded.mask.unsqueeze(-1)

    for layer in range(LAYERS):
        incoming_reading = scope.readings[residual(layer, "input")]
        post_attention_reading = scope.readings[residual(layer, "post-attention")]
        preactivation_reading = scope.readings[preactivation(layer)]
        output_reading = scope.readings[residual(layer, "output")]
        incoming = incoming_reading.tensor
        post_attention = post_attention_reading.tensor
        before_gelu = preactivation_reading.tensor
        output = output_reading.tensor
        expected_post = torch.where(
            live, incoming + merged[layer], torch.zeros_like(incoming)
        )
        expected_preactivation = torch.where(
            live, expanded[layer], torch.zeros_like(expanded[layer])
        )
        expected_output = torch.where(
            live, post_attention + contracted[layer], torch.zeros_like(post_attention)
        )

        for reading in (
            incoming_reading,
            post_attention_reading,
            preactivation_reading,
            output_reading,
        ):
            assert reading.mask is not None
            assert torch.equal(reading.mask, encoded.mask)

        assert torch.equal(post_attention, expected_post)
        assert torch.equal(before_gelu, expected_preactivation)
        assert torch.equal(activated[layer], torch.nn.functional.gelu(before_gelu))
        assert torch.equal(output, expected_output)
        assert torch.equal(
            output[~encoded.mask], torch.zeros_like(output[~encoded.mask])
        )

        if layer:
            prior = scope.readings[residual(layer - 1, "output")].tensor

            assert torch.equal(incoming, prior)


def test_nested_instrumentation_scopes_both_receive_their_selected_readings() -> None:
    subject = build()
    outer_site = residual(0, "input")
    inner_site = attention_weights(1, 0)

    with (
        subject.instrument(frozenset({outer_site}), patches={}) as outer,
        subject.instrument(frozenset({inner_site}), patches={}) as inner,
    ):
        subject.encode(TERMS)

    assert set(outer.readings) == {outer_site}
    assert set(inner.readings) == {inner_site}


def test_a_scope_detaches_and_the_encoder_is_reusable_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = build()
    site = residual(0, "input")

    def fail_after_encoding() -> Never:
        subject.encode_nodes(TERMS)
        message = "planted failure"
        raise RuntimeError(message)

    failed = subject.instrument(frozenset({site}), patches={})

    with (
        pytest.raises(RuntimeError, match="planted failure"),
        failed,
    ):
        fail_after_encoding()

    recorded_before = failed.readings[site]
    shortened = BatchEncoding((TERMS.rows[0],))

    def forbidden_observation(
        role: Role,
        key: tuple[int | str, ...],
        tensor: torch.Tensor,
        mask: torch.Tensor | None,
        /,
    ) -> Never:
        del role, key, tensor, mask
        pytest.fail("a closed scope kept observation active")

    with monkeypatch.context() as patch:
        patch.setattr(subject, "_observe", forbidden_observation)
        subject.encode_nodes(shortened)

    assert failed.readings[site] is recorded_before

    with subject.instrument(frozenset({site}), patches={}) as recovered:
        subject.encode_nodes(shortened)

    assert set(recovered.readings) == {site}
    assert recovered.readings[site] is not recorded_before
    assert recovered.readings[site].tensor.shape[0] == 1


class _IdentityPatch:
    def stands_in_for(self, reading: Reading, /) -> torch.Tensor:
        return reading.tensor


def test_unknown_sites_and_every_patch_are_refused_by_site_name() -> None:
    subject = build()
    unknown = attention_logits(LAYERS, 0)
    published = attention_logits(0, 0)

    with pytest.raises(ValueError, match="ATTENTION_LOGITS") as absent:
        subject.instrument(frozenset({unknown}), patches={})

    assert str(LAYERS) in str(absent.value)

    with pytest.raises(ValueError, match="ATTENTION_LOGITS") as patch:
        subject.instrument(
            frozenset(),
            patches={published: _IdentityPatch()},
        )

    assert "observation-only" in str(patch.value)


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

    monkeypatch.setattr(transformer_module, "Site", forbidden_site)
    monkeypatch.setattr(transformer_module, "_Reading", forbidden_reading)
    monkeypatch.setattr(subject, "_observe", forbidden_observation)

    subject.encode(TERMS)


def test_a_transformer_encoder_map_refuses_swapped_term_axes() -> None:
    sites = build().site_map()
    changed = dict(sites)
    changed[TERM_SITE] = SiteSpec(
        ("width", "term"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE,
    )

    with pytest.raises(ValueError, match=r"TERM_ENCODING.*runs over"):
        ensure_publishes(SiteMap(COMPONENT, ComponentKind.TERM_ENCODER, changed))


def test_the_encoder_retains_its_codec_and_width() -> None:
    subject = TransformerTermEncoder(
        codec=VIEW,
        component=COMPONENT,
        settings=SETTINGS,
        structure=POSITIONS,
    )

    assert subject.codec is VIEW
    assert subject.output_width == WIDTH
