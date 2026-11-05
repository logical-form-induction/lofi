from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.term_encoder import TermEncoderConformance, TermEncoderSuite
from torch import nn

from lofi._types import ModuleSharing
from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders import (
    HolographicCodebook,
    HolographicEncoderSettings,
    HrrTermEncoder,
    SharedSubmodule,
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
from lofi.nn.vsa import AtomTable, bundle_norm, bundle_scale, unbind
from lofi.probe import ensure_publishes
from lofi.terms import (
    BatchEncoding,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn._values import Encoded

WIDTH: Final = 16
WIDE: Final = 1024

COMPONENT: Final = ComponentId("hrr")
TERM_SITE: Final = Site(COMPONENT, Role.TERM_ENCODING)
NODE_SITE: Final = Site(COMPONENT, Role.NODE_ENCODING)

ATOM: Final = Operator("a", "T", family="primitive")
OTHER_ATOM: Final = Operator("b", "T", family="primitive")
UNARY: Final = Operator("raise", "T", arg_sorts=("T",), family="constructor")
BINARY: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
OTHER_BINARY: Final = Operator("join", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature(
    "hrr-fixture", ("T",), (ATOM, OTHER_ATOM, UNARY, BINARY, OTHER_BINARY)
)
INDEX: Final = OperatorIndex(SIGNATURE)
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=3)
CODEC: Final = TermCodec(INDEX, LAYOUT)
VIEW: Final = NeuralCodecView(CODEC)
CAPACITY: Final = VIEW.capacity
N_SYMBOLS: Final = VIEW.n_symbols

PARENTS: Final = LAYOUT.skeleton.parents
OTHER_PARENTS: Final = (Encoding.NO_PARENT, 0, 0, 1, 2, 1, 2)
DEPTHS: Final = LAYOUT.skeleton.depths
LIVE: Final = (True,) * CAPACITY

A: Final = INDEX.index(ATOM)
B: Final = INDEX.index(OTHER_ATOM)
RAISE: Final = INDEX.index(UNARY)
PAIR: Final = INDEX.index(BINARY)
JOIN: Final = INDEX.index(OTHER_BINARY)
SHAPE: Final = (JOIN, PAIR, RAISE, A, B, A, B)


def row(
    symbols: tuple[int, ...],
    *,
    parents: tuple[int, ...] = PARENTS,
    depths: tuple[int, ...] = DEPTHS,
    occupied: tuple[bool, ...] = LIVE,
) -> Encoding:
    return Encoding(symbols, parents, depths, occupied)


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

TERNARY: Final = BatchEncoding(
    (
        row(
            SHAPE,
            parents=(Encoding.NO_PARENT, 0, 0, 0, 1, 1, 1),
            depths=(0, 1, 1, 1, 2, 2, 2),
        ),
    )
)

CHAIN: Final = BatchEncoding(
    (
        row(
            SHAPE,
            parents=(Encoding.NO_PARENT, 0, 1, 2, 3, 4, 5),
            depths=(0, 1, 2, 3, 4, 5, 6),
        ),
    )
)

SINGLE: Final = BatchEncoding(
    (
        row(
            (A, *((OperatorIndex.PAD,) * (CAPACITY - 1))),
            occupied=(True, *((False,) * (CAPACITY - 1))),
        ),
    )
)


def settings(
    *, width: int = WIDTH, magnitude: float = 1.0, floor: float = 1e-3
) -> HolographicEncoderSettings:
    return HolographicEncoderSettings(width=width, magnitude=magnitude, floor=floor)


def build(*, width: int = WIDTH, seed: int = 17) -> HrrTermEncoder:
    torch.manual_seed(seed)

    return HrrTermEncoder(
        codec=VIEW,
        component=COMPONENT,
        settings=settings(width=width),
    )


def set_magnitudes(table: AtomTable, values: Tensor) -> None:
    (magnitudes,) = table.parameters()

    with torch.no_grad():
        magnitudes.copy_(values.expand_as(magnitudes))


def unequal(subject: HrrTermEncoder) -> HrrTermEncoder:
    """Hold every role at its own magnitude and every code at its own."""
    roles = subject.codebook.roles
    codes = subject.codebook.codes
    set_magnitudes(
        roles, torch.arange(1, roles.count + 1).float().mul(0.25).unsqueeze(-1)
    )
    set_magnitudes(
        codes, torch.arange(1, codes.count + 1).float().mul(0.5).unsqueeze(-1)
    )

    return subject


def children_of(term: Encoding, slot: int) -> tuple[int, ...]:
    return tuple(
        child
        for child, parent in enumerate(term.parents)
        if parent == slot and term.occupied[child]
    )


def reference_node(subject: HrrTermEncoder, term: Encoding, slot: int) -> Tensor:
    """Encode one slot by the recursion written out over the codebook's atoms."""
    codes = subject.codebook.codes.atoms.detach()
    roles = subject.codebook.roles.atoms.detach()
    own = roles[slot] * codes[term.symbols[slot]]
    children = children_of(term, slot)

    if not children:
        return own

    below = torch.stack([reference_node(subject, term, child) for child in children])

    return own + below.sum(dim=0) / bundle_scale(len(children))


def reference_flat(subject: HrrTermEncoder, term: Encoding) -> Tensor:
    """Encode a term as the scaled superposition over its occupied addresses."""
    codes = subject.codebook.codes.atoms.detach()
    roles = subject.codebook.roles.atoms.detach()
    total = torch.zeros(subject.output_width)

    for slot, occupied in enumerate(term.occupied):
        if not occupied:
            continue

        scale = 1.0

        for ancestor in term.ancestors(slot):
            scale /= bundle_scale(len(children_of(term, ancestor)))

        total = total + scale * roles[slot] * codes[term.symbols[slot]]

    return total


def observations(subject: HrrTermEncoder) -> MappingProxyType[Site, Encoded]:
    return MappingProxyType(
        {
            TERM_SITE: subject.encode(TERMS),
            NODE_SITE: subject.encode_nodes(TERMS),
        }
    )


def conformance() -> TermEncoderConformance[HrrTermEncoder]:
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


class TestHrrTermEncoder(TermEncoderSuite):
    @pytest.fixture
    def conformance(self) -> TermEncoderConformance[HrrTermEncoder]:
        return conformance()


def test_the_fixture_is_what_the_codec_encodes() -> None:
    assert TERMS.rows[0].symbols == (JOIN, PAIR, JOIN, A, B, B, A)
    assert TERMS.rows[0].occupied == LIVE
    assert TERMS.rows[1].symbols == (PAIR, B, A, *((OperatorIndex.PAD,) * 4))
    assert TERMS.rows[1].occupied == (True, True, True, False, False, False, False)
    assert TERMS.rows[0].parents == TERMS.rows[1].parents == PARENTS


def test_the_output_width_is_the_atoms_dimension() -> None:
    narrow = build(width=4)
    wide = build(width=32)

    assert narrow.output_width == narrow.codebook.width == 4
    assert wide.output_width == wide.codebook.width == 32
    assert narrow.encode(TERMS).size("width") == 4
    assert wide.encode_nodes(TERMS).size("width") == 32


def test_the_codebook_holds_a_code_per_symbol_and_a_role_per_slot() -> None:
    codebook = build().codebook

    assert codebook.codes.count == N_SYMBOLS
    assert codebook.roles.count == CAPACITY
    assert codebook.codes.dimension == codebook.roles.dimension == WIDTH
    assert codebook.codes.magnitude == codebook.roles.magnitude == 1.0
    assert codebook.codes.floor == codebook.roles.floor == 1e-3


def test_the_settings_reach_both_atom_tables() -> None:
    torch.manual_seed(3)
    subject = HrrTermEncoder(
        codec=VIEW,
        component=COMPONENT,
        settings=settings(width=WIDE, magnitude=2.0, floor=0.5),
    )

    for table in (subject.codebook.codes, subject.codebook.roles):
        assert table.magnitude == 2.0
        assert table.floor == 0.5
        assert float(table.atoms.detach().abs().mean()) == pytest.approx(2.0, abs=0.05)


def test_every_node_is_its_binding_plus_its_corrected_children() -> None:
    subject = unequal(build(width=WIDE))

    for batch in (TERMS, TERNARY, CHAIN):
        nodes = subject.encode_nodes(batch).tensor

        for term, row_ in enumerate(batch.rows):
            for slot, occupied in enumerate(row_.occupied):
                if occupied:
                    assert torch.allclose(
                        nodes[term, slot],
                        reference_node(subject, row_, slot),
                        atol=1e-5,
                    )
                else:
                    assert torch.equal(
                        nodes[term, slot], torch.zeros_like(nodes[term, slot])
                    )


def test_a_term_is_the_scaled_superposition_over_its_addresses() -> None:
    subject = unequal(build(width=WIDE))

    for batch in (TERMS, TERNARY, CHAIN):
        encoded = subject.encode(batch).tensor

        for term, row_ in enumerate(batch.rows):
            assert torch.allclose(
                encoded[term], reference_flat(subject, row_), atol=1e-5
            )


def test_a_term_encodes_to_the_vector_at_its_root() -> None:
    subject = build()
    encoded = subject.encode(TERMS).tensor
    nodes = subject.encode_nodes(TERMS).tensor

    assert torch.allclose(encoded, nodes[:, 0, :], atol=1e-6)


LEAVES_BELOW_THE_ROOT: Final = (
    (
        1,
        row(
            SHAPE,
            parents=(Encoding.NO_PARENT, 0, 0, 0, 1, 1, 1),
            depths=(0, 1, 1, 1, 2, 2, 2),
            occupied=(True, True, False, False, False, False, False),
        ),
    ),
    (2, TERMS.rows[1]),
    (
        3,
        row(
            SHAPE,
            parents=(Encoding.NO_PARENT, 0, 0, 0, 1, 1, 1),
            depths=(0, 1, 1, 1, 2, 2, 2),
            occupied=(True, True, True, True, False, False, False),
        ),
    ),
)


def own_binding(subject: HrrTermEncoder, term: Encoding, slot: int) -> Tensor:
    codes = subject.codebook.codes.atoms.detach()
    roles = subject.codebook.roles.atoms.detach()

    return roles[slot] * codes[term.symbols[slot]]


def test_the_children_of_a_node_sit_at_the_norm_of_one_binding() -> None:
    subject = build(width=WIDE)
    set_magnitudes(subject.codebook.codes, torch.tensor(1.5))
    set_magnitudes(subject.codebook.roles, torch.tensor(1.5))
    one = bundle_norm(magnitude=1.5, arity=1, dimension=WIDE)
    alone = subject.encode_nodes(SINGLE).tensor[0, 0].detach()

    assert float(alone.norm()) == pytest.approx(one, rel=0.05)

    for arity, term in LEAVES_BELOW_THE_ROOT:
        nodes = subject.encode_nodes(BatchEncoding((term,))).tensor[0].detach()
        below = nodes[0] - own_binding(subject, term, 0)
        leaves = torch.stack([nodes[child] for child in children_of(term, 0)])
        superposed = leaves.sum(dim=0)
        uncorrected = bundle_norm(magnitude=1.5, arity=arity, dimension=WIDE)

        assert leaves.shape[0] == arity
        assert torch.allclose(below, superposed / bundle_scale(arity), atol=1e-5)
        assert float(superposed.norm()) == pytest.approx(uncorrected, rel=0.05)
        assert float(below.norm()) == pytest.approx(
            uncorrected / bundle_scale(arity), rel=0.05
        )
        assert float(below.norm()) == pytest.approx(one, rel=0.05)


def test_unbinding_an_address_recovers_its_constructor() -> None:
    subject = unequal(build(width=WIDE))
    codes = subject.codebook.codes.atoms.detach()
    roles = subject.codebook.roles.atoms.detach()

    for batch in (TERMS, TERNARY, CHAIN):
        encoded = subject.encode(batch).tensor.detach()

        for term, row_ in enumerate(batch.rows):
            for slot, occupied in enumerate(row_.occupied):
                if not occupied:
                    continue

                recovered = unbind(encoded[term], roles[slot])
                nearest = int((recovered @ codes.T).argmax())

                assert nearest == row_.symbols[slot]


def test_cross_talk_is_larger_under_a_weaker_role() -> None:
    subject = build(width=WIDE)
    set_magnitudes(subject.codebook.codes, torch.tensor(1.0))
    set_magnitudes(
        subject.codebook.roles,
        torch.tensor([1.0, 1.0, 1.0, 0.25, 1.0, 1.0, 4.0]).unsqueeze(-1),
    )
    codes = subject.codebook.codes.atoms.detach()
    roles = subject.codebook.roles.atoms.detach()
    encoded = subject.encode(TERMS).tensor.detach()[0]
    term = TERMS.rows[0]

    def similarity(slot: int) -> float:
        recovered = unbind(encoded, roles[slot])

        return float(
            torch.nn.functional.cosine_similarity(
                recovered, codes[term.symbols[slot]], dim=0
            )
        )

    weak, plain, strong = similarity(3), similarity(4), similarity(6)

    assert weak < plain < strong
    assert strong > 0.5


def test_a_swapped_role_is_told_apart_by_unequal_magnitudes() -> None:
    subject = unequal(build(width=WIDE))
    codes = subject.codebook.codes.atoms.detach()
    roles = subject.codebook.roles.atoms.detach()
    encoded = subject.encode(SINGLE).tensor[0]

    assert torch.allclose(encoded, roles[0] * codes[A], atol=1e-6)

    for other in range(1, CAPACITY):
        assert not torch.allclose(encoded, roles[other] * codes[A], atol=1e-3)
        assert not torch.allclose(encoded, roles[0] * codes[B], atol=1e-3)


def test_the_codebook_is_published_as_the_shared_submodule() -> None:
    subject = build()
    shared = subject.shared

    assert set(shared) == {"codebook"}
    assert shared["codebook"].module is subject.codebook
    assert isinstance(shared["codebook"].module, HolographicCodebook)
    assert any(module is subject.codebook for module in subject.modules())

    with pytest.raises(TypeError):
        cast("dict[str, SharedSubmodule[nn.Module]]", shared)["other"] = (
            SharedSubmodule(
                nn.Identity(),
                ModuleSharing(in_parameters=False, in_state_dict=False),
            )
        )


def test_the_encoder_declares_that_a_holder_trains_and_saves_its_codebook() -> None:
    subject = build()
    module, sharing = subject.shared["codebook"]

    assert module is subject.codebook
    assert sharing == ModuleSharing(in_parameters=True, in_state_dict=True)
    assert tuple(
        SharedSubmodule(
            module,
            ModuleSharing(in_parameters=False, in_state_dict=True),
        )
    ) == (
        module,
        ModuleSharing(in_parameters=False, in_state_dict=True),
    )


class Holder(nn.Module):
    def __init__(self, codebook: nn.Module) -> None:
        super().__init__()
        self.codebook = codebook


def test_a_holder_of_the_codebook_shares_exactly_its_state() -> None:
    subject = build()
    holder = Holder(subject.shared["codebook"].module)
    codebook_keys = set(subject.codebook.state_dict())
    codebook_parameters = {id(p) for p in subject.codebook.parameters()}
    holder_parameters = {id(p) for p in holder.parameters()}
    encoder_parameters = {id(p) for p in subject.parameters()}

    assert codebook_keys
    assert set(holder.state_dict()) == {f"codebook.{key}" for key in codebook_keys}
    assert set(subject.state_dict()) == {f"_codebook.{key}" for key in codebook_keys}
    assert holder_parameters == codebook_parameters
    assert holder_parameters & encoder_parameters == codebook_parameters
    assert all(module is not subject for module in holder.modules())


def test_the_codebook_holds_no_reference_to_its_encoder() -> None:
    subject = build()
    codebook = subject.codebook

    assert all(module is not subject for module in codebook.modules())
    assert set(codebook.state_dict()) == {
        f"{table}.{key}"
        for table, held in (("_codes", codebook.codes), ("_roles", codebook.roles))
        for key in held.state_dict()
    }


def test_a_holder_and_its_encoder_move_together_after_a_reload() -> None:
    written = build(seed=5)
    read = build(seed=6)
    holder = Holder(read.shared["codebook"].module)

    assert not torch.equal(written.encode(TERMS).tensor, read.encode(TERMS).tensor)

    holder.load_state_dict(
        {f"codebook.{k}": v for k, v in written.codebook.state_dict().items()}
    )

    assert torch.equal(written.encode(TERMS).tensor, read.encode(TERMS).tensor)


def test_parent_relations_change_the_node_encodings() -> None:
    batch = BatchEncoding((row(SHAPE), row(SHAPE, parents=OTHER_PARENTS)))
    subject = build()
    nodes = subject.encode_nodes(batch).tensor

    assert not torch.allclose(nodes[0, 1], nodes[1, 1], atol=1e-6)
    assert not torch.allclose(nodes[0, 2], nodes[1, 2], atol=1e-6)
    assert torch.allclose(nodes[0, 3:], nodes[1, 3:], atol=1e-6)


def test_a_term_reads_its_structure_through_the_arity_corrections() -> None:
    batch = BatchEncoding(
        (
            row(SHAPE),
            row(SHAPE, parents=OTHER_PARENTS),
            row(
                SHAPE,
                parents=(Encoding.NO_PARENT, 0, 0, 2, 1, 1, 1),
                depths=(0, 1, 1, 2, 2, 2, 2),
            ),
        )
    )
    subject = build()
    encoded = subject.encode(batch).tensor

    assert torch.allclose(encoded[0], encoded[1], atol=1e-6)
    assert not torch.allclose(encoded[0], encoded[2], atol=1e-6)


def test_a_child_reaches_its_ancestors_and_nothing_else() -> None:
    subject = build()
    original = subject.encode_nodes(TERMS).tensor[0]
    changed = subject.encode_nodes(
        BatchEncoding((row((JOIN, PAIR, JOIN, B, B, B, A)),))
    ).tensor[0]

    assert not torch.allclose(original[3], changed[3], atol=1e-6)
    assert not torch.allclose(original[1], changed[1], atol=1e-6)
    assert not torch.allclose(original[0], changed[0], atol=1e-6)

    for slot in (2, 4, 5, 6):
        assert torch.allclose(original[slot], changed[slot], atol=1e-6)


def test_a_slot_with_children_binds_its_own_constructor_code() -> None:
    subject = build()
    original = subject.encode_nodes(TERMS).tensor[0]
    inner = subject.encode_nodes(
        BatchEncoding((row((JOIN, JOIN, JOIN, A, B, B, A)),))
    ).tensor[0]

    assert not torch.allclose(original[1], inner[1], atol=1e-6)
    assert not torch.allclose(original[0], inner[0], atol=1e-6)

    for slot in (2, 3, 4, 5, 6):
        assert torch.allclose(original[slot], inner[slot], atol=1e-6)


def test_an_unoccupied_slot_is_the_child_of_nothing() -> None:
    occupied = (True, True, True, False, False, False, False)
    held = BatchEncoding((row(SHAPE, occupied=occupied),))
    reparented = BatchEncoding(
        (
            row(
                SHAPE,
                parents=(Encoding.NO_PARENT, 0, 0, 1, 3, 3, 3),
                occupied=occupied,
            ),
        )
    )
    subject = build()

    assert torch.allclose(
        subject.encode(held).tensor, subject.encode(reparented).tensor, atol=1e-6
    )
    assert torch.allclose(
        subject.encode_nodes(held).tensor,
        subject.encode_nodes(reparented).tensor,
        atol=1e-6,
    )


def test_padded_symbol_codes_cannot_change_live_outputs() -> None:
    occupied = (True, True, True, False, False, False, False)
    original = BatchEncoding(
        (row((5, 4, 3, *((OperatorIndex.PAD,) * 4)), occupied=occupied),)
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
    encoded = build().encode(empty)

    assert encoded.mask is not None
    assert encoded.mask.tolist() == [False]
    assert torch.equal(encoded.tensor, torch.zeros_like(encoded.tensor))


def test_a_batch_whose_rows_all_hold_a_term_comes_back_unmasked() -> None:
    assert build().encode(TERMS).mask is None


def test_a_batch_of_no_term_encodes_to_no_row() -> None:
    subject = build()
    encoded = subject.encode(BatchEncoding(()))

    assert encoded.tensor.shape == (0, WIDTH)
    assert subject.encode_nodes(BatchEncoding(())).tensor.shape == (
        0,
        CAPACITY,
        WIDTH,
    )


def test_input_placement_follows_model_parameters() -> None:
    for dtype in (torch.float32, torch.float64):
        subject = build().to(dtype=dtype)
        hard = subject.encode(TERMS)
        nodes = subject.encode_nodes(TERMS)

        assert hard.tensor.dtype == nodes.tensor.dtype == dtype


def test_gradient_reaches_the_roles_and_the_codes_that_were_bound() -> None:
    subject = build()
    subject.encode(SINGLE).tensor.square().sum().backward()
    (roles,) = subject.codebook.roles.parameters()
    (codes,) = subject.codebook.codes.parameters()

    assert roles.grad is not None
    assert codes.grad is not None
    assert bool(roles.grad[0].ne(0.0).all())
    assert torch.equal(roles.grad[1:], torch.zeros_like(roles.grad[1:]))
    assert bool(codes.grad[A].ne(0.0).all())
    assert torch.equal(codes.grad[B], torch.zeros_like(codes.grad[B]))


def test_the_site_map_publishes_the_term_and_node_sites() -> None:
    sites = build().site_map()

    assert sites.kind is ComponentKind.TERM_ENCODER
    assert ensure_publishes(sites) is sites
    assert set(sites) == {TERM_SITE, NODE_SITE}
    assert sites[TERM_SITE].axes == ("term", "width")
    assert sites[NODE_SITE].axes == ("term", "slot", "width")

    for site in (TERM_SITE, NODE_SITE):
        assert sites[site].dtype_class is DTypeClass.REAL
        assert sites[site].differentiable
        assert sites[site].access == Access.OBSERVE


def test_a_holographic_encoder_map_refuses_permuted_node_axes() -> None:
    sites = build().site_map()
    changed = dict(sites)
    changed[NODE_SITE] = SiteSpec(
        ("term", "width", "slot"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE,
    )

    with pytest.raises(ValueError, match=r"NODE_ENCODING.*runs over"):
        ensure_publishes(SiteMap(COMPONENT, ComponentKind.TERM_ENCODER, changed))


def test_the_encoder_retains_the_codec_view() -> None:
    subject = HrrTermEncoder(codec=VIEW, component=COMPONENT, settings=settings())

    assert subject.codec is VIEW


def test_construction_refuses_a_bad_width_or_atom_setting() -> None:
    with pytest.raises(ValueError, match="dimension"):
        build_with(settings(width=0))

    with pytest.raises(TypeError):
        build_with(settings(width=cast("int", 2.5)))

    with pytest.raises(ValueError, match="floor"):
        build_with(settings(floor=0.0))

    with pytest.raises(ValueError, match="floor"):
        build_with(settings(magnitude=1.0, floor=2.0))

    with pytest.raises(ValueError, match="magnitude"):
        build_with(settings(magnitude=float("inf")))


def build_with(held: HolographicEncoderSettings) -> HrrTermEncoder:
    return HrrTermEncoder(codec=VIEW, component=COMPONENT, settings=held)
