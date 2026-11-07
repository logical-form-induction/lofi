"""The holographic decoder exposes one payload-aware choice parameterization."""

from typing import Final

import pytest
import torch
from tests.conformance.decoder import (
    DecoderAgreementSuite,
    DecoderConformance,
    FrontierDecoderSuite,
    MarginalDecoderSuite,
    TeacherForcedDecoderSuite,
)

from lofi.nn._values import Encoded
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import HolographicTermDecoder
from lofi.nn.decoders.hrr import _PathAddressedHolographicTermDecoder
from lofi.nn.encoders import HolographicEncoderSettings, HrrTermEncoder
from lofi.nn.ports import FrontierDecoder, MarginalDecoder, TeacherForcedDecoder
from lofi.nn.sites import ComponentId, Role, Site
from lofi.nn.strategies import greedy, score
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Frontier,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
    TermStore,
    Vocabulary,
)

SIGNATURE: Final = Signature(
    "holographic-choice-decoder",
    ("Cat",),
    (
        Operator(
            "/",
            "Cat",
            arg_sorts=("Cat", "Cat"),
            family="constructor",
        ),
    ),
    vocabularies=(Vocabulary("Cat", ("N", "NP", "S")),),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": 3})
CODEC: Final = TermCodec(
    INDEX,
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3),
)
VIEW: Final = NeuralCodecView(CODEC)
STORE: Final = TermStore(SIGNATURE)
VOCABULARY: Final = SIGNATURE.vocabulary("Cat")
CONSTANT: Final = SIGNATURE.constant("Cat")
SLASH: Final = SIGNATURE.operator(SIGNATURE.code("/"))
N: Final = STORE.op(CONSTANT, payload=VOCABULARY.id("N"))
NP: Final = STORE.op(CONSTANT, payload=VOCABULARY.id("NP"))
S: Final = STORE.op(CONSTANT, payload=VOCABULARY.id("S"))
FUNCTOR: Final = STORE.op(SLASH, S, NP)
SPARSE: Final = STORE.op(SLASH, N, STORE.op(SLASH, NP, S))
TERMS: Final = CODEC.encode_batch((FUNCTOR, N))
INPUTS: Final = Encoded(
    torch.tensor(
        (
            (0.25, -0.5, 0.75, 1.0, -0.25, 0.5, 0.1, -0.3),
            (-0.25, 0.5, 1.0, -0.75, 0.4, -0.1, 0.3, 0.2),
        ),
        dtype=torch.float64,
    ),
    ("term", "width"),
)
ENCODER_COMPONENT: Final = ComponentId("holographic-encoder")
DECODER_COMPONENT: Final = ComponentId("holographic-decoder")


def pair() -> tuple[HrrTermEncoder, _PathAddressedHolographicTermDecoder]:
    """Return a deterministic encoder and its codebook-sharing decoder."""
    torch.manual_seed(17)
    encoder = HrrTermEncoder(
        codec=VIEW,
        component=ENCODER_COMPONENT,
        settings=HolographicEncoderSettings(
            width=INPUTS.tensor.shape[1], magnitude=1.0, floor=1e-3
        ),
    ).to(torch.float64)
    decoder = _PathAddressedHolographicTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=DECODER_COMPONENT,
        codebook=encoder.codebook,
        choice_heads="flat",
    ).to(torch.float64)
    return encoder, decoder


def build() -> _PathAddressedHolographicTermDecoder:
    """Return one deterministic fixture decoder."""
    return pair()[1]


class TestHolographicTermDecoder(
    TeacherForcedDecoderSuite,
    MarginalDecoderSuite,
    FrontierDecoderSuite,
    DecoderAgreementSuite,
):
    """Apply every protocol law the holographic decoder implements."""

    @pytest.fixture
    def conformance(
        self,
    ) -> DecoderConformance[_PathAddressedHolographicTermDecoder]:
        frontier = Frontier(
            TermStore(SIGNATURE.with_holes()).hole("Cat"),
            {(): object()},
            0.0,
        )
        return DecoderConformance(
            build,
            INPUTS,
            TERMS,
            CHOICES,
            frontier.context(VIEW.capacity),
            "Cat",
            VIEW.capacity,
        )


def test_holographic_decoder_declares_the_current_choice_interfaces() -> None:
    decoder = build()

    assert isinstance(decoder, FrontierDecoder)
    assert isinstance(decoder, TeacherForcedDecoder)
    assert isinstance(decoder, MarginalDecoder)
    assert not hasattr(decoder, "score")
    assert not hasattr(decoder, "distribution")
    assert not hasattr(decoder, "draw")


def test_preorder_holographic_decoder_exposes_only_teacher_forcing() -> None:
    codec = NeuralCodecView(TermCodec(INDEX, PreorderArityLayout(3)))
    encoder = HrrTermEncoder(
        codec=codec,
        component=ENCODER_COMPONENT,
        settings=HolographicEncoderSettings(
            width=INPUTS.tensor.shape[1], magnitude=1.0, floor=1e-3
        ),
    ).to(torch.float64)
    decoder = HolographicTermDecoder(
        codec=codec,
        choices=CHOICES,
        component=DECODER_COMPONENT,
        codebook=encoder.codebook,
        choice_heads="flat",
    ).to(torch.float64)
    preorder = codec.codec.encode_batch((FUNCTOR,))
    inputs = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    rows = decoder.rows(inputs, terms=preorder)
    scored = score(decoder, inputs, preorder, index=CHOICES)

    assert isinstance(decoder, TeacherForcedDecoder)
    assert not isinstance(decoder, FrontierDecoder)
    assert not isinstance(decoder, MarginalDecoder)
    assert rows.tensor.shape == (1, codec.capacity, CHOICES.width)
    assert bool(torch.isfinite(scored.tensor).all())


def test_teacher_rows_use_the_language_choice_width() -> None:
    rows = build().rows(INPUTS, terms=TERMS)

    assert CHOICES.width > VIEW.n_symbols
    assert rows.axes == ("term", "slot", "choice")
    assert rows.normalized_over == "choice"
    assert rows.tensor.shape == (2, VIEW.capacity, CHOICES.width)
    assert torch.allclose(
        rows.tensor[:, 0].logsumexp(dim=-1),
        torch.zeros(2, dtype=torch.float64),
    )


def test_frontier_teacher_and_marginal_rows_share_the_root_head() -> None:
    decoder = build()
    teacher = decoder.rows(INPUTS, terms=TERMS).tensor[:, 0]
    marginal = decoder.marginal_rows(INPUTS, root_sort="Cat")
    frontier = Frontier(
        TermStore(SIGNATURE.with_holes()).hole("Cat"),
        {(): object()},
        0.0,
    )
    root = decoder.weigh(
        decoder.start(Encoded(INPUTS.tensor[:1], INPUTS.axes)),
        context=frontier.context(VIEW.capacity),
    )

    assert marginal.axes == ("term", "slot", "choice")
    assert marginal.normalized_over == "choice"
    assert torch.equal(root.tensor, teacher[0])
    assert torch.equal(marginal.tensor[:, 0], teacher)


def test_marginal_rows_condition_legality_on_the_requested_root_sort() -> None:
    signature = Signature(
        "holographic-multisort",
        ("Left", "Right"),
        (
            Operator("left-leaf", "Left"),
            Operator("left-wrap", "Left", arg_sorts=("Left",)),
            Operator("right-leaf", "Right"),
            Operator("right-wrap", "Right", arg_sorts=("Right",)),
        ),
    )
    index = OperatorIndex(signature)
    choices = ChoiceIndex(signature, depth_bounds={"Left": 2, "Right": 2})
    codec = NeuralCodecView(
        TermCodec(
            index,
            CompleteTree(signature, signature.operators, order="df", depth=2),
        )
    )
    encoder = HrrTermEncoder(
        codec=codec,
        component=ComponentId("multisort-encoder"),
        settings=HolographicEncoderSettings(width=8, magnitude=1.0, floor=1e-3),
    ).to(torch.float64)
    decoder = _PathAddressedHolographicTermDecoder(
        codec=codec,
        choices=choices,
        component=ComponentId("multisort-decoder"),
        codebook=encoder.codebook,
        choice_heads="flat",
    ).to(torch.float64)
    inputs = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    left = decoder.marginal_rows(inputs, root_sort="Left")
    right = decoder.marginal_rows(inputs, root_sort="Right")

    assert left.mask is not None
    assert right.mask is not None
    assert torch.equal(left.mask[0, 0], torch.tensor(choices.legal_choices["Left"]))
    assert torch.equal(right.mask[0, 0], torch.tensor(choices.legal_choices["Right"]))


def test_shared_score_selects_the_teacher_forced_choice_rows() -> None:
    decoder = build()
    rows = decoder.rows(INPUTS, terms=TERMS)
    choices = torch.tensor(tuple(CHOICES.codes(row) for row in TERMS.rows))
    occupied = torch.tensor(TERMS.mask)
    selected = rows.tensor.gather(2, choices.unsqueeze(-1)).squeeze(-1)
    expected = torch.where(occupied, selected, selected.new_zeros(())).sum(dim=1)

    assert torch.equal(score(decoder, INPUTS, TERMS, index=CHOICES).tensor, expected)


def test_frontier_decoding_tracks_sparse_fixed_layout_paths() -> None:
    decoder = build()
    inputs = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    decoded = greedy(
        decoder,
        inputs,
        index=CHOICES,
        store=STORE,
        root_sort="Cat",
        max_nodes=VIEW.capacity,
        clamp=SPARSE,
    )

    assert decoded.terms == (SPARSE,)
    teacher = score(
        decoder, inputs, CODEC.encode_batch((SPARSE,)), index=CHOICES
    ).tensor[0]
    assert torch.allclose(decoded.log_weights[0], teacher)


def test_teacher_supervision_reaches_the_choice_head_and_shared_codebook() -> None:
    encoder, decoder = pair()
    loss = -score(decoder, INPUTS, TERMS, index=CHOICES).tensor.sum()
    loss.backward()

    assert decoder.codebook is encoder.codebook
    gradients = tuple(
        parameter.grad
        for parameter in decoder.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(bool(gradient.isfinite().all()) for gradient in gradients)
    assert decoder.get_parameter("_heads._flat.weight").grad is not None


def test_the_distribution_site_names_choice_rows() -> None:
    decoder = build()
    site = Site(DECODER_COMPONENT, Role.SYMBOL_DISTRIBUTION)

    assert decoder.site_map()[site].axes == ("term", "slot", "choice")
