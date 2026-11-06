"""The diffusion decoder reports payload-aware denoising choice rows."""

from typing import Final

import pytest
import torch
from tests.conformance.decoder import (
    DecoderConformance,
    FrontierDecoderSuite,
    TeacherForcedDecoderSuite,
)

from lofi.nn._values import Encoded
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders.diffusion import (
    DiffusionDecoderSettings,
    DiffusionTermDecoder,
    NoiseSchedule,
)
from lofi.nn.ports import FrontierDecoder, MarginalDecoder, TeacherForcedDecoder
from lofi.nn.sites import ComponentId, Role, Site
from lofi.nn.strategies import greedy, score
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Frontier,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
    Vocabulary,
)

SIGNATURE: Final = Signature(
    "diffusion-choice-decoder",
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
        ((0.25, -0.5, 0.75, 1.0), (-0.25, 0.5, 1.0, -0.75)),
        dtype=torch.float64,
    ),
    ("term", "width"),
)
COMPONENT: Final = ComponentId("diffusion-decoder")


def build() -> DiffusionTermDecoder:
    """Return one deterministic fixture decoder."""
    torch.manual_seed(17)
    return DiffusionTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        settings=DiffusionDecoderSettings(
            input_width=4,
            width=8,
            layers=2,
            feedforward_width=16,
            activation="gelu",
        ),
        schedule=NoiseSchedule.linear(steps=4, first=0.25),
        choice_heads="flat",
    ).to(torch.float64)


class TestDiffusionTermDecoder(TeacherForcedDecoderSuite, FrontierDecoderSuite):
    """Apply the frontier and teacher-forcing laws diffusion implements."""

    @pytest.fixture
    def conformance(self) -> DecoderConformance[DiffusionTermDecoder]:
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


def test_diffusion_declares_only_its_current_choice_interfaces() -> None:
    decoder = build()

    assert isinstance(decoder, FrontierDecoder)
    assert isinstance(decoder, TeacherForcedDecoder)
    assert not isinstance(decoder, MarginalDecoder)
    assert not hasattr(decoder, "score")
    assert not hasattr(decoder, "distribution")
    assert not hasattr(decoder, "draw")
    assert not hasattr(decoder, "sample")


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


def test_frontier_and_teacher_forcing_share_the_root_choice_row() -> None:
    decoder = build()
    teacher = decoder.rows(INPUTS, terms=TERMS).tensor[0, 0]
    frontier = Frontier(
        TermStore(SIGNATURE.with_holes()).hole("Cat"),
        {(): object()},
        0.0,
    )
    root = decoder.weigh(
        decoder.start(Encoded(INPUTS.tensor[:1], INPUTS.axes)),
        context=frontier.context(VIEW.capacity),
    )

    assert root.axes == ("choice",)
    assert root.normalized_over == "choice"
    assert torch.equal(root.tensor, teacher)


def test_teacher_feedback_cannot_leak_the_current_choice() -> None:
    decoder = build()
    first = CODEC.encode_batch((STORE.op(SLASH, S, NP),))
    second = CODEC.encode_batch((STORE.op(SLASH, NP, S),))
    occupied = tuple(slot for slot, live in enumerate(first.mask[0]) if live)
    one = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    before = decoder.rows(one, terms=first).tensor
    after = decoder.rows(one, terms=second).tensor

    assert torch.equal(before[:, occupied[0]], after[:, occupied[0]])
    assert torch.equal(before[:, occupied[1]], after[:, occupied[1]])
    assert bool(before[:, occupied[2]].ne(after[:, occupied[2]]).any())


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


def test_teacher_supervision_reaches_feedback_and_choice_parameters() -> None:
    decoder = build()
    loss = -score(decoder, INPUTS, TERMS, index=CHOICES).tensor.sum()
    loss.backward()

    head = decoder.get_parameter("_choice_heads._flat.weight")
    choices = decoder.get_parameter("_choices_table.weight")
    assert head.grad is not None
    assert bool(head.grad.isfinite().all())
    assert bool(head.grad.ne(0).any())
    assert choices.grad is not None
    assert bool(choices.grad.isfinite().all())
    assert bool(choices.grad.ne(0).any())


def test_denoised_rows_use_choice_axes_and_normalization() -> None:
    decoder = build()
    revealed = torch.zeros((2, VIEW.capacity), dtype=torch.bool)
    rows = decoder.denoised(
        INPUTS,
        terms=TERMS,
        steps=torch.tensor((1, 2), dtype=torch.int64),
        revealed=revealed,
    )

    assert rows.axes == ("term", "slot", "choice")
    assert rows.normalized_over == "choice"
    assert rows.tensor.shape == (2, VIEW.capacity, CHOICES.width)


def test_noise_schedule_reaches_a_fully_masked_final_step() -> None:
    schedule = NoiseSchedule.linear(steps=4, first=0.25)

    assert schedule.betas == pytest.approx((0.25, 0.5, 0.75, 1.0))
    assert schedule.alpha_bars[-1] == 0.0
    assert schedule.previous_alpha_bar(0) == 1.0
    assert schedule.reveal_probability(0) == 1.0


def test_the_distribution_site_names_choice_rows() -> None:
    decoder = build()
    site = Site(COMPONENT, Role.SYMBOL_DISTRIBUTION)

    assert decoder.site_map()[site].axes == ("term", "slot", "choice")
