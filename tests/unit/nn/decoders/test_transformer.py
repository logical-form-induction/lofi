"""The Transformer decoder reports payload-aware autoregressive choice rows."""

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
from lofi.nn.decoders.transformer import (
    TransformerDecoderSettings,
    TransformerTermDecoder,
)
from lofi.nn.ports import (
    FrontierDecoder,
    MarginalDecoder,
    StandardDeclaration,
    TeacherForcedDecoder,
)
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
    "transformer-choice-decoder",
    ("Cat",),
    (
        Operator(
            "/",
            "Cat",
            arg_sorts=("Cat", "Cat"),
            family="constructor",
        ),
        Operator(
            "combine",
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
COMBINE: Final = SIGNATURE.operator(SIGNATURE.code("combine"))
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
COMPONENT: Final = ComponentId("transformer-decoder")


def build(
    *, structure: StandardDeclaration = StandardDeclaration.PARENT_RECURRENCE
) -> TransformerTermDecoder:
    """Return one deterministic fixture decoder."""
    torch.manual_seed(17)
    return TransformerTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        settings=TransformerDecoderSettings(
            input_width=4,
            width=8,
            heads=2,
            layers=2,
            feedforward_width=16,
            activation="gelu",
        ),
        structure=structure.declaration,
        choice_heads="flat",
    ).to(torch.float64)


class TestTransformerTermDecoder(
    TeacherForcedDecoderSuite,
    MarginalDecoderSuite,
    FrontierDecoderSuite,
    DecoderAgreementSuite,
):
    """Apply every protocol law the Transformer implements."""

    @pytest.fixture
    def conformance(self) -> DecoderConformance[TransformerTermDecoder]:
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


def test_transformer_declares_the_current_choice_interfaces() -> None:
    decoder = build()

    assert isinstance(decoder, FrontierDecoder)
    assert isinstance(decoder, TeacherForcedDecoder)
    assert isinstance(decoder, MarginalDecoder)
    assert not hasattr(decoder, "score")
    assert not hasattr(decoder, "distribution")
    assert not hasattr(decoder, "draw")


@pytest.mark.parametrize(
    "structure",
    [
        StandardDeclaration.POSITIONS,
        StandardDeclaration.ANCESTOR_MASK,
        StandardDeclaration.ANCESTOR_AND_SIBLING_BIAS,
        StandardDeclaration.PARENT_RECURRENCE,
    ],
)
def test_every_structural_variant_reports_normalized_choice_rows(
    structure: StandardDeclaration,
) -> None:
    rows = build(structure=structure).rows(INPUTS, terms=TERMS)

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
    assert torch.allclose(root.tensor, teacher[0])
    assert torch.equal(marginal.tensor[:, 0], teacher)


def test_marginal_rows_condition_legality_on_the_requested_root_sort() -> None:
    signature = Signature(
        "transformer-multisort",
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
    decoder = TransformerTermDecoder(
        codec=codec,
        choices=choices,
        component=ComponentId("multisort-transformer"),
        settings=TransformerDecoderSettings(4, 8, 2, 1, 16, "gelu"),
        structure=StandardDeclaration.PARENT_RECURRENCE.declaration,
        choice_heads="flat",
    ).to(torch.float64)
    left = decoder.marginal_rows(INPUTS, root_sort="Left")
    right = decoder.marginal_rows(INPUTS, root_sort="Right")

    assert left.mask is not None
    assert right.mask is not None
    assert torch.equal(left.mask[0, 0], torch.tensor(choices.legal_choices["Left"]))
    assert torch.equal(right.mask[0, 0], torch.tensor(choices.legal_choices["Right"]))


def test_teacher_feedback_changes_only_rows_after_the_changed_choice() -> None:
    decoder = build()
    first = CODEC.encode_batch((STORE.op(SLASH, S, NP),))
    second = CODEC.encode_batch((STORE.op(COMBINE, S, NP),))
    occupied = tuple(slot for slot, live in enumerate(first.mask[0]) if live)
    one = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    before = decoder.rows(one, terms=first).tensor
    after = decoder.rows(one, terms=second).tensor

    assert torch.equal(before[:, occupied[0]], after[:, occupied[0]])
    assert bool(before[:, occupied[1]].ne(after[:, occupied[1]]).any())


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


def test_the_distribution_site_names_choice_rows() -> None:
    decoder = build()
    site = Site(COMPONENT, Role.SYMBOL_DISTRIBUTION)

    assert decoder.site_map()[site].axes == ("term", "slot", "choice")
