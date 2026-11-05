"""The recursive decoder uses one payload-aware choice parameterization."""

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
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.ports import (
    FrontierDecoder,
    MarginalDecoder,
    TeacherForcedDecoder,
)
from lofi.nn.sites import ComponentId
from lofi.nn.strategies import greedy, score
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.terms import (
    BatchEncoding,
    BindingSite,
    ChoiceIndex,
    CompleteTree,
    Frontier,
    HoleContext,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
    Vocabulary,
)

SIGNATURE: Final = Signature(
    "recursive-choice-decoder",
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
TERMS: Final = CODEC.encode_batch((FUNCTOR, N))
INPUTS: Final = Encoded(
    torch.tensor(
        ((0.25, -0.5, 0.75, 1.0), (-0.25, 0.5, 1.0, -0.75)),
        dtype=torch.float64,
    ),
    ("term", "width"),
)


def build() -> RecursiveTermDecoder:
    """Return one deterministic fixture decoder."""
    torch.manual_seed(17)
    return RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("recursive"),
        settings=TreeDecoderSettings(input_width=4, width=6),
        choice_heads="flat",
    ).to(torch.float64)


class TestARecursiveTermDecoder(
    TeacherForcedDecoderSuite,
    MarginalDecoderSuite,
    FrontierDecoderSuite,
    DecoderAgreementSuite,
):
    @pytest.fixture
    def conformance(self) -> DecoderConformance[RecursiveTermDecoder]:
        frontier = Frontier(
            TermStore(SIGNATURE.with_holes()).hole("Cat"),
            {(): object()},
            0.0,
        )
        return DecoderConformance(
            build=build,
            inputs=INPUTS,
            terms=TERMS,
            index=CHOICES,
            root=frontier.context(VIEW.capacity),
            root_sort="Cat",
            slots=VIEW.capacity,
        )


def _selected(rows: torch.Tensor, terms: BatchEncoding) -> torch.Tensor:
    """Return each occupied teacher choice's log weight."""
    codes = torch.tensor(tuple(CHOICES.codes(row) for row in terms.rows))
    occupied = torch.tensor(terms.mask)
    selected = rows.gather(2, codes.unsqueeze(-1)).squeeze(-1)
    return torch.where(occupied, selected, selected.new_zeros(()))


def test_recursive_decoder_declares_the_current_choice_interfaces() -> None:
    decoder = build()

    assert isinstance(decoder, FrontierDecoder)
    assert isinstance(decoder, TeacherForcedDecoder)
    assert isinstance(decoder, MarginalDecoder)
    assert not hasattr(decoder, "score")
    assert not hasattr(decoder, "distribution")
    assert not hasattr(decoder, "draw")


def test_teacher_rows_use_the_language_choice_width_not_the_operator_width() -> None:
    rows = build().rows(INPUTS, terms=TERMS)

    assert CHOICES.width > VIEW.n_symbols
    assert rows.axes == ("term", "slot", "choice")
    assert rows.normalized_over == "choice"
    assert rows.tensor.shape == (2, VIEW.capacity, CHOICES.width)
    assert torch.allclose(
        rows.tensor[:, 0].exp().sum(dim=-1),
        torch.ones(2, dtype=torch.float64),
    )


def test_the_shared_strategy_selects_the_teacher_forced_choice_rows() -> None:
    decoder = build()
    rows = decoder.rows(INPUTS, terms=TERMS)
    expected = _selected(rows.tensor, TERMS).sum(dim=1)
    shared = score(decoder, INPUTS, TERMS, index=CHOICES)

    assert torch.equal(shared.tensor, expected)


def test_marginal_rows_are_payload_aware_and_agree_at_the_root() -> None:
    decoder = build()
    marginal = decoder.marginal_rows(INPUTS, root_sort="Cat")
    teacher = decoder.rows(INPUTS, terms=TERMS)

    assert marginal.axes == ("term", "slot", "choice")
    assert marginal.normalized_over == "choice"
    assert marginal.tensor.shape == (2, VIEW.capacity, CHOICES.width)
    assert torch.equal(marginal.tensor[:, 0], teacher.tensor[:, 0])


def test_marginal_rows_mark_slots_below_nullary_choices_unreachable() -> None:
    signature = Signature(
        "nullary-marginal",
        ("Cat",),
        (Operator("leaf", "Cat"),),
    )
    index = OperatorIndex(signature)
    choices = ChoiceIndex(signature, depth_bounds={"Cat": 2})
    codec = NeuralCodecView(
        TermCodec(
            index,
            CompleteTree(signature, signature.operators, order="df", depth=2),
        )
    )
    decoder = RecursiveTermDecoder(
        codec=codec,
        choices=choices,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("nullary"),
        settings=TreeDecoderSettings(input_width=4, width=6),
        choice_heads="flat",
    ).to(torch.float64)

    rows = decoder.marginal_rows(INPUTS, root_sort="Cat")

    assert rows.mask is not None
    assert bool(rows.mask[:, 0].any())
    assert torch.allclose(
        rows.tensor[:, 0].exp().sum(dim=1),
        torch.ones(2, dtype=rows.tensor.dtype),
    )
    assert not bool(rows.mask[:, 1:].any())
    assert bool(rows.tensor[:, 1:].isneginf().all())


def test_teacher_supervision_reaches_the_head_frontier_search_reads() -> None:
    decoder = build()
    loss = -score(decoder, INPUTS, TERMS, index=CHOICES).tensor.sum()
    loss.backward()

    head = decoder.get_parameter("_choice_heads._flat.weight")
    symbols = decoder.get_parameter("_choice_symbols.weight")

    assert head.grad is not None
    assert bool(head.grad.isfinite().all())
    assert bool(head.grad.ne(0).any())
    assert symbols.grad is not None
    assert bool(symbols.grad.isfinite().all())
    assert bool(symbols.grad.ne(0).any())


def test_frontier_rows_and_teacher_rows_share_the_root_head() -> None:
    decoder = build()
    teacher = decoder.rows(INPUTS, terms=TERMS).tensor[0, 0]
    frontier = Frontier(
        TermStore(SIGNATURE.with_holes()).hole("Cat"),
        {(): object()},
        0.0,
    )
    row = decoder.weigh(
        decoder.start(Encoded(INPUTS.tensor[:1], INPUTS.axes)),
        context=frontier.context(VIEW.capacity),
    )

    assert torch.equal(row.tensor, teacher)


def test_frontier_choices_cannot_grow_beyond_the_fixed_layout() -> None:
    decoder = build()
    state = decoder.start(Encoded(INPUTS.tensor[:1], INPUTS.axes))
    slash = CHOICES.index(SLASH)
    noun = CHOICES.index(CONSTANT, VOCABULARY.id("N"))
    state = decoder.fill(state, choices=torch.tensor(slash))
    state = decoder.fill(state, choices=torch.tensor(slash))
    state = decoder.fill(state, choices=torch.tensor(noun))
    context = HoleContext((0, 0), "Cat", {}, {}, VIEW.capacity)
    row = decoder.weigh(state, context=context)

    assert row.mask is not None
    assert not bool(row.mask[slash])
    assert bool(row.mask[noun])

    with pytest.raises(ValueError, match="exceeds the decoder's fixed term layout"):
        decoder.fill(state, choices=torch.tensor(slash))


def test_teacher_rows_apply_the_same_fixed_layout_bound() -> None:
    decoder = build()
    deep = STORE.op(SLASH, FUNCTOR, N)
    encoded = CODEC.encode(deep)
    rows = decoder.rows(
        Encoded(INPUTS.tensor[:1], INPUTS.axes),
        terms=BatchEncoding((encoded,)),
    )
    slash = CHOICES.index(SLASH)
    leaves = tuple(
        slot
        for slot, (occupied, depth) in enumerate(
            zip(encoded.occupied, encoded.depths, strict=True)
        )
        if occupied and depth == 2
    )

    assert leaves
    assert all(bool(rows.tensor[0, slot, slash].isneginf()) for slot in leaves)


def test_payload_choices_are_distinct_columns_of_one_operator_symbol() -> None:
    decoder = build()
    rows = decoder.rows(INPUTS, terms=TERMS).tensor
    constant_codes = tuple(
        CHOICES.index(CONSTANT, VOCABULARY.id(name)) for name in VOCABULARY.names
    )

    assert len(constant_codes) == len(VOCABULARY)
    assert len({float(rows[1, 0, code].detach()) for code in constant_codes}) > 1


def test_teacher_feedback_changes_descendants_but_not_the_root() -> None:
    decoder = build()
    first = CODEC.encode_batch((STORE.op(SLASH, S, NP),))
    second = CODEC.encode_batch((STORE.op(SLASH, NP, S),))
    one = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    before = decoder.rows(one, terms=first).tensor
    after = decoder.rows(one, terms=second).tensor

    assert torch.equal(before[:, 0], after[:, 0])
    assert bool(before[:, 1:].ne(after[:, 1:]).any())


def test_masked_inputs_mask_choice_rows_and_scores() -> None:
    decoder = build()
    masked = Encoded(INPUTS.tensor, INPUTS.axes, torch.tensor((True, False)))
    rows = decoder.rows(masked, terms=TERMS)
    weights = score(decoder, masked, TERMS, index=CHOICES)

    assert rows.mask is not None
    assert torch.equal(rows.mask, torch.tensor((True, False)))
    assert bool(rows.tensor[1].isneginf().all())
    assert weights.mask is not None
    assert bool(weights.tensor[1].isneginf())


def test_marginal_rows_refuse_a_root_sort_reaching_binding_operators() -> None:
    signature = Signature(
        "binding-marginal",
        ("Cat",),
        (
            Operator(
                "bind",
                "Cat",
                arg_sorts=("Cat",),
                binding=(BindingSite((0,), "Cat"),),
            ),
            Operator("leaf", "Cat"),
        ),
    )
    index = OperatorIndex(signature)
    choices = ChoiceIndex(signature, depth_bounds={"Cat": 2})
    codec = NeuralCodecView(
        TermCodec(
            index,
            CompleteTree(signature, signature.operators, order="df", depth=2),
        )
    )

    decoder = RecursiveTermDecoder(
        codec=codec,
        choices=choices,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("binding"),
        settings=TreeDecoderSettings(input_width=4, width=6),
        choice_heads="flat",
    )

    with pytest.raises(ValueError, match="reach a binding operator"):
        decoder.marginal_rows(INPUTS, root_sort="Cat")


def test_marginal_rows_refuse_a_choice_dependent_child_sort() -> None:
    signature = Signature(
        "mixed-child-sort-marginal",
        ("Root", "Left", "Right"),
        (
            Operator("left", "Root", arg_sorts=("Left",)),
            Operator("right", "Root", arg_sorts=("Right",)),
            Operator("l", "Left"),
            Operator("r", "Right"),
        ),
    )
    index = OperatorIndex(signature)
    choices = ChoiceIndex(
        signature,
        depth_bounds={"Root": 2, "Left": 1, "Right": 1},
    )
    codec = NeuralCodecView(
        TermCodec(
            index,
            CompleteTree(signature, signature.operators, order="df", depth=2),
        )
    )

    decoder = RecursiveTermDecoder(
        codec=codec,
        choices=choices,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("mixed"),
        settings=TreeDecoderSettings(input_width=4, width=6),
        choice_heads="flat",
    )

    with pytest.raises(ValueError, match="one fixed sort"):
        decoder.marginal_rows(INPUTS, root_sort="Root")


def test_a_malformed_teacher_is_refused_before_choice_recurrence() -> None:
    malformed = TERMS.rows[0]._replace(symbols=TERMS.rows[0].symbols[:-1])

    with pytest.raises(ValueError, match="symbols cover"):
        build().rows(
            Encoded(INPUTS.tensor[:1], INPUTS.axes),
            terms=BatchEncoding((malformed,)),
        )


def test_greedy_decoding_uses_the_same_payload_aware_choice_index() -> None:
    decoder = build()
    decoded = greedy(
        decoder,
        Encoded(INPUTS.tensor[:1], INPUTS.axes),
        index=CHOICES,
        store=STORE,
        root_sort="Cat",
        max_nodes=VIEW.capacity,
    )

    assert len(decoded.terms) == 1
    assert decoded.terms[0].op.sort == "Cat"


def test_placement_follows_module_dtype() -> None:
    decoder = build().to(torch.float32)
    rows = decoder.rows(INPUTS, terms=TERMS)

    assert rows.tensor.dtype is torch.float32
