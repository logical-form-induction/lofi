from typing import TYPE_CHECKING, Final, Literal

import pytest
import torch
from tests.conformance.decoder import (
    DecoderConformance,
    FrontierDecoderSuite,
    TeacherForcedDecoderSuite,
)
from tests.unit.nn.decoders.test_rnn import (
    CHOICES,
    FUNCTOR,
    INDEX,
    INPUTS,
    SIGNATURE,
    STORE,
    N,
)

from lofi.nn._values import Encoded
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders.linear import LinearRnnDecoder
from lofi.nn.embeddings import ChoiceEmbedding
from lofi.nn.ports import (
    FrontierDecoder,
    Reading,
    TeacherForcedDecoder,
)
from lofi.nn.sites import ComponentId, Site
from lofi.nn.strategies import score
from lofi.spec.decoders.linear import LinearRnnDecoderSettings
from lofi.spec.layers import BUILTIN_RECURRENT_CELLS, RecurrentCell
from lofi.terms import (
    ChoiceIndex,
    Frontier,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

WIDTH: Final = 6
LAYOUT: Final = PreorderArityLayout(7)
CODEC: Final = TermCodec(INDEX, LAYOUT)
VIEW: Final = NeuralCodecView(CODEC)
TERMS: Final = CODEC.encode_batch((FUNCTOR, N))
COMPONENT: Final = ComponentId("linear-decoder")

CONFORMANCE_SIGNATURE: Final = Signature(
    "linear-decoder-conformance",
    ("term",),
    (
        Operator("a", "term", family="primitive"),
        Operator("b", "term", family="primitive"),
        Operator(
            "pair",
            "term",
            arg_sorts=("term", "term"),
            family="constructor",
        ),
    ),
)
CONFORMANCE_INDEX: Final = OperatorIndex(CONFORMANCE_SIGNATURE)
CONFORMANCE_CHOICES: Final = ChoiceIndex(
    CONFORMANCE_SIGNATURE, depth_bounds={"term": 2}
)
CONFORMANCE_CODEC: Final = TermCodec(CONFORMANCE_INDEX, PreorderArityLayout(5))
CONFORMANCE_VIEW: Final = NeuralCodecView(CONFORMANCE_CODEC)
CONFORMANCE_STORE: Final = TermStore(CONFORMANCE_SIGNATURE)
CONFORMANCE_A: Final = CONFORMANCE_STORE.op(CONFORMANCE_SIGNATURE.operator(1))
CONFORMANCE_B: Final = CONFORMANCE_STORE.op(CONFORMANCE_SIGNATURE.operator(2))
CONFORMANCE_TERMS: Final = CONFORMANCE_CODEC.encode_batch(
    (
        CONFORMANCE_STORE.op(
            CONFORMANCE_SIGNATURE.operator(3), CONFORMANCE_A, CONFORMANCE_B
        ),
        CONFORMANCE_A,
    )
)


def build(
    *,
    cell: RecurrentCell = "gru",
    layers: int = 2,
    heads: Literal["factored", "flat"] = "flat",
    embedding: ChoiceEmbedding | None = None,
) -> LinearRnnDecoder:
    torch.manual_seed(23)
    return LinearRnnDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        input_width=4,
        settings=LinearRnnDecoderSettings(WIDTH, cell, layers, heads, None),
        embedding=embedding,
    ).to(torch.float64)


def build_conformance() -> LinearRnnDecoder:
    """Return a recurrence over the frontier suite's payload-free language."""
    torch.manual_seed(23)
    return LinearRnnDecoder(
        codec=CONFORMANCE_VIEW,
        choices=CONFORMANCE_CHOICES,
        component=COMPONENT,
        input_width=4,
        settings=LinearRnnDecoderSettings(WIDTH, "gru", 2, "flat", None),
        embedding=None,
    ).to(torch.float64)


class TestALinearRnnDecoder(TeacherForcedDecoderSuite, FrontierDecoderSuite):
    @pytest.fixture
    def conformance(self) -> DecoderConformance[LinearRnnDecoder]:
        frontier = Frontier(
            TermStore(CONFORMANCE_SIGNATURE.with_holes()).hole("term"),
            {(): object()},
            0.0,
        )
        return DecoderConformance(
            build=build_conformance,
            inputs=INPUTS,
            terms=CONFORMANCE_TERMS,
            index=CONFORMANCE_CHOICES,
            root=frontier.context(CONFORMANCE_VIEW.capacity),
            root_sort="term",
            slots=CONFORMANCE_VIEW.capacity,
        )


@pytest.mark.parametrize("cell", BUILTIN_RECURRENT_CELLS)
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("heads", ["flat", "factored"])
def test_every_cell_layer_and_head_trains_on_prefix_arity_terms(
    cell: RecurrentCell,
    layers: int,
    heads: Literal["factored", "flat"],
) -> None:
    decoder = build(cell=cell, layers=layers, heads=heads)
    loss = -score(decoder, INPUTS, TERMS, index=CHOICES).tensor.mean()
    loss.backward()

    assert isinstance(decoder, FrontierDecoder)
    assert isinstance(decoder, TeacherForcedDecoder)
    assert bool(loss.isfinite())
    gradients = tuple(
        parameter.grad
        for parameter in decoder.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(bool(gradient.isfinite().all()) for gradient in gradients)
    assert decoder.get_parameter("_root_hidden").grad is not None


def test_linear_decoder_exposes_only_its_capability_protocols() -> None:
    decoder = build()

    assert isinstance(decoder, FrontierDecoder)
    assert isinstance(decoder, TeacherForcedDecoder)
    assert not hasattr(decoder, "score")
    assert not hasattr(decoder, "distribution")
    assert not hasattr(decoder, "draw")


def test_frontier_and_teacher_forcing_share_the_root_choice_row() -> None:
    decoder = build()
    teacher = decoder.rows(INPUTS, terms=TERMS).tensor[0, 0]
    frontier_store = TermStore(SIGNATURE.with_holes())
    frontier = Frontier(
        frontier_store.hole("Cat"),
        {(): object()},
        0.0,
    )
    context = frontier.context(VIEW.capacity)
    state = decoder.start(Encoded(INPUTS.tensor[:1], INPUTS.axes))

    torch.testing.assert_close(
        decoder.weigh(state, context=context).tensor, teacher, rtol=1e-12, atol=1e-12
    )


def test_learned_root_states_exist_for_every_declared_layer() -> None:
    decoder = build(cell="lstm", layers=3)

    assert decoder.get_parameter("_root_hidden").shape == (3, WIDTH)
    assert decoder.get_parameter("_root_memory").shape == (3, WIDTH)
    state = decoder.start(type(INPUTS)(INPUTS.tensor[:1], INPUTS.axes))
    assert state.hidden.shape == state.memory.shape == (3, 1, WIDTH)


@pytest.mark.parametrize("cell", ["elman-tanh", "elman-relu", "gru"])
def test_a_cell_without_separate_memory_learns_no_root_memory(
    cell: RecurrentCell,
) -> None:
    decoder = build(cell=cell)
    names = {name for name, _ in decoder.named_parameters()}

    assert decoder.cell == cell
    assert "_root_memory" not in names
    assert not any(name.startswith("_memory_projection") for name in names)


def test_a_declared_tie_uses_the_exact_published_choice_embedding() -> None:
    embedding = ChoiceEmbedding(index=CHOICES, width=WIDTH, component=COMPONENT)
    decoder = build(embedding=embedding)

    assert decoder.get_submodule("_embedding") is embedding


def test_teacher_feedback_changes_only_rows_after_the_changed_choice() -> None:
    decoder = build()
    first = CODEC.encode_batch((FUNCTOR,))
    swapped = CODEC.encode_batch(
        (STORE.op(FUNCTOR.op, FUNCTOR.args[1], FUNCTOR.args[0]),)
    )
    one = Encoded(INPUTS.tensor[:1], INPUTS.axes)
    before = decoder.rows(one, terms=first).tensor
    after = decoder.rows(one, terms=swapped).tensor

    assert torch.equal(before[:, 0], after[:, 0])
    assert bool(before[:, 1:].ne(after[:, 1:]).any())


def observations(subject: LinearRnnDecoder) -> Mapping[Site, Reading]:
    """Return every value the decoder publishes during teacher forcing."""
    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        subject.rows(INPUTS, terms=CONFORMANCE_TERMS)

    return scope.readings


def test_teacher_forcing_scores_only_through_the_shared_strategy() -> None:
    decoder = build_conformance()
    rows = decoder.rows(INPUTS, terms=CONFORMANCE_TERMS)
    choices = torch.tensor(
        tuple(CONFORMANCE_CHOICES.codes(row) for row in CONFORMANCE_TERMS.rows)
    )
    occupied = torch.tensor(CONFORMANCE_TERMS.mask)
    selected = rows.tensor.gather(2, choices.unsqueeze(-1)).squeeze(-1)
    expected = torch.where(occupied, selected, selected.new_zeros(())).sum(dim=1)

    assert torch.equal(
        score(
            decoder,
            INPUTS,
            CONFORMANCE_TERMS,
            index=CONFORMANCE_CHOICES,
        ).tensor,
        expected,
    )
