from itertools import product
from typing import Final, cast

import pytest
import torch
from tests.conformance.component_plugin import plugin_context

from lofi._types import ComponentId
from lofi.nn._values import Encoded
from lofi.nn.build import build_component
from lofi.nn.loss._ports import SpanTokenDecoder
from lofi.nn.policy import DevicePolicy
from lofi.nn.tokens.decoder import RecurrentSpanDecoder
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.spec.layers import (
    BUILTIN_RECURRENT_CELLS,
    Activation,
    FeedForwardLayer,
    FeedForwardLayerSpec,
    RecurrentCell,
)
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec

VOCABULARY: Final = TokenVocabulary(
    ("a", "b", "c"),
    reserved=ReservedTokens({"<pad>": 0, "<unk>": 1}),
    unknown="<unk>",
)
WIDTH: Final = 3
COMPONENT: Final = ComponentId("reconstruction")
SPAN_END: Final = "<stop>"


def decoder(
    cell: RecurrentCell = "gru",
    layers: int = 1,
    head: tuple[FeedForwardLayer, ...] = (),
    *,
    state_width: int | None = None,
) -> RecurrentSpanDecoder:
    torch.manual_seed(0)

    return RecurrentSpanDecoder(
        VOCABULARY,
        width=WIDTH,
        cell=cell,
        layers=layers,
        state_width=state_width,
        head=head,
        end=SPAN_END,
        component=COMPONENT,
    )


def prefix(spans: int = 2) -> Encoded:
    torch.manual_seed(1)

    return Encoded(torch.randn(spans, WIDTH, requires_grad=True), ("span", "width"))


def test_the_vocabulary_reserves_the_end_token_after_every_run_row() -> None:
    vocabulary = decoder().vocabulary

    assert vocabulary.size == VOCABULARY.size + 1
    assert vocabulary.reserved.index_of(SPAN_END) == VOCABULARY.size
    assert vocabulary.unknown_index == VOCABULARY.unknown_index
    assert all(
        vocabulary.index_of(token) == VOCABULARY.index_of(token)
        for token in (*VOCABULARY.tokens, "<pad>", "<unk>", "unseen")
    )


def test_each_position_is_a_distribution_over_every_vocabulary_row() -> None:
    built = decoder()
    targets = torch.tensor([[2, 3, 5], [4, 5, 0]])

    scored = built.score_tokens(prefix(), targets=targets)

    assert isinstance(built, SpanTokenDecoder)
    assert scored.axes == ("span", "position", "token")
    assert scored.normalized_over == "token"
    assert tuple(scored.tensor.shape) == (2, 3, built.vocabulary.size)
    assert torch.allclose(scored.tensor.exp().sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_a_position_reads_only_the_span_vector_and_the_words_before_it() -> None:
    built = decoder()
    vectors = prefix()
    first = built.score_tokens(vectors, targets=torch.tensor([[2, 3, 5], [4, 2, 5]]))
    changed = built.score_tokens(vectors, targets=torch.tensor([[2, 4, 3], [4, 3, 2]]))

    assert torch.equal(first.tensor[:, :2], changed.tensor[:, :2])
    assert not torch.equal(first.tensor[:, 2], changed.tensor[:, 2])


ARCHITECTURES: Final = list(
    product(
        BUILTIN_RECURRENT_CELLS,
        (1, 2),
        ((), (FeedForwardLayer(4, "relu", 0.0), FeedForwardLayer(3, "gelu", 0.1))),
    )
)


@pytest.mark.parametrize(("cell", "layers", "head"), ARCHITECTURES)
def test_every_declared_architecture_reads_only_earlier_words(
    cell: RecurrentCell, layers: int, head: tuple[FeedForwardLayer, ...]
) -> None:
    built = decoder(cell, layers, head)
    built.eval()
    vectors = prefix()
    first = built.score_tokens(vectors, targets=torch.tensor([[2, 3, 5], [4, 2, 5]]))
    changed = built.score_tokens(vectors, targets=torch.tensor([[2, 4, 3], [4, 3, 2]]))

    assert torch.equal(first.tensor[:, :2], changed.tensor[:, :2])
    assert not torch.equal(first.tensor[:, 2], changed.tensor[:, 2])
    assert torch.allclose(first.tensor.exp().sum(dim=-1), torch.ones(2, 3), atol=1e-6)


GATES: Final = {
    "gru": 3,
    "lstm": 4,
    "elman-tanh": 1,
    "elman-relu": 1,
}


@pytest.mark.parametrize(("cell", "layers", "head"), ARCHITECTURES)
def test_the_recurrence_and_head_are_the_declared_ones(
    cell: RecurrentCell, layers: int, head: tuple[FeedForwardLayer, ...]
) -> None:
    state = decoder(cell, layers, head, state_width=5).state_dict()
    rows = VOCABULARY.size + 1
    reads = [5, *(layer.width for layer in head)]
    head_weights = [
        tuple(value.shape)
        for name, value in state.items()
        if name.startswith("_score.") and name.endswith(".weight")
    ]

    assert state["_recur.weight_ih_l0"].shape == (GATES[cell] * 5, WIDTH)
    assert state["_recur.weight_hh_l0"].shape == (GATES[cell] * 5, 5)
    assert f"_recur.weight_ih_l{layers - 1}" in state
    assert f"_recur.weight_ih_l{layers}" not in state
    assert "_recur.weight_ih_l0_reverse" not in state
    assert head_weights == [
        *zip(reads[1:], reads[:-1], strict=True),
        (rows, reads[-1]),
    ]


def test_each_hidden_layer_has_its_own_declared_width() -> None:
    state = decoder(
        head=(FeedForwardLayer(7, "tanh", 0.0), FeedForwardLayer(5, "relu", 0.1))
    ).state_dict()

    assert state["_recur.weight_hh_l0"].shape == (3 * WIDTH, WIDTH)
    assert state["_score.0.weight"].shape == (7, WIDTH)
    assert state["_score.3.weight"].shape == (5, 7)
    assert state["_score.6.weight"].shape == (VOCABULARY.size + 1, 5)


def test_the_two_ungated_cells_differ_only_in_their_nonlinearity() -> None:
    squashing = decoder("elman-tanh")
    rectifying = decoder("elman-relu")
    rectifying.load_state_dict(squashing.state_dict())
    targets = torch.tensor([[2, 3, 5], [4, 5, 0]])

    assert not torch.allclose(
        squashing.score_tokens(prefix(), targets=targets).tensor,
        rectifying.score_tokens(prefix(), targets=targets).tensor,
    )


@pytest.mark.parametrize(
    ("kind", "module"),
    [
        ("tanh", torch.nn.Tanh),
        ("relu", torch.nn.ReLU),
        ("gelu", torch.nn.GELU),
        ("silu", torch.nn.SiLU),
    ],
)
def test_each_hidden_head_layer_is_followed_by_its_declared_nonlinearity(
    kind: Activation, module: type[torch.nn.Module]
) -> None:
    built = decoder(
        head=(FeedForwardLayer(4, "tanh", 0.0), FeedForwardLayer(3, kind, 0.0))
    )

    assert isinstance(built.get_submodule("_score.1"), torch.nn.Tanh)
    assert isinstance(built.get_submodule("_score.4"), module)


def test_the_span_vector_and_every_parameter_take_a_gradient() -> None:
    built = decoder("lstm", 2, (FeedForwardLayer(4, "tanh", 0.1),), state_width=5)
    vectors = prefix()
    targets = torch.tensor([[2, 3, 5], [4, 5, 0]])

    scored = built.score_tokens(vectors, targets=targets)
    scored.tensor.gather(2, targets.unsqueeze(2)).sum().backward()

    assert vectors.tensor.grad is not None
    assert bool(vectors.tensor.grad.abs().sum() > 0.0)
    assert all(parameter.grad is not None for parameter in built.parameters())


def test_the_decoder_names_the_end_token_its_vocabulary_reserves() -> None:
    built = decoder()

    assert built.end == SPAN_END
    assert built.vocabulary.reserved.holds(built.end)


def test_a_vocabulary_already_holding_the_end_token_is_refused() -> None:
    held = TokenVocabulary(
        ("a",),
        reserved=ReservedTokens({"<unk>": 0, SPAN_END: 1}),
        unknown="<unk>",
    )

    with pytest.raises(ValueError, match="already holds the end token"):
        RecurrentSpanDecoder(
            held,
            width=WIDTH,
            cell="gru",
            layers=1,
            state_width=None,
            head=(),
            end=SPAN_END,
            component=COMPONENT,
        )


def test_vectors_over_other_axes_or_widths_and_unmatched_targets_are_refused() -> None:
    built = decoder()
    targets = torch.tensor([[2, 5], [3, 5]])

    with pytest.raises(ValueError, match="reads vectors over"):
        built.score_tokens(
            Encoded(torch.zeros(2, WIDTH), ("term", "width")), targets=targets
        )

    with pytest.raises(ValueError, match="reads vectors over"):
        built.score_tokens(
            Encoded(torch.zeros(2, WIDTH + 1), ("span", "width")), targets=targets
        )

    with pytest.raises(ValueError, match="do not give positions"):
        built.score_tokens(prefix(3), targets=targets)


class Cells:
    """A producer of chart cells at the test width."""

    output_width = WIDTH


def test_the_plugin_places_a_decoder_reading_spans_at_the_encoders_width() -> None:
    policy = DevicePolicy(torch.device("cpu"), torch.float64)
    built = cast(
        "RecurrentSpanDecoder",
        build_component(
            RnnSpanDecoderSpec(),
            plugin_context(
                component=COMPONENT,
                producers={"cells": Cells()},
                policy=policy,
                vocabulary=VOCABULARY,
            ),
        ),
    )

    assert built.input_width == WIDTH
    assert built.vocabulary.size == VOCABULARY.size + 1
    assert all(parameter.dtype == torch.float64 for parameter in built.parameters())
    assert COMPONENT in repr(built)
    assert isinstance(built.get_submodule("_recur"), torch.nn.GRU)
    assert set(built.get_submodule("_score").state_dict()) == {"0.weight", "0.bias"}


def test_the_plugin_builds_the_declared_recurrence_and_head() -> None:
    declared = RnnSpanDecoderSpec(
        cell="elman-relu",
        layers=3,
        state_width=7,
        head=(FeedForwardLayerSpec(width=4, activation="gelu", dropout=0.1),),
    )
    built = cast(
        "RecurrentSpanDecoder",
        build_component(
            declared,
            plugin_context(
                component=COMPONENT,
                producers={"cells": Cells()},
                vocabulary=VOCABULARY,
            ),
        ),
    )
    recurrence = built.get_submodule("_recur")

    assert isinstance(recurrence, torch.nn.RNN)
    assert recurrence.nonlinearity == "relu"
    assert recurrence.num_layers == 3
    assert recurrence.hidden_size == 7
    assert isinstance(built.get_submodule("_score.1"), torch.nn.GELU)
    assert built.state_dict()["_score.0.weight"].shape == (4, 7)
