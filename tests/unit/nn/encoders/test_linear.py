from typing import TYPE_CHECKING, Final, Literal

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.term_encoder import TermEncoderConformance, TermEncoderSuite
from tests.unit.nn.encoders.test_rnn import (
    BINARY,
    CHOICES,
    COMPONENT,
    INDEX,
    SIGNATURE,
    STORE,
    A,
    B,
)

from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders.linear import LinearRnnEncoder
from lofi.nn.ports import Reading, TermEncoder
from lofi.nn.sites import Role, Site
from lofi.spec.encoders.linear import LinearRnnEncoderSettings
from lofi.spec.layers import BUILTIN_RECURRENT_CELLS, RecurrentCell
from lofi.terms import PreorderArityLayout, TermCodec

if TYPE_CHECKING:
    from collections.abc import Mapping

WIDTH: Final = 8
CODEC: Final = TermCodec(INDEX, PreorderArityLayout(7))
VIEW: Final = NeuralCodecView(CODEC)
LEFT: Final = STORE.op(
    BINARY, STORE.op(SIGNATURE.operator(A)), STORE.op(SIGNATURE.operator(B))
)
RIGHT: Final = STORE.op(
    BINARY, STORE.op(SIGNATURE.operator(B)), STORE.op(SIGNATURE.operator(A))
)
TERMS: Final = CODEC.encode_batch((LEFT, RIGHT))


def build(
    *,
    cell: RecurrentCell = "gru",
    layers: int = 2,
    direction: Literal["forward", "bidirectional"] = "forward",
    variables: Literal["index", "level"] = "index",
    merge: Literal["mean", "sum", "concatenation"] | None = None,
) -> LinearRnnEncoder:
    torch.manual_seed(19)
    return LinearRnnEncoder(
        codec=VIEW,
        choices=CHOICES,
        component=COMPONENT,
        settings=LinearRnnEncoderSettings(
            WIDTH,
            cell,
            layers,
            direction,
            variables,
            "mean" if direction == "bidirectional" and merge is None else merge,
        ),
    ).to(torch.float64)


@pytest.mark.parametrize("cell", BUILTIN_RECURRENT_CELLS)
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_every_cell_layer_and_direction_encodes_prefix_arity_terms(
    cell: RecurrentCell,
    layers: int,
    direction: Literal["forward", "bidirectional"],
) -> None:
    subject = build(cell=cell, layers=layers, direction=direction)
    encoded = subject.encode(TERMS)
    nodes = subject.encode_nodes(TERMS)

    assert isinstance(subject, TermEncoder)
    assert encoded.axes == ("term", "width")
    assert encoded.tensor.shape == (2, WIDTH)
    assert nodes.axes == ("term", "slot", "width")
    assert nodes.tensor.shape == (2, VIEW.capacity, WIDTH)
    assert bool(encoded.tensor.isfinite().all())
    assert bool(nodes.tensor.isfinite().all())


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
def test_constructor_identity_and_argument_order_change_the_encoding(
    direction: Literal["forward", "bidirectional"],
) -> None:
    encoded = build(direction=direction).encode(TERMS).tensor

    assert not torch.allclose(encoded[0], encoded[1])


def test_the_declared_merge_combines_the_two_directions() -> None:
    mean = build(direction="bidirectional", merge="mean")
    summed = build(direction="bidirectional", merge="sum")
    joined = build(direction="bidirectional", merge="concatenation")

    assert (mean.output_width, summed.output_width) == (WIDTH, WIDTH)
    assert joined.output_width == 2 * WIDTH
    assert joined.encode(TERMS).tensor.shape == (2, 2 * WIDTH)
    assert joined.encode_nodes(TERMS).tensor.shape == (2, VIEW.capacity, 2 * WIDTH)
    torch.testing.assert_close(
        summed.encode(TERMS).tensor, 2.0 * mean.encode(TERMS).tensor
    )
    forward, backward = joined.encode(TERMS).tensor.split(WIDTH, dim=-1)
    torch.testing.assert_close(forward + backward, summed.encode(TERMS).tensor)


def test_the_encoder_publishes_and_observes_its_tied_choice_table() -> None:
    subject = build()
    sites = frozenset(subject.site_map())

    with subject.instrument(sites, patches={}) as scope:
        subject.encode(TERMS)

    assert "choice-embedding" in subject.shared
    assert {site.role for site in scope.readings} >= {
        Role.PARAMETER,
        Role.NODE_ENCODING,
        Role.TERM_ENCODING,
        Role.RECURRENT_HIDDEN,
    }


def test_level_and_index_are_both_explicit_current_variable_treatments() -> None:
    assert build(variables="index").variables == "index"
    assert build(variables="level").variables == "level"


def observations(subject: LinearRnnEncoder) -> Mapping[Site, Reading]:
    """Return every value the encoder publishes during one pass."""
    with subject.instrument(frozenset(subject.site_map()), patches={}) as scope:
        subject.encode(TERMS)

    return scope.readings


def conformance() -> TermEncoderConformance[LinearRnnEncoder]:
    """Return the shared term-encoder laws over the linear recurrence."""
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


class TestLinearRnnEncoder(TermEncoderSuite):
    """Apply every term-encoder law to the linear recurrence."""

    @pytest.fixture
    def conformance(self) -> TermEncoderConformance[LinearRnnEncoder]:
        """Return one linear-recurrence conformance fixture."""
        return conformance()
