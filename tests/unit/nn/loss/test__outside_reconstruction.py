from typing import TYPE_CHECKING, Final, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.objective import ObjectiveLawSuite

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._outside_reconstruction import OutsideReconstructionObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.spec.losses.outside_reconstruction import OutsideReconstructionLossSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import (
        LossComponentBindings,
        LossTerm,
        ProofRewardBindings,
        SpanTokenDecoder,
    )
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec

type Outputs = ObjectiveOutputs[Words]

RESERVED: Final = ReservedTokens({"<pad>": 0, "<eos>": 1, "<unk>": 2})
VOCABULARY: Final = TokenVocabulary(("a", "b", "c"), reserved=RESERVED, unknown="<unk>")
END: Final = "<eos>"
END_ID: Final = RESERVED.index_of(END)
WIDTH: Final = VOCABULARY.size
SPANS: Final = SpanIndex(3)
ITEMS: Final = (("a", "b", "c"), ("c", "a"))
PADDING: Final = -7.0


class Words:
    def __init__(self, items: tuple[tuple[str, ...], ...]) -> None:
        self._items = items

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        return self._items


class TableDecoder:
    def __init__(
        self, vocabulary: TokenVocabulary = VOCABULARY, *, end: str = END
    ) -> None:
        self._vocabulary = vocabulary
        self._end = end
        self.prefixes: list[Encoded] = []
        self.targets: list[Tensor] = []

    @property
    def vocabulary(self) -> TokenVocabulary:
        return self._vocabulary

    @property
    def end(self) -> str:
        return self._end

    def score_tokens(self, prefix: Encoded, /, *, targets: Tensor) -> LogDistribution:
        self.prefixes.append(prefix)
        self.targets.append(targets.detach().clone())
        logits = prefix.tensor.unsqueeze(1).expand(-1, targets.shape[1], -1)

        return LogDistribution(
            logits.log_softmax(dim=2),
            ("span", "position", "token"),
            normalized_over="token",
        )


def objective(
    decoder: TableDecoder | None = None,
) -> OutsideReconstructionObjective[Words]:
    return OutsideReconstructionObjective(
        decoder if decoder is not None else TableDecoder(),
        name="outside-reconstruction",
    )


def live_spans() -> Tensor:
    mask = torch.zeros(len(ITEMS), SPANS.num_spans, dtype=torch.bool)

    for item, words in enumerate(ITEMS):
        for start, end in SPANS.spans_by_length():
            if end <= len(words):
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def outside_tensor() -> Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        len(ITEMS) * SPANS.num_spans * WIDTH,
        dtype=torch.float64,
    ).reshape(len(ITEMS), SPANS.num_spans, WIDTH)

    return torch.where(LIVE.unsqueeze(-1), values, PADDING).detach().requires_grad_()


def chart() -> CellChart:
    inside = torch.where(LIVE, -1.0, PADDING).to(torch.float64)

    return CellChart(LogDistribution(inside, SPAN_AXES, LIVE))


def bundle() -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        outside_cells=Encoded(outside_tensor(), (*SPAN_AXES, "width"), LIVE),
    )


def filled(values: Tensor, items: tuple[tuple[str, ...], ...] = ITEMS) -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(items),
        outside_cells=Encoded(values, (*SPAN_AXES, "width"), LIVE),
    )


def aligned(sign: float) -> Outputs:
    values = torch.zeros(len(ITEMS), SPANS.num_spans, WIDTH, dtype=torch.float64)

    for item, words in enumerate(ITEMS):
        for position, word in enumerate(words):
            span = SPANS.span_id(position, position + 1)
            values[item, span, VOCABULARY.index_of(word)] = sign * 4.0
            values[item, span, END_ID] = sign * 4.0

    padded = torch.where(LIVE.unsqueeze(-1), values, PADDING)

    return filled(padded.detach().requires_grad_())


def empty_bundle() -> Outputs:
    mask = torch.zeros_like(LIVE)
    values = torch.full(
        (len(ITEMS), SPANS.num_spans, WIDTH),
        PADDING,
        dtype=torch.float64,
        requires_grad=True,
    )

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        outside_cells=Encoded(values, (*SPAN_AXES, "width"), mask),
    )


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    outside = outputs.outside_cells

    assert outside is not None

    return (outside.tensor,)


def reckoning(outputs: Outputs) -> float:
    outside = outputs.outside_cells

    assert outside is not None
    assert outside.mask is not None

    losses: list[float] = []

    for item, words in enumerate(outputs.batch.words):
        for position, word in enumerate(words):
            span = SPANS.span_id(position, position + 1)

            if not bool(outside.mask[item, span]):
                continue

            logits = outside.tensor[item, span].detach().log_softmax(dim=0)
            losses.append(
                float(-(logits[VOCABULARY.index_of(word)] + logits[END_ID]) / 2.0)
            )

    return sum(losses) / len(losses) if losses else 0.0


def permuted(outputs: Outputs) -> Outputs:
    outside = outputs.outside_cells
    order = torch.tensor([1, 0])

    assert outside is not None
    assert outside.mask is not None

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(tuple(reversed(outputs.batch.words))),
        outside_cells=Encoded(
            outside.tensor[order].detach().requires_grad_(),
            outside.axes,
            outside.mask[order],
        ),
    )


def refilled(outputs: Outputs) -> Outputs:
    outside = outputs.outside_cells

    assert outside is not None
    assert outside.mask is not None

    values = torch.where(
        outside.mask.unsqueeze(-1), outside.tensor.detach(), PADDING / 3.0
    )

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=outputs.batch,
        outside_cells=Encoded(values.requires_grad_(), outside.axes, outside.mask),
    )


def models_built(call: Callable[[], LossTerm]) -> int:
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1
        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


def fixture() -> ObjectiveFixture[Words]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=reckoning,
        monotone=lambda: (aligned(-1.0), aligned(1.0)),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
    )


class TestOutsideReconstructionObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Words]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        the_value_is_monotone_where_the_fixture_says(conformance)


class Bindings:
    def __init__(self, decoder: SpanTokenDecoder) -> None:
        self._decoder = decoder
        self.requests: list[str] = []

    @property
    def name(self) -> str:
        return "outside-reconstruction"

    @property
    def readouts(self) -> dict[str, TermsReadout | Space]:
        return {}

    @property
    def spaces(self) -> dict[str, TermsReadout | Space]:
        return {}

    def terms_readout(self, role: str, /) -> TermsReadout:
        del role
        raise AssertionError

    def space(self, role: str, /) -> Space:
        del role
        raise AssertionError

    def readout_space(self, role: str, /) -> Space:
        del role
        raise AssertionError

    def relation(self, role: str, /) -> SpaceRelationBinding:
        del role
        raise AssertionError

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        self.requests.append(role)

        if role != "reconstruction":
            raise ValueError(role)

        return self._decoder

    def proof_reward(self) -> ProofRewardBindings:
        raise AssertionError


class TestConfiguredOutsideReconstruction(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        decoder = TableDecoder()
        context = Bindings(cast("SpanTokenDecoder", decoder))

        def observe(built: object) -> dict[str, bool]:
            held = cast("OutsideReconstructionObjective[Words]", built)

            return {
                "named for its term": held.name == "outside-reconstruction",
                "reads the declared decoder": context.requests == ["reconstruction"],
                "holds the decoder vocabulary": held.vocabulary is VOCABULARY,
            }

        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                OutsideReconstructionLossSpec(decoder="reconstruction"),
            ),
            context=cast("LossComponentBindings", context),
            expected_type=OutsideReconstructionObjective,
            observe=observe,
            unchanged=lambda: not decoder.prefixes,
        )


def test_the_objective_reads_only_contextual_outside_vectors() -> None:
    built = objective()

    assert built.requires == frozenset({"spans", "outside_cells", "batch"})


def test_missing_outside_vectors_are_refused_by_name() -> None:
    outputs = ObjectiveOutputs(spans=SPANS, chart=chart(), batch=Words(ITEMS))

    with pytest.raises(MissingOutputError) as raised:
        objective().evaluate(outputs)

    assert raised.value.missing == frozenset({"outside_cells"})


def test_only_live_leaf_vectors_receive_gradient() -> None:
    outputs = bundle()
    outside = outputs.outside_cells

    assert outside is not None

    objective().evaluate(outputs).value.backward()
    gradient = outside.tensor.grad

    assert gradient is not None

    for item, words in enumerate(ITEMS):
        for start, end in SPANS.spans_by_length():
            span = SPANS.span_id(start, end)
            expected = end - start == 1 and end <= len(words)

            assert bool((gradient[item, span] != 0.0).any()) is expected


def test_the_decoder_receives_every_live_word_in_one_batched_call() -> None:
    decoder = TableDecoder()
    term = objective(decoder).evaluate(bundle())

    assert term.count == sum(map(len, ITEMS))
    assert len(decoder.prefixes) == 1
    assert decoder.prefixes[0].tensor.shape == (term.count, WIDTH)
    assert decoder.targets[0].shape == (term.count, 2)
    assert bool((decoder.targets[0][:, 1] == END_ID).all())


def test_the_end_row_cannot_also_be_the_unknown_row() -> None:
    reserved = ReservedTokens({"<unk>": 0})
    vocabulary = TokenVocabulary(("a",), reserved=reserved, unknown="<unk>")

    with pytest.raises(ValueError, match="unknown"):
        OutsideReconstructionObjective(
            cast("SpanTokenDecoder", TableDecoder(vocabulary, end="<unk>")),
            name="outside-reconstruction",
        )


def test_the_name_end_vocabulary_and_repr_are_public() -> None:
    built = objective()

    assert built.name == "outside-reconstruction"
    assert built.end == END
    assert built.vocabulary is VOCABULARY
    assert "outside-reconstruction" in repr(built)
    assert repr(END) in repr(built)
