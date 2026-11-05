import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.loss._autoencoder import TypeAutoencoderObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.sites import ComponentId
from lofi.nn.strategies import score
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import LossTerm
    from lofi.nn.ports import TeacherForcedDecoder, TermEncoder
    from lofi.terms import Term


SIGNATURE: Final = Signature(
    "autoencoded",
    ("T",),
    (
        Operator("A", "T", family="atom"),
        Operator("B", "T", family="atom"),
        Operator("f", "T", arg_sorts=("T", "T"), family="functor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CODEC: Final = TermCodec(
    INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3)
)
STORE: Final = TermStore(SIGNATURE)
CAPACITY: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
CHOICES: Final = ChoiceIndex(
    SIGNATURE, depth_bounds=dict.fromkeys(SIGNATURE.sorts, CODEC.capacity)
)
A: Final = STORE.op(SIGNATURE.by_name("A"))
B: Final = STORE.op(SIGNATURE.by_name("B"))
FAVORED: Final = 3.0


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


class Types(NamedTuple):
    terms: BatchEncoding


class IndicatorEncoder:
    """Encodes a type as its slot-by-symbol indicators, each scaled by a weight."""

    def __init__(self) -> None:
        weight = torch.ones(CAPACITY, SYMBOLS, dtype=torch.float64)
        weight[:, INDEX.index(SIGNATURE.by_name("A"))] = FAVORED
        self.weight = weight.requires_grad_()

    @property
    def output_width(self) -> int:
        return CAPACITY * SYMBOLS

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        indicators = torch.nn.functional.one_hot(symbols, SYMBOLS).to(torch.float64)
        vectors = (indicators * self.weight).reshape(
            symbols.shape[0], CAPACITY * SYMBOLS
        )
        live = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        held = live.any(dim=1)

        return Encoded(vectors, ("term", "width"), None if bool(held.all()) else held)


class ReadingDecoder:
    """Weighs a type by reading each slot's logits off its encoding, at a gain."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = torch.tensor(gain, dtype=torch.float64).requires_grad_()

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        rows = encoded.tensor.shape[0]
        logits = encoded.tensor.reshape(rows, CAPACITY, SYMBOLS) * self.gain
        del terms
        mask = encoded.mask

        if mask is not None:
            logits = logits.masked_fill(~mask.reshape(-1, 1, 1), -torch.inf)

        return LogDistribution(
            logits.log_softmax(dim=-1),
            ("term", "slot", "choice"),
            mask,
            normalized_over="choice",
        )


class WideDecoder(ReadingDecoder):
    @override
    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        scored = super().rows(encoded, terms=terms)

        return LogDistribution(
            scored.tensor.unsqueeze(-1), ("term", "slot", "choice", "wide")
        )


ENCODER: Final = IndicatorEncoder()
DECODER: Final = ReadingDecoder()


def objective(
    encoder: IndicatorEncoder = ENCODER,
    decoder: ReadingDecoder = DECODER,
) -> TypeAutoencoderObjective[Types]:
    return TypeAutoencoderObjective(
        cast("TermEncoder", encoder),
        cast("TeacherForcedDecoder", decoder),
        CHOICES,
    )


def dead_row() -> Encoding:
    """Return the codec's empty row, which holds no term."""
    skeleton = CODEC.skeleton

    if skeleton is None:
        msg = "the fixture codec has no fixed empty layout"
        raise ValueError(msg)

    return skeleton


def bundle_of(rows: tuple[Encoding, ...]) -> ObjectiveOutputs[Types]:
    return ObjectiveOutputs(
        spans=SpanIndex(1),
        chart=CellChart(
            LogDistribution(torch.zeros(len(rows), 1, dtype=torch.float64), SPAN_AXES)
        ),
        batch=Types(BatchEncoding(rows)),
    )


def bundle_over(*terms: Term) -> ObjectiveOutputs[Types]:
    return bundle_of(CODEC.encode_batch(terms).rows)


def bundle() -> ObjectiveOutputs[Types]:
    return bundle_over(A, f(A, B), f(f(A, B), B))


def permuted(outputs: ObjectiveOutputs[Types]) -> ObjectiveOutputs[Types]:
    rows = outputs.batch.terms.rows

    return bundle_of((rows[2], rows[0], rows[1]))


def refilled(outputs: ObjectiveOutputs[Types]) -> ObjectiveOutputs[Types]:
    return bundle_of((*outputs.batch.terms.rows, dead_row()))


def empty_bundle() -> ObjectiveOutputs[Types]:
    return bundle_of((dead_row(), dead_row()))


def inputs_of(outputs: ObjectiveOutputs[Types]) -> tuple[Tensor, ...]:
    del outputs

    return (ENCODER.weight, DECODER.gain)


def slot_nll(slot: int, symbol: int, *, gain: float = 1.0) -> float:
    """Return the negative log probability of one symbol at one slot, by hand."""
    logits = [
        gain * (ENCODER.weight[slot, symbol].item() if other == symbol else 0.0)
        for other in range(SYMBOLS)
    ]

    return math.log(sum(math.exp(logit) for logit in logits)) - logits[symbol]


def by_hand(outputs: ObjectiveOutputs[Types]) -> float:
    rows = [row for row in outputs.batch.terms.rows if any(row.occupied)]
    total = sum(
        slot_nll(slot, symbol)
        for row in rows
        for slot, (symbol, occupied) in enumerate(
            zip(row.symbols, row.occupied, strict=True)
        )
        if occupied
    )

    return total / len(rows) if rows else 0.0


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


def fixture() -> ObjectiveFixture[Types]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (bundle_over(B, f(B, B)), bundle_over(A, f(A, A))),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
        tolerance=1e-9,
    )


class TestTypeAutoencoderObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Types]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that types read back worse score above types read back well."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_value_is_the_mean_over_types_of_the_slot_summed_nll() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.count == 3
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.diagnostics["per_slot"].item() == pytest.approx(
        by_hand(outputs) / CAPACITY, rel=1e-12
    )
    assert 0.0 <= term.diagnostics["exact_match"].item() <= 1.0
    assert term.diagnostics["exact_match"].grad_fn is None
    assert term.diagnostics["per_slot"].grad_fn is None


def shipped() -> tuple[TreeRnnEncoder, RecursiveTermDecoder]:
    """Return the tree recurrence encoder and decoder over the fixture codec."""
    view = NeuralCodecView(CODEC)
    torch.manual_seed(3)
    encoder = TreeRnnEncoder(
        codec=view,
        choices=CHOICES,
        component=ComponentId("tree-encoder"),
        scheme=ArityHeadScheme(width=8, reached=SIGNATURE.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    ).to(torch.float64)
    decoder = RecursiveTermDecoder(
        codec=view,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("tree-decoder"),
        settings=TreeDecoderSettings(encoder.output_width, 6),
        choice_heads="flat",
    ).to(torch.float64)

    return encoder, decoder


def test_the_value_over_the_shipped_recurrences_is_the_mean_of_their_weights() -> None:
    encoder, decoder = shipped()
    built = TypeAutoencoderObjective(encoder, decoder, CHOICES)
    outputs = bundle()
    terms = outputs.batch.terms

    term = built.evaluate(outputs)
    padded = built.evaluate(refilled(outputs))
    weights = score(decoder, encoder.encode(terms), terms, index=CHOICES)

    assert term.count == len(terms.rows)
    assert term.value.item() == pytest.approx(-weights.tensor.mean().item(), rel=1e-12)
    assert padded.count == term.count
    assert padded.value.item() == pytest.approx(term.value.item(), rel=1e-12)


def test_the_value_over_the_shipped_recurrences_reaches_both_parameter_sets() -> None:
    encoder, decoder = shipped()

    TypeAutoencoderObjective(encoder, decoder, CHOICES).evaluate(
        refilled(bundle())
    ).value.backward()

    for module in (encoder, decoder):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]

        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert any(bool((gradient != 0.0).any()) for gradient in gradients)


def test_a_decoder_more_confident_in_the_truth_scores_lower() -> None:
    outputs = bundle()

    unsure = objective(decoder=ReadingDecoder(0.5)).evaluate(outputs).value.item()
    sure = objective(decoder=ReadingDecoder(2.0)).evaluate(outputs).value.item()

    assert sure < unsure


def test_the_value_reaches_the_encoder_and_the_decoder() -> None:
    encoder = IndicatorEncoder()
    decoder = ReadingDecoder()

    objective(encoder, decoder).evaluate(bundle()).value.backward()

    assert encoder.weight.grad is not None
    assert decoder.gain.grad is not None
    assert bool(encoder.weight.grad.abs().sum() > 0.0)
    assert decoder.gain.grad.item() != 0.0


def test_a_row_holding_no_type_is_neither_scored_nor_counted() -> None:
    rows = CODEC.encode_batch((A, f(A, B))).rows

    complete = objective().evaluate(bundle_of(rows))
    padded = objective().evaluate(bundle_of((rows[0], dead_row(), rows[1])))

    assert padded.count == 2
    assert padded.value.item() == pytest.approx(complete.value.item(), rel=1e-12)


def test_a_batch_holding_no_type_is_inapplicable() -> None:
    term = objective().evaluate(empty_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


def test_a_decoder_returning_nonstandard_teacher_axes_is_refused() -> None:
    with pytest.raises(ValueError, match="teacher-forced rows run over"):
        objective(decoder=WideDecoder()).evaluate(bundle())


def test_the_objective_declares_what_it_reads() -> None:
    built = objective()

    assert built.requires == frozenset({"batch"})
    assert built.name == "type-autoencoder"
    assert "IndicatorEncoder" in repr(built)
    assert "ReadingDecoder" in repr(built)


def test_a_blank_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        TypeAutoencoderObjective(
            cast("TermEncoder", ENCODER),
            cast("TeacherForcedDecoder", DECODER),
            CHOICES,
            name="",
        )


def test_the_term_reaches_the_suite_under_its_name() -> None:
    outputs = bundle()
    built = objective()

    result = ObjectiveSuite([(2.0, built)]).evaluate(outputs)

    assert result.total.item() == pytest.approx(
        2.0 * built.evaluate(outputs).value.item(), rel=1e-12
    )
    assert set(result.diagnostics) == {
        "type-autoencoder/exact_match",
        "type-autoencoder/loss",
        "type-autoencoder/per_slot",
    }
