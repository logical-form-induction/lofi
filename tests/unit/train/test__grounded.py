from typing import Final, NamedTuple

import pytest
import torch
from tests.language import language

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.lf import THREE_WAY, ArgumentRenaming, InferenceCell, InferenceLabel
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.entailment import EntailmentHead
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import (
    CellMass,
    EmbeddingVectors,
    EvidenceLedger,
    ExpectedEmbeddingGrounding,
    GradientPath,
    GroundedBatch,
    GroundingInput,
    GroundingMetadata,
    GroundingQuery,
    Marginal,
    Phase,
)
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._readout import (
    CELL_AXES,
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs, SpanReadout, SpanTerms
from lofi.nn.sites import ComponentId, SiteMap
from lofi.nn.spaces import Space
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)
from lofi.train._grounded import GroundedForward, grounding_metrics

LANGUAGE: Final = language(("rain",))
FORM: Final = LANGUAGE.constant("rain")
SPANS: Final = SpanIndex(1)
LEFT: Final = ReadoutId("left")
RIGHT: Final = ReadoutId("right")
FORMS: Final = ReadoutId("forms")

SOFT_ATOM: Final = Operator("atom", "T")
SOFT_PAIR: Final = Operator("pair", "T", arg_sorts=("T", "T"))
SOFT_SIGNATURE: Final = Signature("grounded-soft-path", ("T",), (SOFT_ATOM, SOFT_PAIR))
SOFT_LAYOUT: Final = CompleteTree(
    SOFT_SIGNATURE, SOFT_SIGNATURE.operators, order="bf", depth=2
)
SOFT_VIEW: Final = NeuralCodecView(
    TermCodec(OperatorIndex(SOFT_SIGNATURE), SOFT_LAYOUT)
)
SOFT_CHOICES: Final = ChoiceIndex(SOFT_SIGNATURE, depth_bounds={"T": SOFT_LAYOUT.depth})


class Batch(NamedTuple):
    premises: tuple[ReadoutId, ...]
    hypotheses: tuple[ReadoutId, ...]
    alignments: tuple[tuple[ArgumentRenaming, ...], ...]
    labels: tuple[InferenceLabel, ...]
    words: tuple[tuple[str, ...], ...]
    premise_rows: tuple[int, ...]
    hypothesis_rows: tuple[int, ...]


def batch() -> Batch:
    return Batch(
        (LEFT,),
        (RIGHT,),
        ((ArgumentRenaming.identity(),),),
        (InferenceLabel.ENTAILS,),
        (("rain",), ("rain",)),
        (0,),
        (1,),
    )


def prediction(*, ledger: EvidenceLedger | None = None) -> GroundedBatch:
    decided = torch.tensor([True])
    cells = torch.zeros(1, len(InferenceCell))
    cells[0, tuple(InferenceCell).index(InferenceCell.DEGENERATE)] = 0.25
    return GroundedBatch(
        LogDistribution(
            torch.tensor([[0.7, 0.2, 0.1]]).log(),
            ("pair", "label"),
            decided,
            normalized_over="label",
        ),
        labels=THREE_WAY.labels,
        complete=False,
        residual=torch.tensor([0.1]),
        decided=decided,
        cells=CellMass(cells),
        gradient_path=GradientPath.PATHWISE,
        baseline=None,
        auxiliary=None,
        ledger=EvidenceLedger.empty() if ledger is None else ledger,
        variance=None,
        reward=None,
    )


class Inner:
    publishes = frozenset({"terms", "cells", "outside_cells", "choice_rows"})

    def publish(self, held: Batch, /) -> ObjectiveOutputs[Batch]:
        weights = torch.zeros(1, requires_grad=True)
        forms = WeightedTerms(
            (FORM,), weights, torch.tensor(-torch.inf), Estimate.EXACT
        )
        return ObjectiveOutputs(
            spans=SPANS,
            chart=CellChart(LogDistribution(torch.zeros(2, 1), SPAN_AXES)),
            batch=held,
            terms={FORMS: SpanTerms(SPANS, {(0, 0): forms, (1, 0): forms})},
            choice_rows={
                FORMS: SpanReadout(
                    SPANS,
                    LogDistribution(
                        torch.tensor([[[[0.8, 0.2]]], [[[0.3, 0.7]]]]).log(),
                        (*SPAN_AXES, "slot", "choice"),
                        normalized_over="choice",
                    ),
                )
            },
            cells=Encoded(
                torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
                ("item", "span", "width"),
            ),
            outside_cells=Encoded(
                torch.tensor([[[-1.0, -2.0]], [[-3.0, -4.0]]]),
                ("item", "span", "width"),
            ),
        )


class Grounding:
    labels = THREE_WAY.labels
    grounding_metadata = GroundingMetadata(
        "fixture",
        scorer="fixture",
        source="fixture",
        prover=None,
        prover_version=None,
        budget=None,
    )

    def __init__(self, requires: frozenset[str]) -> None:
        self.requires = requires
        self.received = None
        self.opened = 0

    def ground(self, received: GroundingInput, /) -> GroundedBatch:
        self.received = received
        return prediction()

    def open_step(self) -> None:
        self.opened += 1

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ComponentId("fixture-grounding"), {})


def test_grounded_forward_publishes_one_batch_and_opens_the_next_step() -> None:
    grounding = Grounding(frozenset({"forms"}))
    forward = GroundedForward(
        Inner(),
        grounding,
        query=GroundingQuery.OBSERVED_LABEL,
        premise_readout=FORMS,
        hypothesis_readout=FORMS,
        phase=Phase.TRAIN,
    )
    outputs = forward.publish(batch())

    assert outputs.entailment == prediction()
    assert grounding.received is not None
    assert grounding.received.phase is Phase.TRAIN
    assert grounding.received.sentences is None
    assert grounding.received.forms.keys() == {LEFT, RIGHT}
    assert grounding.received.forms[LEFT] is outputs.terms[FORMS].at(0, 0)
    assert grounding.received.forms[RIGHT] is outputs.terms[FORMS].at(1, 0)
    assert outputs.outside_cells is not None
    assert outputs.outside_cells.tensor.tolist() == [
        [[-1.0, -2.0]],
        [[-3.0, -4.0]],
    ]

    forward.open_step()

    assert grounding.opened == 1


def test_sentence_grounding_pairs_vectors_by_the_declared_side_ids() -> None:
    grounding = Grounding(frozenset({"sentences"}))
    forward = GroundedForward(Inner(), grounding, query=GroundingQuery.OBSERVED_LABEL)

    forward.publish(batch())

    assert grounding.received is not None
    assert grounding.received.sentences is not None
    assert grounding.received.sentences.premise.tensor.tolist() == [[1.0, 2.0]]
    assert grounding.received.sentences.hypothesis.tensor.tolist() == [[3.0, 4.0]]


def test_soft_grounding_pairs_root_choice_rows_by_the_declared_side_ids() -> None:
    grounding = Grounding(frozenset({"choice_rows"}))
    forward = GroundedForward(
        Inner(),
        grounding,
        query=GroundingQuery.OBSERVED_LABEL,
        premise_readout=FORMS,
        hypothesis_readout=FORMS,
        phase=Phase.TRAIN,
    )

    forward.publish(batch())

    assert grounding.received is not None
    assert set(grounding.received.choice_rows) == {LEFT, RIGHT}
    assert grounding.received.choice_rows[LEFT].axes == ("term", "slot", "choice")
    assert torch.allclose(
        grounding.received.choice_rows[LEFT].tensor.exp(),
        torch.tensor([[[0.8, 0.2]]]),
    )
    assert torch.allclose(
        grounding.received.choice_rows[RIGHT].tensor.exp(),
        torch.tensor([[[0.3, 0.7]]]),
    )


def test_soft_grounding_preserves_the_decoder_to_classifier_gradient_path() -> None:
    torch.manual_seed(23)
    decoder = RecursiveTermDecoder(
        codec=SOFT_VIEW,
        choices=SOFT_CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("grounded-soft-decoder"),
        settings=TreeDecoderSettings(input_width=2, width=4),
        choice_heads="flat",
    )
    encoder = TreeRnnEncoder(
        codec=SOFT_VIEW,
        choices=SOFT_CHOICES,
        component=ComponentId("grounded-soft-encoder"),
        scheme=ArityHeadScheme(width=5, reached=SOFT_SIGNATURE.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )
    space = Space(
        "T",
        codec=SOFT_VIEW,
        choices=SOFT_CHOICES,
        store=TermStore(SOFT_SIGNATURE),
        max_nodes=SOFT_VIEW.capacity,
        encoder=encoder,
        identity=decoder,
        combinators=None,
    )
    readout = TermsReadout(
        decoder,
        codec=SOFT_VIEW,
        index=SOFT_CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=FORMS,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="T",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )
    cells = torch.randn(2, SPANS.num_spans, 2, requires_grad=True)

    class SoftInner:
        publishes = frozenset({"choice_rows"})

        def publish(self, held: Batch, /) -> ObjectiveOutputs[Batch]:
            publication = readout.publish_with_choices(
                Encoded(cells, CELL_AXES), spans=SPANS
            )
            return ObjectiveOutputs(
                spans=SPANS,
                chart=CellChart(LogDistribution(torch.zeros(2, 1), SPAN_AXES)),
                batch=held,
                choice_rows={FORMS: publication.choices},
            )

    grounding = ExpectedEmbeddingGrounding(
        EntailmentHead(THREE_WAY.labels, width=encoder.output_width, hidden=6),
        vectors=EmbeddingVectors.EXPECTED_FORM,
        expectation="soft",
        composition=Marginal(),
        space=space,
    )
    forward = GroundedForward(
        SoftInner(),
        grounding,
        query=GroundingQuery.FULL,
        premise_readout=FORMS,
        hypothesis_readout=FORMS,
        phase=Phase.TRAIN,
    )
    outputs = forward.publish(batch())
    grounded = outputs.entailment

    assert grounded is not None
    (-grounded.log_probabilities.tensor[0, 0]).backward()

    assert cells.grad is not None
    assert bool(cells.grad.ne(0).any())
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for parameter in decoder.parameters()
    )
    assert any(
        parameter.grad is not None and bool(parameter.grad.ne(0).any())
        for parameter in encoder.parameters()
    )


def test_sentence_grounding_refuses_a_forward_without_chart_cells() -> None:
    class NoCells:
        publishes = frozenset({"terms"})

        def publish(self, held: Batch, /) -> ObjectiveOutputs[Batch]:
            return ObjectiveOutputs(
                spans=SPANS,
                chart=CellChart(LogDistribution(torch.zeros(2, 1), SPAN_AXES)),
                batch=held,
            )

    forward = GroundedForward(
        NoCells(),
        Grounding(frozenset({"sentences"})),
        query=GroundingQuery.OBSERVED_LABEL,
    )

    with pytest.raises(MissingOutputError, match="sentences"):
        forward.publish(batch())


def test_grounding_metrics_report_the_complete_current_schema() -> None:
    metrics = grounding_metrics(prediction(), (InferenceLabel.ENTAILS,))

    assert metrics["grounded/decided"] == 1.0
    assert metrics["grounded/degenerate_mass"] == pytest.approx(0.25)
    assert metrics["grounded/questions_per_second"] == 0.0
    assert metrics["grounded/memo_hit_rate"] == 0.0
    assert metrics["grounded/process_starts"] == 0.0
    assert metrics["grounded/label_agreement"] == 1.0
    assert metrics["grounded/out_of_fragment_share"] == 0.0


def test_grounding_metrics_measure_decided_agreement_with_symbolic_labels() -> None:
    metrics = grounding_metrics(prediction(), (InferenceLabel.NEUTRAL,))

    assert metrics["grounded/label_agreement"] == 0.0


def test_grounding_metrics_report_a_custom_abstention_cause() -> None:
    ledger = EvidenceLedger(
        {},
        {"license-expired": 2},
        calls=2,
        memo_hits=0,
        cpu_seconds=0.0,
        questions=2,
        wall_seconds=0.1,
        process_starts=None,
        verified_share=None,
    )

    metrics = grounding_metrics(
        prediction(ledger=ledger),
        (InferenceLabel.ENTAILS,),
    )

    assert metrics["grounded/abstention/license-expired"] == 2.0


def test_grounding_metrics_measure_the_out_of_fragment_question_share() -> None:
    ledger = EvidenceLedger(
        {},
        {"out-of-fragment": 1},
        calls=1,
        memo_hits=0,
        cpu_seconds=0.0,
        questions=2,
        wall_seconds=0.1,
        process_starts=None,
        verified_share=None,
    )

    metrics = grounding_metrics(
        prediction(ledger=ledger),
        (InferenceLabel.ENTAILS,),
    )

    assert metrics["grounded/out_of_fragment_share"] == 0.5
