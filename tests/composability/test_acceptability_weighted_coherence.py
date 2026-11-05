"""The public judgment scale composes with the coherence weight channel."""

import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import torch

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.judgment import DesignCodec, JudgmentRow, RowColumns, ridit_scores
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.loss._coherence import CoherenceObjective
from lofi.nn.loss._divergence import KL
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._readout import (
    CELL_AXES,
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs
from lofi.nn.spaces import CombinatorFamily, Concatenate, Space
from lofi.spec.decoding import DecodingSettings
from lofi.spec.losses.coherence import CoherenceEstimator
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    HoleContext,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from lofi.judge import RuleSet
    from lofi.nn.ports import TermEncoder
    from lofi.terms import Term


class Batch(NamedTuple):
    """The item order and step consumed by the objective."""

    items: tuple[str, ...]
    step: int


ITEMS = ("low", "middle", "high")
SPANS = SpanIndex(2)
IDENTITY = Action("identity")
COMBINE = Action("combine", "forward-application")
READOUT_KEY = ReadoutId("types")
ATOM_A: Final = Operator("a", "T", family="primitive")
ATOM_B: Final = Operator("b", "T", family="primitive")
SIGNATURE: Final = Signature(
    "weighted-coherence", ("T",), (ATOM_A, ATOM_B)
).with_holes()
STORE: Final = TermStore(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": 1})
CODEC: Final = NeuralCodecView(
    TermCodec(
        OperatorIndex(SIGNATURE),
        CompleteTree(SIGNATURE, (ATOM_A, ATOM_B), order="bf", depth=1),
    )
)
DECODING: Final = DecodingSettings(
    "exhaustive", None, 0.0, 8, None, None, None, None, 1e-6
)


class BinaryRule:
    name = "forward-application"
    arity = 2


class State(NamedTuple):
    score: torch.Tensor


class Encoder:
    """Minimal encoder identity for the represented test space."""

    output_width = 2
    codec = CODEC


class Decoder:
    """One-step frontier decoder driven by the sum of its input vector."""

    def start(self, inputs: Encoded, /) -> State:
        return State(inputs.tensor.sum())

    def weigh(self, state: State, /, *, context: HoleContext) -> LogDistribution:
        del context
        logits = state.score.new_full((CHOICES.width,), -torch.inf)
        logits[CHOICES.index("a")] = state.score
        logits[CHOICES.index("b")] = -state.score
        normalized = logits.log_softmax(0)

        return LogDistribution(
            normalized,
            ("choice",),
            normalized.isfinite(),
            normalized_over="choice",
        )

    def fill(self, state: State, /, *, choices: torch.Tensor) -> State:
        del choices
        return state

    def admits(self, term: object, /) -> bool:
        """Admit every completed term in the fixture's finite language."""
        del term
        return True

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        scores = inputs.tensor.sum(dim=-1)
        logits = scores.new_full((*scores.shape, 1, CHOICES.width), -torch.inf)
        logits[..., 0, CHOICES.index("a")] = scores
        logits[..., 0, CHOICES.index("b")] = -scores
        normalized = logits.log_softmax(dim=-1)

        return LogDistribution(
            normalized,
            ("term", "slot", "choice"),
            normalized.isfinite(),
            normalized_over="choice",
        )

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms

        return self.marginal_rows(inputs, root_sort="T")


def objective() -> CoherenceObjective[Batch]:
    actions = ActionSet((IDENTITY, COMBINE), IDENTITY)
    decoder = Decoder()
    encoder = cast("TermEncoder", Encoder())
    family = CombinatorFamily(
        {COMBINE: decoder},
        actions=actions,
        rules=cast("RuleSet[Term]", (BinaryRule(),)),
        merge=Concatenate(2),
        codec=CODEC.codec,
        encoder=encoder,
        identity=decoder,
        index=CHOICES,
        root_sort="T",
    )
    space = Space(
        "T",
        codec=CODEC,
        choices=CHOICES,
        store=STORE,
        max_nodes=1,
        encoder=encoder,
        identity=decoder,
        combinators=family,
    )
    readout = TermsReadout(
        decoder,
        codec=CODEC,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=DECODING,
        key=READOUT_KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=STORE,
        root_sort="T",
        max_nodes=1,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    return CoherenceObjective(
        readout=readout,
        space=space,
        relation=None,
        warmup=None,
        divergence=KL,
        estimator=CoherenceEstimator.EXACT,
        name="coherence",
        approximation=None,
    )


def outputs() -> ObjectiveOutputs[Batch]:
    item_count = len(ITEMS)
    live = torch.ones(item_count, SPANS.num_spans, dtype=torch.bool)
    valid = (
        torch.tensor(SPANS.split_mask, dtype=torch.bool)
        .unsqueeze(0)
        .expand(item_count, -1, -1)
    )
    built = torch.zeros_like(live)
    root = SPANS.span_id(0, 2)
    built[:, root] = True
    split_values = torch.where(
        valid,
        torch.zeros(valid.shape, dtype=torch.float64),
        torch.full(valid.shape, -math.inf, dtype=torch.float64),
    )
    cells = torch.tensor(
        [
            [[0.2, -0.6], [0.9, 0.1], [0.0, 0.0]],
            [[-0.4, 0.7], [0.5, 0.3], [0.0, 0.0]],
            [[0.8, -0.2], [-0.3, 0.6], [0.0, 0.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    constituent = torch.zeros(item_count, SPANS.num_spans, dtype=torch.float64)
    constituent[:, root] = 1.0

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(
                torch.zeros(item_count, SPANS.num_spans, dtype=torch.float64),
                SPAN_AXES,
                live,
            )
        ),
        batch=Batch(ITEMS, step=0),
        splits=SplitPosterior(
            SPANS,
            LogDistribution(
                split_values,
                SPLIT_AXES,
                valid,
                normalized_over="split",
            ),
        ),
        marginals=SpanMarginals(
            SPANS,
            Encoded(constituent, SPAN_AXES, built),
            Encoded(valid.to(torch.float64), SPLIT_AXES, valid),
            torch.full((item_count,), 2, dtype=torch.int64),
        ),
        actions={COMBINE: LogDistribution(split_values, SPLIT_AXES, valid)},
        cells=Encoded(cells, CELL_AXES, live),
    )


def cell_gradient(
    term: torch.Tensor, published: ObjectiveOutputs[Batch]
) -> torch.Tensor:
    cells = published.cells

    assert cells is not None

    (gradient,) = torch.autograd.grad(term, (cells.tensor,))

    return gradient


def test_public_ridit_scores_weight_coherence_per_item() -> None:
    design = DesignCodec(7).encode(
        JudgmentRow("annotator", item, response, RowColumns({}, {}))
        for item, response in zip(ITEMS, (0, 3, 6), strict=True)
    )
    unweighted_outputs = outputs()
    identity_outputs = outputs()
    ridit_outputs = outputs()
    per_row = ridit_scores(design)
    ridit_by_item = {
        design.item_levels[int(code)]: per_row[position]
        for position, code in enumerate(design.items)
    }
    ridit = torch.stack(
        tuple(ridit_by_item[item] for item in ridit_outputs.batch.items)
    )
    subject = objective()
    unweighted = subject.evaluate(unweighted_outputs)
    identity = subject.evaluate(
        identity_outputs,
        item_weights=torch.ones(len(ITEMS), dtype=torch.float64),
    )
    weighted = subject.evaluate(ridit_outputs, item_weights=ridit)
    unweighted_gradient = cell_gradient(unweighted.value, unweighted_outputs)
    identity_gradient = cell_gradient(identity.value, identity_outputs)
    weighted_gradient = cell_gradient(weighted.value, ridit_outputs)

    assert design.item_levels == tuple(sorted(ITEMS))
    torch.testing.assert_close(identity.value, unweighted.value, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        identity_gradient,
        unweighted_gradient,
        rtol=0.0,
        atol=1e-6,
    )
    assert not torch.allclose(weighted_gradient, unweighted_gradient)

    for item, weight in enumerate(ridit):
        torch.testing.assert_close(
            weighted_gradient[item],
            unweighted_gradient[item] * weight,
            rtol=1e-10,
            atol=1e-10,
        )
