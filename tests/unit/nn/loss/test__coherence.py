from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import pytest
import torch
from torch import nn

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._coherence import CoherenceObjective, MarginalWarmup
from lofi.nn.loss._divergence import CrossEntropy
from lofi.nn.loss._mapping import SOFTMAX
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._readout import (
    TermsReadout,
)
from lofi.nn.loss._values import CellChart, ObjectiveOutputs
from lofi.nn.spaces import (
    CombinatorFamily,
    Concatenate,
    RelationKind,
    Space,
    SpaceRelationBinding,
)
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
    Term,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from lofi.judge import RuleSet
    from lofi.nn.ports import FrontierDecoder, TermEncoder


class Batch(NamedTuple):
    items: tuple[str, ...]
    step: int = 0


class State(NamedTuple):
    score: torch.Tensor


ATOM_A: Final = Operator("a", "T", family="primitive")
ATOM_B: Final = Operator("b", "T", family="primitive")
SIGNATURE: Final = Signature("coherence", ("T",), (ATOM_A, ATOM_B)).with_holes()
STORE: Final = TermStore(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": 1})
CODEC: Final = NeuralCodecView(
    TermCodec(
        OperatorIndex(SIGNATURE),
        CompleteTree(SIGNATURE, (ATOM_A, ATOM_B), order="bf", depth=1),
    )
)
IDENTITY: Final = Action("identity")
APPLY: Final = Action("apply", "application")
RAISE: Final = Action("raise", "raise")
ACTIONS: Final = ActionSet((IDENTITY, APPLY), IDENTITY)
UNARY_ACTIONS: Final = ActionSet((IDENTITY, RAISE), IDENTITY)
SPANS: Final = SpanIndex(2)
DECODING: Final = DecodingSettings(
    "exhaustive", None, 0.0, 8, None, None, None, None, 1e-6
)


class BinaryRule:
    name = "application"
    arity = 2


class PartialUnaryRule:
    """Map only the first atomic category to two conclusions."""

    name = "raise"
    arity = 1
    directions = frozenset({"forward"})
    forward_is_complete = True

    def apply(self, inputs: tuple[Term, ...], /) -> tuple[Term, ...]:
        source = STORE.op(ATOM_A)
        target = STORE.op(ATOM_B)
        return (source, target) if inputs == (source,) else ()


class EmptyUnaryRule(PartialUnaryRule):
    """Have no conclusions in the decoded frontier."""

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> tuple[Term, ...]:
        del inputs
        return ()


class Encoder:
    """Minimal codec-bound encoder capability for a represented space."""

    output_width = 1
    codec = CODEC

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        values = torch.tensor(
            [[float(row.symbols[0])] for row in terms.rows], dtype=torch.float32
        )
        return Encoded(values.reshape(len(terms.rows), 1), ("term", "width"))


class Decoder:
    """One-step frontier decoder whose input controls the two atom weights."""

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

    def admits(self, term: Term, /) -> bool:
        """Return whether the completed frontier is one atomic reading."""
        return term.size == 1

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


def space(*, combinators: bool) -> Space:
    identity = Decoder()
    encoder = cast("TermEncoder", Encoder())
    family = (
        CombinatorFamily(
            {APPLY: identity},
            actions=ACTIONS,
            rules=cast("RuleSet[Term]", (BinaryRule(),)),
            merge=Concatenate(1),
            codec=CODEC.codec,
            encoder=encoder,
            identity=identity,
            index=CHOICES,
            root_sort="T",
        )
        if combinators
        else None
    )

    return Space(
        "T",
        codec=CODEC,
        choices=CHOICES,
        store=STORE,
        max_nodes=1,
        encoder=encoder,
        identity=identity,
        combinators=family,
    )


def unary_space(*, empty: bool = False) -> Space:
    """Return a space whose one unary rule is defined on part of its frontier."""
    identity = Decoder()
    encoder = cast("TermEncoder", Encoder())
    family = CombinatorFamily(
        {},
        actions=UNARY_ACTIONS,
        rules=cast(
            "RuleSet[Term]", ((EmptyUnaryRule() if empty else PartialUnaryRule()),)
        ),
        merge=None,
        codec=CODEC.codec,
        encoder=encoder,
        identity=identity,
        index=CHOICES,
        root_sort="T",
    )
    return Space(
        "T",
        codec=CODEC,
        choices=CHOICES,
        store=STORE,
        max_nodes=1,
        encoder=encoder,
        identity=identity,
        combinators=family,
    )


def readout(*, cap: int = 8) -> TermsReadout:
    settings = DECODING._replace(cap=cap)

    return TermsReadout(
        Decoder(),
        codec=CODEC,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=settings,
        key=ReadoutId("terms"),
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


def parsed() -> tuple[SplitPosterior, SpanMarginals]:
    valid = torch.tensor([[[False], [False], [True]]])
    split = SplitPosterior(
        SPANS,
        LogDistribution(
            torch.where(valid, torch.zeros(1, 3, 1), -torch.inf),
            SPLIT_AXES,
            valid,
            normalized_over="split",
        ),
    )
    built = valid.any(2)
    constituent = torch.tensor([[0.0, 0.0, 1.0]])
    rule = torch.tensor([[[0.0], [0.0], [1.0]]])

    return split, SpanMarginals(
        SPANS,
        Encoded(constituent, SPAN_AXES, built),
        Encoded(rule, SPLIT_AXES, valid),
        torch.tensor([2]),
    )


def outputs() -> ObjectiveOutputs[Batch]:
    splits, marginals = parsed()
    cells = Encoded(torch.tensor([[[0.2], [-0.1], [0.4]]]), (*SPAN_AXES, "width"))
    chart = CellChart(LogDistribution(torch.zeros(1, 3), SPAN_AXES))

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart,
        batch=Batch(("item",)),
        splits=splits,
        marginals=marginals,
        actions=MappingProxyType(
            {APPLY: LogDistribution(torch.zeros(1, 3), SPAN_AXES)}
        ),
        cells=cells,
    )


def unary_outputs() -> ObjectiveOutputs[Batch]:
    """Publish the same chart under the partial unary action."""
    published = outputs()
    return ObjectiveOutputs(
        spans=published.spans,
        chart=published.chart,
        batch=published.batch,
        splits=published.splits,
        marginals=published.marginals,
        actions=MappingProxyType(
            {RAISE: LogDistribution(torch.zeros(1, 3), SPAN_AXES)}
        ),
        cells=published.cells,
    )


def objective(
    *,
    estimator: CoherenceEstimator = CoherenceEstimator.EXACT,
    cap: int = 8,
    bound: float = 0.99,
    relation: SpaceRelationBinding | None = None,
    represented: Space | None = None,
) -> CoherenceObjective[Batch]:
    return CoherenceObjective(
        readout=readout(cap=cap),
        space=represented or space(combinators=True),
        relation=relation,
        divergence=CrossEntropy(),
        warmup=None,
        estimator=estimator,
        name="coherence",
        approximation=(
            ApproximationBound(bound)
            if estimator is CoherenceEstimator.TRUNCATED
            else None
        ),
    )


def test_exact_complete_term_coherence_has_zero_residual() -> None:
    result = objective().evaluate(outputs())

    assert torch.isfinite(result.value)
    assert result.count == 1
    assert result.residual is not None
    assert result.residual.item() == pytest.approx(0.0)


def test_unary_coherence_reencodes_all_terms_in_a_partial_rule_domain() -> None:
    result = objective(represented=unary_space()).evaluate(unary_outputs())

    assert torch.isfinite(result.value)
    assert result.count == 1
    assert result.residual is not None
    assert result.residual.item() == pytest.approx(0.0)


def test_unary_coherence_skips_a_cell_outside_the_rule_domain() -> None:
    result = objective(represented=unary_space(empty=True)).evaluate(unary_outputs())

    assert result.value.item() == pytest.approx(0.0)
    assert result.count == 0
    assert result.residual is not None
    assert result.residual.item() == pytest.approx(0.0)


def test_partial_unary_coherence_scores_the_rule_domain_and_shared_residual_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = WeightedTerms(
        (STORE.op(ATOM_A), STORE.op(ATOM_B)),
        torch.tensor((0.4, 0.4)).log(),
        torch.tensor(0.2).log(),
        Estimate.TRUNCATED,
    )
    conclusion = WeightedTerms(
        (STORE.op(ATOM_B),),
        torch.tensor((0.75,)).log(),
        torch.tensor(0.25).log(),
        Estimate.TRUNCATED,
    )
    decoded = iter((source, conclusion, conclusion))

    def decode_input[StateT](
        self: TermsReadout,
        decoder: FrontierDecoder[StateT],
        inputs: Encoded,
        /,
        *,
        space: Space,
        approximation: ApproximationBound | None = None,
    ) -> WeightedTerms:
        del self, decoder, inputs, space, approximation
        return next(decoded)

    monkeypatch.setattr(TermsReadout, "decode_input", decode_input)

    result = objective(
        estimator=CoherenceEstimator.TRUNCATED,
        bound=0.75,
        represented=unary_space(),
    ).evaluate(unary_outputs())

    assert result.residual is not None
    assert result.residual.item() == pytest.approx(0.3)


def test_held_out_evaluation_records_each_actual_action_and_span_contribution() -> None:
    built = objective()
    published = outputs()

    (evaluation,) = built.evaluation(published)
    result = built.evaluate(published)

    assert evaluation.item == 0
    assert evaluation.contributions
    assert {action for action, _, _ in evaluation.contributions} == {"apply"}
    assert {length for _, length, _ in evaluation.contributions} == {2}
    assert sum(value for _, _, value in evaluation.contributions) == pytest.approx(
        result.value.item()
    )


def test_truncated_complete_term_coherence_persists_residual_mass() -> None:
    result = objective(estimator=CoherenceEstimator.TRUNCATED, cap=1).evaluate(
        outputs()
    )

    assert torch.isfinite(result.value)
    assert result.residual is not None
    assert 0.0 < result.residual.item() <= 1.0


def test_binary_coherence_divides_its_residual_budget_between_frontiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired_inputs = TermsReadout.paired_inputs
    received: list[ApproximationBound | None] = []

    def recording_paired_inputs[LeftStateT, RightStateT](
        self: TermsReadout,
        reference: FrontierDecoder[LeftStateT],
        candidate: FrontierDecoder[RightStateT],
        reference_inputs: Encoded,
        candidate_inputs: Encoded,
        /,
        *,
        space: Space,
        approximation: ApproximationBound | None = None,
    ) -> tuple[WeightedTerms, WeightedTerms]:
        received.append(approximation)
        return paired_inputs(
            self,
            reference,
            candidate,
            reference_inputs,
            candidate_inputs,
            space=space,
            approximation=approximation,
        )

    monkeypatch.setattr(TermsReadout, "paired_inputs", recording_paired_inputs)
    bound = ApproximationBound(0.75)

    result = objective(
        estimator=CoherenceEstimator.TRUNCATED,
        cap=1,
        bound=bound.residual_mass,
    ).evaluate(outputs())

    assert len(received) == 1
    assert received[0] is not None
    assert received[0].residual_mass == pytest.approx(
        bound.paired_member().residual_mass
    )
    assert result.residual is not None
    assert 0.0 < result.residual.item() <= bound.residual_mass


def test_consistency_and_combinator_residuals_share_one_objective_bound() -> None:
    source = space(combinators=False)
    relation = SpaceRelationBinding(
        source,
        space(combinators=True),
        kind=RelationKind.CONSISTENCY,
        mapping=None,
        erasure=lambda term: term,
    )
    bound = 0.99

    result = objective(
        estimator=CoherenceEstimator.TRUNCATED,
        cap=1,
        bound=bound,
        relation=relation,
        represented=source,
    ).evaluate(outputs())

    assert result.residual is not None
    assert 0.0 < result.residual.item() <= bound


def test_exact_coherence_refuses_a_truncated_paired_frontier() -> None:
    with pytest.raises(ValueError, match="declared bound"):
        objective(cap=1).evaluate(outputs())


def test_marginal_warmup_names_its_endpoints() -> None:
    warmup = MarginalWarmup(10)

    assert warmup.fraction(0) == 0.0
    assert warmup.fraction(5) == 0.5
    assert warmup.fraction(10) == 1.0
    assert warmup.fraction(20) == 1.0


class CountingErasure:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, term: Term, /) -> Term:
        self.calls += 1
        return term


class CountingMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def apply_to(self, vectors: Encoded, /) -> Encoded:
        self.calls += 1
        return vectors


@pytest.mark.parametrize("kind", list(RelationKind))
def test_each_space_relation_changes_the_runtime_coherence_route(
    kind: RelationKind,
) -> None:
    source = space(combinators=kind is RelationKind.ERASE_THEN_DECODE)
    target = space(combinators=True)
    erasure = CountingErasure()
    mapping = CountingMap()
    relation = SpaceRelationBinding(
        source,
        target,
        kind=kind,
        mapping=mapping if kind is RelationKind.MAPPED else None,
        erasure=None if kind is RelationKind.MAPPED else erasure,
    )

    result = objective(
        estimator=CoherenceEstimator.TRUNCATED,
        relation=relation,
        represented=source,
    ).evaluate(outputs())

    assert torch.isfinite(result.value)

    if kind in {RelationKind.ERASE_THEN_DECODE, RelationKind.CONSISTENCY}:
        assert erasure.calls > 0
        assert mapping.calls == 0
    else:
        assert erasure.calls == 0
        assert mapping.calls == 1
