from math import prod
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Final, Self

import pytest
import torch
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.deduce import SpanIndex
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._chart import CutpointEntropy, SpanLengthBias
from lofi.probe._constructors import ConstructorEmbedding
from lofi.probe._controller_choice import ControllerChoice
from lofi.probe._controller_relations import ControllerRelation, ControllerRelations
from lofi.probe._diagnose import (
    Aggregation,
    DiagnosticContext,
    DiagnosticResult,
    Measurement,
)
from lofi.probe._differentiation import AtomicDifferentiation
from lofi.probe._diora import SplitPosteriorKL
from lofi.probe._grading import DepthGrading
from lofi.probe._marginals import SpanAggregate, SpanWeighting
from lofi.probe._observation_groups import ObservationGrouping
from lofi.probe._recurrent import (
    GatePatterns,
    HiddenStateEvolution,
    MemoryCellDynamics,
    RecurrentConnectionAnalysis,
)
from lofi.probe._recursion import RecursionSpread
from lofi.probe._selectivity import (
    ClassifierProbe,
    ControlTask,
    ProbeCapacity,
    ProbeResult,
    SpanInventory,
)
from lofi.probe._structure import NODE_AXES
from lofi.probe._superposition import SuperpositionInterference
from lofi.probe._transformer import (
    AttentionPatterns,
    FeedForwardSpecialization,
    ResidualContributions,
)
from lofi.probe.laws import (
    SCALINGS,
    ClassifierFixture,
    DiagnosticFixture,
    a_reading_of_any_magnitude_measures_to_a_number,
    a_result_carries_the_seed_its_control_labels_were_drawn_under,
    a_result_reports_selectivity_as_the_gap_between_its_accuracies,
    an_aggregate_over_spans_is_reported_under_both_weightings,
    both_tasks_are_fitted_at_the_capacity_the_result_reports,
    check_classifier_laws,
    check_diagnostic_laws,
    every_score_lies_in_the_unit_interval,
    scaling_the_readings_leaves_every_score_in_range,
)
from lofi.terms import BatchEncoding, Encoding

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from hypothesis.strategies import SearchStrategy
    from torch import Tensor

    from lofi.probe._values import Trace

PARSER: Final = ComponentId("cky-parser")
DECIDING: Final = ComponentId("combinator-controller")
ENCODING: Final = ComponentId("hrr")

CELL: Final = Site(PARSER, Role.CHART_CELL, (0, 2))
OUTSIDE: Final = Site(PARSER, Role.OUTSIDE_SCORE, (0, 2))
WEIGHTS: Final = Site(DECIDING, Role.COMBINATOR_LOGITS)
SPLITS: Final = Site(PARSER, Role.SPLIT_DISTRIBUTION, ("alpha",))
POSTERIOR_SPLITS: Final = Site(PARSER, Role.SPLIT_DISTRIBUTION, ("posterior",))
POSTERIOR_INSIDE: Final = Site(PARSER, Role.INSIDE_SCORE, ("posterior",))
POSTERIOR_OUTSIDE: Final = Site(PARSER, Role.OUTSIDE_SCORE, ("posterior",))
MARGINALS: Final = Site(PARSER, Role.CONSTITUENT_MARGINAL, ("mu",))
ENCODED: Final = Site(ENCODING, Role.TERM_ENCODING, ("node",))
HIDDEN: Final = Site(ENCODING, Role.RECURRENT_HIDDEN, ("layer",))
CELL_STATE: Final = Site(ENCODING, Role.RECURRENT_CELL, ("layer",))
GATE: Final = Site(ENCODING, Role.GATE, ("layer", "gate"))
ATTENTION_LOGITS: Final = tuple(
    Site(ENCODING, Role.ATTENTION_LOGITS, (0, head)) for head in range(2)
)
ATTENTION_WEIGHTS: Final = tuple(
    Site(ENCODING, Role.ATTENTION_WEIGHTS, (0, head)) for head in range(2)
)
PREACTIVATION: Final = Site(ENCODING, Role.FFN_PREACTIVATION, (0,))
RESIDUALS: Final = tuple(
    Site(ENCODING, Role.RESIDUAL_STREAM, (0, stage))
    for stage in ("input", "post-attention", "output")
)

REAL: Final = SiteSpec(
    ("item", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

TERM_REAL: Final = SiteSpec(
    ("term", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

RECURRENT_REAL: Final = SiteSpec(
    ("term", "slot", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

ATTENTION_REAL: Final = SiteSpec(
    ("batch", "position", "attended-position"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

ATTENTION_PROBABILITY: Final = SiteSpec(
    ("batch", "position", "attended-position"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)

TRANSFORMER_REAL: Final = SiteSpec(
    ("batch", "position", "feature"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

WEIGHED: Final = SiteSpec(
    ("span", "action"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

SPLIT_LOG_PROBABILITIES: Final = SiteSpec(
    ("item", "span", "split"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)

SPAN_REAL: Final = SiteSpec(
    ("item", "span"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

MARGINAL_PROBABILITIES: Final = SiteSpec(
    ("item", "span"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)

NODES: Final = SiteSpec(
    NODE_AXES,
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

EXAMPLES: Final = 15

# the derivation weighting of the three spans a margin fixture reads
BELIEVED: Final = SpanWeighting(torch.tensor([0.2, 0.3, 0.5]), axes=("span",))

# magnitude that remains finite under the tested scaling factors
ORDINARY: Final = 1e3

# finite exponential before scaling and non-finite after scaling
MODEST: Final = 10.0

# non-finite reading values
NOT_NUMBERS: Final = (float("nan"), float("inf"), -float("inf"))

# non-finite positions per generated reading
HOLES: Final = 2


def tensors(shape: tuple[int, ...], low: float, high: float) -> SearchStrategy[Tensor]:
    numbers = st.floats(
        min_value=low,
        max_value=high,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    count = prod(shape)

    return st.lists(numbers, min_size=count, max_size=count).map(
        lambda drawn: torch.tensor(drawn, dtype=torch.float32).reshape(shape)
    )


def holed(tensor: Tensor, where: tuple[int, ...], what: tuple[float, ...]) -> Tensor:
    flattened = tensor.flatten()

    for position, value in zip(where, what, strict=True):
        flattened[position] = value

    return flattened.reshape(tensor.shape)


def tensors_with_holes(shape: tuple[int, ...]) -> SearchStrategy[Tensor]:
    count = prod(shape)

    return st.tuples(
        tensors(shape, -ORDINARY, ORDINARY),
        st.lists(
            st.integers(min_value=0, max_value=count - 1),
            min_size=HOLES,
            max_size=HOLES,
            unique=True,
        ).map(tuple),
        st.lists(st.sampled_from(NOT_NUMBERS), min_size=HOLES, max_size=HOLES).map(
            tuple
        ),
    ).map(lambda drawn: holed(drawn[0], drawn[1], drawn[2]))


def a_chart_reading(
    low: float = -ORDINARY, high: float = ORDINARY
) -> SearchStrategy[Mapping[Site, Tensor]]:
    return st.tuples(tensors((2, 3), low, high), tensors((2, 3), low, high)).map(
        lambda drawn: {CELL: drawn[0], OUTSIDE: drawn[1]}
    )


def a_holed_chart_reading() -> SearchStrategy[Mapping[Site, Tensor]]:
    return st.tuples(tensors_with_holes((2, 3)), tensors_with_holes((2, 3))).map(
        lambda drawn: {CELL: drawn[0], OUTSIDE: drawn[1]}
    )


def a_chain(length: int, slots: int) -> Encoding:
    filling = slots - length

    return Encoding(
        tuple(range(1, length + 1)) + (0,) * filling,
        tuple(range(-1, length - 1)) + (Encoding.NO_PARENT,) * filling,
        tuple(range(length)) + (0,) * filling,
        (True,) * length + (False,) * filling,
    )


# two terms of three slots, the second of them padded, for the structural laws
STRUCTURE: Final = BatchEncoding((a_chain(3, 3), a_chain(2, 3)))


def a_node_reading() -> SearchStrategy[Mapping[Site, Tensor]]:
    return tensors((2, 3, 4), -ORDINARY, ORDINARY).map(lambda drawn: {ENCODED: drawn})


def a_structural_fixture(
    diagnostic: AtomicDifferentiation
    | ConstructorEmbedding
    | DepthGrading
    | RecursionSpread,
) -> DiagnosticFixture:
    return DiagnosticFixture(
        diagnostic,
        DiagnosticContext({ENCODED: NODES}),
        a_node_reading(),
        EXAMPLES,
    )


def superposition_interference() -> DiagnosticFixture:
    return DiagnosticFixture(
        SuperpositionInterference(),
        DiagnosticContext({ENCODED: TERM_REAL}),
        tensors((6, 4), 1.0, ORDINARY).map(lambda drawn: {ENCODED: drawn}),
        EXAMPLES,
    )


def recurrent_reading() -> SearchStrategy[Tensor]:
    return tensors((2, 3, 4), 1.0, ORDINARY)


def hidden_state_evolution() -> DiagnosticFixture:
    return DiagnosticFixture(
        HiddenStateEvolution((Encoding.NO_PARENT, 0, 0)),
        DiagnosticContext({HIDDEN: RECURRENT_REAL}),
        recurrent_reading().map(lambda drawn: {HIDDEN: drawn}),
        EXAMPLES,
    )


def gate_patterns() -> DiagnosticFixture:
    return DiagnosticFixture(
        GatePatterns(),
        DiagnosticContext({GATE: RECURRENT_REAL}),
        recurrent_reading().map(lambda drawn: {GATE: drawn}),
        EXAMPLES,
    )


def memory_cell_dynamics() -> DiagnosticFixture:
    return DiagnosticFixture(
        MemoryCellDynamics((Encoding.NO_PARENT, 0, 0)),
        DiagnosticContext(
            {HIDDEN: RECURRENT_REAL, CELL_STATE: RECURRENT_REAL, GATE: RECURRENT_REAL}
        ),
        st.tuples(recurrent_reading(), recurrent_reading(), recurrent_reading()).map(
            lambda drawn: {
                HIDDEN: drawn[0],
                CELL_STATE: drawn[1],
                GATE: drawn[2],
            }
        ),
        EXAMPLES,
    )


def recurrent_connection_analysis() -> DiagnosticFixture:
    return DiagnosticFixture(
        RecurrentConnectionAnalysis((Encoding.NO_PARENT, 0, 0)),
        DiagnosticContext({HIDDEN: RECURRENT_REAL}),
        recurrent_reading().map(lambda drawn: {HIDDEN: drawn}),
        EXAMPLES,
    )


def attention_patterns() -> DiagnosticFixture:
    shape = (2, 3, 3)
    logits = torch.zeros(shape)
    weights = torch.full(shape, 1.0 / shape[-1])
    readings = {
        ATTENTION_LOGITS[0]: logits,
        ATTENTION_WEIGHTS[0]: weights,
        ATTENTION_LOGITS[1]: logits.clone(),
        ATTENTION_WEIGHTS[1]: weights.clone(),
    }

    return DiagnosticFixture(
        AttentionPatterns(),
        DiagnosticContext(
            {
                ATTENTION_LOGITS[0]: ATTENTION_REAL,
                ATTENTION_WEIGHTS[0]: ATTENTION_PROBABILITY,
                ATTENTION_LOGITS[1]: ATTENTION_REAL,
                ATTENTION_WEIGHTS[1]: ATTENTION_PROBABILITY,
            }
        ),
        st.just(readings),
        EXAMPLES,
    )


def feed_forward_specialization() -> DiagnosticFixture:
    return DiagnosticFixture(
        FeedForwardSpecialization(),
        DiagnosticContext(
            {PREACTIVATION: TRANSFORMER_REAL},
            groupings={
                PREACTIVATION: ObservationGrouping(
                    6,
                    ((0, 1, 2), (3, 4, 5)),
                )
            },
        ),
        tensors((2, 3, 4), 1.0, ORDINARY).map(lambda drawn: {PREACTIVATION: drawn}),
        EXAMPLES,
    )


def residual_contributions() -> DiagnosticFixture:
    return DiagnosticFixture(
        ResidualContributions(),
        DiagnosticContext(dict.fromkeys(RESIDUALS, TRANSFORMER_REAL)),
        st.tuples(
            tensors((2, 3, 4), 1.0, ORDINARY),
            tensors((2, 3, 4), 1.0, ORDINARY),
            tensors((2, 3, 4), 1.0, ORDINARY),
        ).map(lambda drawn: dict(zip(RESIDUALS, drawn, strict=True))),
        EXAMPLES,
    )


def finite_readings() -> DiagnosticFixture:
    return DiagnosticFixture(
        installed_diagnostics().kind("finite-readings")(),
        DiagnosticContext({CELL: REAL, OUTSIDE: REAL}),
        a_holed_chart_reading(),
        EXAMPLES,
    )


def action_margin() -> DiagnosticFixture:
    return DiagnosticFixture(
        installed_diagnostics().kind("action-margin")(),
        DiagnosticContext({WEIGHTS: WEIGHED}, weighting=BELIEVED),
        tensors((3, 4), -ORDINARY, ORDINARY).map(lambda drawn: {WEIGHTS: drawn}),
        EXAMPLES,
    )


def controller_choice() -> DiagnosticFixture:
    reading = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ]
    )

    return DiagnosticFixture(
        ControllerChoice(),
        DiagnosticContext({WEIGHTS: WEIGHED}, weighting=BELIEVED),
        st.just({WEIGHTS: reading}),
        EXAMPLES,
    )


def cutpoint_entropy() -> DiagnosticFixture:
    index = SpanIndex(4)
    reading = torch.full((1, index.num_spans, index.max_splits), -torch.inf)

    for start, end in index.spans_by_length(min_length=2):
        span = index.span_id(start, end)
        choices = end - start - 1
        reading[0, span, :choices] = torch.linspace(-0.1, -2.0, choices)

    reading[0, index.span_id(0, 3), :2] = torch.tensor([0.5, 0.5]).log()
    reading[0, index.span_id(1, 4), :2] = torch.tensor([1.0, 0.0]).log()

    weighting = SpanWeighting(
        torch.arange(1, index.num_spans + 1, dtype=torch.float64).reshape(1, -1),
        axes=("item", "span"),
    )

    return DiagnosticFixture(
        CutpointEntropy(),
        DiagnosticContext(
            {SPLITS: SPLIT_LOG_PROBABILITIES},
            span_index=index,
            weighting=weighting,
        ),
        st.just({SPLITS: reading}),
        EXAMPLES,
    )


def span_length_bias() -> DiagnosticFixture:
    index = SpanIndex(4)
    values = torch.linspace(1e-7, 1e-6, index.num_spans).reshape(1, -1)

    return DiagnosticFixture(
        SpanLengthBias(),
        DiagnosticContext({MARGINALS: MARGINAL_PROBABILITIES}, span_index=index),
        st.just({MARGINALS: values}),
        EXAMPLES,
    )


def split_posterior_kl() -> DiagnosticFixture:
    index = SpanIndex(4)
    attention = torch.full((1, index.num_spans, index.max_splits), -torch.inf)

    for start, end in index.spans_by_length(min_length=2):
        choices = end - start - 1
        attention[0, index.span_id(start, end), :choices] = -torch.log(
            torch.tensor(float(choices))
        )

    scores = torch.zeros((1, index.num_spans))
    readings = {
        POSTERIOR_SPLITS: attention,
        POSTERIOR_INSIDE: scores,
        POSTERIOR_OUTSIDE: scores.clone(),
    }

    return DiagnosticFixture(
        SplitPosteriorKL(),
        DiagnosticContext(
            {
                POSTERIOR_SPLITS: SPLIT_LOG_PROBABILITIES,
                POSTERIOR_INSIDE: SPAN_REAL,
                POSTERIOR_OUTSIDE: SPAN_REAL,
            },
            span_index=index,
        ),
        st.just(readings),
        EXAMPLES,
    )


def controller_relations() -> DiagnosticFixture:
    return DiagnosticFixture(
        ControllerRelations(),
        DiagnosticContext(
            {WEIGHTS: WEIGHED},
            relations={
                WEIGHTS: ControllerRelation(
                    3,
                    sequences=((0, 1, 2),),
                    similar_inputs=((0, 2),),
                )
            },
        ),
        tensors((3, 4), -ORDINARY, ORDINARY).map(lambda drawn: {WEIGHTS: drawn}),
        EXAMPLES,
    )


# one conformance fixture per registered diagnostic
FIXTURES: Final = {
    "attention-patterns": attention_patterns,
    "feed-forward-specialization": feed_forward_specialization,
    "finite-readings": finite_readings,
    "gate-patterns": gate_patterns,
    "hidden-state-evolution": hidden_state_evolution,
    "memory-cell-dynamics": memory_cell_dynamics,
    "action-margin": action_margin,
    "atomic-complex-differentiation": lambda: a_structural_fixture(
        AtomicDifferentiation(STRUCTURE)
    ),
    "constructor-embedding": lambda: a_structural_fixture(
        ConstructorEmbedding(STRUCTURE)
    ),
    "controller-choice": controller_choice,
    "controller-relations": controller_relations,
    "cutpoint-entropy": cutpoint_entropy,
    "depth-graded-encoding": lambda: a_structural_fixture(DepthGrading(STRUCTURE)),
    "recursion-spread": lambda: a_structural_fixture(RecursionSpread(STRUCTURE)),
    "residual-contributions": residual_contributions,
    "span-length-bias": span_length_bias,
    "split-posterior-kl": split_posterior_kl,
    "recurrent-connection-analysis": recurrent_connection_analysis,
    "superposition-interference": superposition_interference,
}


class Unbounded:
    """A diagnostic that reports an unbounded count as a score."""

    tag: ClassVar[str] = "unbounded"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset({"share"})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's element count as a share."""
        return DiagnosticResult(
            self.tag,
            [
                Measurement(site, "share", float(reading.tensor.numel()))
                for site, reading in context.readings(trace)
            ],
        )


class Saturating:
    """A diagnostic normalized by a constant."""

    tag: ClassVar[str] = "saturating"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset({"share"})

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's mean over an assumed size."""
        return DiagnosticResult(
            self.tag,
            [
                Measurement(
                    site, "share", float(reading.tensor.abs().mean()) / ORDINARY
                )
                for site, reading in context.readings(trace)
            ],
        )


class Exploding:
    """A diagnostic that may overflow."""

    tag: ClassVar[str] = "exploding"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return the exponential of each reading's maximum."""
        return DiagnosticResult(
            self.tag,
            [
                Measurement(site, "mass", float(reading.tensor.exp().max()))
                for site, reading in context.readings(trace)
            ],
        )


def a_row_mean(reading: Tensor, weights: Tensor) -> tuple[float, float]:
    rows = reading.to(torch.float64).sum(dim=1)

    return float((rows * weights).sum() / weights.sum()), float(rows.mean())


class HalfWeighted:
    """A diagnostic reporting a weighted mean over spans as a single number."""

    tag: ClassVar[str] = "half-weighted"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's row sums averaged under the weighting alone."""
        measured: list[Measurement] = []

        for site, reading in context.readings(trace):
            weights = context.weights_over(reading.axes[:1], reading.tensor.shape[:1])
            weighted, _ = a_row_mean(reading.tensor, weights)

            measured.append(Measurement(site, "rows", weighted))

        return DiagnosticResult(self.tag, measured)


class Copying:
    """A diagnostic whose weighted half is its uniform half again."""

    tag: ClassVar[str] = "copying"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's row sums averaged uniformly, twice."""
        aggregated: list[Aggregation] = []

        for site, reading in context.readings(trace):
            weights = context.weights_over(reading.axes[:1], reading.tensor.shape[:1])
            _, uniform = a_row_mean(reading.tensor, weights)
            count = int(reading.tensor.shape[0])

            aggregated.append(
                Aggregation(site, "rows", SpanAggregate(uniform, uniform, count))
            )

        return DiagnosticResult(self.tag, [], aggregated)


class Wobbling:
    """A diagnostic whose uniform half moves with the weighting."""

    tag: ClassVar[str] = "wobbling"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's row sums averaged under the weighting, twice."""
        aggregated: list[Aggregation] = []

        for site, reading in context.readings(trace):
            weights = context.weights_over(reading.axes[:1], reading.tensor.shape[:1])
            weighted, _ = a_row_mean(reading.tensor, weights)
            count = int(reading.tensor.shape[0])

            aggregated.append(
                Aggregation(site, "rows", SpanAggregate(weighted, weighted, count))
            )

        return DiagnosticResult(self.tag, [], aggregated)


class Unweighed:
    """A diagnostic aggregating over spans without asking for the weighting."""

    tag: ClassVar[str] = "unweighed"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's row sums averaged uniformly, as an aggregate."""
        aggregated: list[Aggregation] = []

        for site, reading in context.readings(trace):
            count = int(reading.tensor.shape[0])
            _, uniform = a_row_mean(reading.tensor, torch.ones(count))

            aggregated.append(
                Aggregation(site, "rows", SpanAggregate(uniform, uniform, count))
            )

        return DiagnosticResult(self.tag, [], aggregated)


class Honest:
    """A diagnostic aggregating each reading's row sums under both weightings."""

    tag: ClassVar[str] = "honest"
    requires: ClassVar[frozenset[Role]] = frozenset()
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return each reading's row sums under the weighting and uniformly."""
        aggregated: list[Aggregation] = []

        for site, reading in context.readings(trace):
            weights = context.weights_over(reading.axes[:1], reading.tensor.shape[:1])
            weighted, uniform = a_row_mean(reading.tensor, weights)
            count = int(reading.tensor.shape[0])

            aggregated.append(
                Aggregation(site, "rows", SpanAggregate(weighted, uniform, count))
            )

        return DiagnosticResult(self.tag, [], aggregated)


class MarginalBlind:
    """A diagnostic whose observed-marginal aggregate ignores its marginals."""

    tag: ClassVar[str] = "marginal-blind"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.CONSTITUENT_MARGINAL})
    scores: ClassVar[frozenset[str]] = frozenset()

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return every marginal row's uniform position mean under both weightings."""
        aggregated: list[Aggregation] = []

        for site, reading in context.readings(trace):
            values = torch.arange(reading.tensor.numel(), dtype=torch.float64)
            uniform = float(values.mean())
            aggregated.append(
                Aggregation(
                    site,
                    "positions",
                    SpanAggregate(uniform, uniform, reading.tensor.numel()),
                )
            )

        return DiagnosticResult(self.tag, [], aggregated)


def a_broken_fixture(
    diagnostic: Unbounded | Saturating | Exploding,
    readings: SearchStrategy[Mapping[Site, Tensor]],
) -> DiagnosticFixture:
    return DiagnosticFixture(
        diagnostic,
        DiagnosticContext({CELL: REAL, OUTSIDE: REAL}),
        readings,
        EXAMPLES,
    )


# the derivation weighting of the two items a chart reading holds
EVEN: Final = SpanWeighting(torch.tensor([0.5, 0.5]), axes=("item",))


def a_weighed_fixture(
    diagnostic: HalfWeighted | Copying | Wobbling | Unweighed | Honest,
    *,
    weighting: SpanWeighting | None = EVEN,
) -> DiagnosticFixture:
    return DiagnosticFixture(
        diagnostic,
        DiagnosticContext({CELL: REAL, OUTSIDE: REAL}, weighting=weighting),
        a_chart_reading(),
        EXAMPLES,
    )


@pytest.mark.parametrize("tag", installed_diagnostics().tags)
def test_every_registered_diagnostic_is_held_to_the_laws(tag: str) -> None:
    build = FIXTURES.get(tag)

    if build is None:
        pytest.fail(
            f"{tag} is registered and has no law fixture, so nothing generated "
            f"readings for it; the fixtures name {sorted(FIXTURES)}"
        )

    check_diagnostic_laws(build())


def test_the_laws_reach_every_tag_the_registry_holds() -> None:
    assert set(FIXTURES) == set(installed_diagnostics().tags)


def test_a_count_reported_as_a_score_is_caught() -> None:
    fixture = a_broken_fixture(Unbounded(), a_chart_reading())

    with pytest.raises(LawViolationError, match="score range"):
        every_score_lies_in_the_unit_interval(fixture)


def test_a_score_in_range_at_its_own_scale_is_caught_at_another() -> None:
    fixture = a_broken_fixture(Saturating(), a_chart_reading(1.0, ORDINARY))

    every_score_lies_in_the_unit_interval(fixture)

    with pytest.raises(LawViolationError, match="scale invariance"):
        scaling_the_readings_leaves_every_score_in_range(fixture)


def test_a_diagnostic_that_cannot_measure_an_extreme_reading_is_caught() -> None:
    fixture = a_broken_fixture(Exploding(), a_chart_reading(1.0, MODEST))

    with pytest.raises(LawViolationError, match="finite measurement"):
        a_reading_of_any_magnitude_measures_to_a_number(fixture)


def test_a_diagnostic_reporting_both_halves_genuinely_passes() -> None:
    an_aggregate_over_spans_is_reported_under_both_weightings(
        a_weighed_fixture(Honest())
    )


def test_a_weighted_mean_reported_as_a_single_number_is_caught() -> None:
    with pytest.raises(LawViolationError, match="without its uniform half"):
        an_aggregate_over_spans_is_reported_under_both_weightings(
            a_weighed_fixture(HalfWeighted())
        )


def test_a_uniform_mean_reported_as_both_halves_is_caught() -> None:
    with pytest.raises(LawViolationError, match="uniform mean reported twice"):
        an_aggregate_over_spans_is_reported_under_both_weightings(
            a_weighed_fixture(Copying())
        )


def test_an_observed_marginal_aggregate_that_ignores_its_marginals_is_caught() -> None:
    fixture = DiagnosticFixture(
        MarginalBlind(),
        DiagnosticContext({MARGINALS: MARGINAL_PROBABILITIES}),
        st.just({MARGINALS: torch.tensor([[0.2, 0.3, 0.5]])}),
        EXAMPLES,
    )

    with pytest.raises(LawViolationError, match="uniform mean reported twice"):
        an_aggregate_over_spans_is_reported_under_both_weightings(fixture)


def test_a_uniform_half_that_moves_with_the_weighting_is_caught() -> None:
    with pytest.raises(LawViolationError, match="uniform half"):
        an_aggregate_over_spans_is_reported_under_both_weightings(
            a_weighed_fixture(Wobbling())
        )


def test_an_aggregate_over_spans_under_no_weighting_is_caught() -> None:
    with pytest.raises(LawViolationError, match="under no weighting"):
        an_aggregate_over_spans_is_reported_under_both_weightings(
            a_weighed_fixture(Unweighed(), weighting=None)
        )


def test_a_diagnostic_that_never_aggregates_passes_under_no_weighting() -> None:
    an_aggregate_over_spans_is_reported_under_both_weightings(
        a_broken_fixture(Exploding(), a_chart_reading(1.0, MODEST))
    )


def test_a_score_reported_as_an_aggregate_is_held_to_the_unit_interval() -> None:
    class Overshooting(Honest):
        """An honest aggregate whose halves are called scores."""

        tag: ClassVar[str] = "overshooting"
        scores: ClassVar[frozenset[str]] = frozenset({"rows"})

    with pytest.raises(LawViolationError, match="score range"):
        every_score_lies_in_the_unit_interval(a_weighed_fixture(Overshooting()))


def test_the_extremes_a_law_measures_at_bracket_an_ordinary_reading() -> None:
    smallest, largest = SCALINGS

    assert smallest * ORDINARY > 0.0
    assert largest * ORDINARY < torch.finfo(torch.float32).max


def test_a_fixture_left_to_itself_states_its_own_budget() -> None:
    stated = finite_readings()
    defaulted = DiagnosticFixture(stated.diagnostic, stated.context, stated.readings)

    assert stated.max_examples == EXAMPLES
    assert defaulted.max_examples == 25


# a probe small enough that a law running it six times stays quick
CAPACITY: Final = ProbeCapacity(hidden=8, epochs=40, learning_rate=0.1, seed=3)

CONTROL: Final = ControlTask(seed=11)

# the fields a classifier-probe report carries
REPORTED: Final = (
    "accuracy",
    "control_accuracy",
    "selectivity",
    "control_seed",
    "capacity",
    "parameters",
)


def probed_spans(count: int, seed: int) -> SpanInventory:
    draws = torch.Generator().manual_seed(seed)
    labels = torch.randint(2, (count,), generator=draws).tolist()
    centers = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    spread = torch.randn(count, 3, generator=draws, dtype=torch.float64)
    keys = [f"span-{seed}-{at}" for at in range(count)]

    return SpanInventory(centers[labels] + 0.4 * spread, keys, labels, classes=2)


def a_classifier_fixture(probe: object) -> ClassifierFixture:
    return ClassifierFixture(
        probe,  # ty: ignore[invalid-argument-type]
        probed_spans(40, 1),
        probed_spans(40, 2),
    )


def restated(report: ProbeResult, **fields: object) -> SimpleNamespace:
    held = {name: getattr(report, name) for name in REPORTED}

    return SimpleNamespace(**(held | fields))


def without(report: ProbeResult, dropped: str) -> SimpleNamespace:
    return SimpleNamespace(
        **{name: getattr(report, name) for name in REPORTED if name != dropped}
    )


class Changed:
    """A probe reporting what a change makes of a classifier probe's own report."""

    def __init__(
        self, change: Callable[[ProbeResult, ControlTask], object], control: ControlTask
    ) -> None:
        self._change = change
        self._probe = ClassifierProbe(CAPACITY, control)

    @property
    def capacity(self) -> ProbeCapacity:
        return self._probe.capacity

    @property
    def control(self) -> ControlTask:
        return self._probe.control

    def under(self, control: ControlTask) -> Self:
        return type(self)(self._change, control)

    def run(self, fitting: SpanInventory, held_out: SpanInventory) -> object:
        return self._change(self._probe.run(fitting, held_out), self._probe.control)


class Mismatched:
    """A probe fitting its control task at a capacity other than the declared one."""

    def __init__(self, other: ProbeCapacity, control: ControlTask) -> None:
        self._other = other
        self._probe = ClassifierProbe(CAPACITY, control)

    @property
    def capacity(self) -> ProbeCapacity:
        return self._probe.capacity

    @property
    def control(self) -> ControlTask:
        return self._probe.control

    def under(self, control: ControlTask) -> Self:
        return type(self)(self._other, control)

    def run(self, fitting: SpanInventory, held_out: SpanInventory) -> SimpleNamespace:
        declared = self._probe.run(fitting, held_out)
        elsewhere = ClassifierProbe(self._other, self._probe.control).run(
            fitting, held_out
        )

        return restated(
            declared,
            control_accuracy=elsewhere.control_accuracy,
            selectivity=declared.accuracy - elsewhere.control_accuracy,
        )


def test_a_classifier_probe_satisfies_every_classifier_law() -> None:
    check_classifier_laws(a_classifier_fixture(ClassifierProbe(CAPACITY, CONTROL)))


def test_a_report_without_selectivity_is_refused() -> None:
    probe = Changed(lambda report, _: without(report, "selectivity"), CONTROL)

    with pytest.raises(LawViolationError, match="reports no selectivity"):
        a_result_reports_selectivity_as_the_gap_between_its_accuracies(
            a_classifier_fixture(probe)
        )


def test_a_selectivity_that_is_not_the_gap_is_refused() -> None:
    probe = Changed(
        lambda report, _: restated(report, selectivity=report.selectivity + 0.25),
        CONTROL,
    )

    with pytest.raises(LawViolationError, match="reported a selectivity"):
        a_result_reports_selectivity_as_the_gap_between_its_accuracies(
            a_classifier_fixture(probe)
        )


def test_an_accuracy_that_is_no_proportion_is_refused() -> None:
    probe = Changed(lambda report, _: restated(report, accuracy=1.5), CONTROL)

    with pytest.raises(LawViolationError, match="which is no proportion"):
        a_result_reports_selectivity_as_the_gap_between_its_accuracies(
            a_classifier_fixture(probe)
        )


def test_a_report_of_another_seed_than_the_control_task_drew_under_is_refused() -> None:
    probe = Changed(
        lambda report, _: restated(report, control_seed=report.control_seed + 1),
        CONTROL,
    )

    with pytest.raises(LawViolationError, match="reported the seed"):
        a_result_carries_the_seed_its_control_labels_were_drawn_under(
            a_classifier_fixture(probe)
        )


def test_a_real_fit_that_moves_with_the_control_seed_is_refused() -> None:
    probe = Changed(
        lambda report, control: restated(
            report,
            accuracy=(control.seed % 2) / 2,
            selectivity=(control.seed % 2) / 2 - report.control_accuracy,
        ),
        CONTROL,
    )

    with pytest.raises(LawViolationError, match="reads the control labels"):
        a_result_carries_the_seed_its_control_labels_were_drawn_under(
            a_classifier_fixture(probe)
        )


def test_a_report_of_another_capacity_than_the_probe_fits_at_is_refused() -> None:
    elsewhere = ProbeCapacity(hidden=0, epochs=1, learning_rate=0.1, seed=3)
    probe = Changed(lambda report, _: restated(report, capacity=elsewhere), CONTROL)

    with pytest.raises(LawViolationError, match="reported the capacity"):
        both_tasks_are_fitted_at_the_capacity_the_result_reports(
            a_classifier_fixture(probe)
        )


def test_a_report_of_another_weight_count_is_refused() -> None:
    probe = Changed(
        lambda report, _: restated(report, parameters=report.parameters + 1), CONTROL
    )

    with pytest.raises(LawViolationError, match="weights at"):
        both_tasks_are_fitted_at_the_capacity_the_result_reports(
            a_classifier_fixture(probe)
        )


def test_a_control_task_fitted_at_another_capacity_is_refused() -> None:
    probe = Mismatched(
        ProbeCapacity(hidden=0, epochs=1, learning_rate=0.1, seed=3), CONTROL
    )

    with pytest.raises(LawViolationError, match="not fitted at one capacity"):
        both_tasks_are_fitted_at_the_capacity_the_result_reports(
            a_classifier_fixture(probe)
        )


def test_the_laws_run_in_conformance_order() -> None:
    probe = Changed(
        lambda report, _: restated(report, parameters=report.parameters + 1), CONTROL
    )

    with pytest.raises(LawViolationError, match="matched capacity"):
        check_classifier_laws(a_classifier_fixture(probe))
