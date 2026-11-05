from typing import TYPE_CHECKING, Final, override

import didactic.api as dx
import pytest
import torch

import lofi.nn._values as nn_values_module
from lofi import _checks
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._values import CellChart, ObjectiveOutputs
from lofi.nn.loss.laws import (
    OBJECTIVE_LAWS,
    PARSER_OUTPUT_LAWS,
    ObjectiveFixture,
    ParserOutputFixture,
    check_objective_laws,
    check_parser_output_laws,
    only_the_declared_outputs_are_read,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import Objective, PublishedOutputs

type Batch = tuple[str, ...]
type Outputs = ObjectiveOutputs[Batch]
type Term = Objective[PublishedOutputs[Batch]]

LENGTH: Final = 3
ITEMS: Final = 2
SPANS: Final = SpanIndex(LENGTH)
LENGTHS: Final = (LENGTH, LENGTH - 1)
PADDING: Final = -9.0
BATCH: Final[Batch] = ("first", "second")


def live_spans() -> torch.Tensor:
    mask = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(LENGTHS):
        for i, j in SPANS.spans_by_length():
            if j <= length:
                mask[item, SPANS.span_id(i, j)] = True

    return mask


LIVE: Final = live_spans()


def inside(scale: float = 1.0, fill: float = PADDING) -> LogDistribution:
    steps = torch.arange(ITEMS * SPANS.num_spans, dtype=torch.float64)
    scores = (-1.0 - steps.reshape(ITEMS, SPANS.num_spans) / 10.0) * scale
    padded = torch.where(LIVE, scores, fill).detach().requires_grad_()

    return LogDistribution(padded, SPAN_AXES, LIVE)


def bundle(scale: float = 1.0, fill: float = PADDING) -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(inside(scale, fill)),
        batch=BATCH,
    )


def empty_bundle() -> Outputs:
    blank = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)
    scores = torch.full(
        (ITEMS, SPANS.num_spans),
        PADDING,
        dtype=torch.float64,
    ).requires_grad_()

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(scores, SPAN_AXES, blank)),
        batch=BATCH,
    )


def permuted(outputs: Outputs) -> Outputs:
    order = torch.tensor([1, 0])
    scores = outputs.chart.inside.tensor
    mask = outputs.chart.inside.mask

    assert mask is not None
    moved = scores[order].detach().requires_grad_()

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(LogDistribution(moved, SPAN_AXES, mask[order])),
        batch=tuple(reversed(BATCH)),
    )


def refilled(outputs: Outputs) -> Outputs:
    del outputs

    return bundle(fill=PADDING / 3.0)


def chart_mass(outputs: Outputs) -> float:
    scores = outputs.chart.inside.tensor
    mask = outputs.chart.inside.mask
    total = 0.0

    for item in range(int(scores.shape[0])):
        for span in range(int(scores.shape[1])):
            if mask is None or bool(mask[item, span]):
                total -= float(scores[item, span].detach())

    return total


def whole_chart(outputs: Outputs) -> float:
    scores = outputs.chart.inside.tensor

    return -float(scores.detach().sum())


def positional_mass(outputs: Outputs) -> float:
    scores = outputs.chart.inside.tensor
    mask = outputs.chart.inside.mask
    total = 0.0

    for item in range(int(scores.shape[0])):
        for span in range(int(scores.shape[1])):
            if mask is None or bool(mask[item, span]):
                total -= item * float(scores[item, span].detach())

    return total


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


class Coordinate(dx.Model):
    span: int = dx.field(default=0)


class ChartMass:
    __slots__ = ("name", "requires")

    def __init__(self, name: str = "chart-mass") -> None:
        self.name = name
        self.requires = frozenset({"chart"})

    def live(self, outputs: PublishedOutputs[Batch]) -> tuple[Tensor, int]:
        """Return the chart with its padding zeroed, and how many items are real.

        Parameters
        ----------
        outputs : PublishedOutputs[Batch]
            What the graph published for the step.

        Returns
        -------
        tuple of torch.Tensor and int
            A tensor of rank 2 over item and span, of the chart's floating dtype, and
            the number of items holding any real span.
        """
        scores = outputs.chart.inside
        mask = scores.mask

        if mask is None:
            return scores.tensor, int(scores.tensor.shape[0])

        kept = torch.where(mask, scores.tensor, scores.tensor.new_zeros(()))

        return kept, int(mask.any(dim=1).sum())

    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        """Return the negated inside weight of the step's real spans.

        Parameters
        ----------
        outputs : PublishedOutputs[Batch]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The term, inapplicable where no item holds a real span.
        """
        kept, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        return LossTerm(-kept.sum(), items)


class Rounded(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)

        return LossTerm((-kept.sum()).detach().round().to(torch.int64), items)


class Detached(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)

        return LossTerm((-kept.sum()).detach(), items)


class DetachedZero(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)

        if items == 0:
            return LossTerm(torch.zeros(()), 0)

        return LossTerm(-kept.sum(), items)


class Blind:
    __slots__ = ("name", "requires", "scores")

    def __init__(self) -> None:
        self.name = "blind"
        self.requires = frozenset({"chart"})
        self.scores = inside()

    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        """Return a term computed from the chart this objective was built with.

        Parameters
        ----------
        outputs : PublishedOutputs[Batch]
            What the graph published for the step, which is not read.

        Returns
        -------
        LossTerm
            The unchanged term.
        """
        del outputs
        kept = torch.where(LIVE, self.scores.tensor, self.scores.tensor.new_zeros(()))

        return LossTerm(-kept.sum(), ITEMS)


class Leaky(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        scores = outputs.chart.inside.tensor

        return LossTerm(-scores.sum(), int(scores.shape[0]))


class Antitone(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)

        return LossTerm(kept.sum(), items)


class Scaled(ChartMass):
    __slots__ = ("factor",)

    def __init__(self, factor: float = 2.5) -> None:
        super().__init__("scaled")
        self.factor = factor

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)

        return LossTerm(self.factor * -kept.sum(), items)


class Positional(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)
        positions = torch.arange(kept.shape[0], dtype=kept.dtype).unsqueeze(1)

        return LossTerm(-(kept * positions).sum(), items)


class Jittery(ChartMass):
    __slots__ = ("stream",)

    def __init__(self) -> None:
        super().__init__("jittery")
        self.stream = torch.Generator().manual_seed(0)

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)
        drawn = torch.rand((), generator=self.stream, dtype=kept.dtype)

        return LossTerm(-kept.sum() + drawn * 1e-9, items)


class Modeling(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)
        built = Coordinate(span=int(kept.shape[1]))

        return LossTerm(-kept.sum() * (built.span / built.span), items)


class Peeking(ChartMass):
    __slots__ = ()

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, items = self.live(outputs)

        return LossTerm(-kept.sum() * len(outputs.batch), items)


def fixture_for(
    objective: Term,
    unweighted: Callable[[Outputs], float],
    *,
    inputs: Callable[[Outputs], tuple[Tensor, ...]] | None = None,
    inapplicable: Callable[[], Outputs] | None = None,
) -> ObjectiveFixture[Batch]:
    return ObjectiveFixture(
        objective=objective,
        outputs=bundle,
        inputs=inputs or (lambda outputs: (outputs.chart.inside.tensor,)),
        unweighted=unweighted,
        monotone=lambda: (bundle(scale=4.0), bundle()),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=inapplicable,
    )


def scaled_mass(factor: float) -> Callable[[Outputs], float]:
    return lambda outputs: factor * chart_mass(outputs)


BLIND: Final = Blind()

BLIND_MASS: Final = -float(
    torch.where(LIVE, BLIND.scores.tensor, BLIND.scores.tensor.new_zeros(()))
    .detach()
    .sum(),
)


CONFORMANT: Final[tuple[tuple[str, ObjectiveFixture[Batch]], ...]] = (
    (
        "chart-mass",
        fixture_for(ChartMass(), chart_mass, inapplicable=empty_bundle),
    ),
    (
        "chart-mass-always-applies",
        fixture_for(ChartMass(), chart_mass),
    ),
)

BROKEN: Final[tuple[tuple[str, ObjectiveFixture[Batch], frozenset[str]], ...]] = (
    (
        "rounded",
        fixture_for(Rounded(), lambda outputs: round(chart_mass(outputs))),
        frozenset(
            {
                "the_value_is_a_scalar",
                "the_gradient_reaches_the_declared_inputs",
            },
        ),
    ),
    (
        "detached-zero",
        fixture_for(DetachedZero(), chart_mass, inapplicable=empty_bundle),
        frozenset({"an_inapplicable_term_is_a_connected_zero"}),
    ),
    (
        "blind",
        fixture_for(
            BLIND,
            lambda _outputs: BLIND_MASS,
            inputs=lambda _outputs: (BLIND.scores.tensor,),
        ),
        frozenset(
            {
                "only_the_declared_outputs_are_read",
                "the_value_is_monotone_where_the_fixture_says",
            },
        ),
    ),
    (
        "leaky",
        fixture_for(Leaky(), whole_chart),
        frozenset({"a_masked_position_does_not_move_the_value"}),
    ),
    (
        "antitone",
        fixture_for(Antitone(), scaled_mass(-1.0)),
        frozenset({"the_value_is_monotone_where_the_fixture_says"}),
    ),
    (
        "scaled",
        fixture_for(Scaled(), chart_mass),
        frozenset({"the_objective_applies_no_weight_of_its_own"}),
    ),
    (
        "positional",
        fixture_for(Positional(), positional_mass),
        frozenset({"permuting_the_items_leaves_the_value_alone"}),
    ),
    (
        "jittery",
        fixture_for(Jittery(), chart_mass),
        frozenset({"the_same_seed_gives_the_same_value"}),
    ),
    (
        "modeling",
        fixture_for(Modeling(), chart_mass),
        frozenset({"no_didactic_model_is_built"}),
    ),
    (
        "detached",
        fixture_for(Detached(), chart_mass),
        frozenset({"the_gradient_reaches_the_declared_inputs"}),
    ),
)


def red(fixture: ObjectiveFixture[Batch]) -> frozenset[str]:
    failing: list[str] = []

    for law in OBJECTIVE_LAWS:
        try:
            law(fixture)

        except _checks.LawViolationError:
            failing.append(law.__name__)

    return frozenset(failing)


@pytest.mark.parametrize(
    ("name", "fixture"),
    CONFORMANT,
    ids=[name for name, _ in CONFORMANT],
)
def test_every_objective_law_holds_of_a_conformant_objective(
    name: str,
    fixture: ObjectiveFixture[Batch],
) -> None:
    del name

    assert red(fixture) == frozenset()
    check_objective_laws(fixture)


@pytest.mark.parametrize(
    ("name", "fixture", "expected"),
    BROKEN,
    ids=[name for name, _, _ in BROKEN],
)
def test_a_defect_turns_exactly_the_laws_it_breaks_red(
    name: str,
    fixture: ObjectiveFixture[Batch],
    expected: frozenset[str],
) -> None:
    del name

    assert red(fixture) == expected


def test_every_objective_law_is_broken_by_some_subject() -> None:
    covered = frozenset().union(*(expected for _, _, expected in BROKEN))

    assert covered == frozenset(law.__name__ for law in OBJECTIVE_LAWS)


def test_reading_an_undeclared_field_fails_the_declaration_law() -> None:
    fixture = fixture_for(Peeking(), chart_mass)

    with pytest.raises(_checks.LawViolationError, match="reads batch"):
        only_the_declared_outputs_are_read(fixture)


def test_the_monotonicity_hook_has_no_default_a_fixture_can_fall_back_on() -> None:
    assert "monotone" not in ObjectiveFixture._field_defaults

    with pytest.raises(TypeError, match="got 4"):
        ObjectiveFixture._make((ChartMass(), bundle, chart_mass, models_built))


def cutpoint_mask(index: SpanIndex, items: int) -> torch.Tensor:
    valid = torch.zeros(items, index.num_spans, index.max_splits, dtype=torch.bool)

    for i, j in index.spans_by_length(min_length=2):
        for k in index.splits(i, j):
            span, column = index.split_id(i, j, k)
            valid[:, span, column] = True

    return valid


def uniform_scores(valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=2, keepdim=True).clamp(min=1)

    return torch.where(valid, -counts.to(torch.float).log(), -torch.inf)


def parsed(length: int, items: int = ITEMS) -> tuple[SplitPosterior, SpanMarginals]:
    index = SpanIndex(length)
    valid = cutpoint_mask(index, items)
    posterior = SplitPosterior(
        index,
        LogDistribution(
            uniform_scores(valid),
            SPLIT_AXES,
            valid,
            normalized_over="split",
        ),
    )
    weights = torch.where(valid, posterior.log_probabilities.tensor.exp(), 0.0)
    constituent = torch.zeros(items, index.num_spans)
    rule = torch.zeros(items, index.num_spans, index.max_splits)
    constituent[:, index.span_id(0, length)] = 1.0

    for i, j in index.spans_by_length_desc(min_length=2):
        span = index.span_id(i, j)

        for k in index.splits(i, j):
            _, column = index.split_id(i, j, k)
            share = constituent[:, span] * weights[:, span, column]
            rule[:, span, column] = share

            for start, end in ((i, k), (k, j)):
                if end - start > 1:
                    constituent[:, index.span_id(start, end)] += share

    return posterior, SpanMarginals(
        index,
        Encoded(constituent, SPAN_AXES, valid.any(dim=2)),
        Encoded(rule, SPLIT_AXES, valid),
        torch.full((items,), length),
    )


def leaking(length: int) -> tuple[SplitPosterior, SpanMarginals]:
    posterior, marginals = parsed(length)
    index = posterior.spans

    if length < 3:
        return posterior, marginals

    valid = posterior.valid.clone()
    scores = posterior.log_probabilities.tensor.clone()
    span, column = index.split_id(0, length, 1)
    valid[:, span, column] = False
    scores[:, span, :] = -torch.inf
    scores[:, span, column] = -torch.log(torch.tensor(2.0))
    _, other = index.split_id(0, length, 2)
    scores[:, span, other] = -torch.log(torch.tensor(2.0))

    return SplitPosterior(
        index,
        LogDistribution(scores, SPLIT_AXES, valid, normalized_over="split"),
    ), marginals


def doubled(length: int) -> tuple[SplitPosterior, SpanMarginals]:
    posterior, marginals = parsed(length)

    return posterior, SpanMarginals(
        posterior.spans,
        Encoded(
            marginals.constituent.tensor * 2.0,
            SPAN_AXES,
            marginals.constituent.mask,
        ),
        marginals.rule,
        marginals.lengths,
    )


PARSER_LENGTHS: Final = (1, 2, 3, 4)


def parser_red(fixture: ParserOutputFixture) -> frozenset[str]:
    failing: list[str] = []

    for law in PARSER_OUTPUT_LAWS:
        try:
            law(fixture)

        except _checks.LawViolationError:
            failing.append(law.__name__)

    return frozenset(failing)


def test_both_parser_laws_hold_of_a_well_behaved_parse() -> None:
    fixture = ParserOutputFixture(lengths=PARSER_LENGTHS, parse=parsed)

    assert parser_red(fixture) == frozenset()
    check_parser_output_laws(fixture)


def test_mass_on_a_cutpoint_the_mask_rules_out_fails_the_normalization_law(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(nn_values_module, on=False)
    fixture = ParserOutputFixture(lengths=PARSER_LENGTHS, parse=leaking)

    assert parser_red(fixture) == frozenset(
        {"split_mass_sums_to_one_over_the_valid_cutpoints"},
    )


def test_counting_every_node_twice_fails_the_constituent_law() -> None:
    fixture = ParserOutputFixture(lengths=PARSER_LENGTHS, parse=doubled)

    assert parser_red(fixture) == frozenset(
        {"constituent_marginals_sum_to_one_less_than_the_length"},
    )
