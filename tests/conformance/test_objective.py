from typing import TYPE_CHECKING, Final, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import (
    OBJECTIVE_CONFORMANCE_LAWS,
    ObjectiveLawSuite,
)

from lofi._checks import LawViolationError
from lofi.deduce import SpanIndex
from lofi.nn._values import LogDistribution
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    OBJECTIVE_LAWS,
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import (
        Objective,
        PublishedOutputs,
    )

type Batch = tuple[str, ...]
type Outputs = ObjectiveOutputs[Batch]
type Term = Objective[PublishedOutputs[Batch]]

LENGTH: Final = 3

ITEMS: Final = 3

LENGTHS: Final = (3, 2, 1)

SPANS: Final = SpanIndex(LENGTH)

BATCH: Final[Batch] = ("first", "second", "third")

PADDING: Final = -9.0


def live_spans() -> Tensor:
    """Return which spans are real for each item of the ragged batch.

    Returns
    -------
    torch.Tensor
        A boolean tensor of rank 2 over the axes ``("item", "span")``, true where the
        span ends within that item.
    """
    mask = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(LENGTHS):
        for i, j in SPANS.spans_by_length():
            if j <= length:
                mask[item, SPANS.span_id(i, j)] = True

    return mask


LIVE: Final = live_spans()


def inside(scale: float = 1.0, fill: float = PADDING) -> LogDistribution:
    """Return an inside chart whose live cells scale and whose padding is fill.

    Parameters
    ----------
    scale
        What each live log weight is multiplied by.
    fill
        What each padded cell holds.

    Returns
    -------
    LogDistribution
        Rank 2 over ``("item", "span")`` at double precision, masked over the spans each
        item really has.
    """
    steps = torch.arange(ITEMS * SPANS.num_spans, dtype=torch.float64)
    scores = (-1.0 - steps.reshape(ITEMS, SPANS.num_spans) / 10.0) * scale

    return LogDistribution(
        torch.where(LIVE, scores, fill).detach().requires_grad_(),
        SPAN_AXES,
        LIVE,
    )


def bundle(scale: float = 1.0, fill: float = PADDING) -> Outputs:
    """Return a step's outputs whose chart is the only quantity published.

    Parameters
    ----------
    scale
        What each live log weight is multiplied by.
    fill
        What each padded cell holds.

    Returns
    -------
    ObjectiveOutputs
        The bundle, publishing the spans, the chart and the batch.
    """
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(inside(scale, fill)),
        batch=BATCH,
    )


def empty_bundle() -> Outputs:
    """Return a step whose chart holds no real span, which nothing applies to.

    Returns
    -------
    ObjectiveOutputs
        The bundle, whose chart mask excludes every cell.
    """
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
    """Return the same step with its items in the reverse order.

    Parameters
    ----------
    outputs
        The bundle to reorder.

    Returns
    -------
    ObjectiveOutputs
        A bundle over the same items, ordered last to first.
    """
    order = torch.tensor([2, 1, 0])
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
    """Return the same step with the padding of its chart written over.

    Parameters
    ----------
    outputs
        The bundle whose padding is replaced, which fixes nothing else.

    Returns
    -------
    ObjectiveOutputs
        A bundle whose live cells are those of the original and whose padded cells hold
        another number.
    """
    del outputs

    return bundle(fill=PADDING / 4.0)


def cells_of(outputs: Outputs) -> tuple[list[float], list[int]]:
    """Return the live log weights of a bundle and the item each belongs to.

    Parameters
    ----------
    outputs
        The bundle to read.

    Returns
    -------
    tuple of list of float and list of int
        The live weights, read off the device one at a time, and the row each was read
        from.
    """
    scores = outputs.chart.inside.tensor
    mask = outputs.chart.inside.mask
    weights: list[float] = []
    rows: list[int] = []

    for item in range(int(scores.shape[0])):
        for span in range(int(scores.shape[1])):
            if mask is None or bool(mask[item, span]):
                weights.append(float(scores[item, span].detach()))
                rows.append(item)

    return weights, rows


def live_mean(outputs: Outputs) -> float:
    """Return the negated mean log weight of the live cells, summed by hand.

    Parameters
    ----------
    outputs
        The bundle to score.

    Returns
    -------
    float
        The quantity a conformant objective reports for this bundle.
    """
    weights, _ = cells_of(outputs)

    return -sum(weights) / len(weights) if weights else 0.0


def whole_mean(outputs: Outputs) -> float:
    """Return the negated mean log weight of every cell, padding included.

    Parameters
    ----------
    outputs
        The bundle to score.

    Returns
    -------
    float
        The quantity an objective that reads through the mask reports.
    """
    scores = outputs.chart.inside.tensor

    return -float(scores.detach().sum()) / scores.numel()


def positional_mean(outputs: Outputs) -> float:
    """Return the live mean with each cell weighted by the row it sits in.

    Parameters
    ----------
    outputs
        The bundle to score.

    Returns
    -------
    float
        The quantity an objective that reads a position rather than an item reports.
    """
    weights, rows = cells_of(outputs)
    weighted = sum(row * weight for row, weight in zip(rows, weights, strict=True))

    return -weighted / len(weights) if weights else 0.0


def antitone_mean(outputs: Outputs) -> float:
    """Return the live mean with its sign the other way round.

    Parameters
    ----------
    outputs
        The bundle to score.

    Returns
    -------
    float
        The quantity an objective ordering batches the wrong way reports.
    """
    return -live_mean(outputs)


class Coordinate(dx.Model):
    span: int = dx.field(default=0)


def models_built(call: Callable[[], LossTerm]) -> int:
    """Return how many didactic models one call constructed.

    Parameters
    ----------
    call
        The call to run under the counter.

    Returns
    -------
    int
        How many models were built while it ran.
    """
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


class SpanMean:
    __slots__ = ("name", "requires")

    def __init__(self, name: str = "span-mean") -> None:
        self.name = name
        self.requires = frozenset({"chart"})

    def live(self, outputs: PublishedOutputs[Batch]) -> tuple[Tensor, int, int]:
        """Return the chart with its padding zeroed, its live cells and items.

        Parameters
        ----------
        outputs
            What the graph published for the step.

        Returns
        -------
        tuple of torch.Tensor, int and int
            A tensor of rank 2 over item and span, of the chart's floating dtype,
            holding zero where a cell is padding.
        """
        scores = outputs.chart.inside
        mask = scores.mask

        if mask is None:
            return scores.tensor, scores.tensor.numel(), int(scores.tensor.shape[0])

        kept = torch.where(mask, scores.tensor, scores.tensor.new_zeros(()))

        return kept, int(mask.sum()), int(mask.any(dim=1).sum())

    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        """Return the negated mean log weight of the step's live spans.

        Parameters
        ----------
        outputs
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The term, inapplicable where no item holds a live span.
        """
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        return LossTerm(-kept.sum() / cells, items)


class Leaky(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("leaky")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        scores = outputs.chart.inside.tensor
        _, _, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(scores)

        return LossTerm(-scores.sum() / scores.numel(), items)


class Positional(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("positional")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        rows = torch.arange(kept.shape[0], dtype=kept.dtype).unsqueeze(1)

        return LossTerm(-(kept * rows).sum() / cells, items)


class Scaled(SpanMean):
    __slots__ = ("factor",)

    def __init__(self, factor: float = 2.5) -> None:
        super().__init__("scaled")
        self.factor = factor

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        return LossTerm(self.factor * -kept.sum() / cells, items)


class Detached(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("detached")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        return LossTerm((-kept.sum() / cells).detach(), items)


class Antitone(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("antitone")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        return LossTerm(kept.sum() / cells, items)


class Rounded(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("rounded")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        whole = (-kept.sum() / cells).detach().round()

        return LossTerm(whole.to(torch.int64), items)


class DetachedZero(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("detached-zero")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm(torch.zeros(()), 0)

        return LossTerm(-kept.sum() / cells, items)


class Jittery(SpanMean):
    __slots__ = ("stream",)

    def __init__(self) -> None:
        super().__init__("jittery")
        self.stream = torch.Generator().manual_seed(0)

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        drawn = torch.rand((), generator=self.stream, dtype=kept.dtype)

        return LossTerm(-kept.sum() / cells + drawn * 1e-9, items)


class Modeling(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("modeling")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        built = Coordinate(span=cells)

        return LossTerm(-kept.sum() / built.span, items)


class Peeking(SpanMean):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("peeking")

    @override
    def evaluate(self, outputs: PublishedOutputs[Batch], /) -> LossTerm:
        kept, cells, items = self.live(outputs)

        if items == 0:
            return LossTerm.inapplicable(outputs.chart.inside.tensor)

        return LossTerm(-kept.sum() / cells, len(outputs.batch))


def fixture_for(
    objective: Term,
    unweighted: Callable[[Outputs], float],
) -> ObjectiveFixture[Batch]:
    """Return the fixture the laws run one of these objectives through.

    Parameters
    ----------
    objective
        The objective under test.
    unweighted
        The quantity that objective is supposed to report, reckoned without it.

    Returns
    -------
    ObjectiveFixture
        The fixture, over the ragged batch the bundle above builds.
    """
    return ObjectiveFixture(
        objective=objective,
        outputs=bundle,
        inputs=lambda outputs: (outputs.chart.inside.tensor,),
        unweighted=unweighted,
        monotone=lambda: (bundle(scale=4.0), bundle()),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
    )


class DeclaredMonotone(ObjectiveLawSuite):
    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that the batch of larger log weights scores higher.

        Parameters
        ----------
        conformance
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the bundle the fixture names as scoring higher does not.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestAnObjectiveThatMeansOverTheSpansItIsHanded(DeclaredMonotone):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        return fixture_for(SpanMean(), live_mean)


def red(conformance: ObjectiveFixture[Batch]) -> frozenset[str]:
    """Return the name of every law of the suite that fails against a fixture.

    Parameters
    ----------
    conformance
        The objective and what the laws need to run against it.

    Returns
    -------
    frozenset of str
        The names of the failing methods.
    """
    suite = DeclaredMonotone()
    hook = ObjectiveLawSuite.test_the_value_is_monotone_where_the_fixture_says
    stated = DeclaredMonotone.test_the_value_is_monotone_where_the_fixture_says
    failing: list[str] = []

    for law in OBJECTIVE_CONFORMANCE_LAWS:
        running = stated if law is hook else law

        try:
            running(suite, conformance)

        except LawViolationError:
            failing.append(law.__name__)

    return frozenset(failing)


UNDER_THE_PROXY: Final = "test_only_the_declared_outputs_are_read"

_BROKEN: Final[tuple[tuple[str, ObjectiveFixture[Batch], frozenset[str]], ...]] = (
    (
        "leaky",
        fixture_for(Leaky(), whole_mean),
        frozenset({"test_a_masked_position_does_not_move_the_value"}),
    ),
    (
        "positional",
        fixture_for(Positional(), positional_mean),
        frozenset({"test_permuting_the_items_leaves_the_value_alone"}),
    ),
    (
        "scaled",
        fixture_for(Scaled(), live_mean),
        frozenset({"test_the_objective_applies_no_weight_of_its_own"}),
    ),
    (
        "detached",
        fixture_for(Detached(), live_mean),
        frozenset({"test_the_gradient_reaches_the_declared_inputs"}),
    ),
    (
        "antitone",
        fixture_for(Antitone(), antitone_mean),
        frozenset({"test_the_value_is_monotone_where_the_fixture_says"}),
    ),
    (
        "rounded",
        fixture_for(Rounded(), lambda outputs: float(round(live_mean(outputs)))),
        frozenset(
            {
                "test_the_value_is_a_scalar",
                "test_the_gradient_reaches_the_declared_inputs",
            },
        ),
    ),
    (
        "detached-zero",
        fixture_for(DetachedZero(), live_mean),
        frozenset({"test_an_inapplicable_term_is_a_connected_zero"}),
    ),
    (
        "jittery",
        fixture_for(Jittery(), live_mean),
        frozenset({"test_the_same_seed_gives_the_same_value"}),
    ),
    (
        "modeling",
        fixture_for(Modeling(), live_mean),
        frozenset({"test_no_didactic_model_is_built"}),
    ),
)


def test_every_law_holds_of_an_objective_that_means_over_its_live_spans() -> None:
    assert red(fixture_for(SpanMean(), live_mean)) == frozenset()


@pytest.mark.parametrize(
    ("name", "conformance", "expected"),
    _BROKEN,
    ids=[name for name, _, _ in _BROKEN],
)
def test_a_defect_turns_exactly_the_laws_it_breaks_red(
    name: str,
    conformance: ObjectiveFixture[Batch],
    expected: frozenset[str],
) -> None:
    del name

    assert red(conformance) == expected


def test_reading_an_undeclared_field_turns_the_declaration_law_red() -> None:
    conformance = fixture_for(Peeking(), live_mean)

    with pytest.raises(LawViolationError, match="reads batch"):
        DeclaredMonotone().test_only_the_declared_outputs_are_read(conformance)


def test_every_method_of_the_suite_is_turned_red_by_some_subject() -> None:
    covered = frozenset({UNDER_THE_PROXY}).union(
        *(expected for _, _, expected in _BROKEN)
    )

    assert covered == {law.__name__ for law in OBJECTIVE_CONFORMANCE_LAWS}


def test_the_suite_binds_a_method_to_every_objective_law() -> None:
    bound = {
        method.__name__.removeprefix("test_") for method in OBJECTIVE_CONFORMANCE_LAWS
    }

    assert bound == {law.__name__ for law in OBJECTIVE_LAWS}


def test_a_subclass_stating_no_monotonicity_is_refused_where_it_is_defined() -> None:
    with pytest.raises(TypeError, match="monotonicity hook"):
        type("Unstated", (ObjectiveLawSuite,), {})


def test_a_subclass_inheriting_a_stated_monotonicity_is_accepted() -> None:
    inheriting = type("Inheriting", (DeclaredMonotone,), {})

    assert issubclass(inheriting, ObjectiveLawSuite)


def test_the_hook_the_mixin_declares_stands_in_for_no_ordering() -> None:
    hook = ObjectiveLawSuite.test_the_value_is_monotone_where_the_fixture_says

    with pytest.raises(NotImplementedError, match="per-objective"):
        hook(ObjectiveLawSuite(), fixture_for(SpanMean(), live_mean))


def test_the_model_counter_reports_what_a_call_constructed() -> None:
    scores = bundle().chart.inside.tensor

    def builds_one() -> LossTerm:
        return LossTerm(-scores.sum() * Coordinate(span=1).span, ITEMS)

    assert models_built(lambda: LossTerm(-scores.sum(), ITEMS)) == 0
    assert models_built(builds_one) == 1


def test_a_shorter_item_leaves_the_spans_past_it_as_padding() -> None:
    published = bundle().chart.inside

    assert published.mask is not None
    assert int(published.mask[0].sum()) == SPANS.num_spans
    assert int(published.mask[2].sum()) == 1
    assert float(published.tensor[2, SPANS.span_id(0, LENGTH)].detach()) == PADDING
