from typing import TYPE_CHECKING, Final, cast, override

import didactic.api as dx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.objective import ObjectiveLawSuite

import lofi.nn.loss._scale as scale_module
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._scale import ChartScaleObjective
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.spec.losses.chart_scale import ChartScaleLossSpec, ScalePenaltyName

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import (
        LossTerm,
        ProofRewardBindings,
        SpanTokenDecoder,
    )
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec

type Batch = tuple[str, ...]
type Outputs = ObjectiveOutputs[Batch]

LENGTH: Final = 3
ITEMS: Final = 3
LENGTHS: Final = (3, 2, 1)
WIDTH: Final = 4
SPANS: Final = SpanIndex(LENGTH)
BATCH: Final[Batch] = ("first", "second", "third")
PADDING: Final = -9.0
EXAMPLES: Final = 50
TARGET: Final = 1.5


def regularizer(
    target: float = TARGET,
    *,
    over: str = "width",
    name: str = "chart-scale",
    penalty: ScalePenaltyName = ScalePenaltyName.SQUARED_RATIO,
) -> ChartScaleObjective[Batch]:
    """Return a chart-scale objective under explicit runtime settings."""
    return ChartScaleObjective(target=target, over=over, name=name, penalty=penalty)


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
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def cell_tensor(scale: float = 1.0, fill: float = PADDING) -> Tensor:
    """Return cell vectors whose live positions scale and whose padding is fill.

    Parameters
    ----------
    scale : float
        What every live coordinate is multiplied by.
    fill : float
        What every padded coordinate holds.

    Returns
    -------
    torch.Tensor
        Rank 3 over ``("item", "span", "width")`` at double precision, a leaf requiring
        a gradient.
    """
    steps = torch.arange(ITEMS * SPANS.num_spans * WIDTH, dtype=torch.float64)
    values = (0.5 + steps.reshape(ITEMS, SPANS.num_spans, WIDTH) / 40.0) * scale

    return torch.where(LIVE.unsqueeze(-1), values, fill).detach().requires_grad_()


def cells(scale: float = 1.0, fill: float = PADDING) -> Encoded:
    """Return the encoded cells of a step.

    Parameters
    ----------
    scale : float
        What every live coordinate is multiplied by.
    fill : float
        What every padded coordinate holds.

    Returns
    -------
    Encoded
        Rank 3 over ``("item", "span", "width")``, masked over the spans each item
        really has.
    """
    return Encoded(cell_tensor(scale, fill), ("item", "span", "width"), LIVE)


def chart() -> CellChart:
    """Return an inside chart with the bundle's cell shape.

    Returns
    -------
    CellChart
        Rank 2 over ``("item", "span")``, masked over the live spans, with no outside
        pass.
    """
    scores = torch.where(LIVE, -1.0, PADDING).to(torch.float64)

    return CellChart(LogDistribution(scores, SPAN_AXES, LIVE))


def bundle(scale: float = 1.0, fill: float = PADDING) -> Outputs:
    """Return a step publishing cells, a chart, the spans and the batch.

    Parameters
    ----------
    scale : float
        What every live coordinate is multiplied by.
    fill : float
        What every padded coordinate holds.

    Returns
    -------
    ObjectiveOutputs
        The bundle the objective is run against.
    """
    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=cells(scale, fill),
    )


def at_target(target: float = TARGET) -> Outputs:
    """Return a step whose every live cell has exactly the target norm.

    Parameters
    ----------
    target : float
        The norm every live cell is given.

    Returns
    -------
    ObjectiveOutputs
        The bundle, whose padded coordinates hold another number entirely.
    """
    one_hot = torch.zeros(ITEMS, SPANS.num_spans, WIDTH, dtype=torch.float64)
    one_hot[:, :, 0] = target
    values = torch.where(LIVE.unsqueeze(-1), one_hot, PADDING)

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=Encoded(
            values.detach().requires_grad_(),
            ("item", "span", "width"),
            LIVE,
        ),
    )


def empty_bundle() -> Outputs:
    """Return a step whose cell mask excludes every position.

    Returns
    -------
    ObjectiveOutputs
        The bundle, which the objective does not apply to.
    """
    blank = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)
    values = torch.full(
        (ITEMS, SPANS.num_spans, WIDTH),
        PADDING,
        dtype=torch.float64,
    ).requires_grad_()

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=Encoded(values, ("item", "span", "width"), blank),
    )


def permuted(outputs: Outputs) -> Outputs:
    """Return the same step with its items in the reverse order.

    Parameters
    ----------
    outputs : Outputs
        The bundle to reorder.

    Returns
    -------
    ObjectiveOutputs
        A bundle over the same items, ordered last to first.
    """
    order = torch.tensor([2, 1, 0])
    published = outputs.cells

    assert published is not None
    assert published.mask is not None

    moved = published.tensor[order].detach().requires_grad_()

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=tuple(reversed(BATCH)),
        cells=Encoded(moved, published.axes, published.mask[order]),
    )


def refilled(outputs: Outputs) -> Outputs:
    """Return the same step with the padding of its cells written over.

    Parameters
    ----------
    outputs : Outputs
        The bundle with only its padding replaced.

    Returns
    -------
    ObjectiveOutputs
        A bundle whose live cells are those of the original and whose padded coordinates
        hold another number.
    """
    del outputs

    return bundle(fill=PADDING / 3.0)


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    """Return the leaf a gradient must reach.

    Parameters
    ----------
    outputs : Outputs
        The bundle the term was computed from.

    Returns
    -------
    tuple of torch.Tensor
        The cell tensor, which is the objective's one declared input.
    """
    published = outputs.cells

    assert published is not None

    return (published.tensor,)


def mean_departure(outputs: Outputs, target: float = TARGET) -> float:
    """Return the quantity the objective reports, summed in Python.

    Parameters
    ----------
    outputs : Outputs
        The bundle to score.
    target : float
        The norm the objective was built with.

    Returns
    -------
    float
        The mean over live cells of the squared difference between a cell's squared norm
        and the target's, in units of the target's.
    """
    published = outputs.cells

    assert published is not None

    values = published.tensor.detach()
    mask = published.mask
    total = 0.0
    count = 0

    for item in range(ITEMS):
        for span in range(SPANS.num_spans):
            if mask is not None and not bool(mask[item, span]):
                continue

            squares = sum(
                float(values[item, span, position]) ** 2 for position in range(WIDTH)
            )
            total += (squares / (target * target) - 1.0) ** 2
            count += 1

    return total / count if count else 0.0


def mean_norm(outputs: Outputs) -> float:
    """Return the root mean square norm of the live cells, summed in Python.

    Parameters
    ----------
    outputs : Outputs
        The bundle to read.

    Returns
    -------
    float
        The square root of the mean squared norm over the live cells.
    """
    published = outputs.cells

    assert published is not None

    values = published.tensor.detach()
    mask = published.mask
    total = 0.0
    count = 0

    for item in range(ITEMS):
        for span in range(SPANS.num_spans):
            if mask is not None and not bool(mask[item, span]):
                continue

            total += sum(
                float(values[item, span, position]) ** 2 for position in range(WIDTH)
            )
            count += 1

    return (total / count) ** 0.5 if count else 0.0


def models_built(call: Callable[[], LossTerm]) -> int:
    """Return how many didactic models one call constructed.

    Parameters
    ----------
    call : Callable[[], LossTerm]
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


def fixture() -> ObjectiveFixture[Batch]:
    """Return everything the objective laws need to run against the term.

    Returns
    -------
    ObjectiveFixture
        The objective, the bundles it is run against and the reckoning of what it should
        report.
    """
    return ObjectiveFixture(
        objective=regularizer(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=mean_departure,
        monotone=lambda: (bundle(scale=2.0), bundle()),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
    )


class TestChartScaleObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        """Return the fixture the laws run against.

        Returns
        -------
        ObjectiveFixture
            The regularizer and the bundles it is scored on.
        """
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that scaling every cell up scores above leaving it alone.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the scaled batch does not score above the unscaled one.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_term_is_exactly_zero_where_every_cell_sits_at_the_target() -> None:
    objective = regularizer()

    term = objective.evaluate(at_target())

    assert term.value.item() == 0.0
    assert term.count == int(LIVE.sum())


def test_the_term_is_exactly_zero_at_a_target_the_cells_were_built_for() -> None:
    objective = regularizer(0.25)

    assert objective.evaluate(at_target(0.25)).value.item() == 0.0


def test_a_cell_off_the_target_moves_the_term_off_zero() -> None:
    objective = regularizer()
    outputs = at_target()
    published = outputs.cells

    assert published is not None

    moved = published.tensor.detach().clone()
    moved[0, 0, 0] = TARGET * 2.0
    nudged = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=Encoded(moved.requires_grad_(), published.axes, published.mask),
    )

    term = objective.evaluate(nudged)

    assert term.value.item() == pytest.approx(9.0 / int(LIVE.sum()))


def test_each_declared_penalty_form_changes_an_off_target_loss() -> None:
    outputs = scaled(2.0)
    values = {
        penalty: regularizer(penalty=penalty).evaluate(outputs).value.item()
        for penalty in ScalePenaltyName
    }

    assert len(set(values.values())) == len(ScalePenaltyName)


def scaled(scale: float) -> Outputs:
    """Return a step whose cells sit at the target norm times a scale.

    Parameters
    ----------
    scale : float
        The multiplier for each live cell's norm.

    Returns
    -------
    ObjectiveOutputs
        The bundle, whose live cells all have norm ``scale`` times the target.
    """
    outputs = at_target()
    published = outputs.cells

    assert published is not None

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=Encoded(
            (published.tensor.detach() * scale).requires_grad_(),
            published.axes,
            published.mask,
        ),
    )


@settings(max_examples=EXAMPLES)
@given(
    scale=st.floats(min_value=1.0, max_value=3.0),
    gap=st.floats(min_value=0.05, max_value=1.0),
)
def test_the_term_grows_with_the_departure_across_generated_scalings(
    scale: float,
    gap: float,
) -> None:
    objective = regularizer()

    near = objective.evaluate(scaled(scale)).value.item()
    far = objective.evaluate(scaled(scale + gap)).value.item()

    assert far > near


def test_the_term_matches_a_reckoning_made_one_coordinate_at_a_time() -> None:
    objective = regularizer()
    outputs = bundle()

    term = objective.evaluate(outputs)

    assert term.value.item() == pytest.approx(mean_departure(outputs))


def test_the_term_reports_the_root_mean_square_norm_of_the_live_cells() -> None:
    objective = regularizer()
    outputs = bundle()

    term = objective.evaluate(outputs)

    assert set(term.diagnostics) == {"norm"}
    assert term.diagnostics["norm"].item() == pytest.approx(mean_norm(outputs))


def test_the_padding_reaches_neither_the_term_nor_the_norm_it_reports() -> None:
    objective = regularizer()

    term = objective.evaluate(bundle())
    refilled_term = objective.evaluate(refilled(bundle()))

    assert refilled_term.value.item() == pytest.approx(term.value.item())
    assert refilled_term.diagnostics["norm"].item() == pytest.approx(
        term.diagnostics["norm"].item()
    )


def test_an_unmasked_bundle_counts_every_cell() -> None:
    objective = regularizer()
    values = torch.full((2, 3, WIDTH), 0.5, dtype=torch.float64)
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=Encoded(values, ("item", "span", "width")),
    )

    term = objective.evaluate(outputs)

    assert term.count == 6
    assert term.value.item() == pytest.approx((1.0 / (TARGET * TARGET) - 1.0) ** 2)


def test_a_mask_over_the_items_alone_covers_every_span_beneath_it() -> None:
    objective = regularizer()
    values = torch.full((3, 2, WIDTH), 0.5, dtype=torch.float64)
    values[2] = 4.0
    items = torch.tensor([True, True, False])
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=BATCH,
        cells=Encoded(values, ("item", "span", "width"), items),
    )

    term = objective.evaluate(outputs)

    assert term.count == 4
    assert term.value.item() == pytest.approx((1.0 / (TARGET * TARGET) - 1.0) ** 2)
    assert term.diagnostics["norm"].item() == pytest.approx(1.0)


def test_a_step_with_no_live_cell_is_inapplicable() -> None:
    objective = regularizer()

    term = objective.evaluate(empty_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


def test_a_step_publishing_no_cell_is_refused_by_name() -> None:
    objective = regularizer()
    outputs = ObjectiveOutputs(spans=SPANS, chart=chart(), batch=BATCH)

    with pytest.raises(MissingOutputError, match="chart-scale reads cells"):
        objective.evaluate(outputs)


def test_the_axis_a_cell_runs_along_is_read_by_name() -> None:
    objective = regularizer(over="feature")

    with pytest.raises(ValueError, match="no axis named 'feature'"):
        objective.evaluate(bundle())


def test_an_axis_the_mask_covers_is_not_a_cell_s_features() -> None:
    objective = regularizer(over="span")

    with pytest.raises(ValueError, match="addresses the batch"):
        objective.evaluate(bundle())


@pytest.mark.parametrize(
    ("target", "raised", "message"),
    [
        (0.0, ValueError, "positive length"),
        (-1.0, ValueError, "positive length"),
        (float("inf"), ValueError, "finite"),
        (float("nan"), ValueError, "finite"),
        (True, TypeError, "not a number"),
    ],
)
def test_a_target_no_cell_could_sit_at_is_refused(
    target: float,
    raised: type[Exception],
    message: str,
) -> None:
    with pytest.raises(raised, match=message):
        regularizer(target)


@pytest.mark.parametrize(
    "build",
    [
        lambda: regularizer(1.0, over=""),
        lambda: regularizer(1.0, name=""),
    ],
    ids=["over", "name"],
)
def test_an_empty_name_is_refused(
    build: Callable[[], ChartScaleObjective[Batch]],
) -> None:
    with pytest.raises(ValueError, match="empty"):
        build()


def test_the_objective_publishes_what_it_reads_and_how_it_was_built() -> None:
    objective = regularizer(name="scale")

    assert objective.name == "scale"
    assert objective.requires == frozenset({"cells"})
    assert objective.target == TARGET
    assert objective.over == "width"


def test_a_suite_leaving_the_regularizer_off_takes_nothing_from_it() -> None:
    outputs = bundle()
    objective = regularizer()
    off = ObjectiveSuite([(0.0, objective)], published=outputs.published)
    on = ObjectiveSuite([(1.0, objective)], published=outputs.published)

    silent = off.evaluate(outputs)
    heard = on.evaluate(outputs)

    assert silent.total.item() == 0.0
    assert heard.total.item() == pytest.approx(mean_departure(outputs))
    assert silent.terms["chart-scale"].value.item() == pytest.approx(
        heard.terms["chart-scale"].value.item()
    )
    assert set(silent.diagnostics) == {"chart-scale/loss", "chart-scale/norm"}


def test_the_regularizer_renders_its_target_and_its_axis() -> None:
    objective = regularizer()

    assert repr(objective) == (
        "ChartScaleObjective(name='chart-scale', target=1.5, over='width', "
        "penalty=<ScalePenaltyName.SQUARED_RATIO: 'squared-ratio'>)"
    )


class NoBindings:
    """A binding context whose every lookup fails, for an objective reading none."""

    @property
    def name(self) -> str:
        return "regularized-scale"

    def terms_readout(self, role: str, /) -> TermsReadout:
        msg = f"the regularizer must not resolve term readout {role!r}"
        raise AssertionError(msg)

    def space(self, role: str, /) -> Space:
        msg = f"the regularizer must not resolve represented space {role!r}"
        raise AssertionError(msg)

    def readout_space(self, role: str, /) -> Space:
        msg = f"the regularizer must not resolve a readout space for {role!r}"
        raise AssertionError(msg)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        msg = f"the regularizer must not resolve space relation {role!r}"
        raise AssertionError(msg)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        msg = f"the regularizer must not resolve span decoder {role!r}"
        raise AssertionError(msg)

    def proof_reward(self) -> ProofRewardBindings:
        msg = "the regularizer must not resolve proof-reward bindings"
        raise AssertionError(msg)


def configured(target: float = TARGET) -> ChartScaleObjective[Batch]:
    """Return the regularizer the installed plugin builds from a declaration.

    Parameters
    ----------
    target : float
        The declared target norm.
    """
    return scale_module.PLUGIN.construct(
        ChartScaleLossSpec(target=target), NoBindings()
    )


class TestConfiguredChartScaleLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        """Return the laws' fixture over the factory-built regularizer."""
        return fixture()._replace(objective=configured())

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that scaling every cell up scores above leaving it alone.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the laws run against.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestChartScalePlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        def observe(built: object) -> dict[str, bool]:
            held = cast("ChartScaleObjective[Batch]", built)

            return {
                "named for its term": held.name == "regularized-scale",
                "holds the declared target": held.target == 2.5,
                "runs along the cell width": held.over == "width",
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", ChartScaleLossSpec(target=2.5)),
            context=NoBindings(),
            expected_type=ChartScaleObjective,
            observe=observe,
            unchanged=lambda: True,
        )


def test_the_declared_default_target_reaches_the_built_regularizer() -> None:
    built = scale_module.PLUGIN.construct(ChartScaleLossSpec(), NoBindings())

    assert built.target == 1.0


def test_the_plugin_builds_the_regularizer_under_its_term_and_declared_target() -> None:
    built = configured(2.5)
    outputs = bundle()

    assert built.name == "regularized-scale"
    assert built.target == 2.5
    assert built.over == "width"
    assert built.requires == frozenset({"cells"})
    assert built.evaluate(outputs).value.item() == pytest.approx(
        regularizer(2.5).evaluate(outputs).value.item(),
        rel=1e-12,
    )


def test_the_plugin_preserves_the_declared_penalty_form() -> None:
    declaration = ChartScaleLossSpec.model_validate(
        {"target": 2.5, "penalty": "absolute-distance"}
    )
    built = scale_module.PLUGIN.construct(declaration, NoBindings())

    assert built.penalty is ScalePenaltyName.ABSOLUTE_DISTANCE
