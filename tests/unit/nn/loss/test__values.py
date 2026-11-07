import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from tests.language import language

import lofi.nn.loss._values as loss_values
from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.judge import Action
from lofi.lf import THREE_WAY, InferenceCell
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import CellMass, EvidenceLedger, GradientPath, GroundedBatch
from lofi.nn.loss._errors import (
    MissingOutputError,
    UncomputedCellError,
    UndeclaredOutputError,
)
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
    SplitPosterior,
)
from lofi.nn.loss._ports import LossTerm, Objective, PublishedOutputs
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    OUTPUT_FIELDS,
    OUTPUT_NAMES,
    OUTSIDE_CHART,
    CellChart,
    ObjectiveOutputs,
    RecordingOutputs,
    SpanReadout,
    SpanTerms,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.nn._values import DerivationScoreChart


LENGTH = 3
SPANS = SpanIndex(LENGTH)
ITEMS = 2


LANGUAGE = language(("bo",))


def scored_forms(mass: float = 0.5) -> WeightedTerms:
    """Return weighted forms holding one constant at the given mass."""
    return WeightedTerms(
        (LANGUAGE.constant("bo"),),
        torch.tensor([mass], dtype=torch.float64).log(),
        torch.tensor(1.0 - mass, dtype=torch.float64).log(),
        Estimate.TRUNCATED,
    )


def grounded() -> GroundedBatch:
    """Return a complete grounded distribution over both items."""
    log_probabilities = torch.zeros(ITEMS, len(THREE_WAY.labels)).log_softmax(dim=1)

    return GroundedBatch(
        LogDistribution(
            log_probabilities,
            ("pair", "label"),
            normalized_over="label",
        ),
        labels=THREE_WAY.labels,
        complete=True,
        residual=torch.zeros(ITEMS),
        decided=torch.ones(ITEMS, dtype=torch.bool),
        cells=CellMass(
            torch.full(
                (ITEMS, len(InferenceCell)),
                1.0 / len(InferenceCell),
            )
        ),
        gradient_path=GradientPath.PATHWISE,
        baseline=None,
        auxiliary=None,
        ledger=EvidenceLedger.empty(),
        variance=None,
        reward=None,
    )


class ReadingForms:
    """An objective that declares the scored forms and reads nothing else."""

    name = "inference"
    requires = frozenset({"forms"})

    def evaluate(self, outputs: RecordingOutputs[tuple[str, ...]], /) -> LossTerm:
        weights = torch.cat([beam.log_weights for beam in outputs.forms.values()])

        return LossTerm(-weights.sum(), len(outputs.forms))


def reading_forms() -> Objective[PublishedOutputs[tuple[str, ...]]]:
    """Return the objective as the suite takes it."""
    return cast("Objective[PublishedOutputs[tuple[str, ...]]]", ReadingForms())


def cutpoint_mask() -> torch.Tensor:
    valid = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.bool)

    for i, j in SPANS.spans_by_length(min_length=2):
        for k in SPANS.splits(i, j):
            span, column = SPANS.split_id(i, j, k)
            valid[:, span, column] = True

    return valid


def span_scores(fill: float = -1.0) -> LogDistribution:
    return LogDistribution(torch.full((ITEMS, SPANS.num_spans), fill), SPAN_AXES)


def span_readout(fill: float = -1.0) -> SpanReadout:
    return SpanReadout(SPANS, span_scores(fill))


def split_posterior() -> SplitPosterior:
    valid = cutpoint_mask()
    scores = torch.where(valid, torch.zeros_like(valid, dtype=torch.float), -torch.inf)
    normalized = torch.where(
        valid, scores.log_softmax(dim=2), scores.new_full((), -torch.inf)
    )

    return SplitPosterior(
        SPANS,
        LogDistribution(normalized, SPLIT_AXES, valid, normalized_over="split"),
    )


def span_marginals() -> SpanMarginals:
    posterior = split_posterior()
    valid = posterior.valid
    weights = torch.where(valid, posterior.log_probabilities.tensor.exp(), 0.0)
    constituent = torch.zeros(ITEMS, SPANS.num_spans)
    rule = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits)
    constituent[:, SPANS.span_id(0, LENGTH)] = 1.0

    for i, j in SPANS.spans_by_length_desc(min_length=2):
        span = SPANS.span_id(i, j)

        for k in SPANS.splits(i, j):
            _, column = SPANS.split_id(i, j, k)
            share = constituent[:, span] * weights[:, span, column]
            rule[:, span, column] = share

            for start, end in ((i, k), (k, j)):
                if end - start > 1:
                    constituent[:, SPANS.span_id(start, end)] += share

    return SpanMarginals(
        SPANS,
        Encoded(constituent, SPAN_AXES, valid.any(dim=2)),
        Encoded(rule, SPLIT_AXES, valid),
        torch.full((ITEMS,), LENGTH),
    )


def bundle(
    *,
    chart: CellChart | None = None,
    splits: SplitPosterior | None = None,
    actions: Mapping[Action, LogDistribution] | None = None,
    readouts: Mapping[ReadoutId, SpanReadout] | None = None,
    forms: Mapping[ReadoutId, WeightedTerms] | None = None,
) -> ObjectiveOutputs[tuple[str, ...]]:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(span_scores()) if chart is None else chart,
        batch=("first", "second"),
        splits=splits,
        actions=actions,
        readouts=readouts,
        forms=forms,
    )


def test_the_vocabulary_is_what_a_bundle_carrying_everything_publishes() -> None:
    filled = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(span_scores(-1.0), span_scores(-2.0)),
        batch=("first", "second"),
        splits=split_posterior(),
        scores=LogDistribution(torch.full((ITEMS,), -3.0), ("item",)),
        marginals=span_marginals(),
        actions={Action("apply"): span_scores(-4.0)},
        readouts={ReadoutId("span-head"): span_readout(-5.0)},
        choice_rows={ReadoutId("span-choices"): span_readout(-5.5)},
        terms={ReadoutId("cell-terms"): SpanTerms(SPANS, {(0, 0): scored_forms()})},
        forms={ReadoutId("root-forms"): scored_forms()},
        entailment=grounded(),
        cells=Encoded(torch.zeros(ITEMS, SPANS.num_spans, 4), (*SPAN_AXES, "width")),
        outside_cells=Encoded(
            torch.zeros(ITEMS, SPANS.num_spans, 4), (*SPAN_AXES, "width")
        ),
        entropy=Encoded(torch.zeros(ITEMS), ("item",)),
        mbr=Encoded(torch.zeros(ITEMS, SPANS.num_spans), SPAN_AXES),
        derivations=cast("DerivationScoreChart[object]", object()),
    )

    assert filled.published == OUTPUT_NAMES
    assert frozenset(OUTPUT_FIELDS) < OUTPUT_NAMES
    assert OUTSIDE_CHART not in OUTPUT_FIELDS
    assert "forms" in OUTPUT_FIELDS


def test_a_chart_names_its_two_passes() -> None:
    inside = span_scores(-1.0)
    outside = span_scores(-2.0)

    chart = CellChart(inside, outside)

    assert chart.inside is inside
    assert chart.outside is outside


def test_a_chart_without_an_outside_pass_says_so() -> None:
    assert CellChart(span_scores()).outside is None


def test_the_debug_checks_reject_two_passes_over_different_axes(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_values, on=True)
    other = LogDistribution(
        torch.full((ITEMS, SPANS.num_spans), -1.0), ("item", "cell")
    )

    with pytest.raises(ValueError, match="outside pass is over the axes"):
        loss_values.CellChart(span_scores(), other)


def test_a_bundle_reads_back_what_the_graph_published() -> None:
    spans = SPANS
    chart = CellChart(span_scores())
    splits = split_posterior()
    marginals = span_marginals()
    cells = Encoded(torch.zeros(ITEMS, SPANS.num_spans, 4), ("item", "span", "width"))
    outside_cells = Encoded(
        torch.ones(ITEMS, SPANS.num_spans, 4), ("item", "span", "width")
    )
    action = Action("forward_application", "fa")
    weights = span_scores(-3.0)
    readout = ReadoutId("types")

    outputs = ObjectiveOutputs(
        spans=spans,
        chart=chart,
        batch=("first", "second"),
        splits=splits,
        scores=LogDistribution(torch.full((ITEMS,), -4.0), ("item",)),
        marginals=marginals,
        actions={action: weights},
        readouts={readout: span_readout(-5.0)},
        cells=cells,
        outside_cells=outside_cells,
        truncated=((), (1, 2)),
    )

    assert outputs.spans is spans
    assert outputs.chart is chart
    assert outputs.splits is splits
    assert outputs.marginals is marginals
    assert outputs.cells is cells
    assert outputs.outside_cells is outside_cells
    assert outputs.batch == ("first", "second")
    assert outputs.scores is not None
    assert outputs.scores.tensor[0].item() == -4.0
    assert outputs.actions[Action("forward_application", "fa")] is weights
    assert outputs.readouts[readout].at(0, 0, 1).tensor.item() == -5.0
    assert outputs.truncated == ((), (1, 2))
    assert bundle().truncated == ()
    assert not hasattr(outputs, "__dict__")


def test_a_bundle_publishes_the_names_it_carries_a_value_for() -> None:
    assert bundle().published == frozenset({"spans", "chart", "batch"})


def test_a_bundle_and_recording_view_publish_derivation_scoring() -> None:
    scoring = cast("DerivationScoreChart[object]", object())
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(span_scores()),
        batch=("first", "second"),
        derivations=scoring,
    )
    view = RecordingOutputs(
        outputs,
        objective="derivational",
        requires=frozenset({"derivations"}),
    )

    assert outputs.derivations is scoring
    assert "derivations" in outputs.published
    assert view.derivations is scoring
    assert view.touched == frozenset({"derivations"})


def test_contextual_outside_vectors_have_their_own_publication_name() -> None:
    outside = Encoded(torch.ones(ITEMS, SPANS.num_spans, 4), ("item", "span", "width"))
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(span_scores()),
        batch=("first", "second"),
        outside_cells=outside,
    )
    view = RecordingOutputs(
        outputs,
        objective="outside-reconstruction",
        requires=frozenset({"outside_cells"}),
    )

    assert outputs.published == frozenset({"spans", "chart", "outside_cells", "batch"})
    assert view.outside_cells is outside
    assert view.touched == frozenset({"outside_cells"})


def test_a_bundle_reads_back_the_scored_forms_by_readout_id() -> None:
    beam = scored_forms()
    readout = ReadoutId("root-forms")

    outputs = bundle(forms={readout: beam})

    assert outputs.forms[readout] is beam
    assert "forms" in outputs.published
    assert "forms" not in bundle(forms={}).published
    assert "forms" not in bundle().published


def test_the_scored_forms_mapping_is_a_snapshot_rather_than_the_callers() -> None:
    readout = ReadoutId("root-forms")
    source = {readout: scored_forms()}

    outputs = bundle(forms=source)
    source[ReadoutId("other")] = scored_forms(0.25)

    assert len(outputs.forms) == 1

    with pytest.raises(TypeError, match="does not support item assignment"):
        cast("dict[ReadoutId, WeightedTerms]", outputs.forms)[readout] = scored_forms()


def test_the_debug_checks_reject_an_empty_readout_id_on_the_scored_forms(
    reload_with_checks: Callable[..., None],
) -> None:
    reload_with_checks(loss_values, on=True)

    with pytest.raises(ValueError, match="readout id"):
        loss_values.ObjectiveOutputs(
            spans=SPANS,
            chart=loss_values.CellChart(span_scores()),
            batch=("first", "second"),
            forms={ReadoutId(""): scored_forms()},
        )


def test_an_objective_reading_the_scored_forms_sees_them_through_the_proxy() -> None:
    outputs = bundle(forms={ReadoutId("root-forms"): scored_forms()})
    view = RecordingOutputs(
        outputs, objective="inference", requires=ReadingForms().requires
    )

    term = ReadingForms().evaluate(view)

    assert view.touched == frozenset({"forms"})
    assert float(term.value) == pytest.approx(-math.log(0.5))
    assert term.count == 1


def test_an_objective_declaring_forms_against_a_graph_publishing_none_is_refused() -> (
    None
):
    with pytest.raises(MissingOutputError) as raised:
        ObjectiveSuite([(1.0, reading_forms())], published=bundle().published)

    assert raised.value.objective == "inference"
    assert raised.value.missing == frozenset({"forms"})


def test_an_objective_declaring_forms_against_a_graph_publishing_them_builds() -> None:
    published = bundle(forms={ReadoutId("root-forms"): scored_forms()}).published

    suite = ObjectiveSuite([(1.0, reading_forms())], published=published)

    assert suite.names == ("inference",)


def test_a_bundle_publishes_the_outside_pass_under_its_own_name() -> None:
    published = bundle(chart=CellChart(span_scores(), span_scores(-2.0))).published

    assert OUTSIDE_CHART in published
    assert "chart" in published


def test_an_empty_mapping_publishes_nothing() -> None:
    action = Action("identity")

    assert "actions" not in bundle(actions={}).published
    assert "actions" in bundle(actions={action: span_scores()}).published


def test_the_mappings_are_snapshots_rather_than_the_callers() -> None:
    action = Action("identity")
    source = {action: span_scores()}

    outputs = bundle(actions=source)
    source[Action("forward_application", "fa")] = span_scores(-2.0)

    assert len(outputs.actions) == 1
    assert action in outputs.actions


def checked_bundle(
    reload_with_checks: Callable[..., None],
    *,
    readouts: Mapping[ReadoutId, SpanReadout] | None = None,
) -> ObjectiveOutputs[tuple[str, ...]]:
    reload_with_checks(loss_values, on=True)

    return loss_values.ObjectiveOutputs(
        spans=SPANS,
        chart=loss_values.CellChart(span_scores()),
        batch=("first", "second"),
        readouts=readouts,
    )


def test_the_debug_checks_reject_an_empty_readout_id(
    reload_with_checks: Callable[..., None],
) -> None:
    with pytest.raises(ValueError, match="readout id"):
        checked_bundle(reload_with_checks, readouts={ReadoutId(""): span_readout()})


def test_the_debug_checks_reject_a_readout_over_another_span_addressing(
    reload_with_checks: Callable[..., None],
) -> None:
    other = SpanIndex(LENGTH + 1)
    readout = SpanReadout(
        other, LogDistribution(torch.full((ITEMS, other.num_spans), -1.0), SPAN_AXES)
    )

    with pytest.raises(ValueError, match="span-head"):
        checked_bundle(reload_with_checks, readouts={ReadoutId("span-head"): readout})


def test_a_readout_over_the_bundle_addressing_is_accepted_by_the_debug_checks(
    reload_with_checks: Callable[..., None],
) -> None:
    outputs = checked_bundle(
        reload_with_checks,
        readouts={
            ReadoutId("span-head"): SpanReadout(SpanIndex(LENGTH), span_scores())
        },
    )

    assert outputs.readouts[ReadoutId("span-head")].spans == outputs.spans


def test_a_proxy_hands_over_a_field_the_objective_declared() -> None:
    outputs = bundle(splits=split_posterior())

    view = RecordingOutputs(
        outputs,
        objective="coherence",
        requires=frozenset({"splits", "spans"}),
    )

    assert view.splits is outputs.splits
    assert view.spans is outputs.spans
    assert view.touched == frozenset({"splits", "spans"})


def test_a_proxy_records_nothing_until_something_is_read() -> None:
    view = RecordingOutputs(
        bundle(), objective="coherence", requires=frozenset({"chart"})
    )

    assert view.touched == frozenset()


def test_a_proxy_refuses_a_field_the_objective_did_not_declare() -> None:
    view = RecordingOutputs(
        bundle(), objective="coherence", requires=frozenset({"chart"})
    )

    with pytest.raises(UndeclaredOutputError) as raised:
        _ = view.batch

    assert raised.value.objective == "coherence"
    assert raised.value.field == "batch"
    assert view.touched == frozenset()


@pytest.mark.parametrize("field", [name for name in OUTPUT_FIELDS if name != "batch"])
def test_a_proxy_guards_every_field_of_the_bundle(field: str) -> None:
    view = RecordingOutputs(
        bundle(), objective="coherence", requires=frozenset({"batch"})
    )

    with pytest.raises(UndeclaredOutputError, match=field):
        _ = getattr(view, field)


def test_declaring_the_outside_pass_admits_the_chart_it_is_part_of() -> None:
    outputs = bundle(chart=CellChart(span_scores(), span_scores(-2.0)))

    view = RecordingOutputs(
        outputs,
        objective="reconstruction",
        requires=frozenset({OUTSIDE_CHART}),
    )

    assert view.chart.outside is not None
    assert view.touched == frozenset({"chart"})


def cell_distribution(mask: torch.Tensor | None = None) -> LogDistribution:
    """Return a per-cell distribution over two slots and three symbols."""
    logits = torch.arange(ITEMS * SPANS.num_spans * 2 * 3, dtype=torch.float64).reshape(
        ITEMS, SPANS.num_spans, 2, 3
    )
    scores = logits.log_softmax(dim=3)

    if mask is not None:
        widened = mask.reshape(tuple(mask.shape) + (1,) * (4 - mask.ndim))
        scores = torch.where(widened, scores, -torch.inf)

    return LogDistribution(
        scores,
        (*SPAN_AXES, "slot", "symbol"),
        mask,
        normalized_over="symbol",
    )


def test_a_readout_publishes_its_axes_and_reads_one_cell_back() -> None:
    readout = SpanReadout(SPANS, cell_distribution())

    cell = readout.at(1, 0, 2)

    assert readout.axes == ("item", "span", "slot", "symbol")
    assert readout.items == ITEMS
    assert readout.computed.shape == (ITEMS, SPANS.num_spans)
    assert bool(readout.computed.all())
    assert cell.axes == ("slot", "symbol")
    assert cell.normalized_over == "symbol"
    assert torch.equal(cell.tensor, readout.distribution.tensor[1, SPANS.span_id(0, 2)])


def test_a_readout_masked_over_the_items_marks_every_cell_of_a_masked_item() -> None:
    mask = torch.tensor([True, False])
    readout = SpanReadout(
        SPANS,
        cell_distribution(mask.reshape(ITEMS, 1).expand(ITEMS, SPANS.num_spans)),
    )

    assert bool(readout.computed[0].all())
    assert not bool(readout.computed[1].any())


def test_reading_a_cell_the_readout_never_computed_is_a_named_error() -> None:
    mask = torch.ones(ITEMS, SPANS.num_spans, dtype=torch.bool)
    mask[0, SPANS.span_id(1, 3)] = False
    readout = SpanReadout(SPANS, cell_distribution(mask))

    with pytest.raises(UncomputedCellError) as caught:
        readout.at(0, 1, 3)

    assert caught.value.item == 0
    assert caught.value.span == (1, 3)
    assert "[1, 3)" in str(caught.value)
    assert readout.at(1, 1, 3).axes == ("slot", "symbol")


@pytest.mark.parametrize("item", [-1, -ITEMS, ITEMS])
def test_a_readout_refuses_an_item_outside_the_batch(item: int) -> None:
    readout = SpanReadout(SPANS, cell_distribution())

    with pytest.raises(IndexError, match=f"no item {item}"):
        readout.at(item, 0, 1)


def test_a_readout_refuses_a_span_outside_the_input() -> None:
    readout = SpanReadout(SPANS, cell_distribution())

    with pytest.raises(ValueError, match="not a span"):
        readout.at(0, 2, 2)

    with pytest.raises(ValueError, match="not a span"):
        readout.at(0, 0, LENGTH + 1)


def test_a_gather_reads_several_cells_in_the_given_order() -> None:
    readout = SpanReadout(SPANS, cell_distribution())
    cells = [(1, 0, 2), (0, 0, 1), (1, 0, 2), (0, 1, 3)]

    rows = readout.gather(cells)

    assert rows.axes == ("cell", "slot", "symbol")
    assert rows.normalized_over == "symbol"
    assert rows.tensor.shape == (len(cells), 2, 3)
    assert all(
        torch.equal(rows.tensor[row], readout.at(item, i, j).tensor)
        for row, (item, i, j) in enumerate(cells)
    )


def test_a_gather_of_no_cells_is_empty() -> None:
    readout = SpanReadout(SPANS, cell_distribution())

    rows = readout.gather([])

    assert rows.axes == ("cell", "slot", "symbol")
    assert rows.tensor.shape == (0, 2, 3)


def test_a_gather_stays_attached_to_the_published_graph() -> None:
    scores = cell_distribution().tensor.detach().requires_grad_()
    readout = SpanReadout(
        SPANS, LogDistribution(scores, (*SPAN_AXES, "slot", "symbol"))
    )

    readout.gather([(0, 0, 1), (1, 1, 2)]).tensor.sum().backward()

    assert scores.grad is not None
    assert bool(scores.grad[0, SPANS.span_id(0, 1)].eq(1.0).all())
    assert bool(scores.grad[1, SPANS.span_id(1, 2)].eq(1.0).all())
    assert scores.grad.sum().item() == 2 * 2 * 3


def test_a_gather_names_the_first_uncomputed_cell_in_the_given_order() -> None:
    mask = torch.ones(ITEMS, SPANS.num_spans, dtype=torch.bool)
    mask[0, SPANS.span_id(1, 3)] = False
    mask[1, SPANS.span_id(0, 1)] = False
    readout = SpanReadout(SPANS, cell_distribution(mask))

    with pytest.raises(UncomputedCellError) as caught:
        readout.gather([(0, 0, 1), (1, 0, 1), (0, 1, 3)])

    assert caught.value.item == 1
    assert caught.value.span == (0, 1)
    assert readout.gather([(0, 0, 1), (1, 1, 3)]).tensor.shape == (2, 2, 3)


def test_a_gather_refuses_a_bad_address_before_reading_anything() -> None:
    readout = SpanReadout(SPANS, cell_distribution())

    with pytest.raises(IndexError, match=f"no item {ITEMS}"):
        readout.gather([(0, 0, 1), (ITEMS, 0, 1)])

    with pytest.raises(ValueError, match="not a span"):
        readout.gather([(0, 0, 1), (0, 0, LENGTH + 1)])


def test_a_readout_must_run_over_the_item_and_then_the_span() -> None:
    scores = torch.zeros(SPANS.num_spans, ITEMS, dtype=torch.float64)

    with pytest.raises(ValueError, match="outermost"):
        SpanReadout(SPANS, LogDistribution(scores, ("span", "item")))


def test_a_readout_must_hold_one_cell_per_span_of_the_index() -> None:
    scores = torch.zeros(ITEMS, SPANS.num_spans + 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="addresses"):
        SpanReadout(SPANS, LogDistribution(scores, SPAN_AXES))


def test_a_readout_refuses_a_mask_reaching_into_a_cell() -> None:
    mask = torch.ones(ITEMS, SPANS.num_spans, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="part of a cell"):
        SpanReadout(SPANS, cell_distribution(mask))


def test_a_readout_with_no_inner_axis_reads_a_scalar_cell() -> None:
    readout = span_readout(-2.5)

    cell = readout.at(0, 0, 1)

    assert cell.axes == ()
    assert cell.tensor.item() == -2.5
