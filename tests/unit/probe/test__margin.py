from typing import Final

import pytest
import torch
from tests._controllers import mlp_controller

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._controller_relations import ControllerRelation
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._margin import ActionMargin
from lofi.probe._marginals import SpanWeighting
from lofi.probe._values import Observation, Trace

WIRED: Final = ComponentId("chart-controller")
WEIGHING: Final = Site(WIRED, Role.COMBINATOR_LOGITS)
CELL: Final = Site(WIRED, Role.CHART_CELL, (0, 2))

SCORES: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
REAL: Final = SiteSpec(
    ("cell", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

LOGITS: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

# the derivation weighting of a two-cell reading, and of a one-cell reading
BELIEVED: Final = (0.25, 0.75)
ALONE: Final = (1.0,)


def weighed(
    marginals: tuple[float, ...] = BELIEVED, spec: SiteSpec = SCORES
) -> DiagnosticContext:
    weighting = SpanWeighting(torch.tensor(marginals), axes=("cell",))

    return DiagnosticContext({WEIGHING: spec, CELL: REAL}, weighting=weighting)


CONTEXT: Final = weighed()
UNNORMALIZED: Final = weighed(spec=LOGITS)


def a_trace(*readings: Observation) -> Trace:
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def weights(
    numbers: list[list[float]], mask: torch.Tensor | None = None
) -> Observation:
    return Observation(WEIGHING, torch.tensor(numbers), axes=SCORES.axes, mask=mask)


def test_the_margin_is_the_gap_between_the_two_best_weighted_actions() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])
    result = ActionMargin().run(a_trace(reading), context=CONTEXT)
    margin = result.aggregate(WEIGHING, "margin")

    # gaps 1.9 and 1.0: their mean, and their mean under weights of 0.25 and 0.75
    assert result.tag == "action-margin"
    assert margin.uniform == pytest.approx(1.45)
    assert margin.weighted == pytest.approx(0.25 * 1.9 + 0.75 * 1.0)
    assert margin.count == 2
    assert result.value(WEIGHING, "rows") == 2.0


def test_the_weighted_margin_is_not_the_uniform_one_reported_twice() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])
    believing_the_first = weighed((0.9, 0.1))
    believing_the_second = weighed((0.1, 0.9))
    first = ActionMargin().run(a_trace(reading), context=believing_the_first)
    second = ActionMargin().run(a_trace(reading), context=believing_the_second)

    assert first.aggregate(WEIGHING, "margin").weighted > 1.45
    assert second.aggregate(WEIGHING, "margin").weighted < 1.45
    assert first.aggregate(WEIGHING, "margin").uniform == pytest.approx(1.45)
    assert second.aggregate(WEIGHING, "margin").uniform == pytest.approx(1.45)


def test_a_context_without_a_weighting_cannot_report_the_margin() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])
    unweighted = DiagnosticContext({WEIGHING: SCORES, CELL: REAL})

    with pytest.raises(ValueError, match="no derivation weighting"):
        ActionMargin().run(a_trace(reading), context=unweighted)


def test_a_weighting_over_other_rows_than_the_reading_has_is_refused() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])

    with pytest.raises(ValueError, match="cannot weigh a reading of shape"):
        ActionMargin().run(a_trace(reading), context=weighed(ALONE))


def test_actions_weighed_alike_leave_no_margin() -> None:
    reading = weights([[-1.0, -1.0, -1.0]])
    result = ActionMargin().run(a_trace(reading), context=weighed(ALONE))

    assert result.aggregate(WEIGHING, "margin").uniform == pytest.approx(0.0)
    assert result.aggregate(WEIGHING, "margin").weighted == pytest.approx(0.0)


def test_the_margin_does_not_move_when_the_columns_are_permuted() -> None:
    ordered = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])
    permuted = weights([[-2.0, -0.1, -3.0], [-1.5, -4.0, -0.5]])
    first = ActionMargin().run(a_trace(ordered), context=CONTEXT)
    second = ActionMargin().run(a_trace(permuted), context=CONTEXT)

    assert first.aggregate(WEIGHING, "margin") == second.aggregate(WEIGHING, "margin")


def test_a_masked_off_pair_is_left_out_of_the_mean() -> None:
    reading = weights(
        [[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]], mask=torch.tensor([True, False])
    )
    result = ActionMargin().run(a_trace(reading), context=CONTEXT)
    margin = result.aggregate(WEIGHING, "margin")

    assert margin.uniform == pytest.approx(1.9)
    assert margin.weighted == pytest.approx(1.9)
    assert margin.count == 1
    assert result.value(WEIGHING, "rows") == 1.0


def test_a_reading_weighing_one_action_has_no_gap_to_report() -> None:
    reading = weights([[-0.1], [-0.4]])

    with pytest.raises(ValueError, match="no gap"):
        ActionMargin().run(a_trace(reading), context=CONTEXT)


def test_a_reading_whose_every_pair_is_masked_off_is_refused() -> None:
    reading = weights(
        [[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]], mask=torch.tensor([False, False])
    )

    with pytest.raises(ValueError, match="masked off"):
        ActionMargin().run(a_trace(reading), context=CONTEXT)


def test_a_mask_that_differs_from_one_action_to_the_next_is_refused() -> None:
    varying = torch.tensor([[True, True, False], [True, False, True]])
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]], mask=varying)

    with pytest.raises(ValueError, match="differs from one action"):
        ActionMargin().run(a_trace(reading), context=CONTEXT)


def test_a_reading_declaring_another_axis_name_names_what_it_does_declare() -> None:
    reading = Observation(WEIGHING, torch.zeros(2, 3), axes=("cell", "combinator"))

    with pytest.raises(KeyError, match="combinator"):
        ActionMargin().run(a_trace(reading), context=CONTEXT)


def test_the_axis_the_actions_run_along_is_the_one_the_diagnostic_was_given() -> None:
    reading = Observation(
        WEIGHING, torch.tensor([[-0.1, -3.0, -2.0]]), axes=("cell", "combinator")
    )
    result = ActionMargin("combinator").run(a_trace(reading), context=weighed(ALONE))

    assert ActionMargin("combinator").over == "combinator"
    assert result.aggregate(WEIGHING, "margin").uniform == pytest.approx(1.9)


def test_a_diagnosis_that_read_no_decision_point_says_so() -> None:
    reading = Observation(CELL, torch.zeros(2, 3), axes=("cell", "width"))

    with pytest.raises(ValueError, match="no decision to measure"):
        ActionMargin().run(a_trace(reading), context=CONTEXT)


def test_a_model_publishing_no_decision_point_leaves_nothing_to_measure() -> None:
    context = DiagnosticContext({CELL: REAL})
    reading = Observation(CELL, torch.zeros(2, 3), axes=("cell", "width"))
    result = ActionMargin().run(a_trace(reading), context=context)

    assert result.measurements == ()
    assert result.aggregations == ()


def test_the_diagnostic_requires_the_combinator_role_and_names_itself() -> None:
    assert ActionMargin.requires == frozenset({Role.COMBINATOR_LOGITS})
    assert ActionMargin.tag == "action-margin"
    assert repr(ActionMargin()) == "ActionMargin('action')"


def test_the_margin_reads_a_real_controller_through_its_published_map() -> None:
    join, keep, apply = Action("join", "concatenation"), Action("keep"), Action("apply")
    actions = ActionSet((join, keep, apply), keep)
    torch.manual_seed(5)
    controller = mlp_controller(actions, width=4, component=WIRED, hidden=6)

    cells = Encoded(torch.linspace(-1.0, 1.0, 12).reshape(3, 4), ("cell", "width"))
    scored = controller.weigh(cells, right=cells)
    believed = torch.tensor([0.2, 0.3, 0.5])
    context = DiagnosticContext.observing(
        controller,
        relations={
            WEIGHING: ControllerRelation(
                3,
                sequences=((0, 1, 2),),
                similar_inputs=((0, 2),),
            )
        },
        weighting=SpanWeighting(believed, axes=("cell",)),
    )
    trace = a_trace(Observation(WEIGHING, scored.tensor, axes=scored.axes))

    resolved = installed_diagnostics().matching(context)
    reported = {one.tag: one.run(trace, context=context) for one in resolved}

    ordered = scored.tensor.detach().sort(dim=1, descending=True).values
    gaps = ordered[:, 0] - ordered[:, 1]
    margin = reported["action-margin"].aggregate(WEIGHING, "margin")

    assert sorted(reported) == [
        "action-margin",
        "controller-choice",
        "controller-relations",
        "finite-readings",
    ]
    assert margin.uniform == pytest.approx(float(gaps.mean()))
    assert margin.weighted == pytest.approx(float((gaps * believed).sum()))
    assert reported["action-margin"].value(WEIGHING, "rows") == 3.0
    assert reported["finite-readings"].value(WEIGHING, "nonfinite") == 0.0


def test_a_pair_weighing_one_possible_action_is_counted_rather_than_averaged() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-0.5, -float("inf"), -float("inf")]])
    result = ActionMargin().run(a_trace(reading), context=CONTEXT)

    assert result.aggregate(WEIGHING, "margin").uniform == pytest.approx(1.9)
    assert result.aggregate(WEIGHING, "margin").weighted == pytest.approx(1.9)
    assert result.value(WEIGHING, "rows") == 1.0
    assert result.value(WEIGHING, "certain") == 1.0


def test_no_pair_ruled_out_leaves_the_count_at_zero() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])
    result = ActionMargin().run(a_trace(reading), context=CONTEXT)

    assert result.value(WEIGHING, "certain") == 0.0
    assert result.value(WEIGHING, "rows") == 2.0


def test_a_pair_the_mask_drops_is_not_counted_as_a_settled_decision() -> None:
    reading = weights(
        [[-0.1, -3.0, -2.0], [-0.5, -float("inf"), -float("inf")]],
        mask=torch.tensor([True, False]),
    )
    result = ActionMargin().run(a_trace(reading), context=CONTEXT)

    assert result.value(WEIGHING, "certain") == 0.0
    assert result.value(WEIGHING, "rows") == 1.0


def test_a_reading_whose_every_pair_weighs_one_possible_action_is_refused() -> None:
    reading = weights([[-0.5, -float("inf")], [-0.25, -float("inf")]])

    with pytest.raises(ValueError, match="one possible action"):
        ActionMargin().run(a_trace(reading), context=CONTEXT)


def test_a_pair_weighing_nothing_possible_is_counted_rather_than_averaged() -> None:
    nothing = -float("inf")
    reading = weights([[-0.1, -3.0, -2.0], [nothing, nothing, nothing]])
    result = ActionMargin().run(a_trace(reading), context=CONTEXT)

    assert result.aggregate(WEIGHING, "margin").uniform == pytest.approx(1.9)
    assert result.value(WEIGHING, "certain") == 1.0


def test_a_point_weighing_unnormalized_scores_reports_no_settled_count() -> None:
    reading = weights([[-0.1, -3.0, -2.0], [-4.0, -0.5, -1.5]])
    result = ActionMargin().run(a_trace(reading), context=UNNORMALIZED)

    assert result.aggregate(WEIGHING, "margin").uniform == pytest.approx(1.45)

    with pytest.raises(KeyError, match="certain"):
        result.value(WEIGHING, "certain")


def test_an_infinite_gap_where_no_log_was_declared_is_not_a_measurement() -> None:
    nothing = -float("inf")
    reading = weights([[-0.1, nothing, nothing], [-4.0, -0.5, -1.5]])

    with pytest.raises(ValueError, match="not finite"):
        ActionMargin().run(a_trace(reading), context=UNNORMALIZED)


def test_the_settled_count_reads_the_class_the_real_controller_published() -> None:
    join, keep, apply = Action("join", "concatenation"), Action("keep"), Action("apply")
    actions = ActionSet((join, keep, apply), keep)
    torch.manual_seed(5)
    controller = mlp_controller(actions, width=4, component=WIRED, hidden=6)

    cells = Encoded(torch.linspace(-1.0, 1.0, 12).reshape(3, 4), ("cell", "width"))
    scored = controller.weigh(cells, right=cells)
    context = DiagnosticContext.observing(
        controller, weighting=SpanWeighting(torch.ones(3), axes=("cell",))
    )
    published = context.spec_at(WEIGHING)
    trace = a_trace(Observation(WEIGHING, scored.tensor, axes=scored.axes))
    result = ActionMargin().run(trace, context=context)

    assert published.dtype_class is DTypeClass.LOG_PROBABILITY
    assert result.value(WEIGHING, "certain") == 0.0
    assert result.value(WEIGHING, "rows") == 3.0
