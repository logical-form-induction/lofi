from math import log
from typing import Final

import pytest
import torch
from tests._controllers import mlp_controller

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._controller_choice import ControllerChoice
from lofi.probe._controller_relations import ControllerRelation
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._margin import ActionMargin
from lofi.probe._marginals import SpanWeighting
from lofi.probe._scores import certainty
from lofi.probe._values import Observation, Trace

WIRED: Final = ComponentId("chart-controller")
WEIGHING: Final = Site(WIRED, Role.COMBINATOR_LOGITS)
CELL: Final = Site(WIRED, Role.CHART_CELL, (0, 2))

LOG_PROBABILITIES: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
PROBABILITIES: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
LOGITS: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
COUNTS: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.INTEGER,
    differentiable=False,
    access=Access.OBSERVE,
)
CELLS: Final = SiteSpec(
    ("cell", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

BELIEVED: Final = (0.25, 0.75)
ALONE: Final = (1.0,)


def context(
    marginals: tuple[float, ...] = BELIEVED,
    spec: SiteSpec = LOG_PROBABILITIES,
) -> DiagnosticContext:
    """Return a context weighing the controller's cells.

    Parameters
    ----------
    marginals : tuple[float, ...]
        The weight of each cell.
    spec : SiteSpec
        What the controller point publishes.
    """
    weighting = SpanWeighting(torch.tensor(marginals), axes=("cell",))

    return DiagnosticContext({WEIGHING: spec, CELL: CELLS}, weighting=weighting)


def a_trace(*readings: Observation) -> Trace:
    """Return a closed trace holding the readings in order.

    Parameters
    ----------
    *readings : Observation
        The readings to record.
    """
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def logged(numbers: list[list[float]], mask: torch.Tensor | None = None) -> Observation:
    """Return a controller reading from probabilities.

    Parameters
    ----------
    numbers : list[list[float]]
        One probability row per cell.
    mask : torch.Tensor | None
        Which rows or action positions hold data.
    """
    return Observation(
        WEIGHING,
        torch.tensor(numbers).log(),
        axes=LOG_PROBABILITIES.axes,
        mask=mask,
    )


def aggregate(result: DiagnosticResult, name: str) -> tuple[float, float, int]:
    """Return the weighted mean, uniform mean and row count of one score.

    Parameters
    ----------
    result : DiagnosticResult
        The controller-choice result.
    name : str
        The score to read.
    """
    found = result.aggregate(WEIGHING, name)

    return found.weighted, found.uniform, found.count


def test_the_three_choice_scores_match_hand_computed_probabilities() -> None:
    probabilities = [[0.5, 0.25, 0.25], [0.8, 0.1, 0.1]]
    result = ControllerChoice().run(a_trace(logged(probabilities)), context=context())
    first_entropy = -(0.5 * log(0.5) + 0.5 * log(0.25)) / log(3)
    second_entropy = -(0.8 * log(0.8) + 0.2 * log(0.1)) / log(3)

    assert aggregate(result, "confidence") == pytest.approx((0.725, 0.65, 2))
    assert aggregate(result, "normalized-entropy") == pytest.approx(
        (
            0.25 * first_entropy + 0.75 * second_entropy,
            (first_entropy + second_entropy) / 2.0,
            2,
        )
    )
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (0.5875, 0.475, 2)
    )
    assert result.value(WEIGHING, "rows") == 2.0


def test_confidence_is_not_entropy_complement_and_the_margin_is_not_a_log_gap() -> None:
    reading = logged([[0.5, 0.25, 0.25]])
    measured = ControllerChoice().run(a_trace(reading), context=context(ALONE))
    raw = ActionMargin().run(a_trace(reading), context=context(ALONE))
    entropy_complement = certainty(reading, DTypeClass.LOG_PROBABILITY, over="action")

    assert measured.aggregate(WEIGHING, "confidence").uniform == pytest.approx(0.5)
    assert float(entropy_complement.tensor[0]) != pytest.approx(0.5)
    assert measured.aggregate(
        WEIGHING, "top-two-probability-margin"
    ).uniform == pytest.approx(0.25)
    assert raw.aggregate(WEIGHING, "margin").uniform == pytest.approx(log(2))


def test_a_point_mass_has_full_confidence_no_entropy_and_full_margin() -> None:
    result = ControllerChoice().run(
        a_trace(logged([[1.0, 0.0, 0.0]])), context=context(ALONE)
    )

    assert aggregate(result, "confidence") == pytest.approx((1.0, 1.0, 1))
    assert aggregate(result, "normalized-entropy") == pytest.approx((0.0, 0.0, 1))
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (1.0, 1.0, 1)
    )


def test_a_uniform_choice_has_full_entropy_and_no_margin() -> None:
    result = ControllerChoice().run(
        a_trace(logged([[0.25, 0.25, 0.25, 0.25]])), context=context(ALONE)
    )

    assert aggregate(result, "confidence") == pytest.approx((0.25, 0.25, 1))
    assert aggregate(result, "normalized-entropy") == pytest.approx((1.0, 1.0, 1))
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (0.0, 0.0, 1)
    )


def test_a_tie_for_the_largest_probability_has_no_margin() -> None:
    result = ControllerChoice().run(
        a_trace(logged([[0.4, 0.4, 0.2]])), context=context(ALONE)
    )

    assert aggregate(result, "confidence") == pytest.approx((0.4, 0.4, 1))
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (0.0, 0.0, 1)
    )


def test_the_entropy_normalizer_counts_only_live_action_positions() -> None:
    mask = torch.tensor([[True, True, False]])
    result = ControllerChoice().run(
        a_trace(logged([[0.5, 0.5, 0.0]], mask)), context=context(ALONE)
    )

    assert aggregate(result, "confidence") == pytest.approx((0.5, 0.5, 1))
    assert aggregate(result, "normalized-entropy") == pytest.approx((1.0, 1.0, 1))
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (0.0, 0.0, 1)
    )


def test_one_live_action_has_no_entropy_and_full_margin() -> None:
    mask = torch.tensor([[True, False, False]])
    result = ControllerChoice().run(
        a_trace(logged([[1.0, 0.0, 0.0]], mask)), context=context(ALONE)
    )

    assert aggregate(result, "confidence") == pytest.approx((1.0, 1.0, 1))
    assert aggregate(result, "normalized-entropy") == pytest.approx((0.0, 0.0, 1))
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (1.0, 1.0, 1)
    )


def test_a_masked_cell_is_left_out_of_every_mean() -> None:
    reading = logged([[0.5, 0.25, 0.25], [0.0, 0.0, 0.0]], torch.tensor([True, False]))
    result = ControllerChoice().run(a_trace(reading), context=context())

    assert aggregate(result, "confidence") == pytest.approx((0.5, 0.5, 1))
    assert aggregate(result, "top-two-probability-margin") == pytest.approx(
        (0.25, 0.25, 1)
    )
    assert result.value(WEIGHING, "rows") == 1.0


def test_a_reading_with_no_live_action_is_refused() -> None:
    reading = logged([[0.0, 0.0, 0.0]], torch.tensor([[False, False, False]]))

    with pytest.raises(ValueError, match="every span is masked off"):
        ControllerChoice().run(a_trace(reading), context=context(ALONE))


@pytest.mark.parametrize("width", [0, 1])
def test_fewer_than_two_physical_action_columns_are_refused(width: int) -> None:
    reading = Observation(
        WEIGHING,
        torch.empty((1, width)),
        axes=LOG_PROBABILITIES.axes,
    )

    with pytest.raises(ValueError, match="no gap"):
        ControllerChoice().run(a_trace(reading), context=context(ALONE))


def test_permuting_action_columns_leaves_every_score_unchanged() -> None:
    ordered = logged([[0.5, 0.25, 0.25], [0.8, 0.1, 0.1]])
    permuted = logged([[0.25, 0.5, 0.25], [0.1, 0.1, 0.8]])
    first = ControllerChoice().run(a_trace(ordered), context=context())
    second = ControllerChoice().run(a_trace(permuted), context=context())

    for name in ControllerChoice.scores:
        first_summary = first.aggregate(WEIGHING, name)
        second_summary = second.aggregate(WEIGHING, name)

        assert first_summary.weighted == pytest.approx(second_summary.weighted)
        assert first_summary.uniform == pytest.approx(second_summary.uniform)
        assert first_summary.count == second_summary.count


def test_each_declared_numeric_domain_reaches_the_same_probabilities() -> None:
    probabilities = torch.tensor([[0.5, 0.25, 0.25]])
    readings = (
        (LOG_PROBABILITIES, probabilities.log()),
        (PROBABILITIES, probabilities),
        (LOGITS, probabilities.log() + 7.0),
    )
    results = []

    for spec, tensor in readings:
        reading = Observation(WEIGHING, tensor, axes=spec.axes)
        results.append(
            ControllerChoice().run(a_trace(reading), context=context(ALONE, spec=spec))
        )

    for name in ControllerChoice.scores:
        values = [result.aggregate(WEIGHING, name).uniform for result in results]

        assert values == pytest.approx([values[0]] * len(values))


def test_a_non_numeric_declaration_cannot_weigh_choices() -> None:
    reading = Observation(WEIGHING, torch.tensor([[1, 2, 3]]), axes=COUNTS.axes)

    with pytest.raises(ValueError, match="weigh no outcomes"):
        ControllerChoice().run(a_trace(reading), context=context(ALONE, spec=COUNTS))


def test_a_missing_action_axis_names_the_axes_that_were_read() -> None:
    reading = Observation(WEIGHING, torch.zeros(1, 3), axes=("cell", "combinator"))

    with pytest.raises(KeyError, match="combinator"):
        ControllerChoice().run(a_trace(reading), context=context(ALONE))


def test_a_context_without_derivation_weights_cannot_aggregate_cells() -> None:
    unweighted = DiagnosticContext({WEIGHING: LOG_PROBABILITIES, CELL: CELLS})

    with pytest.raises(ValueError, match="no derivation weighting"):
        ControllerChoice().run(a_trace(logged([[0.5, 0.25, 0.25]])), context=unweighted)


def test_weights_over_another_number_of_cells_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot weigh a reading of shape"):
        ControllerChoice().run(
            a_trace(logged([[0.5, 0.25, 0.25], [0.8, 0.1, 0.1]])),
            context=context(ALONE),
        )


def test_weights_over_another_axis_are_refused_by_axis_name() -> None:
    weighting = SpanWeighting(torch.tensor(ALONE), axes=("span",))
    mismatched = DiagnosticContext(
        {WEIGHING: LOG_PROBABILITIES, CELL: CELLS}, weighting=weighting
    )

    with pytest.raises(ValueError, match=r"span.*cell"):
        ControllerChoice().run(a_trace(logged([[0.5, 0.25, 0.25]])), context=mismatched)


def test_a_published_decision_point_that_was_not_read_is_named() -> None:
    reading = Observation(CELL, torch.zeros(1, 3), axes=CELLS.axes)

    with pytest.raises(ValueError, match=r"COMBINATOR_LOGITS.*no choice to measure"):
        ControllerChoice().run(a_trace(reading), context=context(ALONE))


def test_a_model_without_a_decision_role_has_no_choice_measurement() -> None:
    context_without = DiagnosticContext({CELL: CELLS})
    reading = Observation(CELL, torch.zeros(1, 3), axes=CELLS.axes)
    result = ControllerChoice().run(a_trace(reading), context=context_without)

    assert result.measurements == ()
    assert result.aggregations == ()


def test_the_diagnostic_declares_its_capability_scores_and_axis() -> None:
    assert ControllerChoice.tag == "controller-choice"
    assert ControllerChoice.requires == frozenset({Role.COMBINATOR_LOGITS})
    assert ControllerChoice.scores == frozenset(
        {"confidence", "normalized-entropy", "top-two-probability-margin"}
    )
    assert ControllerChoice("combinator").over == "combinator"
    assert repr(ControllerChoice()) == "ControllerChoice('action')"


def test_the_result_contains_no_composite_or_relational_quantity() -> None:
    result = ControllerChoice().run(
        a_trace(logged([[0.5, 0.25, 0.25]])), context=context(ALONE)
    )

    assert {entry.name for entry in result.aggregations} == ControllerChoice.scores
    assert {entry.name for entry in result.measurements} == {"rows"}


def test_a_real_controller_resolves_and_matches_its_action_probabilities() -> None:
    apply = Action("apply", "application")
    compose = Action("compose", "composition")
    keep = Action("keep")
    actions = ActionSet((apply, compose, keep), keep)
    torch.manual_seed(5)
    controller = mlp_controller(actions, width=4, component=WIRED, hidden=6)
    cells = Encoded(torch.linspace(-1.0, 1.0, 12).reshape(3, 4), ("cell", "width"))
    scored = controller.weigh(cells, right=cells)
    believed = torch.tensor([0.2, 0.3, 0.5])
    actual_context = DiagnosticContext.observing(
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
    resolved = installed_diagnostics().matching(actual_context)
    reported = {one.tag: one.run(trace, context=actual_context) for one in resolved}
    probabilities = scored.tensor.detach().exp()
    confidence = probabilities.amax(dim=1)
    entropy = -torch.xlogy(probabilities, probabilities).sum(dim=1) / log(
        len(actions.actions)
    )
    best = probabilities.topk(2, dim=1).values
    margin = best[:, 0] - best[:, 1]

    assert set(reported) == {
        "action-margin",
        "controller-choice",
        "controller-relations",
        "finite-readings",
    }

    for name, values in (
        ("confidence", confidence),
        ("normalized-entropy", entropy),
        ("top-two-probability-margin", margin),
    ):
        summary = reported["controller-choice"].aggregate(WEIGHING, name)

        assert summary.uniform == pytest.approx(float(values.mean()))
        assert summary.weighted == pytest.approx(float((values * believed).sum()))

    assert reported["controller-choice"].value(WEIGHING, "rows") == 3.0
