from copy import copy, deepcopy
from math import exp
from typing import Final

import pytest
import torch
from tests._controllers import mlp_controller

from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._controller_relations import ControllerRelation, ControllerRelations
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._values import Observation, Trace

CONTROLLER: Final = ComponentId("chart-controller")
WEIGHING: Final = Site(CONTROLLER, Role.COMBINATOR_LOGITS)
CELL: Final = Site(CONTROLLER, Role.CHART_CELL, (0, 2))

LOG_PROBABILITIES: Final = SiteSpec(
    ("row", "action"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
PROBABILITIES: Final = SiteSpec(
    ("row", "action"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
LOGITS: Final = SiteSpec(
    ("row", "action"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
CELLS: Final = SiteSpec(
    ("row", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

RELATION: Final = ControllerRelation(
    4,
    sequences=((0, 1, 2), (3,)),
    similar_inputs=((0, 2), (1, 3)),
)


def context(
    relation: ControllerRelation = RELATION,
    spec: SiteSpec = LOG_PROBABILITIES,
) -> DiagnosticContext:
    """Return a context with one declared controller relation.

    Parameters
    ----------
    relation : ControllerRelation
        The row memberships to supply.
    spec : SiteSpec
        What the controller point publishes.
    """
    return DiagnosticContext(
        {WEIGHING: spec, CELL: CELLS}, relations={WEIGHING: relation}
    )


def a_trace(reading: Observation) -> Trace:
    """Return a closed trace containing one reading.

    Parameters
    ----------
    reading : Observation
        The reading to record.
    """
    trace = Trace()
    trace.record(reading)
    trace.close()

    return trace


def probabilities(
    rows: list[list[float]], mask: torch.Tensor | None = None
) -> Observation:
    """Return one probability reading.

    Parameters
    ----------
    rows : list[list[float]]
        One action distribution per row.
    mask : torch.Tensor | None
        The valid positions, or None where all are valid.
    """
    return Observation(
        WEIGHING,
        torch.tensor(rows),
        axes=PROBABILITIES.axes,
        mask=mask,
    )


def logged(rows: list[list[float]], mask: torch.Tensor | None = None) -> Observation:
    """Return one log-probability reading.

    Parameters
    ----------
    rows : list[list[float]]
        One action distribution per row.
    mask : torch.Tensor | None
        The valid positions, or None where all are valid.
    """
    return Observation(
        WEIGHING,
        torch.tensor(rows).log(),
        axes=LOG_PROBABILITIES.axes,
        mask=mask,
    )


def values(result: DiagnosticResult) -> dict[str, float]:
    """Return the result's measurements by name.

    Parameters
    ----------
    result : DiagnosticResult
        The relational result.
    """
    return {measurement.name: measurement.value for measurement in result.measurements}


def test_a_relation_preserves_sequence_order_and_unordered_similar_groups() -> None:
    relation = ControllerRelation(
        5,
        sequences=([3, 1], [4, 0]),
        similar_inputs=([4, 2, 1], [0, 2]),
    )

    assert relation.rows == 5
    assert relation.sequences == ((3, 1), (4, 0))
    assert relation.similar_inputs == (frozenset({1, 2, 4}), frozenset({0, 2}))
    assert relation.referenced == frozenset(range(5))


def test_a_relation_copies_and_freezes_its_memberships() -> None:
    sequence = [0, 1]
    similar = [1, 2]
    relation = ControllerRelation(3, (sequence,), (similar,))
    sequence.reverse()
    similar.clear()

    assert relation.sequences == ((0, 1),)
    assert relation.similar_inputs == (frozenset({1, 2}),)
    assert copy(relation) is relation
    assert deepcopy(relation) is relation

    with pytest.raises(AttributeError, match="immutable"):
        relation.__setattr__("_rows", 4)


def test_a_relation_is_equal_and_hashes_on_its_declaration() -> None:
    one = ControllerRelation(4, ((0, 1),), ((1, 2),))
    same = ControllerRelation(4, ((0, 1),), ((2, 1),))
    reordered = ControllerRelation(4, ((1, 0),), ((1, 2),))

    assert one == same
    assert hash(one) == hash(same)
    assert one != reordered
    assert one != "relation"
    assert repr(one) == "ControllerRelation(4, ((0, 1),), ((1, 2),))"


@pytest.mark.parametrize("rows", [True, 0, -1])
def test_a_relation_requires_a_positive_non_boolean_row_extent(rows: int) -> None:
    with pytest.raises((TypeError, ValueError), match="row extent"):
        ControllerRelation(rows)


@pytest.mark.parametrize("row", [True, -1, 3])
def test_memberships_require_non_boolean_rows_inside_the_extent(row: int) -> None:
    with pytest.raises((TypeError, ValueError), match="row"):
        ControllerRelation(3, sequences=((row,),))

    with pytest.raises((TypeError, ValueError), match="row"):
        ControllerRelation(3, similar_inputs=((0, row),))


def test_a_sequence_must_be_nonempty_unique_and_disjoint_from_every_other() -> None:
    with pytest.raises(ValueError, match="sequence 0 is empty"):
        ControllerRelation(3, sequences=((),))

    with pytest.raises(ValueError, match="repeats"):
        ControllerRelation(3, sequences=((0, 1, 0),))

    with pytest.raises(ValueError, match=r"overlaps.*1"):
        ControllerRelation(3, sequences=((0, 1), (1, 2)))


def test_a_singleton_sequence_is_legal() -> None:
    assert ControllerRelation(2, sequences=((1,),)).sequences == ((1,),)


def test_similar_groups_are_distinct_sets_of_at_least_two_rows() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        ControllerRelation(3, similar_inputs=((0,),))

    with pytest.raises(ValueError, match="repeats a row"):
        ControllerRelation(3, similar_inputs=((0, 0),))

    with pytest.raises(ValueError, match="repeats an earlier group"):
        ControllerRelation(3, similar_inputs=((0, 1), (1, 0)))


def test_distinct_similar_groups_may_overlap() -> None:
    relation = ControllerRelation(4, similar_inputs=((0, 1, 2), (1, 2, 3)))

    assert relation.similar_inputs == (
        frozenset({0, 1, 2}),
        frozenset({1, 2, 3}),
    )


def test_constant_rows_score_one_under_every_declared_relation() -> None:
    reading = logged([[0.75, 0.25]] * 4)
    measured = values(ControllerRelations().run(a_trace(reading), context=context()))

    assert measured == {
        "sequence-consistency": pytest.approx(1.0),
        "sequence-smoothness": pytest.approx(1.0),
        "similar-input-consistency": pytest.approx(1.0),
        "sequences": 2.0,
        "similar-input-groups": 2.0,
    }


def test_alternation_lowers_smoothness_and_disagreement_lowers_consistency() -> None:
    relation = ControllerRelation(
        3,
        sequences=((0, 1, 2),),
        similar_inputs=((0, 1),),
    )
    reading = probabilities([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    measured = values(
        ControllerRelations().run(
            a_trace(reading), context=context(relation, PROBABILITIES)
        )
    )

    assert measured["sequence-consistency"] == pytest.approx(exp(-2.0 / 9.0))
    assert measured["sequence-smoothness"] == pytest.approx(exp(-1.0))
    assert measured["similar-input-consistency"] == pytest.approx(exp(-0.25))
    assert measured["sequence-smoothness"] < 1.0


def test_group_scores_are_arithmetic_means_and_not_weighted_aggregates() -> None:
    relation = ControllerRelation(
        4,
        sequences=((0, 1), (2, 3)),
        similar_inputs=((0, 1), (2, 3)),
    )
    reading = probabilities([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = ControllerRelations().run(
        a_trace(reading), context=context(relation, PROBABILITIES)
    )
    measured = values(result)
    expected = (1.0 + exp(-0.25)) / 2.0

    assert measured["sequence-consistency"] == pytest.approx(expected)
    assert measured["similar-input-consistency"] == pytest.approx(expected)
    assert result.aggregations == ()


def test_empty_relation_families_have_neutral_scores_and_zero_counts() -> None:
    relation = ControllerRelation(2)
    reading = logged([[0.75, 0.25], [0.25, 0.75]])
    measured = values(
        ControllerRelations().run(a_trace(reading), context=context(relation))
    )

    assert measured == {
        "sequence-consistency": 1.0,
        "sequence-smoothness": 1.0,
        "similar-input-consistency": 1.0,
        "sequences": 0.0,
        "similar-input-groups": 0.0,
    }


def test_identical_probabilities_report_differently_under_different_declarations() -> (
    None
):
    reading = probabilities([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    adjacent = ControllerRelation(3, sequences=((0, 1),), similar_inputs=((0, 1),))
    alike = ControllerRelation(3, sequences=((0, 2),), similar_inputs=((0, 2),))
    first = values(
        ControllerRelations().run(
            a_trace(reading), context=context(adjacent, PROBABILITIES)
        )
    )
    second = values(
        ControllerRelations().run(
            a_trace(reading), context=context(alike, PROBABILITIES)
        )
    )

    assert first["sequence-consistency"] < second["sequence-consistency"]
    assert first["sequence-smoothness"] < second["sequence-smoothness"]
    assert first["similar-input-consistency"] < second["similar-input-consistency"]


def test_permuting_rows_and_the_declaration_together_preserves_every_score() -> None:
    reading = probabilities([[0.7, 0.3], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]])
    permuted = probabilities([[0.6, 0.4], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]])
    moved = ControllerRelation(
        4,
        sequences=((1, 3, 0), (2,)),
        similar_inputs=((1, 0), (3, 2)),
    )
    first = values(
        ControllerRelations().run(a_trace(reading), context=context(spec=PROBABILITIES))
    )
    second = values(
        ControllerRelations().run(
            a_trace(permuted), context=context(moved, PROBABILITIES)
        )
    )

    assert first == pytest.approx(second)


def test_each_probability_domain_reaches_the_same_relation_scores() -> None:
    distribution = torch.tensor([[0.7, 0.3], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]])
    readings = (
        (LOG_PROBABILITIES, distribution.log()),
        (PROBABILITIES, distribution),
        (LOGITS, distribution.log() + 4.0),
    )
    reports = []

    for spec, tensor in readings:
        observation = Observation(WEIGHING, tensor, axes=spec.axes)
        reports.append(
            values(
                ControllerRelations().run(
                    a_trace(observation), context=context(spec=spec)
                )
            )
        )

    for name in ControllerRelations.scores:
        assert [report[name] for report in reports] == pytest.approx(
            [reports[0][name]] * len(reports)
        )


def test_a_reading_must_be_rank_two_with_the_exact_row_extent_and_actions() -> None:
    rank_three = Observation(
        WEIGHING,
        torch.zeros(4, 1, 2),
        axes=("row", "split", "action"),
    )

    with pytest.raises(ValueError, match="rank-two"):
        ControllerRelations().run(a_trace(rank_three), context=context())

    too_short = Observation(WEIGHING, torch.zeros(3, 2), axes=("row", "action"))

    with pytest.raises(ValueError, match=r"declares 4 rows.*holds 3"):
        ControllerRelations().run(a_trace(too_short), context=context())

    no_actions = Observation(WEIGHING, torch.empty(4, 0), axes=("row", "action"))

    with pytest.raises(ValueError, match="holds no actions"):
        ControllerRelations().run(a_trace(no_actions), context=context())


def test_a_missing_action_axis_names_the_axes_observed() -> None:
    reading = Observation(WEIGHING, torch.zeros(4, 2), axes=("row", "combinator"))

    with pytest.raises(KeyError, match="combinator"):
        ControllerRelations().run(a_trace(reading), context=context())


def test_referenced_rows_must_be_live_but_unreferenced_rows_may_be_masked() -> None:
    relation = ControllerRelation(4, sequences=((0, 2),), similar_inputs=((0, 2),))
    mask = torch.tensor([True, False, True, False])
    reading = logged(
        [[0.75, 0.25], [0.0, 0.0], [0.75, 0.25], [0.0, 0.0]],
        mask,
    )
    result = ControllerRelations().run(a_trace(reading), context=context(relation))

    assert values(result)["sequence-consistency"] == pytest.approx(1.0)

    referenced_mask = torch.tensor([True, False, False, False])
    masked = logged(
        [[0.75, 0.25], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        referenced_mask,
    )

    with pytest.raises(ValueError, match=r"masked rows.*2"):
        ControllerRelations().run(a_trace(masked), context=context(relation))


def test_a_read_controller_point_requires_a_caller_supplied_relation() -> None:
    without = DiagnosticContext({WEIGHING: LOG_PROBABILITIES})

    with pytest.raises(KeyError, match="no controller relation was supplied"):
        ControllerRelations().run(a_trace(logged([[0.75, 0.25]] * 4)), context=without)


def test_a_published_controller_point_that_was_not_read_is_named() -> None:
    trace = Trace()
    trace.record(Observation(CELL, torch.zeros(4, 2), axes=CELLS.axes))
    trace.close()

    with pytest.raises(ValueError, match="were not read"):
        ControllerRelations().run(trace, context=context())


def test_a_context_without_a_controller_role_has_no_relational_measurement() -> None:
    trace = Trace()
    trace.record(Observation(CELL, torch.zeros(4, 2), axes=CELLS.axes))
    trace.close()
    result = ControllerRelations().run(trace, context=DiagnosticContext({CELL: CELLS}))

    assert result.measurements == ()
    assert result.aggregations == ()


def test_the_diagnostic_declares_its_role_scores_and_axis() -> None:
    assert ControllerRelations.tag == "controller-relations"
    assert ControllerRelations.requires == frozenset({Role.COMBINATOR_LOGITS})
    assert ControllerRelations.scores == frozenset(
        {
            "sequence-consistency",
            "sequence-smoothness",
            "similar-input-consistency",
        }
    )
    assert ControllerRelations("combinator").over == "combinator"
    assert repr(ControllerRelations()) == "ControllerRelations('action')"


def test_the_result_has_exactly_three_scores_and_two_counts() -> None:
    result = ControllerRelations().run(
        a_trace(logged([[0.75, 0.25]] * 4)), context=context()
    )

    assert {measurement.name for measurement in result.measurements} == {
        *ControllerRelations.scores,
        "sequences",
        "similar-input-groups",
    }
    assert result.aggregations == ()


def test_a_real_controller_is_measured_through_its_context_relation() -> None:
    apply = Action("apply", "application")
    compose = Action("compose", "composition")
    keep = Action("keep")
    actions = ActionSet((apply, compose, keep), keep)
    torch.manual_seed(5)
    controller = mlp_controller(actions, width=4, component=CONTROLLER, hidden=6)
    cells = Encoded(torch.linspace(-1.0, 1.0, 16).reshape(4, 4), ("cell", "width"))
    scored = controller.weigh(cells, right=cells)
    relation = ControllerRelation(
        4,
        sequences=((0, 1, 2, 3),),
        similar_inputs=((0, 2), (1, 3)),
    )
    actual_context = DiagnosticContext.observing(
        controller, relations={WEIGHING: relation}
    )
    trace = a_trace(Observation(WEIGHING, scored.tensor, axes=scored.axes))
    resolved = installed_diagnostics().matching(actual_context)
    diagnostic = next(one for one in resolved if one.tag == "controller-relations")
    result = diagnostic.run(trace, context=actual_context)

    assert "controller-relations" in {one.tag for one in resolved}
    assert result.value(WEIGHING, "sequences") == 1.0
    assert result.value(WEIGHING, "similar-input-groups") == 2.0
    assert all(
        0.0 <= result.value(WEIGHING, name) <= 1.0
        for name in ControllerRelations.scores
    )
