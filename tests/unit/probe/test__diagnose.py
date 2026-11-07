from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._controller_relations import ControllerRelation
from lofi.probe._diagnose import (
    Aggregation,
    DiagnosticContext,
    DiagnosticResult,
    Measurement,
)
from lofi.probe._marginals import SpanAggregate, SpanWeighting
from lofi.probe._observation_groups import ObservationGrouping
from lofi.probe._values import Observation, Trace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.nn.ports import Observable
    from lofi.nn.sites import SiteMap

    type ObservableFactory = Callable[[Mapping[Site, SiteSpec]], Observable]

PARSER: Final = ComponentId("cky-parser")
CONTROLLER: Final = ComponentId("chart-controller")

CELL: Final = Site(PARSER, Role.CHART_CELL, (0, 2))
SPLIT: Final = Site(PARSER, Role.SPLIT_DISTRIBUTION, (0, 4))
WEIGHING: Final = Site(CONTROLLER, Role.COMBINATOR_LOGITS)

REAL: Final = SiteSpec(
    ("item", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
SCORES: Final = SiteSpec(
    ("cell", "action"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)


# the derivation weighting of the two items a chart-cell reading holds
BELIEVED: Final = SpanWeighting(torch.tensor([0.25, 0.75]), axes=("item",))
INDEX: Final = SpanIndex(2)

GAP: Final = SpanAggregate(1.5, 2.0, 2)
RELATION: Final = ControllerRelation(2, sequences=((0, 1),))
GROUPING: Final = ObservationGrouping(2, ((0,), (1,)))


def a_context() -> DiagnosticContext:
    return DiagnosticContext({CELL: REAL, SPLIT: SCORES})


def a_weighed_context() -> DiagnosticContext:
    return DiagnosticContext({CELL: REAL, SPLIT: SCORES}, weighting=BELIEVED)


def test_a_context_covers_the_roles_of_the_points_it_holds() -> None:
    context = a_context()

    assert context.roles == frozenset({Role.CHART_CELL, Role.SPLIT_DISTRIBUTION})
    assert context.sites_for(Role.CHART_CELL) == (CELL,)
    assert context.sites_for(Role.COMBINATOR_LOGITS) == ()


def test_a_context_hands_out_a_map_the_caller_cannot_write_through() -> None:
    context = a_context()

    assert set(context.published) == {CELL, SPLIT}

    with pytest.raises(TypeError):
        cast("dict[Site, SiteSpec]", context.published)[WEIGHING] = SCORES


def test_a_context_holds_controller_relations_in_a_read_only_map() -> None:
    supplied = {WEIGHING: RELATION}
    context = DiagnosticContext({CELL: REAL, WEIGHING: SCORES}, relations=supplied)
    supplied.clear()

    assert context.relation_at(WEIGHING) is RELATION
    assert context.relations == {WEIGHING: RELATION}
    assert "relations=1" in repr(context)

    with pytest.raises(TypeError):
        cast("dict[Site, ControllerRelation]", context.relations)[WEIGHING] = RELATION


def test_a_context_snapshots_observation_groupings_in_a_read_only_map() -> None:
    supplied = {CELL: GROUPING}
    context = DiagnosticContext({CELL: REAL}, groupings=supplied)
    supplied.clear()

    assert context.grouping_at(CELL) is GROUPING
    assert context.groupings == {CELL: GROUPING}
    assert "groupings=1" in repr(context)

    with pytest.raises(TypeError):
        cast("dict[Site, ObservationGrouping]", context.groupings)[CELL] = GROUPING


def test_a_context_without_groupings_exposes_an_empty_read_only_map() -> None:
    context = a_context()

    assert not context.groupings
    assert "groupings" not in repr(context)

    with pytest.raises(TypeError):
        cast("dict[Site, ObservationGrouping]", context.groupings)[CELL] = GROUPING


def test_a_missing_observation_grouping_names_its_point() -> None:
    with pytest.raises(
        KeyError, match="no observation grouping was supplied"
    ) as missing:
        a_context().grouping_at(CELL)

    assert repr(CELL) in str(missing.value)


def test_a_grouping_at_an_unpublished_point_is_refused() -> None:
    with pytest.raises(ValueError, match="unpublished point") as unpublished:
        DiagnosticContext({CELL: REAL}, groupings={WEIGHING: GROUPING})

    assert repr(WEIGHING) in str(unpublished.value)


def test_a_grouping_key_that_is_not_a_site_is_refused() -> None:
    with pytest.raises(TypeError, match="not a Site"):
        DiagnosticContext(
            {CELL: REAL},
            groupings=cast("dict[Site, ObservationGrouping]", {"cell": GROUPING}),
        )


def test_a_grouping_value_that_is_not_an_observation_grouping_is_refused() -> None:
    with pytest.raises(TypeError, match="is an ObservationGrouping, not") as malformed:
        DiagnosticContext(
            {CELL: REAL},
            groupings=cast("dict[Site, ObservationGrouping]", {CELL: "not-a-grouping"}),
        )

    assert repr(CELL) in str(malformed.value)


def test_a_missing_controller_relation_names_its_point() -> None:
    with pytest.raises(
        KeyError, match="no controller relation was supplied"
    ) as missing:
        a_context().relation_at(WEIGHING)

    assert repr(WEIGHING) in str(missing.value)


def test_a_relation_at_an_unpublished_or_non_controller_point_is_refused() -> None:
    with pytest.raises(ValueError, match="unpublished point"):
        DiagnosticContext({CELL: REAL}, relations={WEIGHING: RELATION})

    with pytest.raises(ValueError, match=r"COMBINATOR_LOGITS.*CHART_CELL"):
        DiagnosticContext({CELL: REAL}, relations={CELL: RELATION})


def test_a_relation_key_that_is_not_a_site_is_refused() -> None:
    with pytest.raises(TypeError, match="not a Site"):
        DiagnosticContext(
            {WEIGHING: SCORES},
            relations=cast("dict[Site, ControllerRelation]", {"controller": RELATION}),
        )


def test_a_relation_value_that_is_not_a_controller_relation_is_refused() -> None:
    with pytest.raises(TypeError, match="is a ControllerRelation, not") as malformed:
        DiagnosticContext(
            {WEIGHING: SCORES},
            relations=cast(
                "dict[Site, ControllerRelation]", {WEIGHING: "not-a-relation"}
            ),
        )

    assert repr(WEIGHING) in str(malformed.value)


def test_a_context_reads_what_is_published_at_one_point() -> None:
    context = a_context()

    assert context.spec_at(SPLIT) is SCORES

    with pytest.raises(KeyError, match="not published"):
        context.spec_at(WEIGHING)


def test_a_context_refuses_a_key_that_is_not_a_point() -> None:
    with pytest.raises(TypeError, match="not a Site"):
        DiagnosticContext(cast("SiteMap", {"cky-parser": REAL}))


def test_a_context_refuses_a_value_that_is_not_a_specification() -> None:
    with pytest.raises(TypeError, match="not a SiteSpec"):
        DiagnosticContext(cast("SiteMap", {CELL: "real"}))


def test_a_context_refuses_a_role_that_is_not_a_role() -> None:
    with pytest.raises(TypeError, match="is a Role, not"):
        a_context().sites_for(cast("Role", "CHART_CELL"))


def test_observing_two_components_takes_the_union_of_their_maps(
    observable: ObservableFactory,
) -> None:
    context = DiagnosticContext.observing(
        observable({CELL: REAL, SPLIT: SCORES}),
        observable({WEIGHING: SCORES}),
    )

    assert set(context.published) == {CELL, SPLIT, WEIGHING}
    assert context.roles == frozenset(
        {Role.CHART_CELL, Role.SPLIT_DISTRIBUTION, Role.COMBINATOR_LOGITS}
    )


def test_observing_nothing_publishes_nothing() -> None:
    context = DiagnosticContext.observing()

    assert context.roles == frozenset()
    assert not context.published


def test_two_components_describing_one_point_differently_are_refused(
    observable: ObservableFactory,
) -> None:
    with pytest.raises(ValueError, match="not settled"):
        DiagnosticContext.observing(
            observable({CELL: REAL}), observable({CELL: SCORES})
        )


def test_a_reading_at_an_unpublished_point_is_refused() -> None:
    trace = Trace()
    trace.record(Observation(WEIGHING, torch.zeros(2, 3), axes=("cell", "action")))
    trace.close()

    with pytest.raises(KeyError, match="not published"):
        a_context().readings(trace)


def test_the_readings_come_back_in_the_order_they_were_taken() -> None:
    trace = Trace()
    trace.record(Observation(SPLIT, torch.zeros(2, 3), axes=("cell", "action")))
    trace.record(Observation(CELL, torch.zeros(2, 3), axes=("item", "width")))
    trace.close()

    assert tuple(site for site, _ in a_context().readings(trace)) == (SPLIT, CELL)


def test_a_measurement_is_equal_on_its_point_its_name_and_its_number() -> None:
    one = Measurement(CELL, "live", 6.0)

    assert one == Measurement(CELL, "live", 6.0)
    assert hash(one) == hash(Measurement(CELL, "live", 6.0))
    assert one != Measurement(SPLIT, "live", 6.0)
    assert one != Measurement(CELL, "nonfinite", 6.0)
    assert one != Measurement(CELL, "live", 5.0)
    assert one != "live"


def test_a_measurement_refuses_what_is_not_a_number_at_a_point() -> None:
    with pytest.raises(TypeError, match="not a Site"):
        Measurement(cast("Site", "cell"), "live", 1.0)

    with pytest.raises(ValueError, match="empty"):
        Measurement(CELL, "", 1.0)

    with pytest.raises(ValueError, match="not finite"):
        Measurement(CELL, "live", float("nan"))


def test_a_result_reads_one_measured_number_back() -> None:
    result = DiagnosticResult(
        "finite-readings",
        [Measurement(CELL, "live", 6.0), Measurement(CELL, "nonfinite", 1.0)],
    )

    assert result.tag == "finite-readings"
    assert result.value(CELL, "nonfinite") == 1.0
    assert result.sites == (CELL,)

    with pytest.raises(KeyError, match="nothing named"):
        result.value(SPLIT, "live")


def test_a_result_refuses_one_point_measured_twice_under_one_name() -> None:
    with pytest.raises(ValueError, match="measured twice"):
        DiagnosticResult(
            "finite-readings",
            [Measurement(CELL, "live", 6.0), Measurement(CELL, "live", 7.0)],
        )


def test_a_result_refuses_an_entry_that_is_not_a_measurement() -> None:
    with pytest.raises(TypeError, match="measurements"):
        DiagnosticResult("finite-readings", cast("list[Measurement]", [6.0]))

    with pytest.raises(ValueError, match="empty"):
        DiagnosticResult("", [])


def test_two_results_holding_the_same_measurements_agree() -> None:
    measured = [Measurement(CELL, "live", 6.0)]
    one = DiagnosticResult("finite-readings", measured)
    other = DiagnosticResult("finite-readings", measured)

    assert one == other
    assert hash(one) == hash(other)
    assert one != DiagnosticResult("action-margin", measured)
    assert one != measured


def test_a_context_says_how_many_points_it_holds() -> None:
    assert "2 sites" in repr(a_context())
    assert "CHART_CELL" in repr(a_context())


def test_a_context_left_unweighted_holds_no_weighting() -> None:
    assert a_context().weighting is None
    assert "weighting" not in repr(a_context())


def test_a_context_holds_the_weighting_it_was_given() -> None:
    context = a_weighed_context()

    assert context.weighting is BELIEVED
    assert "weighting=SpanWeighting(axes=('item',)" in repr(context)


def test_a_context_holds_the_chart_span_index_it_was_given() -> None:
    context = DiagnosticContext({CELL: REAL}, span_index=INDEX)

    assert context.span_index is INDEX
    assert "span_index=SpanIndex(2)" in repr(context)


def test_a_context_without_chart_addressing_holds_no_span_index() -> None:
    assert a_context().span_index is None
    assert "span_index" not in repr(a_context())


def test_a_context_refuses_a_span_index_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="span index is a SpanIndex"):
        DiagnosticContext({CELL: REAL}, span_index=cast("SpanIndex", 2))


def test_a_context_refuses_a_weighting_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="is a SpanWeighting"):
        DiagnosticContext(
            {CELL: REAL}, weighting=cast("SpanWeighting", torch.tensor([0.5, 0.5]))
        )


def test_observing_threads_the_weighting_through(
    observable: ObservableFactory,
) -> None:
    context = DiagnosticContext.observing(observable({CELL: REAL}), weighting=BELIEVED)

    assert context.weighting is BELIEVED


def test_observing_threads_the_span_index_through(
    observable: ObservableFactory,
) -> None:
    context = DiagnosticContext.observing(observable({CELL: REAL}), span_index=INDEX)

    assert context.span_index is INDEX


def test_observing_threads_controller_relations_through(
    observable: ObservableFactory,
) -> None:
    context = DiagnosticContext.observing(
        observable({WEIGHING: SCORES}), relations={WEIGHING: RELATION}
    )

    assert context.relation_at(WEIGHING) is RELATION
    assert DiagnosticContext.observing(observable({CELL: REAL})).weighting is None


def test_observing_threads_groupings_relations_and_weighting_without_changing_them(
    observable: ObservableFactory,
) -> None:
    context = DiagnosticContext.observing(
        observable({CELL: REAL}),
        observable({WEIGHING: SCORES}),
        groupings={CELL: GROUPING},
        relations={WEIGHING: RELATION},
        span_index=INDEX,
        weighting=BELIEVED,
    )

    assert context.grouping_at(CELL) is GROUPING
    assert context.relation_at(WEIGHING) is RELATION
    assert context.span_index is INDEX
    assert context.weighting is BELIEVED


def test_the_weights_over_a_reading_follow_its_leading_axes() -> None:
    weights = a_weighed_context().weights_over(("item", "width"), (2, 3))

    assert tuple(weights.shape) == (2, 3)
    assert torch.equal(weights[0], torch.full((3,), 0.25))
    assert torch.equal(weights[1], torch.full((3,), 0.75))


def test_a_context_without_a_weighting_cannot_weigh_a_reading() -> None:
    with pytest.raises(ValueError, match="no derivation weighting"):
        a_context().weights_over(("item", "width"), (2, 3))


def test_a_reading_the_weighting_does_not_line_up_with_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot weigh a reading over"):
        a_weighed_context().weights_over(("width", "item"), (3, 2))

    with pytest.raises(ValueError, match="cannot weigh a reading of shape"):
        a_weighed_context().weights_over(("item", "width"), (3, 3))


def test_an_aggregation_holds_both_means_at_a_point() -> None:
    aggregation = Aggregation(CELL, "gap", GAP)

    assert aggregation.site == CELL
    assert aggregation.name == "gap"
    assert aggregation.aggregate == GAP
    assert aggregation.aggregate.weighted == 1.5
    assert aggregation.aggregate.uniform == 2.0
    assert repr(aggregation) == f"Aggregation({CELL!r}, 'gap', {GAP!r})"


def test_an_aggregation_is_equal_on_its_point_its_name_and_its_aggregate() -> None:
    aggregation = Aggregation(CELL, "gap", GAP)

    assert aggregation == Aggregation(CELL, "gap", GAP)
    assert hash(aggregation) == hash(Aggregation(CELL, "gap", GAP))
    assert aggregation != Aggregation(SPLIT, "gap", GAP)
    assert aggregation != Aggregation(CELL, "margin", GAP)
    assert aggregation != Aggregation(CELL, "gap", SpanAggregate(1.5, 2.0, 3))
    assert aggregation != "gap"


def test_an_aggregation_refuses_what_is_not_an_aggregate_at_a_point() -> None:
    with pytest.raises(TypeError, match="not a Site"):
        Aggregation(cast("Site", "cell"), "gap", GAP)

    with pytest.raises(ValueError, match="empty"):
        Aggregation(CELL, "", GAP)

    with pytest.raises(TypeError, match="holds a SpanAggregate"):
        Aggregation(CELL, "gap", cast("SpanAggregate", 1.5))


def test_a_result_reads_one_aggregate_back_under_both_weightings() -> None:
    result = DiagnosticResult(
        "action-margin",
        [Measurement(CELL, "rows", 2.0)],
        [Aggregation(CELL, "gap", GAP), Aggregation(SPLIT, "gap", GAP)],
    )

    assert result.aggregations == (
        Aggregation(CELL, "gap", GAP),
        Aggregation(SPLIT, "gap", GAP),
    )
    assert result.aggregate(SPLIT, "gap") == GAP
    assert (
        result.aggregate(CELL, "gap").weighted != result.aggregate(CELL, "gap").uniform
    )
    assert result.sites == (CELL, SPLIT)
    assert "1 measurements, 2 aggregations" in repr(result)

    with pytest.raises(KeyError, match="nothing named"):
        result.aggregate(CELL, "rows")

    with pytest.raises(KeyError, match="nothing named"):
        result.value(CELL, "gap")


def test_a_result_left_without_aggregations_holds_none() -> None:
    result = DiagnosticResult("finite-readings", [Measurement(CELL, "live", 6.0)])

    assert result.aggregations == ()


def test_a_result_refuses_one_point_aggregated_twice_under_one_name() -> None:
    with pytest.raises(ValueError, match="measured twice"):
        DiagnosticResult(
            "action-margin",
            [],
            [Aggregation(CELL, "gap", GAP), Aggregation(CELL, "gap", GAP)],
        )

    with pytest.raises(ValueError, match="measured twice"):
        DiagnosticResult(
            "action-margin",
            [Measurement(CELL, "gap", 2.0)],
            [Aggregation(CELL, "gap", GAP)],
        )


def test_a_result_refuses_an_entry_that_is_not_an_aggregation() -> None:
    with pytest.raises(TypeError, match="aggregations"):
        DiagnosticResult("action-margin", [], cast("list[Aggregation]", [GAP]))


def test_two_results_differing_only_in_their_aggregations_disagree() -> None:
    measured = [Measurement(CELL, "rows", 2.0)]
    one = DiagnosticResult("action-margin", measured, [Aggregation(CELL, "gap", GAP)])
    same = DiagnosticResult("action-margin", measured, [Aggregation(CELL, "gap", GAP)])
    other = DiagnosticResult("action-margin", measured)

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
