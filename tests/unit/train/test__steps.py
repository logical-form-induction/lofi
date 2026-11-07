from math import inf, nan
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.train._steps import (
    EvalResult,
    NumericalFailureError,
    StepEdge,
    StepGraphError,
    StepOutcome,
    TrainResult,
    edges,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

STEP: Final = 12

ITEMS: Final = 5

DROPPED: Final = 2

METRICS: Final = {"loss": 0.5, "accuracy": 0.75}


class Declared:
    __slots__ = ("_name", "_produces", "_requires")

    def __init__(
        self,
        name: str,
        requires: frozenset[str],
        produces: frozenset[str],
    ) -> None:
        self._name = name
        self._requires = requires
        self._produces = produces

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return self._requires

    @property
    def produces(self) -> frozenset[str]:
        return self._produces


FEATURIZING: Final = Declared(
    "featurizing", frozenset({"examples"}), frozenset({"weights"})
)

SCORING: Final = Declared(
    "scoring", frozenset({"examples", "weights"}), frozenset({"scores"})
)

REPORTING: Final = Declared("reporting", frozenset({"scores"}), frozenset({"report"}))

GRAPH: Final = (FEATURIZING, SCORING, REPORTING)

OUTSIDE: Final = frozenset({"examples"})


def test_a_train_result_reads_back_every_field_it_was_built_with() -> None:
    result = TrainResult(STEP, ITEMS, DROPPED, METRICS)

    assert (result.step, result.items, result.dropped) == (STEP, ITEMS, DROPPED)
    assert dict(result.metrics) == METRICS


def test_a_train_result_defaults_to_no_drop_and_no_metric() -> None:
    result = TrainResult(STEP, ITEMS)

    assert result.dropped == 0
    assert dict(result.metrics) == {}


def test_two_train_results_built_alike_are_equal_and_hash_alike() -> None:
    one = TrainResult(STEP, ITEMS, DROPPED, METRICS)
    other = TrainResult(STEP, ITEMS, DROPPED, dict(reversed(list(METRICS.items()))))

    assert one == other
    assert hash(one) == hash(other)


@pytest.mark.parametrize(
    "moved",
    [
        TrainResult(STEP + 1, ITEMS, DROPPED, METRICS),
        TrainResult(STEP, ITEMS + 1, DROPPED, METRICS),
        TrainResult(STEP, ITEMS, DROPPED + 1, METRICS),
        TrainResult(STEP, ITEMS, DROPPED, {"loss": 0.5}),
        TrainResult(STEP, ITEMS, DROPPED, {"loss": 0.25, "accuracy": 0.75}),
    ],
)
def test_moving_any_field_of_a_train_result_moves_it_out_of_its_class(
    moved: TrainResult,
) -> None:
    assert TrainResult(STEP, ITEMS, DROPPED, METRICS) != moved


def test_a_train_result_is_not_an_eval_result_with_a_step_on_it() -> None:
    assert TrainResult(STEP, ITEMS, DROPPED, METRICS) != EvalResult(
        ITEMS, DROPPED, METRICS
    )


def test_an_eval_result_reads_back_every_field_it_was_built_with() -> None:
    result = EvalResult(ITEMS, DROPPED, METRICS)

    assert (result.items, result.dropped) == (ITEMS, DROPPED)
    assert dict(result.metrics) == METRICS


def test_two_eval_results_built_alike_are_equal_and_hash_alike() -> None:
    one = EvalResult(ITEMS, DROPPED, METRICS)
    other = EvalResult(ITEMS, DROPPED, dict(reversed(list(METRICS.items()))))

    assert one == other
    assert hash(one) == hash(other)


@pytest.mark.parametrize(
    "moved",
    [
        EvalResult(ITEMS + 1, DROPPED, METRICS),
        EvalResult(ITEMS, DROPPED + 1, METRICS),
        EvalResult(ITEMS, DROPPED, {"loss": 0.5}),
    ],
)
def test_moving_any_field_of_an_eval_result_moves_it_out_of_its_class(
    moved: EvalResult,
) -> None:
    assert EvalResult(ITEMS, DROPPED, METRICS) != moved


@pytest.mark.parametrize("value", [inf, nan, -inf])
def test_a_metric_that_is_not_finite_is_a_numerical_failure_where_it_enters(
    value: float,
) -> None:
    with pytest.raises(NumericalFailureError, match="loss") as training:
        TrainResult(STEP, ITEMS, DROPPED, {"loss": value})

    with pytest.raises(NumericalFailureError, match="loss") as validation:
        EvalResult(ITEMS, DROPPED, {"loss": value})

    assert training.value.quantity == validation.value.quantity == "metrics['loss']"
    assert str(training.value) == f"metrics['loss'] is not finite: {value!r}"
    assert isinstance(training.value, FloatingPointError)


def test_a_metric_that_is_not_numeric_is_refused_where_it_enters() -> None:
    with pytest.raises(TypeError, match="loss"):
        EvalResult(ITEMS, DROPPED, cast("Mapping[str, float]", {"loss": "small"}))


def test_a_metric_under_an_empty_name_is_refused_where_it_enters() -> None:
    with pytest.raises(ValueError, match="metric name"):
        TrainResult(STEP, ITEMS, DROPPED, {"": 0.5})


@pytest.mark.parametrize("negative", [-1])
def test_a_negative_count_is_refused_where_it_enters(negative: int) -> None:
    with pytest.raises(ValueError, match="dropped"):
        EvalResult(ITEMS, negative)


def test_the_metrics_a_result_hands_out_are_no_container_a_caller_can_write() -> None:
    result = TrainResult(STEP, ITEMS, DROPPED, METRICS)

    assert not isinstance(result.metrics, dict | list | set)


def test_writing_into_the_metrics_a_result_was_built_from_changes_nothing() -> None:
    built = dict(METRICS)
    result = TrainResult(STEP, ITEMS, DROPPED, built)

    built["loss"] = 9.0

    assert result.metrics["loss"] == METRICS["loss"]


@pytest.mark.parametrize("field", ["step", "items", "dropped", "metrics", "extra"])
def test_a_train_result_takes_on_no_attribute_after_it_is_built(field: str) -> None:
    with pytest.raises(AttributeError):
        setattr(TrainResult(STEP, ITEMS, DROPPED, METRICS), field, None)


def test_a_result_reports_the_call_that_rebuilds_it() -> None:
    printed = repr(TrainResult(STEP, ITEMS, DROPPED, {"loss": 0.5}))

    assert printed == (
        f"TrainResult(step={STEP}, items={ITEMS}, dropped={DROPPED}, "
        "metrics={'loss': 0.5})"
    )


def test_an_eval_result_reports_the_call_that_rebuilds_it() -> None:
    printed = repr(EvalResult(ITEMS, DROPPED, {"loss": 0.5}))

    assert printed == (
        f"EvalResult(items={ITEMS}, dropped={DROPPED}, metrics={{'loss': 0.5}})"
    )


def test_an_outcome_holds_what_was_published_beside_what_was_reported() -> None:
    result = TrainResult(STEP, ITEMS, DROPPED, METRICS)
    outcome = StepOutcome((0.5, 0.25), result)

    assert outcome.outputs == (0.5, 0.25)
    assert outcome.result == result
    assert repr(outcome).startswith("StepOutcome((0.5, 0.25), TrainResult(")


@pytest.mark.parametrize("field", ["outputs", "result", "extra"])
def test_an_outcome_takes_on_no_attribute_after_it_is_built(field: str) -> None:
    with pytest.raises(AttributeError):
        setattr(StepOutcome((), EvalResult(0)), field, None)


def test_an_edge_reads_back_the_two_steps_and_the_value_it_joins_them_over() -> None:
    edge = StepEdge("featurizing", "weights", "scoring")

    assert (edge.producer, edge.value, edge.consumer) == (
        "featurizing",
        "weights",
        "scoring",
    )


def test_two_edges_over_the_same_three_names_are_equal_and_hash_alike() -> None:
    one = StepEdge("featurizing", "weights", "scoring")
    other = StepEdge("featurizing", "weights", "scoring")

    assert one == other
    assert hash(one) == hash(other)


@pytest.mark.parametrize(
    "moved",
    [
        StepEdge("elsewhere", "weights", "scoring"),
        StepEdge("featurizing", "examples", "scoring"),
        StepEdge("featurizing", "weights", "reporting"),
    ],
)
def test_moving_any_name_of_an_edge_moves_it_out_of_its_class(moved: StepEdge) -> None:
    assert StepEdge("featurizing", "weights", "scoring") != moved


def test_an_edge_is_not_an_arbitrary_triple() -> None:
    assert StepEdge("featurizing", "weights", "scoring") != (
        "featurizing",
        "weights",
        "scoring",
    )


def test_an_edge_refuses_a_step_reading_from_itself() -> None:
    with pytest.raises(ValueError, match="from itself"):
        StepEdge("scoring", "weights", "scoring")


def test_an_edge_refuses_an_empty_name() -> None:
    with pytest.raises(ValueError, match="value is empty"):
        StepEdge("featurizing", "", "scoring")


def test_an_edge_reports_the_call_that_rebuilds_it() -> None:
    printed = repr(StepEdge("featurizing", "weights", "scoring"))

    assert printed == "StepEdge('featurizing', 'weights', 'scoring')"


def test_a_graph_has_one_edge_per_value_a_step_reads_from_another() -> None:
    assert edges(GRAPH, OUTSIDE) == (
        StepEdge("scoring", "scores", "reporting"),
        StepEdge("featurizing", "weights", "scoring"),
    )


def test_the_edges_come_back_in_consumer_order_whatever_order_the_steps_are_in() -> (
    None
):
    assert edges(tuple(reversed(GRAPH)), OUTSIDE) == edges(GRAPH, OUTSIDE)


def test_a_value_entering_from_outside_reaches_its_reader_on_no_edge() -> None:
    reading = frozenset(edge.value for edge in edges(GRAPH, OUTSIDE))

    assert "examples" not in reading


def test_a_value_nothing_supplies_is_refused_rather_than_left_off_the_graph() -> None:
    with pytest.raises(StepGraphError, match="examples"):
        edges(GRAPH)


def test_two_steps_publishing_one_value_are_refused() -> None:
    twin = Declared("twin", frozenset(), frozenset({"weights"}))

    with pytest.raises(StepGraphError, match="weights"):
        edges((*GRAPH, twin), OUTSIDE)


def test_two_steps_of_one_name_are_refused() -> None:
    twin = Declared("scoring", frozenset(), frozenset({"other"}))

    with pytest.raises(StepGraphError, match="two steps are named"):
        edges((*GRAPH, twin), OUTSIDE)


def test_a_graph_of_one_step_reading_only_from_outside_has_no_edge() -> None:
    assert edges((FEATURIZING,), OUTSIDE) == ()
