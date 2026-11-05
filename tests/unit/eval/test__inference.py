from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.data import RESPONSE_MISSING, Categorical
from lofi.eval._inference import (
    InferenceLabelMetric,
    InferenceMeasure,
    LabelReduction,
    UndeclaredLabelError,
    declared_labels,
)
from lofi.eval._ports import Metric
from lofi.eval._values import ConventionVector
from lofi.eval.laws import MetricFixture, check_metric_laws

if TYPE_CHECKING:
    from collections.abc import Sequence

LABELS: Final = frozenset({"entailment", "neutral", "contradiction"})

TWO_WAY: Final = LabelReduction(
    {
        "entailment": "entailment",
        "neutral": "non-entailment",
        "contradiction": "non-entailment",
    }
)

# asymmetric twelve-item confusion matrix, one row per gold class
#
#   gold \ pred    entailment  neutral  contradiction
#   entailment              3        1              0
#   neutral                 2        2              1
#   contradiction           0        1              2
_MATRIX: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "entailment",
        ("entailment", "entailment", "entailment", "neutral"),
    ),
    (
        "neutral",
        ("entailment", "entailment", "neutral", "neutral", "contradiction"),
    ),
    (
        "contradiction",
        ("neutral", "contradiction", "contradiction"),
    ),
)


def _items() -> tuple[tuple[ItemId, str | None, Categorical], ...]:
    scored: list[tuple[ItemId, str | None, Categorical]] = []

    for gold, predictions in _MATRIX:
        for predicted in predictions:
            key = ItemId(f"i-{len(scored):02d}")
            scored.append((key, predicted, Categorical(gold, LABELS)))

    return tuple(scored)


ITEMS: Final = _items()


def _conventions(coverage: Literal["zero", "ignore"] = "zero") -> ConventionVector:
    return ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage=coverage,
        self_arcs="forbid",
        gold_derivation="canonical",
    )


def _metric(
    measure: InferenceMeasure,
    target: str | None = None,
    *,
    reduction: LabelReduction | None = None,
    coverage: Literal["zero", "ignore"] = "zero",
) -> InferenceLabelMetric:
    return InferenceLabelMetric(
        f"row-{measure}" if target is None else f"row-{measure}-{target}",
        LABELS,
        measure=measure,
        target=target,
        reduction=reduction,
        conventions=_conventions(coverage),
    )


def _scored(
    measure: InferenceMeasure,
    target: str | None = None,
    *,
    reduction: LabelReduction | None = None,
    items: Sequence[tuple[ItemId, str | None, Categorical]] = ITEMS,
    coverage: Literal["zero", "ignore"] = "zero",
) -> InferenceLabelMetric:
    metric = _metric(measure, target, reduction=reduction, coverage=coverage)

    for key, predicted, gold in items:
        metric.update(predicted, gold, key=key)

    return metric


def test_the_twelve_item_matrix_gives_the_hand_computed_accuracy_and_macro_f1() -> None:
    accuracy = _scored("accuracy").compute()
    macro = _scored("macro-f1").compute()

    assert accuracy.value == 7 / 12
    assert accuracy.components == {"correct": 7.0, "scored": 12.0}
    assert macro.value == pytest.approx(16 / 27)
    assert macro.components == {"classes": 3.0}


def test_the_nine_per_class_cells_are_the_hand_computed_ones() -> None:
    classes = ("contradiction", "entailment", "neutral")

    def row(measure: InferenceMeasure) -> dict[str, float | None]:
        return {label: _scored(measure, label).compute().value for label in classes}

    assert row("precision") == {
        "contradiction": 2 / 3,
        "entailment": 0.6,
        "neutral": 0.5,
    }
    assert row("recall") == {"contradiction": 2 / 3, "entailment": 0.75, "neutral": 0.4}
    assert row("f1") == {"contradiction": 2 / 3, "entailment": 2 / 3, "neutral": 4 / 9}


def test_a_cell_reports_the_counts_it_was_read_off() -> None:
    entailment = _scored("precision", "entailment").compute()

    assert entailment.components == {"matched": 3.0, "predicted": 5.0, "gold": 4.0}
    assert entailment.n == 5


def test_the_metric_draws_no_interval_of_its_own() -> None:
    assert _scored("accuracy").compute().ci is None
    assert _scored("recall", "neutral").compute().ci is None


def test_the_two_way_reduction_scores_the_hand_computed_coarser_matrix() -> None:
    accuracy = _scored("accuracy", reduction=TWO_WAY)
    coarse = _scored("f1", "non-entailment", reduction=TWO_WAY).compute()

    assert accuracy.classes == ("entailment", "non-entailment")
    assert accuracy.compute().value == 9 / 12
    assert _scored("macro-f1", reduction=TWO_WAY).compute().value == pytest.approx(
        11 / 15
    )
    assert coarse.value == 0.8


def test_nothing_collapses_two_classes_into_one_unless_a_reduction_says_so() -> None:
    assert _metric("accuracy").classes == ("contradiction", "entailment", "neutral")
    assert _metric("accuracy", reduction=TWO_WAY).labels == LABELS


def test_a_predicted_label_outside_the_declared_inventory_is_refused() -> None:
    metric = _metric("accuracy")

    with pytest.raises(UndeclaredLabelError, match="not in the declared inventory"):
        metric.update("entailed", Categorical("entailment", LABELS), key=ItemId("i-00"))


def test_a_gold_response_from_another_inventory_is_refused() -> None:
    metric = _metric("accuracy")
    elsewhere = frozenset({"yes", "no"})

    with pytest.raises(UndeclaredLabelError, match="declares the inventory"):
        metric.update("entailment", Categorical("yes", elsewhere), key=ItemId("i-00"))


def test_a_reduction_that_is_not_total_over_the_inventory_is_refused() -> None:
    partial = LabelReduction({"entailment": "a", "neutral": "b", "other": "b"})

    with pytest.raises(ValueError, match="is not total over"):
        _metric("accuracy", reduction=partial)


def test_an_inventory_of_one_label_poses_no_classification() -> None:
    with pytest.raises(ValueError, match="needs at least"):
        InferenceLabelMetric(
            "row",
            frozenset({"entailment"}),
            measure="accuracy",
            target=None,
            reduction=None,
            conventions=_conventions(),
        )


def test_the_metric_refuses_a_reduction_of_the_wrong_type() -> None:
    with pytest.raises(TypeError, match="reduction is not one"):
        _metric("accuracy", reduction=cast("LabelReduction", {"entailment": "a"}))


def test_a_target_must_fit_the_measure() -> None:
    with pytest.raises(ValueError, match="not among the scored classes"):
        _metric("precision", "yes")

    with pytest.raises(ValueError, match="not among the scored classes"):
        _metric("recall", "neutral", reduction=TWO_WAY)

    with pytest.raises(ValueError, match="has a class target"):
        _metric("accuracy", "neutral")


def test_a_reduction_that_leaves_the_inventory_where_it_was_is_not_one() -> None:
    with pytest.raises(ValueError, match="reduces nothing"):
        LabelReduction({"a": "x", "b": "y"})

    with pytest.raises(ValueError, match="needs at least"):
        LabelReduction({"a": "x", "b": "x"})

    with pytest.raises(ValueError, match="maps nothing"):
        LabelReduction({})


def test_a_reduction_refuses_a_label_it_does_not_map() -> None:
    with pytest.raises(UndeclaredLabelError, match="not one this reduction maps"):
        TWO_WAY.apply("yes")

    assert TWO_WAY.apply("neutral") == "non-entailment"
    assert TWO_WAY.domain == LABELS
    assert LabelReduction(dict(TWO_WAY.mapping)) == TWO_WAY
    assert hash(TWO_WAY) == hash(LabelReduction(dict(TWO_WAY.mapping)))


def test_a_metric_over_nothing_has_no_value_and_no_items() -> None:
    metric = _metric("accuracy")
    result = metric.compute()

    assert result.value is None
    assert result.n == 0
    assert dict(metric.per_item()) == {}
    assert _metric("macro-f1").compute().value is None


def test_a_class_nothing_was_predicted_for_has_no_precision_rather_than_zero() -> None:
    precision = _scored("precision", "contradiction", items=ITEMS[:4]).compute()

    assert precision.value is None
    assert precision.components["predicted"] == 0.0
    assert _scored("recall", "contradiction", items=ITEMS[:4]).compute().value is None


def test_each_item_decomposes_to_its_share_of_a_target_cell() -> None:
    recall = _scored("recall", "entailment").per_item()

    assert set(recall) == {key for key, _, _ in ITEMS}
    assert recall[ItemId("i-00")].value == 1.0
    assert recall[ItemId("i-03")].value == 0.0
    assert recall[ItemId("i-08")].value is None


def test_scoring_one_id_twice_leaves_the_metric_where_it_was() -> None:
    metric = _scored("accuracy")
    key, predicted, gold = ITEMS[0]
    metric.update(predicted, gold, key=key)

    assert metric.compute().value == 7 / 12
    assert len(metric.per_item()) == 12


def test_scoring_one_id_against_two_pairs_is_refused() -> None:
    metric = _scored("accuracy")

    with pytest.raises(ValueError, match="already scored against a different pair"):
        metric.update("neutral", Categorical("entailment", LABELS), key=ITEMS[0][0])


def test_a_gold_response_of_the_wrong_type_is_refused() -> None:
    metric = _metric("accuracy")

    with pytest.raises(TypeError, match="not a categorical one"):
        metric.update("entailment", cast("Categorical", "entailment"), key=ItemId("a"))


def test_a_missing_gold_label_is_refused_before_scoring() -> None:
    metric = _metric("accuracy")
    missing = Categorical(
        None,
        LABELS,
        features=frozenset({RESPONSE_MISSING}),
    )

    with pytest.raises(ValueError, match="missing gold inference label"):
        metric.update("entailment", missing, key=ItemId("a"))

    assert metric.compute().n == 0


def test_an_abstention_counts_as_wrong_under_zero_coverage_and_drops_otherwise() -> (
    None
):
    abstained = (*ITEMS, (ItemId("i-12"), None, Categorical("neutral", LABELS)))

    assert _scored("accuracy", items=abstained).compute().value == 7 / 13
    assert (
        _scored("accuracy", items=abstained, coverage="ignore").compute().value
        == 7 / 12
    )

    abstention = _scored("abstention", items=abstained).compute()

    assert abstention.value == 1 / 13
    assert abstention.components == {"abstained": 1.0, "total": 13.0}


def test_a_reset_metric_reports_what_a_fresh_one_reports() -> None:
    metric = _scored("accuracy")
    metric.reset()

    assert metric.compute() == _metric("accuracy").compute()
    assert dict(metric.per_item()) == {}


def test_the_order_the_items_arrived_in_does_not_move_the_number() -> None:
    forward = _scored("macro-f1")
    backward = _scored("macro-f1", items=tuple(reversed(ITEMS)))

    assert forward.order_invariant
    assert forward.compute() == backward.compute()


def test_the_inventory_is_read_off_the_responses_the_evaluation_set_declares() -> None:
    assert declared_labels(gold for _, _, gold in ITEMS) == LABELS

    with pytest.raises(UndeclaredLabelError, match="declare 2 inventories"):
        declared_labels(
            [
                Categorical("entailment", LABELS),
                Categorical("yes", frozenset({"yes", "no"})),
            ]
        )

    with pytest.raises(ValueError, match="no inventory was declared"):
        declared_labels([])

    with pytest.raises(TypeError, match="not a categorical one"):
        declared_labels([cast("Categorical", "entailment")])


def _build() -> InferenceLabelMetric:
    return _metric("accuracy")


def test_the_inference_metric_holds_to_every_metric_law() -> None:
    pool = [(str(key), predicted, gold) for key, predicted, gold in ITEMS[:6]]

    check_metric_laws(MetricFixture(_build, st.sampled_from(pool), max_examples=20))


def test_the_metric_is_usable_through_the_protocol_a_report_holds_it_by() -> None:
    held: Metric[str | None, Categorical] = _build()
    key, predicted, gold = ITEMS[0]
    held.update(predicted, gold, key=key)

    assert isinstance(held, Metric)
    assert held.compute().value == 1.0
    assert held.conventions == _conventions()
