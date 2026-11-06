from math import log
from typing import Final, cast

import pytest
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.eval._bracketing import Constituent, GoldBracketing
from lofi.eval._matched import (
    MATCHED_ROWS,
    InducedBracketing,
    RecallHomogeneity,
)
from lofi.eval._ports import Metric
from lofi.eval._values import ConventionVector
from lofi.eval.laws import MetricFixture, check_metric_laws

_KEEPING: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="corpus",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_STRIPPING: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="corpus",
    coverage="ignore",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_GOLD: Final = GoldBracketing(
    6,
    [
        Constituent(0, 6, label="S", depth=0),
        Constituent(1, 6, label="VP", depth=1),
        Constituent(4, 6, label="NP", depth=2),
        Constituent(2, 4, label="PP", depth=2),
    ],
    punctuation=(),
)

_PRED: Final = InducedBracketing(
    6, {(0, 6): "c1", (1, 6): "c1", (4, 6): "c2", (0, 3): "c3"}
)

_HOMOGENEITY: Final = 1.0 - (2.0 / 3.0) * log(2.0) / log(3.0)
_V_MEASURE: Final = 2.0 * _HOMOGENEITY / (_HOMOGENEITY + 1.0)
_RECALL: Final = 3.0 / 4.0


def _metric(
    conventions: ConventionVector = _KEEPING, *, seed: int | None = 5
) -> RecallHomogeneity:
    return RecallHomogeneity(conventions, split="test", seed=seed)


def _scored(pred: InducedBracketing | None = _PRED) -> RecallHomogeneity:
    metric = _metric()
    metric.update(pred, _GOLD, key=ItemId("s-1"))

    return metric


def test_the_fixture_matches_the_value_derived_by_hand() -> None:
    rows = _scored().rows()

    assert rows["unlabeled-recall"].value == _RECALL
    assert rows["homogeneity"].value == pytest.approx(_HOMOGENEITY)
    assert rows["v-measure"].value == pytest.approx(_V_MEASURE)
    assert rows["recall-homogeneity"].value == pytest.approx(_RECALL * _HOMOGENEITY)
    assert rows["recall-v-measure"].value == pytest.approx(_RECALL * _V_MEASURE)


def test_the_matched_set_holds_the_three_spans_the_prediction_found() -> None:
    counts = dict(_scored().contingency().counts)

    assert counts == {("c1", "S"): 1, ("c1", "VP"): 1, ("c2", "NP"): 1}


def test_the_factors_multiply_back_to_each_product() -> None:
    rows = _scored().rows()
    recall = rows["unlabeled-recall"].value
    pure = rows["homogeneity"].value
    mixed = rows["v-measure"].value

    assert recall is not None
    assert pure is not None
    assert mixed is not None
    assert rows["recall-homogeneity"].value == pytest.approx(recall * pure)
    assert rows["recall-v-measure"].value == pytest.approx(recall * mixed)


def test_every_declared_row_is_emitted_in_its_declared_order() -> None:
    rows = _scored().rows()

    assert rows["unlabeled-recall"].resampling.unit == "gold constituent"
    assert rows["homogeneity"].resampling.unit == "matched constituent"
    assert rows["recall-homogeneity"].resampling.unit == "sentence"


def test_the_products_carry_every_factor_among_their_components() -> None:
    components = _scored().compute().components

    assert components["unlabeled_recall"] == _RECALL
    assert components["homogeneity"] == pytest.approx(_HOMOGENEITY)
    assert components["v_measure"] == pytest.approx(_V_MEASURE)
    assert components["recall_v_measure"] == pytest.approx(_RECALL * _V_MEASURE)
    assert components["matched"] == 3.0
    assert components["gold"] == 4.0
    assert components["scored"] == 1.0
    assert components["uncovered"] == 0.0


def test_renaming_the_induced_labels_leaves_both_products_where_they_were() -> None:
    renamed = InducedBracketing(
        6,
        {
            span: {"c1": "z9", "c2": "a0", "c3": "m4"}[label]
            for span, label in _PRED.spans.items()
        },
    )
    before = _scored().rows()
    after = _scored(renamed).rows()

    for name in MATCHED_ROWS:
        assert after[name].value == pytest.approx(before[name].value)


def test_refining_the_induced_labels_does_not_lower_recall_homogeneity() -> None:
    refined = InducedBracketing(
        6,
        {(0, 6): "c1a", (1, 6): "c1b", (4, 6): "c2", (0, 3): "c3"},
    )
    before = _scored().compute().value
    after = _scored(refined).compute().value

    assert before is not None
    assert after is not None
    assert after >= before
    assert after == pytest.approx(_RECALL)


def test_a_split_that_separates_two_gold_labels_raises_homogeneity_to_one() -> None:
    refined = InducedBracketing(
        6,
        {(0, 6): "c1a", (1, 6): "c1b", (4, 6): "c2", (0, 3): "c3"},
    )
    rows = _scored(refined).rows()

    assert rows["homogeneity"].value == pytest.approx(1.0)
    assert rows["unlabeled-recall"].value == _RECALL


def test_a_metric_over_no_update_has_no_value_and_no_items() -> None:
    metric = _metric()
    scored = metric.compute()

    assert scored.value is None
    assert scored.n == 0
    assert not metric.per_item()
    assert metric.rows()["homogeneity"].value is None


def test_a_sentence_nothing_matched_leaves_the_products_absent() -> None:
    metric = _metric()
    metric.update(InducedBracketing(6, {(0, 3): "c3"}), _GOLD, key=ItemId("s-1"))
    rows = metric.rows()

    assert rows["unlabeled-recall"].value == 0.0
    assert rows["homogeneity"].value is None
    assert rows["recall-homogeneity"].value is None
    assert "recall_homogeneity" not in rows["recall-homogeneity"].components


def test_an_uncovered_sentence_is_counted_under_the_zero_penalty() -> None:
    metric = _metric()
    metric.update(None, _GOLD, key=ItemId("s-1"))
    metric.update(_PRED, _GOLD, key=ItemId("s-2"))
    components = metric.compute().components

    assert components["uncovered"] == 1.0
    assert components["gold"] == 8.0
    assert components["matched"] == 3.0
    assert metric.rows()["unlabeled-recall"].value == 3.0 / 8.0


def test_an_uncovered_sentence_is_dropped_under_the_ignore_penalty() -> None:
    metric = _metric(_STRIPPING)
    metric.update(None, _GOLD, key=ItemId("s-1"))
    components = metric.compute().components

    assert components["scored"] == 0.0
    assert components["dropped"] == 1.0
    assert metric.compute().value is None


def test_the_conventions_drop_a_sentence_shorter_than_the_declared_minimum() -> None:
    short = GoldBracketing(2, [Constituent(0, 2, label="S", depth=0)], punctuation=(1,))
    metric = _metric(_STRIPPING)
    metric.update(InducedBracketing(2, {(0, 2): "c1"}), short, key=ItemId("s-1"))

    assert metric.compute().value is None
    assert metric.compute().components["dropped"] == 1.0


def test_stripping_punctuation_reindexes_both_sides() -> None:
    gold = GoldBracketing(
        5,
        [
            Constituent(0, 5, label="S", depth=0),
            Constituent(0, 2, label="NP", depth=1),
            Constituent(2, 4, label="VP", depth=1),
        ],
        punctuation=(4,),
    )
    pred = InducedBracketing(5, {(0, 2): "c1", (2, 4): "c2", (0, 5): "c3"})
    metric = _metric(_STRIPPING)
    metric.update(pred, gold, key=ItemId("s-1"))
    counts = dict(metric.contingency().counts)

    assert counts == {("c1", "NP"): 1, ("c2", "VP"): 1}
    assert metric.rows()["unlabeled-recall"].value == 1.0


def test_a_span_the_conventions_collapse_keeps_the_wider_original_label() -> None:
    gold = GoldBracketing(
        4,
        [
            Constituent(0, 4, label="S", depth=0),
            Constituent(0, 2, label="NP", depth=1),
        ],
        punctuation=(2,),
    )
    pred = InducedBracketing(4, {(0, 2): "narrow", (0, 3): "wide"})
    metric = RecallHomogeneity(
        ConventionVector(
            punctuation="strip",
            unary_chains="keep",
            trivial_spans="keep",
            minimum_length=0,
            aggregation="corpus",
            coverage="zero",
            self_arcs="forbid",
            gold_derivation="canonical",
        ),
        split="dev",
    )
    metric.update(pred, gold, key=ItemId("s-1"))
    counts = dict(metric.contingency().counts)

    assert ("wide", "NP") in counts
    assert ("narrow", "NP") not in counts


def test_per_item_reports_each_sentence_under_its_own_key() -> None:
    metric = _metric()
    metric.update(_PRED, _GOLD, key=ItemId("s-1"))
    metric.update(None, _GOLD, key=ItemId("s-2"))
    items = metric.per_item()

    assert set(items) == {ItemId("s-1"), ItemId("s-2")}
    assert items[ItemId("s-1")].value == pytest.approx(_RECALL * _HOMOGENEITY)
    assert items[ItemId("s-2")].value is None


def test_resetting_forgets_every_sentence() -> None:
    metric = _scored()
    metric.reset()

    assert metric.compute().value is None
    assert not metric.per_item()


def test_scoring_one_id_twice_against_one_analysis_changes_nothing() -> None:
    metric = _scored()
    metric.update(_PRED, _GOLD, key=ItemId("s-1"))

    assert metric.compute().components["scored"] == 1.0


def test_scoring_one_id_against_two_analyses_is_refused() -> None:
    metric = _scored()

    with pytest.raises(ValueError, match="already scored"):
        metric.update(InducedBracketing(6, {(0, 6): "c9"}), _GOLD, key=ItemId("s-1"))


def test_a_prediction_of_another_length_is_refused() -> None:
    metric = _metric()

    with pytest.raises(ValueError, match="tokens was scored against"):
        metric.update(InducedBracketing(5, {(0, 5): "c1"}), _GOLD, key=ItemId("s-1"))


def test_a_prediction_or_gold_analysis_of_the_wrong_kind_is_refused() -> None:
    metric = _metric()

    with pytest.raises(TypeError, match="gold bracketing is not one"):
        metric.update(_PRED, cast("GoldBracketing", _PRED), key=ItemId("s-1"))

    with pytest.raises(TypeError, match="labeled bracketing is not one"):
        metric.update(cast("InducedBracketing", _GOLD), _GOLD, key=ItemId("s-2"))


def test_the_metric_declares_its_conventions_and_its_order_invariance() -> None:
    metric = _metric()

    assert metric.name == "recall-homogeneity"
    assert metric.conventions is _KEEPING
    assert metric.order_invariant
    assert metric.split == "test"
    assert metric.beta == 1.0
    assert isinstance(metric, Metric)


def test_a_metric_built_without_a_vector_is_refused() -> None:
    with pytest.raises(TypeError, match="conventions are not a vector"):
        RecallHomogeneity(cast("ConventionVector", "keep"), split="dev")


def test_a_negative_or_infinite_beta_is_refused() -> None:
    with pytest.raises(ValueError, match="beta is negative"):
        RecallHomogeneity(_KEEPING, split="dev", beta=-1.0)

    with pytest.raises(ValueError, match="finite"):
        RecallHomogeneity(_KEEPING, split="dev", beta=float("inf"))


def test_an_induced_bracketing_refuses_a_span_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="not a start and an end"):
        InducedBracketing(4, {cast("tuple[int, int]", (1, 2, 3)): "c1"})

    with pytest.raises(ValueError, match="covers no token"):
        InducedBracketing(4, {(2, 2): "c1"})

    with pytest.raises(ValueError, match="runs past"):
        InducedBracketing(4, {(2, 9): "c1"})

    with pytest.raises(ValueError, match="induced label is empty"):
        InducedBracketing(4, {(0, 4): ""})


def test_two_induced_bracketings_of_the_same_spans_are_equal_and_hash_alike() -> None:
    first = InducedBracketing(4, {(0, 4): "c1", (0, 2): "c2"})
    second = InducedBracketing(4, {(0, 2): "c2", (0, 4): "c1"})

    assert first == second
    assert hash(first) == hash(second)
    assert first.labels == ("c1", "c2")
    assert first != InducedBracketing(4, {(0, 4): "c1"})
    assert first != "a bracketing"
    assert repr(first).startswith("InducedBracketing(4, {")


def _build() -> RecallHomogeneity:
    return RecallHomogeneity(_KEEPING, split="dev", seed=1)


type _Example = tuple[str, InducedBracketing | None, GoldBracketing]

_POOL: Final[tuple[_Example, ...]] = (
    ("s-a", _PRED, _GOLD),
    ("s-b", InducedBracketing(6, {(0, 6): "c2", (2, 4): "c1"}), _GOLD),
    ("s-c", None, _GOLD),
    ("s-d", InducedBracketing(6, {(4, 6): "c1", (2, 4): "c1"}), _GOLD),
)


def test_the_metric_passes_every_metric_law() -> None:
    check_metric_laws(MetricFixture(_build, st.sampled_from(_POOL), max_examples=25))
