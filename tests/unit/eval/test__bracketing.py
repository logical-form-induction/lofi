from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.eval._bracketing import (
    Bracketing,
    BracketingF1,
    Constituent,
    GoldBracketing,
    balanced,
    left_branching,
    right_branching,
)
from lofi.eval._ports import Metric
from lofi.eval._values import ConventionVector
from lofi.eval.laws import MetricFixture, check_metric_laws

if TYPE_CHECKING:
    from collections.abc import Sequence


_BASE: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_POOLED: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="corpus",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_TRIVIAL_KEPT: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="keep",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_PUNCTUATION_KEPT: Final = ConventionVector(
    punctuation="keep",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_CHAINS_KEPT: Final = ConventionVector(
    punctuation="strip",
    unary_chains="keep",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_SHORT_SCORED: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_SHORT_DROPPED: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="keep",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_COVERAGE_IGNORED: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="ignore",
    self_arcs="forbid",
    gold_derivation="canonical",
)

_SELF_ARCS_ALLOWED: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)

_ORACLE_GOLD: Final = ConventionVector(
    punctuation="strip",
    unary_chains="collapse",
    trivial_spans="drop",
    minimum_length=3,
    aggregation="sentence",
    coverage="zero",
    self_arcs="forbid",
    gold_derivation="oracle",
)

# gold: [DT NN] [V [DT [IN NN]]]. prediction attaches the object one node lower
GOLD_A: Final = GoldBracketing(
    6,
    [
        Constituent(0, 6, label="S", depth=0),
        Constituent(0, 2, label="NP", depth=1),
        Constituent(0, 1, label="DT", depth=2),
        Constituent(1, 2, label="NN", depth=2),
        Constituent(2, 6, label="VP", depth=1),
        Constituent(2, 3, label="V", depth=2),
        Constituent(3, 6, label="NP", depth=2),
        Constituent(3, 4, label="DT", depth=3),
        Constituent(4, 6, label="PP", depth=3),
        Constituent(4, 5, label="IN", depth=4),
        Constituent(5, 6, label="NN", depth=4),
    ],
    punctuation=(),
)
PRED_A: Final = Bracketing(
    6,
    [
        (0, 6),
        (0, 2),
        (2, 6),
        (2, 4),
        (4, 6),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ],
)

# exact four-token prediction
GOLD_B: Final = GoldBracketing(
    4,
    [
        Constituent(0, 4, label="S", depth=0),
        Constituent(0, 1, label="N", depth=1),
        Constituent(1, 4, label="VP", depth=1),
        Constituent(1, 2, label="V", depth=2),
        Constituent(2, 4, label="NP", depth=2),
        Constituent(2, 3, label="D", depth=3),
        Constituent(3, 4, label="N", depth=3),
    ],
    punctuation=(),
)
PRED_B: Final = Bracketing(4, [(0, 4), (1, 4), (2, 4), (0, 1), (1, 2), (2, 3), (3, 4)])

# final punctuation attaches to the object instead of the clause
GOLD_C: Final = GoldBracketing(
    5,
    [
        Constituent(0, 5, label="S", depth=0),
        Constituent(0, 2, label="NP", depth=1),
        Constituent(0, 1, label="D", depth=2),
        Constituent(1, 2, label="N", depth=2),
        Constituent(2, 5, label="SBAR", depth=1),
        Constituent(2, 4, label="VP", depth=2),
        Constituent(2, 3, label="V", depth=3),
        Constituent(3, 4, label="N", depth=3),
        Constituent(4, 5, label="PUNCT", depth=2),
    ],
    punctuation=(4,),
)
PRED_C: Final = Bracketing(
    5, [(0, 5), (0, 2), (2, 5), (3, 5), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
)

# VP over NP unary chain
GOLD_D: Final = GoldBracketing(
    4,
    [
        Constituent(0, 4, label="S", depth=0),
        Constituent(0, 1, label="N", depth=1),
        Constituent(1, 4, label="VP", depth=1),
        Constituent(1, 4, label="NP", depth=2),
        Constituent(1, 2, label="V", depth=3),
        Constituent(2, 4, label="PP", depth=3),
        Constituent(2, 3, label="P", depth=4),
        Constituent(3, 4, label="N", depth=4),
    ],
    punctuation=(),
)
PRED_D: Final = Bracketing(4, [(0, 4), (1, 4), (2, 4), (0, 1), (1, 2), (2, 3), (3, 4)])

# two-token sentence
GOLD_E: Final = GoldBracketing(
    2,
    [
        Constituent(0, 2, label="NP", depth=0),
        Constituent(0, 1, label="D", depth=1),
        Constituent(1, 2, label="N", depth=1),
    ],
    punctuation=(),
)
PRED_E: Final = Bracketing(2, [(0, 1), (1, 2)])

type _Example = tuple[str, Bracketing | None, GoldBracketing]

_POOL: Final[tuple[_Example, ...]] = (
    ("s-a", PRED_A, GOLD_A),
    ("s-b", PRED_B, GOLD_B),
    ("s-c", PRED_C, GOLD_C),
    ("s-d", None, GOLD_D),
    ("s-e", PRED_E, GOLD_E),
)


def _scored(conventions: ConventionVector, items: Sequence[_Example]) -> BracketingF1:
    metric = BracketingF1(conventions)

    for key, pred, gold in items:
        metric.update(pred, gold, key=ItemId(key))

    return metric


def _value(conventions: ConventionVector, items: Sequence[_Example]) -> float | None:
    return _scored(conventions, items).compute().value


_TWO: Final[tuple[_Example, ...]] = (("s-a", PRED_A, GOLD_A), ("s-b", PRED_B, GOLD_B))


def test_a_constituent_reports_its_span_its_width_and_its_depth() -> None:
    entry = Constituent(2, 6, label="VP", depth=1)

    assert entry.span == (2, 6)
    assert entry.width == 4
    assert entry.label == "VP"
    assert entry.depth == 1
    assert not hasattr(entry, "__dict__")


def test_a_constituent_covering_no_token_is_not_one() -> None:
    with pytest.raises(ValueError, match="covers no token"):
        Constituent(3, 3, label="NP", depth=0)

    with pytest.raises(ValueError, match="covers no token"):
        Constituent(4, 2, label="NP", depth=0)


def test_a_constituent_refuses_an_empty_label_and_a_negative_depth() -> None:
    with pytest.raises(ValueError, match="a constituent's label is empty"):
        Constituent(0, 2, label="", depth=0)

    with pytest.raises(ValueError, match="negative"):
        Constituent(0, 2, label="NP", depth=-1)


def test_a_bracketing_holds_its_spans_once_and_in_order() -> None:
    bracketing = Bracketing(4, [(2, 4), (0, 4), (2, 4)])

    assert bracketing.spans == ((0, 4), (2, 4))
    assert bracketing.length == 4


def test_a_bracketing_refuses_a_span_outside_the_sentence() -> None:
    with pytest.raises(ValueError, match="runs past a sentence of 4 tokens"):
        Bracketing(4, [(2, 5)])

    with pytest.raises(ValueError, match="covers no token"):
        Bracketing(4, [(2, 2)])


def test_a_bracketing_refuses_a_pair_that_is_not_a_span() -> None:
    spans = cast("list[tuple[int, int]]", [(1, 2, 3)])

    with pytest.raises(TypeError, match="not a start and an end"):
        Bracketing(4, spans)


def test_a_gold_bracketing_refuses_a_constituent_or_a_mark_outside_it() -> None:
    with pytest.raises(ValueError, match="runs past a sentence of 3 tokens"):
        GoldBracketing(3, [Constituent(0, 4, label="S", depth=0)], punctuation=())

    with pytest.raises(ValueError, match="falls outside a sentence of 3 tokens"):
        GoldBracketing(3, [], punctuation=(3,))

    with pytest.raises(TypeError, match="a gold constituent is not one"):
        GoldBracketing(3, cast("list[Constituent]", [(0, 2)]), punctuation=())


def test_the_right_branching_baseline_is_the_whole_sentence_and_its_suffixes() -> None:
    assert right_branching(5).spans == ((0, 5), (1, 5), (2, 5), (3, 5))
    assert right_branching(1).spans == ()
    assert right_branching(0).spans == ()


def test_the_left_branching_baseline_is_the_whole_sentence_and_its_prefixes() -> None:
    assert left_branching(5).spans == ((0, 2), (0, 3), (0, 4), (0, 5))
    assert left_branching(1).spans == ()
    assert left_branching(0).spans == ()


def test_the_balanced_baseline_recursively_splits_at_the_lower_midpoint() -> None:
    assert balanced(4).spans == ((0, 2), (0, 4), (2, 4))
    assert balanced(5).spans == ((0, 2), (0, 5), (2, 5), (3, 5))
    assert balanced(1).spans == ()
    assert balanced(0).spans == ()


def test_the_right_branching_baseline_scores_the_hand_derived_value() -> None:
    baseline = _scored(_BASE, [("s-a", right_branching(6), GOLD_A)])
    result = baseline.compute()

    assert result.components["gold"] == 4.0
    assert result.components["predicted"] == 4.0
    assert result.components["matched"] == 3.0
    assert result.value == 0.75


def test_the_sentence_and_corpus_aggregations_differ_on_one_fixture() -> None:
    result = _scored(_BASE, _TWO).compute()

    assert result.value == 0.875
    assert result.components["sentence_f1"] == 0.875
    assert result.components["corpus_f1"] == pytest.approx(5 / 6)
    assert result.components["matched"] == 5.0
    assert result.components["predicted"] == 6.0
    assert result.components["gold"] == 6.0
    assert result.n == 2

    pooled = _scored(_POOLED, _TWO).compute()

    assert pooled.value == pytest.approx(5 / 6)
    assert pooled.value != result.value


def test_each_sentence_decomposes_to_its_own_score() -> None:
    metric = _scored(_BASE, _TWO)
    decomposed = metric.per_item()

    assert set(decomposed) == {ItemId("s-a"), ItemId("s-b")}
    assert decomposed[ItemId("s-a")].value == 0.75
    assert decomposed[ItemId("s-b")].value == 1.0
    assert decomposed[ItemId("s-a")].components["gold"] == 4.0


def test_recall_breaks_out_by_gold_label_by_width_and_by_depth() -> None:
    metric = _scored(_BASE, _TWO)

    assert {
        label: result.value for label, result in metric.per_label_recall().items()
    } == {"NP": pytest.approx(2 / 3), "PP": 1.0, "VP": 1.0}
    assert metric.per_label_recall()["NP"].n == 3

    assert {
        width: (result.value, result.n)
        for width, result in metric.recall_by_length().items()
    } == {2: (1.0, 3), 3: (0.5, 2), 4: (1.0, 1)}

    assert {
        depth: (result.value, result.n)
        for depth, result in metric.recall_by_depth().items()
    } == {1: (1.0, 3), 2: (0.5, 2), 3: (1.0, 1)}


def test_keeping_the_trivial_spans_raises_the_score() -> None:
    dropped = _value(_BASE, _TWO)
    kept = _value(_TRIVIAL_KEPT, _TWO)

    assert dropped == 0.875
    assert kept == pytest.approx((10 / 11 + 1.0) / 2)
    assert kept is not None
    assert dropped is not None
    assert kept > dropped


def test_stripping_the_punctuation_raises_the_score_on_a_low_attachment() -> None:
    item: tuple[_Example, ...] = (("s-c", PRED_C, GOLD_C),)
    stripped = _value(_BASE, item)
    kept = _value(_PUNCTUATION_KEPT, item)

    assert stripped == 1.0
    assert kept == pytest.approx(2 / 3)


def test_collapsing_the_gold_unary_chain_raises_recall() -> None:
    item: tuple[_Example, ...] = (("s-d", PRED_D, GOLD_D),)
    collapsed = _scored(_BASE, item).compute()
    kept = _scored(_CHAINS_KEPT, item).compute()

    assert collapsed.value == 1.0
    assert collapsed.components["gold"] == 2.0
    assert kept.value == 0.8
    assert kept.components["gold"] == 3.0
    assert kept.components["matched"] == 2.0


def test_the_minimum_length_drops_a_sentence_after_punctuation_is_removed() -> None:
    items: tuple[_Example, ...] = (("s-a", PRED_A, GOLD_A), ("s-e", PRED_E, GOLD_E))
    scored = _scored(_SHORT_SCORED, items).compute()
    dropped = _scored(_SHORT_DROPPED, items).compute()

    assert scored.n == 2
    assert scored.value == pytest.approx((10 / 11 + 0.8) / 2)
    assert dropped.n == 1
    assert dropped.components["dropped"] == 1.0
    assert dropped.value == pytest.approx(10 / 11)


def test_an_uncovered_sentence_costs_what_the_coverage_penalty_says() -> None:
    items: tuple[_Example, ...] = (("s-a", PRED_A, GOLD_A), ("s-b", None, GOLD_B))
    ignored = _scored(_COVERAGE_IGNORED, items).compute()
    zeroed = _scored(_BASE, items).compute()

    assert ignored.value == 0.75
    assert ignored.n == 1
    assert ignored.components["uncovered"] == 1.0
    assert zeroed.value == 0.375
    assert zeroed.n == 2
    assert zeroed.components["uncovered"] == 1.0


def test_the_two_dependency_conventions_leave_a_bracketing_where_it_was() -> None:
    baseline = _value(_BASE, _TWO)

    assert _value(_SELF_ARCS_ALLOWED, _TWO) == baseline
    assert _value(_ORACLE_GOLD, _TWO) == baseline


def test_a_metric_over_nothing_has_no_value_and_no_items() -> None:
    metric = BracketingF1(_BASE)
    result = metric.compute()

    assert result.value is None
    assert result.n == 0
    assert dict(metric.per_item()) == {}
    assert dict(metric.per_label_recall()) == {}


def test_a_sentence_the_conventions_dropped_has_no_value_of_its_own() -> None:
    metric = _scored(_BASE, (("s-e", PRED_E, GOLD_E),))
    decomposed = metric.per_item()

    assert decomposed[ItemId("s-e")].value is None
    assert decomposed[ItemId("s-e")].n == 0
    assert metric.compute().value is None


def test_scoring_one_id_twice_leaves_the_metric_where_it_was() -> None:
    metric = _scored(_BASE, _TWO)
    before = metric.compute()
    metric.update(PRED_A, GOLD_A, key=ItemId("s-a"))

    assert metric.compute() == before


def test_scoring_one_id_against_two_analyses_is_refused() -> None:
    metric = _scored(_BASE, _TWO)

    with pytest.raises(ValueError, match="already scored against a different"):
        metric.update(right_branching(6), GOLD_A, key=ItemId("s-a"))

    with pytest.raises(ValueError, match="already scored against a different"):
        metric.update(None, GOLD_A, key=ItemId("s-a"))


def test_a_prediction_of_another_length_is_refused() -> None:
    metric = BracketingF1(_BASE)

    with pytest.raises(ValueError, match="was scored against a gold analysis of 6"):
        metric.update(PRED_B, GOLD_A, key=ItemId("s-a"))


def test_a_gold_analysis_or_a_prediction_of_the_wrong_type_is_refused() -> None:
    metric = BracketingF1(_BASE)

    with pytest.raises(TypeError, match="a gold bracketing is not one"):
        metric.update(PRED_A, cast("GoldBracketing", PRED_A), key=ItemId("s-a"))

    with pytest.raises(TypeError, match="a predicted bracketing is not one"):
        metric.update(cast("Bracketing", GOLD_A), GOLD_A, key=ItemId("s-a"))


def test_a_bracketing_metric_cannot_be_built_without_a_convention_vector() -> None:
    with pytest.raises(TypeError, match="are not a vector"):
        BracketingF1(cast("ConventionVector", "strip"))


def test_a_reset_metric_reports_what_a_fresh_one_reports() -> None:
    metric = _scored(_BASE, _TWO)
    metric.reset()

    assert metric.compute() == BracketingF1(_BASE).compute()
    assert dict(metric.per_item()) == {}


def test_the_seed_reaches_the_result_the_metric_hands_back() -> None:
    metric = BracketingF1(_BASE, seed=7)
    metric.update(PRED_A, GOLD_A, key=ItemId("s-a"))
    result = metric.compute()

    assert result.resampling.unit == "sentence"
    assert result.resampling.seed == 7
    assert metric.per_label_recall()["NP"].resampling.unit == "constituent"


def test_the_metric_publishes_the_conventions_it_was_built_under() -> None:
    conventions = _BASE
    metric = BracketingF1(conventions)

    assert metric.name == "unlabeled-bracketing-f1"
    assert metric.conventions == conventions
    assert metric.order_invariant


def _build() -> BracketingF1:
    return BracketingF1(_BASE)


def test_the_bracketing_metric_holds_to_every_metric_law() -> None:
    check_metric_laws(MetricFixture(_build, st.sampled_from(_POOL), max_examples=25))


def test_the_metric_is_usable_through_the_protocol_a_report_holds_it_by() -> None:
    held: Metric[Bracketing | None, GoldBracketing] = BracketingF1(_BASE)
    held.update(PRED_A, GOLD_A, key=ItemId("s-a"))

    assert isinstance(held, Metric)
    assert held.compute().value == 0.75
    assert held.conventions is not None
