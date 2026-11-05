from math import log, nextafter
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.data import Derivation, DerivationNode, TokenSpan
from lofi.eval._derivational import (
    LIKELIHOOD_UNITS,
    DerivationalLikelihood,
    DerivationalScoring,
    LikelihoodRow,
    LikelihoodUnit,
    NormalFormScores,
    RankChurn,
    SentenceLikelihood,
    derivational_likelihood_report,
)
from lofi.eval._ports import Metric
from lofi.eval.laws import MetricFixture, check_metric_laws
from lofi.judge import EisnerNormalForm, HockenmaierBiskNormalForm, NoNormalForm

if TYPE_CHECKING:
    from collections.abc import Sequence

type _Tree = int | tuple[_Tree, _Tree]


def _derivation(tokens: int) -> Derivation[str]:
    nodes: dict[int, DerivationNode[str]] = {
        at: DerivationNode(TokenSpan(at, at + 1), "N") for at in range(tokens)
    }
    root = 0

    for at in range(1, tokens):
        node_id = tokens + at - 1
        nodes[node_id] = DerivationNode(
            TokenSpan(0, at + 1), "S", rule="apply", children=(root, at)
        )
        root = node_id

    return Derivation(root, nodes)


def _trees(leaves: Sequence[int]) -> tuple[_Tree, ...]:
    if len(leaves) == 1:
        return (leaves[0],)

    found: list[_Tree] = []

    for split in range(1, len(leaves)):
        for left in _trees(leaves[:split]):
            found.extend((left, right) for right in _trees(leaves[split:]))

    return tuple(found)


def _root_takes_a_leaf(tree: _Tree) -> bool:
    return isinstance(tree, int) or isinstance(tree[1], int)


_TOKENS: Final = 4
_ALL_TREES: Final = _trees(range(_TOKENS))
_ADMITTED_TREES: Final = tuple(one for one in _ALL_TREES if _root_takes_a_leaf(one))


def _uniform(derivations: int) -> SentenceLikelihood:
    return SentenceLikelihood(0.0, log(derivations))


def _metric() -> DerivationalLikelihood[str]:
    return DerivationalLikelihood("eisner")


def _scoring() -> DerivationalScoring[str]:
    return DerivationalScoring(EisnerNormalForm[str]())


_POOL: Final = (
    ("a", SentenceLikelihood(-1.0, 0.0), _derivation(2)),
    ("b", SentenceLikelihood(-3.5, -0.5), _derivation(3)),
    ("c", SentenceLikelihood(None, -0.25), _derivation(4)),
    ("d", SentenceLikelihood(0.0, 0.0), _derivation(1)),
)


def test_a_sentence_likelihood_subtracts_the_partition_from_the_score() -> None:
    assert SentenceLikelihood(-2.0, -0.5).log_likelihood == pytest.approx(-1.5)


def test_an_absent_gold_derivation_has_no_log_likelihood() -> None:
    absent = SentenceLikelihood(None, -0.5)

    assert absent.score is None
    assert absent.log_likelihood is None
    assert absent.log_partition == pytest.approx(-0.5)


def test_a_gold_derivation_may_carry_the_whole_mass() -> None:
    assert SentenceLikelihood(-2.0, -2.0).log_likelihood == pytest.approx(0.0)


def test_a_score_a_rounding_above_its_partition_is_accepted() -> None:
    total = -0.5
    above = nextafter(total, 1.0)

    assert SentenceLikelihood(above, total).log_likelihood == pytest.approx(0.0)


def test_a_gold_derivation_outscoring_its_set_is_refused() -> None:
    with pytest.raises(ValueError, match="against a derivation set"):
        SentenceLikelihood(0.5, 0.0)


def test_a_sentence_likelihood_is_equal_on_its_two_numbers() -> None:
    one = SentenceLikelihood(-1.0, -0.5)

    assert one == SentenceLikelihood(-1.0, -0.5)
    assert hash(one) == hash(SentenceLikelihood(-1.0, -0.5))
    assert one != SentenceLikelihood(-1.0, 0.0)
    assert one != SentenceLikelihood(None, -0.5)
    assert one.__eq__(-1.0) is NotImplemented
    assert repr(one) == "SentenceLikelihood(-1.0, -0.5)"


def test_a_derivational_likelihood_is_a_metric() -> None:
    metric = _metric()

    assert isinstance(metric, Metric)
    assert metric.name == "derivational-log-likelihood"
    assert metric.normal_form == "eisner"
    assert metric.conventions is None
    assert metric.order_invariant


def test_an_empty_derivational_likelihood_reports_no_value() -> None:
    result = _metric().compute()

    assert result.value is None
    assert result.n == 0
    assert result.components["sentences"] == pytest.approx(0.0)


def test_the_sentence_aggregate_averages_the_log_likelihoods() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))
    metric.update(SentenceLikelihood(-4.0, 0.0), _derivation(3), key=ItemId("b"))
    result = metric.result("sentence")

    assert result.value == pytest.approx(-3.0)
    assert result.n == 2
    assert result.resampling.unit == "sentence"
    assert result.components["tokens"] == pytest.approx(5.0)
    assert result.components["total"] == pytest.approx(-6.0)


def test_the_token_aggregate_divides_by_the_token_count() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))
    metric.update(SentenceLikelihood(-4.0, 0.0), _derivation(3), key=ItemId("b"))
    result = metric.result("token")

    assert result.value == pytest.approx(-6.0 / 5.0)
    assert result.n == 5
    assert result.resampling.unit == "token"


def test_compute_reports_the_sentence_aggregate() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))

    assert metric.compute() == metric.result("sentence")


def test_a_sentence_outside_the_derivation_set_is_counted_not_averaged() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))
    metric.update(SentenceLikelihood(None, 0.0), _derivation(3), key=ItemId("b"))
    result = metric.result("sentence")

    assert result.value == pytest.approx(-2.0)
    assert result.components["sentences"] == pytest.approx(2.0)
    assert result.components["contained"] == pytest.approx(1.0)
    assert result.components["uncontained"] == pytest.approx(1.0)
    assert result.components["tokens"] == pytest.approx(2.0)


def test_a_metric_holding_only_absent_gold_derivations_reports_no_value() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(None, 0.0), _derivation(2), key=ItemId("a"))
    result = metric.result("token")

    assert result.value is None
    assert result.n == 0
    assert result.components["uncontained"] == pytest.approx(1.0)


def test_per_item_reports_each_sentence_and_its_per_token_share() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))
    metric.update(SentenceLikelihood(None, -1.0), _derivation(3), key=ItemId("b"))
    items = metric.per_item()

    assert set(items) == {ItemId("a"), ItemId("b")}
    assert items[ItemId("a")].value == pytest.approx(-2.0)
    assert items[ItemId("a")].n == 1
    assert items[ItemId("a")].components["per_token"] == pytest.approx(-1.0)
    assert items[ItemId("a")].components["log_partition"] == pytest.approx(0.0)
    assert items[ItemId("b")].value is None
    assert items[ItemId("b")].n == 0
    assert "per_token" not in items[ItemId("b")].components


def test_an_undeclared_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="undeclared unit"):
        _metric().result(cast("LikelihoodUnit", "corpus"))


def test_repeating_an_item_against_the_same_sentence_changes_nothing() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))

    assert metric.compute().n == 1


def test_repeating_an_item_against_a_different_sentence_is_refused() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))

    with pytest.raises(ValueError, match="already accumulated"):
        metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(3), key=ItemId("a"))


def test_a_prediction_that_is_not_a_sentence_likelihood_is_refused() -> None:
    with pytest.raises(TypeError, match="is not one"):
        _metric().update(
            cast("SentenceLikelihood", -2.0), _derivation(2), key=ItemId("a")
        )


def test_reset_forgets_every_sentence() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))
    metric.reset()

    assert metric.compute().value is None
    assert dict(metric.per_item()) == {}


def test_a_derivational_likelihood_names_its_normal_form_and_size() -> None:
    metric = _metric()
    metric.update(SentenceLikelihood(-2.0, 0.0), _derivation(2), key=ItemId("a"))

    assert repr(metric) == "DerivationalLikelihood('eisner', items=1)"


def test_an_empty_normal_form_name_is_refused() -> None:
    with pytest.raises(ValueError, match="a normal form"):
        DerivationalLikelihood("")


def test_a_derivational_likelihood_obeys_the_metric_laws() -> None:
    check_metric_laws(MetricFixture(_metric, st.sampled_from(_POOL), max_examples=25))


def test_a_scoring_builds_one_metric_per_derivation_set() -> None:
    scoring = _scoring()

    assert scoring.constrained.normal_form == "eisner"
    assert scoring.unconstrained.normal_form == NoNormalForm[str]().name


def test_a_scoring_takes_the_bounded_hockenmaier_bisk_name() -> None:
    scoring: DerivationalScoring[str] = DerivationalScoring(
        HockenmaierBiskNormalForm[str]()
    )

    assert scoring.constrained.normal_form == "hockenmaier-bisk"


def test_a_scoring_refuses_the_unrestricted_filter_as_its_normal_form() -> None:
    with pytest.raises(ValueError, match="unrestricted filter"):
        DerivationalScoring(NoNormalForm[str]())


def test_a_scoring_feeds_both_metrics_from_one_update() -> None:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(SentenceLikelihood(-1.0, 0.0), SentenceLikelihood(-1.0, -0.5)),
        _derivation(2),
        key=ItemId("a"),
    )

    assert scoring.constrained.compute().value == pytest.approx(-1.0)
    assert scoring.unconstrained.compute().value == pytest.approx(-0.5)


def test_resetting_a_scoring_forgets_both_metrics() -> None:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(SentenceLikelihood(-1.0, 0.0), SentenceLikelihood(-1.0, 0.0)),
        _derivation(2),
        key=ItemId("a"),
    )
    scoring.reset()

    assert scoring.constrained.compute().value is None
    assert scoring.unconstrained.compute().value is None


def test_a_scoring_names_both_metrics() -> None:
    assert repr(_scoring()).startswith("DerivationalScoring(constrained=")


def test_a_scoring_emits_a_row_per_derivation_set_and_unit() -> None:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(SentenceLikelihood(-1.0, 0.0), SentenceLikelihood(-1.0, -0.5)),
        _derivation(2),
        key=ItemId("a"),
    )
    rows = scoring.rows("wide")

    assert len(rows) == 2 * len(LIKELIHOOD_UNITS)
    assert {row.model for row in rows} == {"wide"}
    assert [row.normal_form for row in rows] == [
        "eisner",
        "eisner",
        NoNormalForm[str]().name,
        NoNormalForm[str]().name,
    ]
    assert [row.unit for row in rows] == ["sentence", "token", "sentence", "token"]


def test_an_empty_model_name_is_refused() -> None:
    with pytest.raises(ValueError, match="a model name"):
        _scoring().rows("")


def test_a_likelihood_row_publishes_the_normal_form_column() -> None:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(SentenceLikelihood(-1.0, 0.0), SentenceLikelihood(-1.0, -0.5)),
        _derivation(2),
        key=ItemId("a"),
    )
    spec = scoring.rows("wide")[0].to_spec()

    assert spec["model"] == "wide"
    assert spec["normal_form"] == "eisner"
    assert spec["unit"] == "sentence"
    assert spec["value"] == pytest.approx(-1.0)
    assert spec["n"] == 1
    assert spec["components"] == {
        "sentences": 1.0,
        "contained": 1.0,
        "uncontained": 0.0,
        "tokens": 2.0,
        "total": -1.0,
    }


def test_the_two_derivation_sets_differ_by_the_log_derivation_count_ratio() -> None:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(_uniform(len(_ADMITTED_TREES)), _uniform(len(_ALL_TREES))),
        _derivation(_TOKENS),
        key=ItemId("a"),
    )
    constrained = scoring.constrained.result("sentence").value
    unconstrained = scoring.unconstrained.result("sentence").value

    assert constrained is not None
    assert unconstrained is not None
    assert len(_ADMITTED_TREES) < len(_ALL_TREES)
    assert constrained - unconstrained == log(len(_ALL_TREES)) - log(
        len(_ADMITTED_TREES)
    )
    assert constrained - unconstrained == pytest.approx(
        log(len(_ALL_TREES) / len(_ADMITTED_TREES))
    )


def test_the_ratio_also_holds_per_token() -> None:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(_uniform(len(_ADMITTED_TREES)), _uniform(len(_ALL_TREES))),
        _derivation(_TOKENS),
        key=ItemId("a"),
    )
    constrained = scoring.constrained.result("token").value
    unconstrained = scoring.unconstrained.result("token").value

    assert constrained is not None
    assert unconstrained is not None
    assert constrained - unconstrained == pytest.approx(
        log(len(_ALL_TREES) / len(_ADMITTED_TREES)) / _TOKENS
    )


def test_rank_churn_counts_the_pairs_the_two_rankings_order_differently() -> None:
    churn = RankChurn(
        "sentence", {"a": -1.0, "b": -2.0, "c": -3.0}, {"a": -3.0, "b": -2.0, "c": -1.0}
    )

    assert churn.unit == "sentence"
    assert churn.constrained == ("a", "b", "c")
    assert churn.unconstrained == ("c", "b", "a")
    assert churn.inversions == 3
    assert churn.moved == 2
    assert churn.pairs == 3


def test_rank_churn_of_zero_is_still_reported() -> None:
    churn = RankChurn("token", {"a": -1.0, "b": -2.0}, {"a": -4.0, "b": -8.0})

    assert churn.inversions == 0
    assert churn.moved == 0
    assert churn.pairs == 1
    assert churn.to_spec() == {
        "unit": "token",
        "constrained": ["a", "b"],
        "unconstrained": ["a", "b"],
        "inversions": 0,
        "moved": 0,
        "pairs": 1,
    }


def test_rank_churn_breaks_a_tie_by_model_name() -> None:
    churn = RankChurn("sentence", {"b": -1.0, "a": -1.0}, {"a": -1.0, "b": -1.0})

    assert churn.constrained == ("a", "b")
    assert churn.unconstrained == ("a", "b")


def test_rank_churn_is_equal_on_its_unit_and_rankings() -> None:
    one = RankChurn("sentence", {"a": -1.0, "b": -2.0}, {"a": -1.0, "b": -2.0})
    same = RankChurn("sentence", {"a": -5.0, "b": -6.0}, {"a": -7.0, "b": -8.0})

    assert one == same
    assert hash(one) == hash(same)
    assert one != RankChurn("token", {"a": -1.0, "b": -2.0}, {"a": -1.0, "b": -2.0})
    assert one.__eq__("sentence") is NotImplemented
    assert repr(one).startswith("RankChurn('sentence'")


def test_rank_churn_over_different_model_sets_is_refused() -> None:
    with pytest.raises(ValueError, match="under one"):
        RankChurn("sentence", {"a": -1.0}, {"b": -1.0})


def test_rank_churn_over_no_model_is_refused() -> None:
    with pytest.raises(ValueError, match="ranks no model"):
        RankChurn("sentence", {}, {})


def test_rank_churn_under_an_undeclared_unit_is_refused() -> None:
    with pytest.raises(ValueError, match="undeclared unit"):
        RankChurn(cast("LikelihoodUnit", "corpus"), {"a": -1.0}, {"a": -1.0})


def test_rank_churn_refuses_a_value_that_is_not_finite() -> None:
    with pytest.raises(ValueError, match="the ranked value"):
        RankChurn("sentence", {"a": float("-inf")}, {"a": -1.0})


def _model(constrained: float, unconstrained: float) -> DerivationalScoring[str]:
    scoring = _scoring()
    scoring.update(
        NormalFormScores(
            SentenceLikelihood(constrained, 0.0),
            SentenceLikelihood(unconstrained, 0.0),
        ),
        _derivation(2),
        key=ItemId("a"),
    )

    return scoring


def test_a_report_emits_four_rows_per_model_in_model_order() -> None:
    report = derivational_likelihood_report(
        {"wide": _model(-1.0, -4.0), "narrow": _model(-2.0, -3.0)}
    )
    models = [row.model for row in report.rows]

    assert len(report.rows) == 8
    assert models == ["narrow"] * 4 + ["wide"] * 4
    assert all(isinstance(row, LikelihoodRow) for row in report.rows)


def test_a_report_emits_one_churn_per_unit() -> None:
    report = derivational_likelihood_report(
        {"wide": _model(-1.0, -4.0), "narrow": _model(-2.0, -3.0)}
    )

    assert [entry.unit for entry in report.churn] == list(LIKELIHOOD_UNITS)
    assert [entry.constrained for entry in report.churn] == [
        ("wide", "narrow"),
        ("wide", "narrow"),
    ]
    assert [entry.unconstrained for entry in report.churn] == [
        ("narrow", "wide"),
        ("narrow", "wide"),
    ]
    assert [entry.inversions for entry in report.churn] == [1, 1]


def test_a_report_over_one_model_still_emits_its_zero_churn() -> None:
    report = derivational_likelihood_report({"only": _model(-1.0, -4.0)})

    assert len(report.churn) == len(LIKELIHOOD_UNITS)
    assert [entry.inversions for entry in report.churn] == [0, 0]
    assert [entry.moved for entry in report.churn] == [0, 0]


def test_a_report_serializes_its_rows_and_its_churn() -> None:
    spec = derivational_likelihood_report({"only": _model(-1.0, -4.0)}).to_spec()

    assert isinstance(spec["rows"], list)
    assert isinstance(spec["churn"], list)
    assert len(spec["rows"]) == 4
    assert len(spec["churn"]) == len(LIKELIHOOD_UNITS)


def test_a_report_over_no_model_is_refused() -> None:
    with pytest.raises(ValueError, match="asked for no model"):
        derivational_likelihood_report({})


def test_a_model_with_nothing_to_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="no held-out derivational likelihood"):
        derivational_likelihood_report({"empty": _scoring()})
