from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.eval._entailment_family import (
    EntailmentAccuracy,
    EntailmentPair,
    entailment_family,
)
from lofi.eval._inference import UndeclaredLabelError
from lofi.eval._ports import Metric
from lofi.eval._values import ConventionVector, CoveragePenalty
from lofi.eval.laws import MetricFixture, check_metric_laws

if TYPE_CHECKING:
    from collections.abc import Sequence

LABELS: Final = frozenset({"entails", "neutral", "contradicts"})

_PAIRS = (
    ("p-a", "entails", EntailmentPair("entails", 5, 4)),
    ("p-b", "neutral", EntailmentPair("contradicts", 5, 4)),
    ("p-c", None, EntailmentPair("neutral", 5, 4)),
    ("p-d", "contradicts", EntailmentPair("contradicts", 5, 1)),
)


def _conventions(
    *, minimum_length: int = 0, coverage: CoveragePenalty = "zero"
) -> ConventionVector:
    return ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=minimum_length,
        aggregation="corpus",
        coverage=coverage,
        self_arcs="forbid",
        gold_derivation="canonical",
    )


def _accuracy(
    entries: Sequence[tuple[str, str | None, EntailmentPair]],
    conventions: ConventionVector,
) -> EntailmentAccuracy:
    metric = EntailmentAccuracy(conventions, LABELS)

    for key, pred, gold in entries:
        metric.update(pred, gold, key=ItemId(key))

    return metric


def test_an_abstention_counts_as_wrong_when_coverage_is_zero() -> None:
    result = _accuracy(_PAIRS[:3], _conventions()).compute()

    assert result.value == pytest.approx(1 / 3)
    assert result.n == 3
    assert result.components == {"matched": 1.0, "labeled": 2.0}
    assert result.resampling.unit == "pair"


def test_an_abstention_leaves_accuracy_when_coverage_ignores_it() -> None:
    result = _accuracy(_PAIRS[:3], _conventions(coverage="ignore")).compute()

    assert result.value == 0.5
    assert result.n == 2
    assert result.components == {"matched": 1.0, "labeled": 2.0}


def test_a_pair_is_scored_only_when_both_members_reach_the_minimum_length() -> None:
    short = _accuracy(_PAIRS, _conventions(minimum_length=3)).compute()
    every = _accuracy(_PAIRS, _conventions()).compute()

    assert short.n == 3
    assert short.components["matched"] == 1.0
    assert every.n == 4
    assert every.components["matched"] == 2.0


def test_a_zero_length_member_is_never_scored() -> None:
    result = _accuracy(
        (("p-a", "entails", EntailmentPair("entails", 5, 0)),), _conventions()
    ).compute()

    assert result.n == 0
    assert result.value is None


def test_accuracy_decomposes_over_every_pair_it_was_given() -> None:
    per_item = _accuracy(_PAIRS, _conventions(minimum_length=3)).per_item()

    assert set(per_item) == {ItemId(key) for key, _, _ in _PAIRS}
    assert per_item[ItemId("p-a")].value == 1.0
    assert per_item[ItemId("p-b")].value == 0.0
    assert per_item[ItemId("p-c")].value == 0.0
    assert per_item[ItemId("p-d")].value is None


def test_an_undeclared_label_is_refused_on_either_side() -> None:
    metric = EntailmentAccuracy(_conventions(), LABELS)

    with pytest.raises(UndeclaredLabelError, match="paraphrase"):
        metric.update("paraphrase", EntailmentPair("neutral", 4, 4), key=ItemId("p-a"))

    with pytest.raises(UndeclaredLabelError, match="unknown"):
        metric.update("neutral", EntailmentPair("unknown", 4, 4), key=ItemId("p-a"))

    with pytest.raises(TypeError, match="predicted entailment label is not text"):
        metric.update(
            cast("str", 1), EntailmentPair("neutral", 4, 4), key=ItemId("p-a")
        )

    with pytest.raises(TypeError, match="gold entailment pair is not one"):
        metric.update(
            "neutral", cast("EntailmentPair", ("neutral", 4, 4)), key=ItemId("p-a")
        )


def test_an_entailment_metric_classifies_at_least_two_labels() -> None:
    with pytest.raises(ValueError, match="classifies nothing"):
        EntailmentAccuracy(_conventions(), frozenset({"entails"}))

    with pytest.raises(TypeError, match="not a vector"):
        EntailmentAccuracy(cast("ConventionVector", "c"), LABELS)


def test_accuracy_refuses_a_repeated_key_scored_against_another_pair() -> None:
    metric = EntailmentAccuracy(_conventions(), LABELS)
    metric.update("neutral", EntailmentPair("neutral", 4, 4), key=ItemId("p-a"))
    metric.update("neutral", EntailmentPair("neutral", 4, 4), key=ItemId("p-a"))

    with pytest.raises(ValueError, match="already scored"):
        metric.update("entails", EntailmentPair("neutral", 4, 4), key=ItemId("p-a"))

    with pytest.raises(ValueError, match="premise length"):
        metric.update("neutral", EntailmentPair("neutral", -1, 4), key=ItemId("p-b"))

    with pytest.raises(ValueError, match="hypothesis length"):
        metric.update("neutral", EntailmentPair("neutral", 4, -1), key=ItemId("p-b"))


def test_the_entailment_metric_holds_to_every_metric_law() -> None:
    check_metric_laws(
        MetricFixture(
            lambda: EntailmentAccuracy(_conventions(), LABELS),
            st.sampled_from(_PAIRS),
            max_examples=25,
        )
    )
    assert isinstance(EntailmentAccuracy(_conventions(), LABELS), Metric)


def test_the_family_scores_the_declared_inventory_under_the_declared_conventions() -> (
    None
):
    labels = frozenset({"entails", "non-entails"})
    family = entailment_family(labels=labels, conventions=_conventions(), seed=7)

    assert family.name == "nli-entailment"
    assert isinstance(family.metric, EntailmentAccuracy)
    assert family.metric.labels == labels
    assert family.metric.conventions == _conventions()
