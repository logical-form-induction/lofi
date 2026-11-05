"""Verify the semantic properties of the evaluation benchmark fixtures."""

from random import Random
from statistics import fmean

from tests.benchmarks.test_eval import (
    INVENTORY,
    REPLICATES,
    SECTION_23,
    SENTENCE_LENGTH,
    TYPE_NODES,
    _measure_tree_distance,
    _type_pair,
    dependency_analysis,
    scored_section,
    sentence_scores,
    supertag_contingency,
)

from lofi.eval import OneToOne
from lofi.stats import Bootstrap, paired_permutation
from lofi.terms import TreeEditDistance


def test_the_supertag_fixture_spans_the_whole_inventory() -> None:
    contingency = supertag_contingency(INVENTORY, seed=5)
    planted = [
        contingency.count(f"u{index:04d}", f"g{index:04d}")
        for index in range(INVENTORY)
    ]
    confused = [
        count
        for (induced, gold), count in contingency.counts.items()
        if induced[1:] != gold[1:]
    ]

    assert len(contingency.induced_labels) == INVENTORY
    assert len(contingency.gold_labels) == INVENTORY
    assert contingency.split == "dev"
    assert min(planted) > 2 * max(confused)


def test_the_category_tree_fixture_holds_thirty_one_nodes() -> None:
    left, right = _type_pair()

    assert left.size == TYPE_NODES
    assert right.size == TYPE_NODES
    assert left is not right


def test_the_dependency_fixture_drops_arcs_from_predictions() -> None:
    gold = dependency_analysis(SENTENCE_LENGTH, Random(11), drop=0.0)
    predicted = dependency_analysis(SENTENCE_LENGTH, Random(11), drop=0.08)

    assert len(gold.arcs) == SENTENCE_LENGTH - 1
    assert len(predicted.arcs) < len(gold.arcs)
    assert set(predicted.arcs) <= set(gold.arcs)


def test_bootstrap_interval_reports_its_sample_and_observation() -> None:
    scores = sentence_scores(SECTION_23, seed=11)
    bootstrap = Bootstrap(unit="sentence", replicates=REPLICATES, seed=7, level=0.95)

    result = bootstrap.interval(scores, fmean)

    assert result.unit == "sentence"
    assert result.units == SECTION_23
    assert result.replicates == REPLICATES
    assert result.percentile.low < result.observed < result.percentile.high
    assert result.bca.low < result.observed < result.bca.high


def test_paired_permutation_reports_the_requested_approximation() -> None:
    left = sentence_scores(SECTION_23, seed=11)
    right = sentence_scores(SECTION_23, seed=13)

    probability = paired_permutation(
        left, right, replicates=REPLICATES, seed=7, tail="two-sided"
    )

    assert not probability.provenance.exact
    assert probability.provenance.replicates == REPLICATES
    assert probability.provenance.seed == 7
    assert 0.0 < probability.value < 1.0


def test_optimal_assignment_recovers_the_planted_inventory_mapping() -> None:
    contingency = supertag_contingency(INVENTORY, seed=5)
    alignment = OneToOne(tie_break="least-gold-label").fit(contingency)

    assert alignment.is_injective
    assert alignment.mapping == {
        f"u{index:04d}": f"g{index:04d}" for index in range(INVENTORY)
    }


def test_tree_edit_distance_normalizes_by_the_two_tree_sizes() -> None:
    left, right = _type_pair()
    distance = TreeEditDistance()

    normalized = _measure_tree_distance(distance, left, right)

    assert distance(left, right) == 33
    assert normalized == 33 / (left.size + right.size)


def test_dependency_scoring_reports_the_expected_section_metrics() -> None:
    scorer = scored_section(SECTION_23, SENTENCE_LENGTH, seed=3)
    rows = scorer.rows()
    scored = {row.name: row.result.value for row in rows}
    recall = scored["dependency:labeled-directed:ignore:micro-recall"]

    assert len(rows) == 64
    assert scored["dependency:labeled-directed:ignore:micro-precision"] == 1.0
    assert scored["dependency:labeled-directed:ignore:coverage"] == 1.0
    assert recall is not None
    assert 0.8 < recall < 1.0
