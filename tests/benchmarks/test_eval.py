"""Benchmarks the five evaluation operations a sweep repeats, against their budgets."""

from random import Random
from statistics import fmean
from typing import TYPE_CHECKING, Final

import pytest

from lofi.data import ItemId
from lofi.eval import (
    Contingency,
    OneToOne,
    normalized_tree_distance,
)
from lofi.eval.corpora import (
    DependencyArc,
    DependencyPrediction,
    DependencyScorer,
    DependencyToken,
)
from lofi.stats import Bootstrap, paired_permutation
from lofi.terms import Operator, Signature, TermStore, TreeEditDistance

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.terms import Term

SECTION_23: Final = 2407
"""How many sentences CCGbank section 23 holds."""

REPLICATES: Final = 999
"""The replicate count a published resampling interval or p-value is drawn at."""

INVENTORY: Final = 425
"""The size of the frequency-thresholded CCGbank supertag inventory."""

TYPE_NODES: Final = 31
"""How many nodes each of the two benchmarked category trees holds."""

SENTENCE_LENGTH: Final = 24
"""How many tokens each benchmarked section 23 sentence holds."""

CATEGORIES: Final = ("(S\\NP)/NP", "S\\NP", "NP/N", "((S\\NP)\\(S\\NP))/NP", "N/N")
"""The categories the benchmarked dependency arcs are labeled from."""


SIGNATURE: Final = Signature(
    "category",
    ("Ty",),
    (
        Operator("S", "Ty", family="atom"),
        Operator("NP", "Ty", family="atom"),
        Operator("N", "Ty", family="atom"),
        Operator("PP", "Ty", family="atom"),
        Operator("/", "Ty", arg_sorts=("Ty", "Ty"), family="forward"),
        Operator("\\", "Ty", arg_sorts=("Ty", "Ty"), family="backward"),
    ),
)

STORE: Final = TermStore(SIGNATURE)


def median_ms(benchmark: BenchmarkFixture, budget: str) -> float:
    """Return the measured median in milliseconds, recorded under its budget id.

    Parameters
    ----------
    benchmark : BenchmarkFixture
        The fixture the operation ran through.
    budget : str
        The identifier of the budget row the median is read against.

    Returns
    -------
    float
        The milliseconds this run's median call took.
    """
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the operation ran but pytest-benchmark timed nothing")

    milliseconds = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = budget
    benchmark.extra_info["median_ms"] = milliseconds

    return milliseconds


def sentence_scores(count: int, seed: int) -> tuple[float, ...]:
    """Return one score per sentence, drawn reproducibly from the unit interval.

    Parameters
    ----------
    count : int
        How many sentences to score.
    seed : int
        The seed the draw runs under.

    Returns
    -------
    tuple[float, ...]
        One score per sentence, in sentence order.
    """
    generator = Random(seed)

    return tuple(generator.random() for _ in range(count))


def supertag_contingency(size: int, seed: int) -> Contingency:
    """Return an induced-by-gold supertag table over one inventory size.

    Each induced label takes one planted gold correspondent and three confusions
    drawn from elsewhere in the inventory. A planted count outweighs any two
    confusions, so the planted permutation is the unique optimal assignment.

    Parameters
    ----------
    size : int
        How many induced and gold labels the table ranges over.
    seed : int
        The seed the counts are drawn under.

    Returns
    -------
    Contingency
        The counts, accumulated on the development split.
    """
    generator = Random(seed)
    counts: dict[tuple[str, str], int] = {}

    for index in range(size):
        induced = f"u{index:04d}"
        counts[induced, f"g{index:04d}"] = generator.randint(100, 200)

        for _ in range(3):
            elsewhere = generator.randrange(size - 1)
            confused = elsewhere + 1 if elsewhere >= index else elsewhere
            counts[induced, f"g{confused:04d}"] = generator.randint(1, 20)

    return Contingency(counts, split="dev")


def category_tree(leaves: tuple[str, ...], slashes: tuple[str, ...]) -> Term:
    """Return the balanced category tree over one leaf sequence.

    Parameters
    ----------
    leaves : tuple[str, ...]
        The atom names at the leaves, left to right. Their count is a power of two.
    slashes : tuple[str, ...]
        The slash names the internal nodes cycle through in build order.

    Returns
    -------
    Term
        The root of the tree the leaves and slashes build.
    """
    level = [STORE.op(SIGNATURE.by_name(leaf)) for leaf in leaves]
    built = 0

    while len(level) > 1:
        above = []

        for at in range(0, len(level), 2):
            slash = SIGNATURE.by_name(slashes[built % len(slashes)])
            above.append(STORE.op(slash, level[at], level[at + 1]))
            built += 1

        level = above

    return level[0]


def dependency_analysis(
    length: int, generator: Random, *, drop: float
) -> DependencyPrediction:
    """Return one sentence's dependency analysis, dropping arcs at a given rate.

    Parameters
    ----------
    length : int
        How many tokens the sentence holds.
    generator : Random
        The stream the heads, categories and drops are drawn from.
    drop : float
        The probability that an arc is left out, which is how a predicted
        analysis differs from the gold one drawn from the same stream.

    Returns
    -------
    DependencyPrediction
        The arcs that survived, over tokens in position order.
    """
    tokens = [DependencyToken(at, f"w{at}") for at in range(length)]
    arcs = []

    for at in range(1, length):
        head = generator.randrange(at)
        category = CATEGORIES[generator.randrange(len(CATEGORIES))]
        slot = 1 + at % 2

        if generator.random() >= drop:
            arcs.append(DependencyArc(tokens[head], tokens[at], category, slot, None))

    return DependencyPrediction(arcs)


def scored_section(sentences: int, length: int, seed: int) -> DependencyScorer:
    """Return a scorer holding one predicted and gold analysis per sentence.

    Parameters
    ----------
    sentences : int
        How many sentences to score.
    length : int
        How many tokens each sentence holds.
    seed : int
        The seed each sentence's two analyses are drawn under.

    Returns
    -------
    DependencyScorer
        The scorer, loaded and ready to be asked for its rows.
    """
    scorer = DependencyScorer()

    for at in range(sentences):
        gold = dependency_analysis(length, Random(seed + at), drop=0.0)
        predicted = dependency_analysis(length, Random(seed + at), drop=0.08)
        scorer.update(predicted, gold, key=ItemId(f"wsj_2300.{at}"))

    return scorer


def _type_pair() -> tuple[Term, Term]:
    """Return the two category trees the tree-distance benchmark compares.

    Returns
    -------
    tuple[Term, Term]
        The two roots, which differ in both their leaves and their slashes.
    """
    left = category_tree(
        (
            "S",
            "NP",
            "N",
            "PP",
            "NP",
            "N",
            "S",
            "NP",
            "PP",
            "N",
            "NP",
            "S",
            "N",
            "PP",
            "NP",
            "S",
        ),
        ("/", "\\"),
    )
    right = category_tree(
        (
            "S",
            "NP",
            "PP",
            "N",
            "NP",
            "S",
            "N",
            "NP",
            "PP",
            "N",
            "S",
            "NP",
            "N",
            "PP",
            "S",
            "NP",
        ),
        ("\\", "/"),
    )

    return left, right


def test_a_999_replicate_bootstrap_over_section_23_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    scores = sentence_scores(SECTION_23, seed=11)
    bootstrap = Bootstrap(unit="sentence", replicates=REPLICATES, seed=7, level=0.95)
    benchmark.pedantic(
        bootstrap.interval, args=(scores, fmean), rounds=5, warmup_rounds=1
    )
    median = median_ms(benchmark, "P-42")

    if wall_clock_budget.enforced:
        assert median <= 2800.0


def test_a_999_replicate_paired_permutation_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    left = sentence_scores(SECTION_23, seed=11)
    right = sentence_scores(SECTION_23, seed=13)
    benchmark.pedantic(
        paired_permutation,
        args=(left, right),
        kwargs={"replicates": REPLICATES, "seed": 7, "tail": "two-sided"},
        rounds=5,
        warmup_rounds=1,
    )
    median = median_ms(benchmark, "P-43")

    if wall_clock_budget.enforced:
        assert median <= 1500.0


def test_optimal_assignment_at_the_ccgbank_inventory_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    contingency = supertag_contingency(INVENTORY, seed=5)
    policy = OneToOne(tie_break="least-gold-label")
    benchmark.pedantic(policy.fit, args=(contingency,), rounds=5, warmup_rounds=1)
    median = median_ms(benchmark, "P-44")

    if wall_clock_budget.enforced:
        assert median <= 2900.0


def _measure_tree_distance(
    distance: TreeEditDistance, left: Term, right: Term
) -> float:
    distance.clear()

    return normalized_tree_distance(distance, left, right)


def test_tree_edit_distance_over_a_thirty_one_node_type_pair_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    left, right = _type_pair()
    distance = TreeEditDistance()

    benchmark.pedantic(
        _measure_tree_distance,
        args=(distance, left, right),
        rounds=50,
        warmup_rounds=2,
    )
    median = median_ms(benchmark, "P-45")

    if wall_clock_budget.enforced:
        assert median <= 15.0


def test_dependency_scoring_over_section_23_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    scorer = scored_section(SECTION_23, SENTENCE_LENGTH, seed=3)
    benchmark.pedantic(scorer.rows, rounds=5, warmup_rounds=1)
    median = median_ms(benchmark, "P-46")
    if wall_clock_budget.enforced:
        assert median <= 3800.0
