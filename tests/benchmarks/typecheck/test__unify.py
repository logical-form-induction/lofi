from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.language import language

from lofi.typecheck import Unifier

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.terms import Term

#: Nanoseconds one two-metavariable subsumption match may take.
SUBSUMPTION_NS: Final = 40_000


class Pair(NamedTuple):
    """A two-metavariable subsumption pair."""

    unifier: Unifier
    general: Term
    specific: Term


@pytest.fixture(scope="module")
def pair() -> Pair:
    """Return the fixed subsumption pair."""
    declared = language()
    store = declared.store
    arrow = store.signature.by_name(declared.function_type)
    entity = declared.read_type("e")
    truth = declared.read_type("t")
    argument = store.fvar(declared.type_sort, "argument")
    result = store.fvar(declared.type_sort, "result")

    return Pair(
        Unifier(store),
        store.op(arrow, argument, store.op(arrow, result, argument)),
        store.op(arrow, entity, store.op(arrow, truth, entity)),
    )


def test_two_metavariable_subsumption(
    benchmark: BenchmarkFixture,
    pair: Pair,
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark(pair.unifier.subsumes, pair.general, pair.specific)

    measured = benchmark.stats

    if measured is None:
        pytest.fail("subsumption ran but pytest-benchmark timed nothing")

    nanoseconds = measured.stats.median * 1e9
    benchmark.extra_info["metavariables"] = 2
    benchmark.extra_info["median_ns_per_match"] = nanoseconds
    benchmark.extra_info["mean_ns_per_match"] = measured.stats.mean * 1e9
    benchmark.extra_info["budget_ns"] = SUBSUMPTION_NS

    if wall_clock_budget.enforced:
        assert nanoseconds <= SUBSUMPTION_NS


def test_two_metavariable_subsumption_returns_the_expected_substitution(
    pair: Pair,
) -> None:
    substitution = pair.unifier.subsumes(pair.general, pair.specific)

    assert substitution is not None
    assert len(substitution) == 2
    assert pair.unifier.apply(substitution, pair.general) is pair.specific
