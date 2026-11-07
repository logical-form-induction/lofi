"""Acceptance checks over structural-probes' read-only parse-cost fixture."""

from functools import cache
from os import environ
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Final

import pytest
from tests.support.structural_parse_cost import (
    ParseCostRuntime,
    UnitWeights,
    fixture_composition_orders,
    fixture_declaration,
    fixture_runtime,
    fixture_sentences,
)

from lofi.deduce import (
    Boolean,
    Counting,
    InsideChart,
    ParsePolicy,
    Reachability,
    Retain,
    parse,
)
from lofi.spec import CombinatorSetSpec
from lofi.spec.grammar import HockenmaierBiskNormalFormSpec, NoNormalFormSpec

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.deduce import Cell
    from lofi.terms import Term

_FIXTURE_SETTING: Final = environ.get("LOFI_STRUCTURAL_PARSE_COST")
_FIXTURE: Final = (
    None if _FIXTURE_SETTING is None else Path(_FIXTURE_SETTING).expanduser()
)
_MEDIAN_BUDGET: Final = 1.0
_MAXIMUM_BUDGET: Final = 5.0

pytestmark = pytest.mark.skipif(
    _FIXTURE is None or not _FIXTURE.is_dir(),
    reason="set LOFI_STRUCTURAL_PARSE_COST to structural-probes' parse-cost fixture",
)


def _fixture() -> Path:
    """Return the configured read-only structural parse fixture."""
    if _FIXTURE is None or not _FIXTURE.is_dir():
        pytest.skip(
            "set LOFI_STRUCTURAL_PARSE_COST to structural-probes' parse-cost fixture"
        )

    return _FIXTURE


@cache
def _runtime(kind: str) -> ParseCostRuntime:
    """Build the fixture once under the selected declared normal form."""
    normal_form = (
        HockenmaierBiskNormalFormSpec(
            composition_primary=False,
            unit_primary=False,
        )
        if kind == "hockenmaier-bisk"
        else NoNormalFormSpec()
    )

    return fixture_runtime(_fixture(), normal_form)


def _count_with_reachability(
    tokens: tuple[str, ...], reachability: Reachability
) -> InsideChart[Term, int]:
    """Count normal-form derivations under one declared reachability policy."""
    runtime = _runtime("hockenmaier-bisk")
    unit = UnitWeights(runtime.lexicon, one=1)

    return parse(
        tokens,
        rules=runtime.rules,
        source=unit,
        semiring=Counting(),
        scorer=unit,
        goal=runtime.goal,
        category_measure=runtime.category_measure,
        max_category_arity=runtime.max_category_arity,
        max_category_size=runtime.max_category_size,
        unary_depth=runtime.unary_depth,
        threads=runtime.threads,
        policy=ParsePolicy(Retain.INSIDE, reachability),
    )


def _count(tokens: tuple[str, ...]) -> InsideChart[Term, int]:
    """Count root-reachable normal-form derivations of one fixture sentence."""
    return _count_with_reachability(tokens, Reachability.ROOTS)


def _support(kind: str, tokens: tuple[str, ...]) -> InsideChart[Term, bool]:
    """Compute the exact category support of one fixture sentence."""
    runtime = _runtime(kind)
    unit = UnitWeights(runtime.lexicon, one=True)

    return parse(
        tokens,
        rules=runtime.rules,
        source=unit,
        semiring=Boolean(),
        scorer=unit,
        goal=runtime.goal,
        category_measure=runtime.category_measure,
        max_category_arity=runtime.max_category_arity,
        max_category_size=runtime.max_category_size,
        unary_depth=runtime.unary_depth,
        threads=runtime.threads,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )


def _suffix_categories(chart: InsideChart[Term, bool]) -> frozenset[tuple[Cell, Term]]:
    """Return exact categories on every span ending at the sentence end."""
    end = chart.index.n

    return frozenset(
        ((start, end), category)
        for start in range(end)
        for category, _ in chart.categories_at((start, end))
    )


def _fixture_prefix(length: int) -> tuple[str, ...]:
    """Return a declared fixture prefix of the requested length."""
    return next(
        sentence.tokens[:length]
        for sentence in fixture_sentences(_fixture())
        if sentence.length >= length
    )


def _short_fixture_sentence(maximum_length: int) -> tuple[str, ...]:
    """Return one complete declared sentence within a bounded length."""
    return next(
        sentence.tokens
        for sentence in fixture_sentences(_fixture())
        if sentence.length <= maximum_length
    )


def _long_parse_times() -> tuple[float, ...]:
    """Measure every declared 16- and 17-token fixture sentence."""
    seconds: list[float] = []

    for sentence in fixture_sentences(_fixture()):
        if sentence.length < 16:
            continue

        started = perf_counter()
        _count(sentence.tokens)
        seconds.append(perf_counter() - started)

    return tuple(seconds)


def test_fixture_declaration_has_one_reachable_root_per_matching_class() -> None:
    """Declare one canonical root for each class the reduced fixture can reach."""
    declaration = fixture_declaration(_fixture())

    assert declaration.goal.categories == ("NP", "PP", "S", "S\\NP", "S/NP")


def test_fixture_declaration_canonicalizes_overlapping_crossed_orders() -> None:
    """Declare each unrestricted composition relation exactly once."""
    straight, crossed = fixture_composition_orders(_fixture())
    declaration = fixture_declaration(_fixture())
    (combinators,) = (
        rule for rule in declaration.rules.rules if isinstance(rule, CombinatorSetSpec)
    )

    assert straight == (1, 2)
    assert crossed == (1, 2)
    assert combinators.composition == (1, 2)
    assert combinators.crossed_composition == ()


def test_fixture_runtime_lowers_the_inventory_owned_category_depth() -> None:
    """Lower every declared fixture category within its derived Cat bound."""
    runtime = fixture_runtime(_fixture())

    assert len(runtime.goal.categories) == 5


def test_a_bounded_large_prefix_keeps_exact_unrestricted_suffix_support() -> None:
    """Witness exact normal-form support before long acceptance measurements."""
    tokens = _fixture_prefix(8)
    unrestricted = _support("none", tokens)
    restricted = _support("hockenmaier-bisk", tokens)

    assert _suffix_categories(restricted) == _suffix_categories(unrestricted)


def test_root_reachability_preserves_every_exact_root_count() -> None:
    """Prune interior categories without changing an exact root derivation count."""
    tokens = _short_fixture_sentence(8)
    runtime = _runtime("hockenmaier-bisk")
    unrestricted = _count_with_reachability(tokens, Reachability.ALL)
    rooted = _count_with_reachability(tokens, Reachability.ROOTS)
    root = (0, len(tokens))

    def accepted(chart: InsideChart[Term, int]) -> dict[Term, int]:
        return {
            category: count
            for category, count in chart.categories_at(root)
            if any(
                runtime.goal.matching.equivalent(category, target)
                for target in runtime.goal.categories
            )
        }

    assert accepted(rooted) == accepted(unrestricted)


def test_every_fixture_suffix_category_survives_the_normal_form() -> None:
    """Keep unrestricted exact suffix support under Hockenmaier-Bisk."""
    for sentence in fixture_sentences(_fixture()):
        unrestricted = _support("none", sentence.tokens)
        restricted = _support("hockenmaier-bisk", sentence.tokens)

        assert _suffix_categories(restricted) == _suffix_categories(unrestricted)


def test_the_long_fixture_sentences_meet_the_consumer_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    """Meet structural-probes' median and worst-case parse budgets."""
    seconds = benchmark.pedantic(_long_parse_times, rounds=1, iterations=1)

    if len(seconds) != 6:
        pytest.fail(f"expected six long fixture sentences, measured {len(seconds)}")

    benchmark.extra_info["sentence_seconds"] = seconds
    benchmark.extra_info["median_budget_seconds"] = _MEDIAN_BUDGET
    benchmark.extra_info["maximum_budget_seconds"] = _MAXIMUM_BUDGET

    if wall_clock_budget.enforced:
        assert median(seconds) <= _MEDIAN_BUDGET
        assert max(seconds) <= _MAXIMUM_BUDGET
