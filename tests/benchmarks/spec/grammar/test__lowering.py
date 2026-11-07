"""Benchmark lowering packaged CCGbank to its native assembly."""

from typing import TYPE_CHECKING, Final

import pytest

from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.grammar import GrammarDecl, GrammarIso
from lofi.spec.grammar import _lowering as lowering_module
from lofi.spec.grammar._lowering import _compile_grammar
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.terms import _CompiledGrammar

DECLARATION: Final = ccgbank_grammar()
LANGUAGE: Final = registered_language("propositional")


def _cross_once() -> None:
    """Lower the declaration through a converter that has not seen it."""
    GrammarIso().forward(DECLARATION, LANGUAGE)


def test_a_repeated_crossing_returns_the_converter_cached_grammar() -> None:
    iso = GrammarIso()
    first = iso.forward(DECLARATION, LANGUAGE)
    rebuilt = GrammarDecl.model_validate_json(first.declaration)

    again = iso.forward(rebuilt, LANGUAGE)

    assert again is first


def test_each_cold_sample_compiles_and_assembles_the_prepared_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted(language: bytes, grammar: bytes) -> _CompiledGrammar:
        nonlocal calls
        calls += 1
        return _compile_grammar(language, grammar)

    monkeypatch.setattr(lowering_module, "_compile_grammar", counted)
    prepared = DECLARATION.compile_envelope
    first = GrammarIso().forward(DECLARATION, LANGUAGE)
    second = GrammarIso().forward(DECLARATION, LANGUAGE)

    assert calls == 2
    assert first is not second
    assert first == second
    assert DECLARATION.compile_envelope is prepared


@pytest.mark.benchmark
def test_the_grammar_crossing_meets_budget(
    benchmark: BenchmarkFixture, wall_clock_budget: WallClockBudget
) -> None:
    benchmark.pedantic(_cross_once, rounds=25, warmup_rounds=2, iterations=4)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the grammar crossing ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "grammar-lowering"
    benchmark.extra_info["median_ms"] = median_ms

    # enforce the 5 ms per-distinct-grammar crossing budget
    if wall_clock_budget.enforced:
        assert median_ms <= 5
