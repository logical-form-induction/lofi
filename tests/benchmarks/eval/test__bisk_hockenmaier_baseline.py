from typing import TYPE_CHECKING, Final

import pytest

from lofi._types import ItemId
from lofi.data import Token
from lofi.eval._bisk_hockenmaier_baseline import (
    _AutoSentence,
    _grammar,
    _inducer,
    _item,
)
from lofi.spec.baselines.bisk_hockenmaier import BiskHockenmaierBaselineSpec

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

BUDGET_S: Final = 5.0
"""Budget for the packaged grammar's support parse of one five-token sentence."""

_FORMS: Final = ("Ms.", "Haag", "plays", "Elianti", ".")


@pytest.mark.benchmark
def test_the_packaged_support_parse_of_a_five_token_sentence_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    settings = BiskHockenmaierBaselineSpec(
        ccgbank_source="ccgbank",
        dependency_gold_source="evaluation.conll",
        reference_source="reference.json",
        reference_tolerance=0.0,
        nominal_category="N",
        sentence_category="S",
        coordination_category="conj",
        gold_sentence_atom="S",
        gold_coordination_atom="conj",
        nominal_tags=frozenset({"NN", "NNS", "NNP", "NNPS"}),
        verbal_tags=frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}),
        coordination_tags=frozenset({"CC"}),
        other_tags=frozenset({"JJ", "RB", "."}),
        punctuation_tags=frozenset({"."}),
        headed_or_modifier_arguments=("N", "S"),
        sentence_arguments=("N",),
        em_max_iterations=50,
        convergence_tolerance=1e-6,
    ).settings()
    grammar = _grammar(settings)
    inducer = _inducer(settings, grammar, settings.conventions)
    sentence = _AutoSentence(
        2,
        ItemId("wsj_0200.2"),
        _FORMS,
        ("NNP", "NNP", "VBZ", "NNP", "."),
        tuple(Token(position, form) for position, form in enumerate(_FORMS)),
        ("N/N", "N", "S\\N/N", "N", "."),
    )

    benchmark.pedantic(
        _item, args=(sentence, grammar, settings, inducer), rounds=3, iterations=1
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the support parse ran but pytest-benchmark timed nothing")

    median_s = measured.stats.median
    benchmark.extra_info["budget_s"] = BUDGET_S
    benchmark.extra_info["median_s"] = median_s

    if wall_clock_budget.enforced:
        assert median_s <= BUDGET_S
