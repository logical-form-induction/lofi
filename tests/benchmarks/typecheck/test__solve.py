from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.language import language, lexicon

from lofi.typecheck import LexicalSignature, Scheme, TypeChecker, TypingRule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term

#: Nanoseconds one end-to-end check of the forty-two-node example may take.
CHECK_NS: Final = 25_000_000


class Judgment(NamedTuple):
    """Hold one fixed typing judgment."""

    checker: TypeChecker
    term: Term
    signature: LexicalSignature
    rules: Mapping[str, TypingRule]


def _apply(language: LanguageAssembly, function: Term, argument: Term) -> Term:
    """Build a logical-form application."""
    return language.apply(function, argument)


def _conjoin(language: LanguageAssembly, forms: tuple[Term, ...]) -> Term:
    """Build a right-branching conjunction."""
    head, *tail = forms

    if not tail:
        return head

    return language.apply(
        language.constant("and"),
        head,
        _conjoin(language, tuple(tail)),
    )


@pytest.fixture(scope="module")
def judgment() -> Judgment:
    """Build an exact forty-two-node neo-Davidsonian judgment."""
    declared = language(
        (
            "actor",
            "agent",
            "deliberate",
            "item",
            "location",
            "move",
            "past",
            "place",
            "theme",
        )
    )
    store = declared.store
    entity = declared.read_type("e")
    event_predicate = declared.read_type("v -> t")
    event_relation = declared.read_type("v -> e -> t")
    signature: dict[int, Scheme] = {}

    def symbol(name: str, type_term: Term) -> Term:
        """Declare and build one symbol."""
        term = store.const(declared.term_sort, name)
        payload = term.payload

        assert payload is not None
        signature[payload] = Scheme(type_term)

        return term

    unary_predicates = tuple(
        symbol(name, event_predicate) for name in ("move", "past", "deliberate")
    )
    binary_predicates = tuple(
        symbol(name, event_relation) for name in ("agent", "theme", "location")
    )
    individuals = tuple(symbol(name, entity) for name in ("actor", "item", "place"))
    variable = store.bvar(declared.term_sort, 0)
    atoms = (
        *(_apply(declared, predicate, variable) for predicate in unary_predicates),
        *(
            _apply(
                declared,
                _apply(declared, predicate, variable),
                individual,
            )
            for predicate, individual in zip(
                binary_predicates,
                individuals,
                strict=True,
            )
        ),
    )
    term = declared.apply(
        declared.constant("exists"),
        store.op(
            store.signature.by_name(declared.calculus.lam),
            _conjoin(declared, atoms),
        ),
    )

    assert term.size == 42

    return Judgment(
        TypeChecker(store),
        term,
        lexicon(declared, signature),
        declared.rules,
    )


def test_forty_two_node_neo_davidsonian_check(
    benchmark: BenchmarkFixture,
    judgment: Judgment,
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark(
        judgment.checker.check,
        judgment.term,
        judgment.signature,
        judgment.rules,
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("type checking ran but pytest-benchmark timed nothing")

    nanoseconds = measured.stats.median * 1e9
    benchmark.extra_info["nodes"] = judgment.term.size
    benchmark.extra_info["median_ns_per_check"] = nanoseconds
    benchmark.extra_info["mean_ns_per_check"] = measured.stats.mean * 1e9
    benchmark.extra_info["median_ns_per_node"] = nanoseconds / judgment.term.size
    benchmark.extra_info["budget_ns"] = CHECK_NS

    if wall_clock_budget.enforced:
        assert nanoseconds <= CHECK_NS


def test_forty_two_node_neo_davidsonian_check_assigns_every_node(
    judgment: Judgment,
) -> None:
    result = judgment.checker.check(judgment.term, judgment.signature, judgment.rules)

    assert judgment.term.size == 42
    assert len(result.assignment) == 42
    assert result.is_well_typed
