"""Benchmarks bounded rule-set closure."""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.spec import CompositionSpec, RuleSetSpec
from lofi.spec.grammar import FeatureErasureMatchingSpec, OperatorSpec

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

    from lofi.judge import RuleSet
    from lofi.terms import Functors, Term

_CORE_COUNT: Final = 1 << 6
"""Generated atomic-category count."""

_FEATURES: Final = tuple(f"feature-{index}" for index in range(6))
"""Generated feature values per atom."""

_STRUCTURAL_LAYER_COUNTS: Final = (80, 90, 100, 90, 80, 56)
"""Generated functor counts at depths two through seven."""

_ERASED_SEED_CATEGORIES: Final = _CORE_COUNT + sum(_STRUCTURAL_LAYER_COUNTS)
"""Expected feature-erased seed count."""

_NODE_BUDGET: Final = 768
"""Closure node budget."""

_MAX_DEPTH: Final = 1 << 6
"""Closure step bound."""


class ClosureCase(NamedTuple):
    """Generated CCG-shaped closure inputs."""

    rules: RuleSet[Term]
    seeds: tuple[Term, ...]
    erased_seeds: int


class CategoryPair(NamedTuple):
    """Pair a category with a feature-bearing equivalent."""

    core: Term
    featured: Term


def _next_layer(
    functors: Functors[Term],
    prior: tuple[CategoryPair, ...],
    atoms: tuple[CategoryPair, ...],
    count: int,
) -> tuple[CategoryPair, ...]:
    """Build one deterministic layer split evenly between slash directions."""
    quota = count // 2
    layer: list[CategoryPair] = []

    for slash in (functors.forward_slash, functors.backward_slash):
        built = 0

        for offset in range(1, len(atoms) + 1):
            for at, principal in enumerate(prior):
                argument = atoms[(at + offset) % len(atoms)]
                layer.append(
                    CategoryPair(
                        functors.make(slash, principal.core, argument.core),
                        functors.make(slash, principal.featured, argument.featured),
                    )
                )
                built += 1

                if built == quota:
                    break

            if built == quota:
                break

    return tuple(layer)


def _structural_seeds(
    functors: Functors[Term],
    variants: tuple[tuple[Term, ...], ...],
) -> tuple[Term, ...]:
    """Build feature-paired functors with a corpus-shaped depth distribution."""
    atoms = tuple(
        CategoryPair(group[0], group[1 + at % len(_FEATURES)])
        for at, group in enumerate(variants)
    )
    prior = atoms
    generated: list[CategoryPair] = []

    for count in _STRUCTURAL_LAYER_COUNTS:
        prior = _next_layer(functors, prior, atoms, count)
        generated.extend(prior)

    return tuple(term for pair in generated for term in pair)


def _case() -> ClosureCase:
    """Build feature-bearing categories for both slash directions."""
    names = tuple(f"a{index}" for index in range(_CORE_COUNT))
    adopted = declared_rules(
        ccg_signature(
            *names,
            operators=tuple(
                OperatorSpec(name=f"{name}[{feature}]", sort="Cat", erases_to=name)
                for name in names
                for feature in _FEATURES
            ),
            name="generated-ccg-scale",
        ),
        RuleSetSpec(
            name="generated-ccg-scale",
            rules=tuple(
                CompositionSpec(view=view, order=1, direction="kuhlmann")
                for view in ("/", "\\")
            ),
        ),
        matching=FeatureErasureMatchingSpec(),
    )
    context = adopted.context
    store = context.store
    syntax = context.syntax
    functors, _ = context.functors["/"]
    variants = tuple(
        (
            syntax.parse(store, name),
            *(syntax.parse(store, f"{name}[{feature}]") for feature in _FEATURES),
        )
        for name in names
    )
    atoms = tuple(term for group in variants for term in group)
    seeds = (*atoms, *_structural_seeds(functors, variants))
    inventory = adopted.rule_set

    return ClosureCase(
        inventory, seeds, len({inventory.matching.key(term) for term in seeds})
    )


@pytest.fixture(scope="module")
def closure_case() -> ClosureCase:
    """Return the generated benchmark inputs."""
    return _case()


def test_generated_ccg_scale_closure_records_explicit_exhaustion(
    benchmark: BenchmarkFixture,
    closure_case: ClosureCase,
) -> None:
    """Record indexed composition closure through budget exhaustion."""
    report = benchmark.pedantic(
        closure_case.rules.closure,
        args=(closure_case.seeds,),
        kwargs={"max_depth": _MAX_DEPTH, "budget": _NODE_BUDGET},
        rounds=1,
        iterations=1,
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("closure ran but pytest-benchmark timed nothing")

    benchmark.extra_info["raw_seed_categories"] = len(closure_case.seeds)
    benchmark.extra_info["erased_seed_categories"] = closure_case.erased_seeds
    benchmark.extra_info["reached_categories"] = len(report.reached)
    benchmark.extra_info["rule_count"] = len(closure_case.rules)
    benchmark.extra_info["fired_rule_count"] = len(closure_case.rules) - len(
        report.never_applicable
    )
    benchmark.extra_info["node_budget"] = _NODE_BUDGET
    benchmark.extra_info["max_depth"] = _MAX_DEPTH
    benchmark.extra_info["wall_time_ms"] = measured.stats.median * 1e3
    benchmark.extra_info["outcome"] = "node-budget-exhausted-after-rule-firing"


def test_generated_ccg_scale_closure_exhausts_its_explicit_budget(
    closure_case: ClosureCase,
) -> None:
    assert len(closure_case.seeds) >= _CORE_COUNT * 20
    assert closure_case.erased_seeds == _ERASED_SEED_CATEGORIES
    assert closure_case.erased_seeds < len(closure_case.seeds)
    assert len(closure_case.rules) == 2

    report = closure_case.rules.closure(
        closure_case.seeds,
        max_depth=_MAX_DEPTH,
        budget=_NODE_BUDGET,
    )

    assert not report.saturated
    assert report.category_set == "unknown"
    assert len(report.reached) == report.budget
    assert len(report.reached) > closure_case.erased_seeds
    assert report.escaping
    assert report.never_applicable == ()
