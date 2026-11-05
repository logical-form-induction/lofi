"""Benchmarks proof-action derivation, one rule at a time.

Each case erases one rule's oriented schemata and searches the implicational
fragment for their inhabitants, which is the work the judgment layer does
once per rule when a rule set is compiled.
"""

from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest

from lofi.judge import ProofSearch, derive_action
from lofi.spec import (
    CompositionSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    SubstitutionSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec._rule_laws import LawGrammar
from lofi.terms import (
    BindingSite,
    Calculus,
    Normalization,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.judge import ProofSchema

#: The search bounds every rule below is derived under.
FUEL: Final = 40
MAX_USES: Final = 2

#: Nanoseconds one rule's derivation may take.
DERIVATION_NS: Final = 700_000


class LambdaCalculus:
    """Caller-owned proof-term constructors."""

    def __init__(self, store: TermStore) -> None:
        self._store = store

    @property
    def store(self) -> TermStore:
        return self._store

    @property
    def operators(self) -> Calculus:
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        return self.free_variable(f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        return self._store.fvar("Proof", name)

    def bound_variable(self, index: int, /) -> Term:
        return self._store.bvar("Proof", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        return self._store.op(
            self._store.signature.by_name("app"),
            function,
            argument,
        )

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return self._store.bind("lam", variable, body)


class Layer(NamedTuple):
    """The stores and the erasure every rule below is derived against.

    Attributes
    ----------
    erasure : Erasure
        The category-to-type erasure proof search reads.
    calculus : LambdaCalculus
        The proof-term constructors the search builds inhabitants with.
    """

    erasure: Erasure
    calculus: LambdaCalculus


def _layer() -> Layer:
    """Build the language and proof stores with their structural erasure."""
    language = TermStore(
        Signature(
            "benchmark-language",
            ("Cat", "Type"),
            (
                Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
                Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
                Operator("type-x", "Type"),
                Operator("type-y", "Type"),
                Operator("->", "Type", arg_sorts=("Type", "Type")),
            ),
            vocabularies=(Vocabulary("Cat", ("x", "y")),),
            open_sorts=("Type",),
        )
    )
    proofs = TermStore(
        Signature(
            "benchmark-proofs",
            ("Proof",),
            (
                Operator("app", "Proof", arg_sorts=("Proof", "Proof")),
                Operator(
                    "lam",
                    "Proof",
                    arg_sorts=("Proof",),
                    binding=(BindingSite((0,), "Proof"),),
                ),
            ),
        )
    )
    erasure = Erasure(
        language,
        {
            "x": language.op(language.signature.by_name("type-x")),
            "y": language.op(language.signature.by_name("type-y")),
        },
        {
            name: language.op(
                language.signature.by_name("->"),
                language.fvar("Type", "a1"),
                language.fvar("Type", "a0"),
            )
            for name in ("fwd", "bwd")
        },
        category_sort="Cat",
        function_type="->",
    )

    return Layer(erasure, LambdaCalculus(proofs))


#: One representative of each rule kind the judgment layer derives actions for.
RULE_NAMES: Final = (
    "application",
    "composition-1",
    "crossed-composition-1",
    "substitution-1",
    "type-raising",
)


@pytest.fixture(scope="module")
def layer() -> Layer:
    """Return the stores, the erasure and the calculus the rules derive against."""
    return _layer()


@pytest.fixture(scope="module")
def rules() -> dict[str, ProofSchema]:
    """Return one rule of each kind, keyed by the name its case is reported under.

    The rules are compiled over the law signature, whose ``fwd`` and ``bwd``
    constructors and ``Cat`` sort are the ones the erasure reads, so their
    schemata erase against the benchmark's category store.
    """
    grammar = LawGrammar("direct")
    x = TermPatternSpec(name="X", sort="Cat")
    y = TermPatternSpec(name="Y", sort="Cat")
    declarations = {
        "application": PatternRuleSpec(
            name="application",
            premises=(
                NodePatternSpec(
                    operator=FixedOperatorSpec(name="fwd"), arguments=(x, y)
                ),
                y,
            ),
            conclusion=x,
        ),
        "composition-1": CompositionSpec(view="fwd", order=1, direction="kuhlmann"),
        "crossed-composition-1": CompositionSpec(
            view="fwd", order=1, direction="kuhlmann", crossed=True
        ),
        "substitution-1": SubstitutionSpec(view="fwd", order=1),
        "type-raising": TypeRaisingSpec(view="fwd", arguments=None),
    }

    return {
        name: cast("ProofSchema", grammar.rule(declaration))
        for name, declaration in declarations.items()
    }


def test_every_rule_kind_the_judgment_layer_derives_has_a_case(
    rules: dict[str, ProofSchema],
) -> None:
    assert tuple(rules) == RULE_NAMES
    assert all(rule.proof_schemata for rule in rules.values())


@pytest.mark.parametrize("rule_name", RULE_NAMES)
def test_deriving_one_rules_actions_is_within_budget(
    benchmark: BenchmarkFixture,
    layer: Layer,
    rules: dict[str, ProofSchema],
    rule_name: str,
    wall_clock_budget: WallClockBudget,
) -> None:
    rule = rules[rule_name]
    benchmark(
        derive_action,
        rule,
        layer.erasure,
        calculus=layer.calculus,
        normalization=Normalization(
            fuel=FUEL,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=MAX_USES, max_inhabitants=64
        ),
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("derivation ran but pytest-benchmark timed nothing")

    nanoseconds = measured.stats.median * 1e9
    benchmark.extra_info["rule"] = rule_name
    benchmark.extra_info["schemata"] = len(rule.proof_schemata)
    benchmark.extra_info["median_ns_per_derivation"] = nanoseconds
    benchmark.extra_info["mean_ns_per_derivation"] = measured.stats.mean * 1e9
    benchmark.extra_info["budget_ns"] = DERIVATION_NS

    if wall_clock_budget.enforced:
        assert nanoseconds <= DERIVATION_NS


@pytest.mark.parametrize("rule_name", RULE_NAMES)
def test_deriving_one_rules_actions_returns_inhabited_actions(
    layer: Layer,
    rules: dict[str, ProofSchema],
    rule_name: str,
) -> None:
    actions = derive_action(
        rules[rule_name],
        layer.erasure,
        calculus=layer.calculus,
        normalization=Normalization(
            fuel=FUEL,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=MAX_USES, max_inhabitants=64
        ),
    )

    assert actions
    assert all(action.inhabitants.terms for action in actions)
