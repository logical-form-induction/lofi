from __future__ import annotations

from itertools import product
from math import inf, log
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import pytest
from tests.support.rules import ccg_signature, declared_rules

try:
    from quivers.stochastic import LOG_PROB, DeductionSystem, InferenceRule, Wildcard
except ModuleNotFoundError as error:
    if error.name != "quivers":
        raise
    _QUIVERS_AVAILABLE = False
else:
    from torch import Tensor, float64, tensor

    _QUIVERS_AVAILABLE = True

from lofi._native import SpanFree
from lofi.deduce import LogInside, ParsePolicy, Reachability, Retain, RootGoal, parse
from lofi.lexicon import Entry, StaticLexicon
from lofi.spec import (
    ApplicationSpec,
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
)
from lofi.terms import (
    TargetSet,
    Term,
    TermKey,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.deduce import Cell, Goal
    from lofi.judge import Rule, RuleSet


type SpanItem = tuple[str, int, int, TermKey]
type WeightedItem = tuple[SpanItem, Tensor]
type LexicalTable = Mapping[str, tuple[tuple[TermKey, float], ...]]
type ProductionTable = tuple[tuple[str, str, str, str], ...]


_CFG_NONTERMINALS: Final = ("S", "A", "B", "C", "D")
_CFG_PRODUCTIONS: Final[ProductionTable] = (
    ("s-ab", "S", "A", "B"),
    ("s-ba", "S", "B", "A"),
    ("a-aa", "A", "A", "A"),
    ("a-cd", "A", "C", "D"),
    ("a-dc", "A", "D", "C"),
    ("b-bb", "B", "B", "B"),
    ("b-cc", "B", "C", "C"),
    ("b-dd", "B", "D", "D"),
)
_CFG_GOAL: Final = "S"
_CFG_RULES: Final = declared_rules(
    ccg_signature(*_CFG_NONTERMINALS, name="quivers-cfg"),
    RuleSetSpec(
        name="cfg-productions",
        rules=tuple(
            PatternRuleSpec(
                name=name,
                premises=(
                    ConstantPatternSpec(name=left),
                    ConstantPatternSpec(name=right),
                ),
                conclusion=ConstantPatternSpec(name=parent),
            )
            for name, parent, left, right in _CFG_PRODUCTIONS
        ),
    ),
)
_CFG_STORE: Final = _CFG_RULES.context.store
_CFG_TERMS: Final = {
    name: _CFG_RULES.context.syntax.parse(_CFG_STORE, name)
    for name in _CFG_NONTERMINALS
}
_CFG_LEXICAL_PROBABILITIES: Final = {
    "x": {"C": 0.70, "D": 0.30},
    "y": {"C": 0.40, "D": 0.60},
}
_CFG_SENTENCES: Final = (
    *product(("x", "y"), repeat=4),
    *tuple(product(("x", "y"), repeat=6))[:34],
)


_AB_RULES: Final = declared_rules(
    ccg_signature("s", "np", "other", name="quivers-ab"),
    RuleSetSpec(
        name="ab-application",
        rules=(ApplicationSpec(view="/"), ApplicationSpec(view="\\")),
    ),
)
_AB_STORE: Final = _AB_RULES.context.store
_AB_S: Final = _AB_RULES.context.syntax.parse(_AB_STORE, "s")
_AB_NP: Final = _AB_RULES.context.syntax.parse(_AB_STORE, "np")
_AB_OTHER: Final = _AB_RULES.context.syntax.parse(_AB_STORE, "other")
_AB_FUNCTORS: Final = _AB_RULES.context.functors["/"][0]
_AB_FORWARD: Final = _AB_FUNCTORS.make("/", _AB_S, _AB_NP)
_AB_BACKWARD: Final = _AB_FUNCTORS.make("\\", _AB_S, _AB_NP)


class ABApplication(NamedTuple):
    """One independently encoded AB application declaration."""

    name: str
    slash: str
    functor_side: Literal["left", "right"]


_AB_APPLICATIONS: Final = (
    ABApplication("forward-application", "/", "left"),
    ABApplication("backward-application", "\\", "right"),
)


class ABCase(NamedTuple):
    """One independently weighted applicative sentence."""

    sentence: tuple[str, str]
    first_category: Term
    second_category: Term
    first_probability: float
    second_probability: float


def _ab_cases() -> tuple[ABCase, ...]:
    """Build fifty distinct applicative cases."""
    cases: list[ABCase] = []

    for index in range(25):
        functor_probability = 0.55 + index / 100.0
        argument_probability = 0.80 - index / 200.0
        forward = f"forward-{index}"
        backward = f"backward-{index}"
        forward_argument = f"forward-argument-{index}"
        backward_argument = f"backward-argument-{index}"
        cases.append(
            ABCase(
                (forward, forward_argument),
                _AB_FORWARD,
                _AB_NP,
                functor_probability,
                argument_probability,
            )
        )
        cases.append(
            ABCase(
                (backward_argument, backward),
                _AB_NP,
                _AB_BACKWARD,
                argument_probability,
                functor_probability,
            )
        )

    return tuple(cases)


_AB_CASES: Final = _ab_cases()
_BOUNDARY: Final[TermKey] = ("quivers-boundary", None, ())
_ACCEPT: Final[TermKey] = ("quivers-accept", None, ())


class UnitScorer:
    """Unit lexical and rule weights."""

    lowering = SpanFree()

    def axiom_weight[T](self, position: int, entry: Entry[T], /) -> float:
        """Return the multiplicative unit."""
        del position, entry

        return 0.0

    def edge_weight[T](
        self,
        rule: Rule[T],
        inputs: tuple[T, ...],
        output: T,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the multiplicative unit."""
        del rule, inputs, output, span, split

        return 0.0


def _atom(name: str) -> TermKey:
    """Encode one nullary category structurally."""
    return (name, None, ())


def _goal_rule(goal: TermKey) -> InferenceRule:
    """Build a stateless full-span goal sequent."""
    end = Wildcard("goal-end")

    return InferenceRule(
        "accept",
        (
            ("span", 0, end, goal),
            ("span", end, end, _BOUNDARY),
        ),
        ("span", 0, end, _ACCEPT),
    )


def _is_goal(item: SpanItem) -> bool:
    """Recognize the private accepted-span category."""
    return item[3] == _ACCEPT


def _axiom_injector(
    table: LexicalTable,
) -> Callable[[Sequence[str]], list[WeightedItem]]:
    """Build a lexical and boundary axiom injector."""

    def axioms(tokens: Sequence[str]) -> list[WeightedItem]:
        """Emit lexical spans and the final boundary span."""
        emitted = [
            (
                ("span", position, position + 1, category),
                tensor(log(probability), dtype=float64),
            )
            for position, token in enumerate(tokens)
            for category, probability in table[token]
        ]
        emitted.append(
            (
                ("span", len(tokens), len(tokens), _BOUNDARY),
                tensor(0.0, dtype=float64),
            )
        )

        return emitted

    return axioms


def _cfg_quivers(lexical: LexicalTable) -> DeductionSystem:
    """Compile a CFG declaration to quivers sequents."""
    rules = []

    for name, parent, left, right in _CFG_PRODUCTIONS:
        start = Wildcard(f"{name}-start")
        split = Wildcard(f"{name}-split")
        end = Wildcard(f"{name}-end")
        rules.append(
            InferenceRule(
                name,
                (
                    ("span", start, split, _atom(left)),
                    ("span", split, end, _atom(right)),
                ),
                ("span", start, end, _atom(parent)),
            )
        )

    rules.append(_goal_rule(_atom(_CFG_GOAL)))

    return DeductionSystem(
        tuple(rules),
        LOG_PROB,
        _axiom_injector(lexical),
        _is_goal,
    )


def _ab_quivers(lexical: LexicalTable) -> DeductionSystem:
    """Compile the applicative declaration to quivers sequents."""
    rules = []

    for application in _AB_APPLICATIONS:
        start = Wildcard(f"{application.name}-start")
        split = Wildcard(f"{application.name}-split")
        end = Wildcard(f"{application.name}-end")
        result = Wildcard(f"{application.name}-result")
        argument = Wildcard(f"{application.name}-argument")
        functor = (application.slash, None, (result, argument))
        inputs = (
            (functor, argument)
            if application.functor_side == "left"
            else (argument, functor)
        )
        rules.append(
            InferenceRule(
                application.name,
                (
                    ("span", start, split, inputs[0]),
                    ("span", split, end, inputs[1]),
                ),
                ("span", start, end, result),
            )
        )

    rules.append(_goal_rule(_AB_S.key()))

    return DeductionSystem(
        tuple(rules),
        LOG_PROB,
        _axiom_injector(lexical),
        _is_goal,
    )


def _lofi_weight[T](
    sentence: Sequence[str],
    rules: RuleSet[T],
    source: StaticLexicon[T],
    goal: Goal[T],
) -> float:
    """Compute one LoFI log-inside goal weight."""
    semiring = LogInside()
    forest = parse(
        sentence,
        rules=rules,
        source=source,
        semiring=semiring,
        scorer=UnitScorer(),
        goal=goal,
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    weight = semiring.zero

    for root in goal.roots(forest):
        weight = semiring.plus(weight, forest.inside(root))

    return weight


def _quivers_weight(system: DeductionSystem, sentence: Sequence[str]) -> float:
    """Compute one quivers log-inside goal weight."""
    return float(system(tuple(sentence)).goal_weight().item())


def _cfg_oracle(
    sentence: Sequence[str],
    productions: ProductionTable,
) -> float:
    """Compute the CFG goal weight in primitive probability space."""
    chart: dict[tuple[int, int, str], float] = {}

    for position, token in enumerate(sentence):
        for category, probability in _CFG_LEXICAL_PROBABILITIES[token].items():
            chart[position, position + 1, category] = probability

    for width in range(2, len(sentence) + 1):
        for start in range(len(sentence) - width + 1):
            end = start + width

            for split in range(start + 1, end):
                for _, parent, left_name, right_name in productions:
                    left = chart.get((start, split, left_name), 0.0)
                    right = chart.get((split, end, right_name), 0.0)

                    if left == 0.0 or right == 0.0:
                        continue

                    key = (start, end, parent)
                    chart[key] = chart.get(key, 0.0) + left * right

    probability = chart.get((0, len(sentence), _CFG_GOAL), 0.0)

    return -inf if probability == 0.0 else log(probability)


def _cfg_inputs() -> tuple[StaticLexicon[Term], LexicalTable]:
    """Compile the shared CFG lexical declaration twice."""
    source = StaticLexicon(
        Entry(token, _CFG_TERMS[name], weight=log(probability))
        for token, alternatives in _CFG_LEXICAL_PROBABILITIES.items()
        for name, probability in alternatives.items()
    )
    table = {
        token: tuple(
            (_atom(name), probability) for name, probability in alternatives.items()
        )
        for token, alternatives in _CFG_LEXICAL_PROBABILITIES.items()
    }

    return source, table


def _ab_inputs() -> tuple[StaticLexicon[Term], LexicalTable]:
    """Compile the shared AB lexical declaration twice."""
    entries: list[Entry[Term]] = []
    table: dict[str, tuple[tuple[TermKey, float], ...]] = {}

    for case in _AB_CASES:
        first, second = case.sentence

        for form, category, probability in (
            (first, case.first_category, case.first_probability),
            (second, case.second_category, case.second_probability),
        ):
            alternatives = (
                (category, probability),
                (_AB_OTHER, 1.0 - probability),
            )
            entries.extend(
                Entry(form, term, weight=log(mass)) for term, mass in alternatives
            )
            table[form] = tuple((term.key(), mass) for term, mass in alternatives)

    return StaticLexicon(entries), table


@pytest.mark.integration
@pytest.mark.skipif(
    not _QUIVERS_AVAILABLE,
    reason="requires the lofi[cross-engine] extra",
)
def test_quivers_agrees_with_lofi_and_independent_oracles() -> None:
    """Compare both engines on fifty CFG and fifty AB sentences."""
    assert len(_CFG_SENTENCES) == 50
    assert len(_AB_CASES) == 50

    cfg_rules = _CFG_RULES.rule_set
    cfg_goal = RootGoal(TargetSet((_CFG_TERMS[_CFG_GOAL],), cfg_rules.matching))
    cfg_source, cfg_table = _cfg_inputs()
    cfg_quivers = _cfg_quivers(cfg_table)

    for sentence in _CFG_SENTENCES:
        expected = _cfg_oracle(sentence, _CFG_PRODUCTIONS)
        lofi = _lofi_weight(sentence, cfg_rules, cfg_source, cfg_goal)
        quivers_weight = _quivers_weight(cfg_quivers, sentence)

        assert lofi == pytest.approx(expected, abs=1e-5, rel=0.0)
        assert quivers_weight == pytest.approx(expected, abs=1e-5, rel=0.0)
        assert quivers_weight == pytest.approx(lofi, abs=1e-5, rel=0.0)

    ab_rules = _AB_RULES.rule_set
    ab_goal = RootGoal(TargetSet((_AB_S,), ab_rules.matching))
    ab_source, ab_table = _ab_inputs()
    ab_quivers = _ab_quivers(ab_table)

    for case in _AB_CASES:
        expected = log(case.first_probability) + log(case.second_probability)
        lofi = _lofi_weight(case.sentence, ab_rules, ab_source, ab_goal)
        quivers_weight = _quivers_weight(ab_quivers, case.sentence)

        assert lofi == pytest.approx(expected, abs=1e-5, rel=0.0)
        assert quivers_weight == pytest.approx(expected, abs=1e-5, rel=0.0)
        assert quivers_weight == pytest.approx(lofi, abs=1e-5, rel=0.0)
