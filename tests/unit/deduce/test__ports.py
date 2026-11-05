from math import exp, isclose, log
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi._native import SpanFree
from lofi.deduce._forest import PackedForest
from lofi.deduce._goals import RootGoal
from lofi.deduce._ports import (
    CellPruner,
    Chart,
    EdgeScorer,
    Goal,
    NativeTermRule,
    Normalizer,
    Proposer,
    Semiring,
)
from lofi.deduce._semiring import (
    Boolean,
    Counting,
    Expectation,
    Forest,
    KBest,
    LogInside,
    Viterbi,
)
from lofi.deduce._spans import Cell, SpanIndex
from lofi.deduce._values import Edge, ItemKey
from lofi.judge import HOLE, NormalFormRole, Rule
from lofi.lexicon import Entry
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge import Direction, RuleProbe, RuleRestriction, Slot
    from lofi.terms import Matching


class Identity:
    """Compares category strings directly."""

    @property
    def name(self) -> str:
        """Return the relation's name."""
        return "identity"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, t: str, /) -> str:
        """Return the category itself, which is its own representative."""
        return t

    def equivalent(self, a: str, b: str, /) -> bool:
        """Return whether two categories are the same string."""
        return a == b


MATCHING: Final[Matching[str]] = Identity()


class ForwardApplication:
    """Applies categories written as `result/argument`."""

    @property
    def name(self) -> str:
        """Return the rule's name."""
        return "fa"

    @property
    def arity(self) -> int:
        """Return the rule's number of premises."""
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the directions supported by the rule."""
        return frozenset({"forward", "check", "residual"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return both premise slots."""
        return frozenset({0, 1})

    @property
    def forward_is_complete(self) -> bool:
        """Return whether application enumerates every accepted conclusion."""
        return True

    @property
    def matching(self) -> Matching[str]:
        """Return the congruence the rule reads categories under."""
        return MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        """Declare no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Decline premise-head indexing."""
        return None

    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        """Yield the result of a functor whose argument is the right premise."""
        if len(inputs) != self.arity:
            return

        functor, argument = inputs

        if "/" not in functor:
            return

        result, wanted = functor.split("/", 1)

        if wanted == argument:
            yield result

    def admits(self, inputs: tuple[str, ...], output: str, /) -> bool:
        """Return whether the rule relates these premises to this conclusion."""
        return output in tuple(self.apply(inputs))

    def residuate(self, output: str, known: tuple[Slot[str], ...], /) -> Iterator[str]:
        """Yield the fillers of the single hole in a premise tuple."""
        if len(known) != self.arity or sum(slot is HOLE for slot in known) != 1:
            return

        functor, argument = known

        if functor is HOLE and isinstance(argument, str):
            yield f"{output}/{argument}"

        elif isinstance(functor, str) and "/" in functor:
            result, wanted = functor.split("/", 1)

            if result == output:
                yield wanted

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the declaration this rule was built from."""
        return {"name": "fa", "arity": 2}


FA: Final[Rule[str]] = cast("Rule[str]", ForwardApplication())

DET = ItemKey((0, 1), "NP/N", "lex")
NOUN = ItemKey((1, 2), "N", "lex")
NAME = ItemKey((1, 2), "NP", "lex")
PHRASE = ItemKey((0, 2), "NP", "step")


class Beam:
    """Proposes ranked categories for each span."""

    def __init__(self, ranked: tuple[str, ...]) -> None:
        self._ranked = ranked

    def propose(
        self, rule: Rule[str], inputs: tuple[str, ...], span: Cell, /
    ) -> Iterator[str]:
        """Yield the ranked candidates the rule could draw over the span."""
        width = span[1] - span[0]

        for candidate in self._ranked:
            if len(inputs) == rule.arity and width >= rule.arity:
                yield candidate


class Tabulated:
    """Scores rules and lexical entries from lookup tables."""

    lowering = SpanFree()

    def __init__(self, per_rule: dict[str, float]) -> None:
        self._per_rule = per_rule

    def axiom_weight(self, _position: int, entry: Entry[str], /) -> float:
        """Return the entry's own weight, unchanged by where it landed."""
        return entry.weight

    def edge_weight(
        self,
        rule: Rule[str],
        inputs: tuple[str, ...],
        output: str,
        _span: Cell,
        _split: int | None,
        /,
    ) -> float:
        """Return the rule weight or log zero when inapplicable."""
        if not rule.admits(inputs, output):
            return -float("inf")

        return self._per_rule[rule.name]


class Narrowing:
    """Keeps the first item a cell offers."""

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[str], float]], /
    ) -> tuple[ItemKey[str], ...]:
        """Return the cell's earliest item, dropping every later one."""
        del span

        return tuple(key for key, _ in items[:1])


class FeatureErasure:
    """Removes bracketed category features."""

    def normalize(self, term: str, /) -> str:
        """Return the category with any bracketed feature removed."""
        head, bracket, _ = term.partition("[")

        return head if bracket else term


def filled() -> PackedForest[str, float]:
    chart = PackedForest[str, float](SpanIndex(2), LogInside())
    chart.add(DET, Edge("lex", (), "NP/N", log(0.5)))
    chart.add(NOUN, Edge("lex", (), "N", log(1.0)))
    chart.add(NAME, Edge("lex", (), "NP", log(0.25)))
    chart.add(PHRASE, Edge("fa", (DET, NOUN), "NP", log(0.5), split=1))
    chart.add(PHRASE, Edge("ba", (DET, NAME), "NP", log(0.5), split=1))

    return chart


def read_through[W](chart: Chart[str, W], key: ItemKey[str]) -> tuple[int, int]:
    return (len(chart.items_at(key.span)), len(chart.edges(key)))


def propose_through(proposer: Proposer[str], span: Cell) -> tuple[str, ...]:
    return tuple(proposer.propose(FA, ("NP/N", "N"), span))


def score_through(scorer: EdgeScorer[str, float], output: str) -> float:
    return scorer.edge_weight(FA, ("NP/N", "N"), output, (0, 2), 1)


def roots_through[W](goal: Goal[str], chart: Chart[str, W]) -> tuple[ItemKey[str], ...]:
    return tuple(goal.roots(chart))


def normalize_through(normalizer: Normalizer[str], term: str) -> str:
    return normalizer.normalize(term)


def keep_through[W](
    pruner: CellPruner[str, W], chart: Chart[str, W], span: Cell
) -> tuple[ItemKey[str], ...]:
    items = tuple((key, chart.inside(key)) for key in chart.items_at(span))

    return tuple(pruner.keep(span, items))


def test_the_chart_aggregates_one_inside_weight_for_an_ambiguous_item() -> None:
    chart = filled()
    both = 0.5 * 0.5 * 1.0 + 0.5 * 0.5 * 0.25

    assert isclose(exp(chart.inside(PHRASE)), both)
    assert isclose(exp(chart.inside(DET)), 0.5)


def test_the_chart_keeps_every_step_under_one_item() -> None:
    chart = filled()

    assert read_through(chart, PHRASE) == (1, 2)
    assert read_through(chart, NOUN) == (2, 1)
    assert {edge.rule_name for edge in chart.edges(PHRASE)} == {"fa", "ba"}


def test_the_chart_answers_for_an_item_it_does_not_hold() -> None:
    chart = filled()
    absent = ItemKey((0, 2), "S", "step")

    assert chart.inside(absent) == LogInside().zero
    assert chart.edges(absent) == ()
    assert chart.items_at((0, 1)) == (DET,)

    with pytest.raises(KeyError):
        chart.term(absent)


def test_the_chart_names_the_term_an_item_stands_for() -> None:
    assert filled().term(PHRASE) == "NP"


def test_the_chart_reads_back_one_tree_for_each_way_the_item_was_reached() -> None:
    chart = filled()
    trees = tuple(chart.derivations(PHRASE))

    assert len(trees) == 2
    assert {tree.edge.rule_name for tree in trees} == {"fa", "ba"}
    assert all(tree.key == PHRASE for tree in trees)
    assert all(len(tree.children) == 2 for tree in trees)
    assert all(child.children == () for tree in trees for child in tree.children)


def test_a_proposer_supplies_the_candidates_a_check_only_rule_needs() -> None:
    beam = Beam(("NP", "S"))

    assert isinstance(beam, Proposer)
    assert propose_through(beam, (0, 2)) == ("NP", "S")
    assert propose_through(beam, (0, 1)) == ()
    assert FA.admits(("NP/N", "N"), propose_through(beam, (0, 2))[0])


def test_a_generic_rule_need_not_publish_native_term_capabilities() -> None:
    assert isinstance(FA, Rule)
    assert not isinstance(FA, NativeTermRule)


def test_a_scorer_weighs_one_step_and_never_the_chart() -> None:
    scorer = Tabulated({"fa": log(0.5)})

    assert isinstance(scorer, EdgeScorer)
    assert isclose(exp(score_through(scorer, "NP")), 0.5)
    assert score_through(scorer, "S") == -float("inf")
    assert scorer.axiom_weight(0, Entry("the", "NP/N", weight=log(0.5))) == log(0.5)


def test_a_goal_accepts_only_what_it_enumerates() -> None:
    chart = filled()
    goal = RootGoal(TargetSet(("NP",), MATCHING))

    assert isinstance(goal, Goal)
    assert roots_through(goal, chart) == (PHRASE,)
    assert goal.accepts(chart, PHRASE)
    assert not goal.accepts(chart, NOUN)


def test_a_goal_over_a_category_the_chart_lacks_ends_nowhere() -> None:
    chart = filled()

    assert roots_through(RootGoal(TargetSet(("S",), MATCHING)), chart) == ()


def test_a_normalizer_is_idempotent() -> None:
    normalizer = FeatureErasure()

    assert isinstance(normalizer, Normalizer)
    assert normalize_through(normalizer, "S[dcl]\\NP") == "S"
    assert normalize_through(normalizer, "NP/N") == "NP/N"
    assert normalize_through(normalizer, normalize_through(normalizer, "S[dcl]")) == "S"


def test_a_pruner_chooses_which_of_a_cells_items_the_chart_keeps() -> None:
    chart = filled()
    pruner = Narrowing()

    assert isinstance(pruner, CellPruner)
    assert keep_through(pruner, chart, (1, 2)) == (NOUN,)
    assert keep_through(pruner, chart, (0, 2)) == (PHRASE,)
    assert keep_through(pruner, chart, (0, 3)) == ()


@pytest.mark.parametrize(
    "port",
    [Proposer, EdgeScorer, Goal, Normalizer, CellPruner],
)
def test_a_value_answering_none_of_the_calls_is_not_a_port(port: type) -> None:
    assert not isinstance(object(), port)


def test_every_shipped_algebra_satisfies_the_protocol_at_runtime() -> None:
    assert isinstance(Boolean(), Semiring)
    assert isinstance(LogInside(), Semiring)
    assert isinstance(Viterbi(), Semiring)
    assert isinstance(Counting(), Semiring)
    assert isinstance(KBest(2), Semiring)
    assert isinstance(Forest[str](), Semiring)
    assert isinstance(Expectation(2), Semiring)
    assert not isinstance(object(), Semiring)
