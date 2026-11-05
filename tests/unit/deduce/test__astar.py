from math import isclose
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import SpanFree
from lofi.deduce._astar import (
    BINARY_WIDENING,
    FREE_ROOTS,
    GOAL_ROOTS,
    LINEAR_WIDENING,
    SUPERTAG_BEAM_PROPOSER,
    SUPERTAG_OUTSIDE,
    ZERO_HEURISTIC,
    AStarContext,
    AStarOptions,
    AStarResult,
    astar_parse,
    supertag_outside,
)
from lofi.deduce._engine import parse
from lofi.deduce._goals import RootGoal
from lofi.deduce._policy import (
    CKY,
    CONTIGUOUS_SPANS,
    NATIVE_CHARTS,
    ParsePolicy,
    ParseSettings,
    Reachability,
    Retain,
)
from lofi.deduce._pruners import TopKPruner
from lofi.deduce._semiring import LogInside, Viterbi
from lofi.judge import NoNormalForm, NormalFormRole, RuleSet
from lofi.lexicon import BeamSource, Entry
from lofi.spec import (
    ConstantPatternSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PatternSpec,
    RuleSetSpec,
)
from lofi.terms import TargetSet, Term

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.deduce import Cell
    from lofi.judge import (
        Direction,
        PatternRule,
        Rule,
        RuleProbe,
        RuleRestriction,
        SchematicSequent,
        Slot,
    )
    from lofi.terms import Matching, TermRuleKernel


_CATEGORY_NAMES: Final = (
    "A",
    "B",
    "C",
    "X",
    "Y",
    "NP",
    "S",
    *(f"L{position}" for position in range(3)),
    *(f"R{position}" for position in range(3)),
    *(f"S{left}{right}" for left in range(3) for right in range(3)),
)
EXHAUSTIVE: Final = ParseSettings(
    ParsePolicy(Retain.FOREST, Reachability.ALL),
    CKY,
    CONTIGUOUS_SPANS,
    NATIVE_CHARTS,
    None,
    LogInside(),
    prune_lexical=False,
    measure_pruned_mass=False,
)
VITERBI_BEAM: Final = ParseSettings(
    ParsePolicy(Retain.FOREST, Reachability.ALL),
    CKY,
    CONTIGUOUS_SPANS,
    NATIVE_CHARTS,
    TopKPruner[object](1),
    Viterbi(),
    prune_lexical=False,
    measure_pruned_mass=False,
)
_SIGNATURE: Final = ccg_signature(*_CATEGORY_NAMES, name="astar-categories")
_ANCHOR: Final = declared_rules(_SIGNATURE, RuleSetSpec(name="astar", rules=()))
_STORE: Final = _ANCHOR.context.store
_SYNTAX: Final = _ANCHOR.context.syntax
EXACT: Final[Matching[Term]] = _ANCHOR.rule_set.matching


def _category(name: str) -> Term:
    """Return one declared test category."""
    return _SYNTAX.parse(_STORE, name)


def _label(term: Term) -> str:
    """Return one declared test category's label."""
    return _SYNTAX.render(_STORE, term)


def _pattern(name: str) -> PatternSpec:
    """Return the fixed pattern of one atomic or backward-functor category."""
    if "\\" not in name:
        return ConstantPatternSpec(name=name)

    result, argument = name.split("\\")

    return NodePatternSpec(
        operator=FixedOperatorSpec(name="\\"),
        arguments=(_pattern(result), _pattern(argument)),
    )


def _compiled(
    table: Mapping[tuple[str, str], tuple[str, ...]], name: str
) -> tuple[Rule[Term], ...]:
    """Compile one pattern rule per row and conclusion of a finite relation."""
    rows = tuple(
        (left, right, output)
        for (left, right), outputs in table.items()
        for output in outputs
    )
    adopted = declared_rules(
        _SIGNATURE,
        RuleSetSpec(
            name=name,
            rules=tuple(
                PatternRuleSpec(
                    name=name if len(rows) == 1 else f"{name}-{at}",
                    premises=(_pattern(left), _pattern(right)),
                    conclusion=_pattern(output),
                    head_policy="non-absorbing",
                )
                for at, (left, right, output) in enumerate(rows)
            ),
        ),
    )

    return adopted.rule_set.rules


class CountedRule:
    """A compiled rule that counts the admission checks the search asks of it.

    Parameters
    ----------
    rule
        The compiled rule.
    checked
        Whether the rule offers only the checking direction, so its
        conclusions must be proposed.
    """

    def __init__(self, rule: Rule[Term], *, checked: bool = False) -> None:
        self._rule = rule
        self._checked = checked
        self.admission_calls = 0

    @property
    def name(self) -> str:
        return self._rule.name

    @property
    def arity(self) -> int:
        return self._rule.arity

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset(("check",)) if self._checked else self._rule.directions

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return self._rule.spine_order

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        return not self._checked

    @property
    def matching(self) -> Matching[Term]:
        return EXACT

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        return cast("PatternRule", self._rule).lowering

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return cast("PatternRule", self._rule).reachability_schemata

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        yield from self._rule.apply(inputs)

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        self.admission_calls += 1
        return self._rule.admits(inputs, output)

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        del output, known
        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        return self._rule.to_spec()


class ZeroScorer:
    """Zero-valued factors beyond the supertag weights."""

    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split
        return 0.0


def _context(
    table: Mapping[tuple[str, str], tuple[str, ...]],
    targets: tuple[str, ...] = ("S",),
) -> AStarContext[Term]:
    rules = RuleSet("search", _compiled(table, "combine"), EXACT, NoNormalForm[Term]())

    return AStarContext(
        rules,
        ZeroScorer(),
        RootGoal(TargetSet(tuple(_category(target) for target in targets), EXACT)),
        0,
        1,
        EXHAUSTIVE,
    )


def _beam(
    left: tuple[tuple[str, float], ...],
    right: tuple[tuple[str, float], ...],
) -> BeamSource[Term]:
    return BeamSource(
        (
            tuple(
                Entry("left", _category(category), weight=weight)
                for category, weight in left
            ),
            tuple(
                Entry("right", _category(category), weight=weight)
                for category, weight in right
            ),
        )
    )


def _parse(
    source: BeamSource[Term], strategy: str, *, free: bool = False
) -> AStarResult[Term]:
    return astar_parse(
        ("left", "right"),
        source,
        _context({("NP", "S\\NP"): ("S",)}),
        _options(strategy, free=free),
    )


def _options(
    strategy: str = "linear",
    *,
    free: bool = False,
    zero: bool = False,
) -> AStarOptions:
    """Return one declared test search setup."""
    return AStarOptions(
        LINEAR_WIDENING if strategy == "linear" else BINARY_WIDENING,
        FREE_ROOTS if free else GOAL_ROOTS,
        ZERO_HEURISTIC if zero else SUPERTAG_OUTSIDE,
        None,
    )


def test_outside_estimate_sums_other_positions_best_weights() -> None:
    source = BeamSource(
        (
            (Entry("a", "A", weight=-0.1),),
            (Entry("b", "B", weight=-0.2),),
            (Entry("c", "C", weight=-0.3),),
        )
    )
    outside = supertag_outside(source)

    assert isclose(outside[0], -0.5)
    assert isclose(outside[1], -0.4)
    assert isclose(outside[2], -0.3)


def test_supertag_beam_proposer_ranks_active_categories_without_validating() -> None:
    source = BeamSource(
        (
            (
                Entry("left", _category("X"), weight=-0.4),
                Entry("left", _category("Y"), weight=-0.5),
            ),
            (
                Entry("right", _category("C"), weight=-0.1),
                Entry("right", _category("X"), weight=-0.2),
            ),
        )
    )
    (shared,) = _compiled({("A", "B"): ("X",)}, "shared")
    (shared_other,) = _compiled({("A", "B"): ("Y",)}, "shared")
    rule = CountedRule(shared)
    other = CountedRule(shared_other)
    inputs = (_category("A"), _category("B"))
    proposer = SUPERTAG_BEAM_PROPOSER.build(source)
    expected = (_category("C"), _category("X"), _category("Y"))

    assert tuple(proposer.propose(rule, inputs, (0, 2))) == expected
    assert tuple(proposer.propose(other, inputs, (0, 1))) == expected
    assert rule.admission_calls == other.admission_calls == 0


def test_checked_proposer_tracks_the_active_widening_frontier() -> None:
    (combine,) = _compiled({("A", "B"): ("X",)}, "combine")
    rule = CountedRule(combine, checked=True)
    rules = RuleSet("checked-search", (rule,), EXACT, NoNormalForm[Term]())
    context = AStarContext(
        rules,
        ZeroScorer(),
        RootGoal(TargetSet((_category("X"),), EXACT)),
        0,
        1,
        EXHAUSTIVE,
    )
    source = _beam(
        (("A", -0.1), ("X", -0.3)),
        (("B", -0.2),),
    )
    options = _options()._replace(proposer=SUPERTAG_BEAM_PROPOSER)

    result = astar_parse(("left", "right"), source, context, options)

    assert result.parsed
    assert result.attempts == 2
    assert result.source.frontier == (2, 1)
    assert result.widenings == 1
    assert rule.admission_calls <= 9


@pytest.mark.parametrize("strategy", ["linear", "binary"])
def test_search_reaches_a_third_best_type_in_two_widenings(strategy: str) -> None:
    source = _beam(
        (("X", -0.1), ("Y", -0.2), ("NP", -0.3)),
        (("S\\NP", -0.05),),
    )
    before = source.beams
    result = _parse(source, strategy)

    assert result.parsed
    assert tuple(_label(root.reading) for root in result.roots) == ("S",)
    assert result.source.frontier == (3, 1)
    assert result.widenings == 2
    assert result.outside == supertag_outside(source)
    assert source.beams is before


def test_linear_search_maximizes_candidate_inside_plus_outside() -> None:
    source = _beam(
        (("A", -0.3), ("NP", -0.5)),
        (("B", -0.01), ("C", -0.4)),
    )
    result = astar_parse(
        ("left", "right"),
        source,
        _context({("NP", "B"): ("S",)}),
        _options(),
    )

    assert result.parsed
    assert result.source.frontier == (2, 1)
    assert result.widenings == 1


@given(
    left_weights=st.lists(
        st.integers(min_value=1, max_value=6), min_size=3, max_size=3
    ),
    right_weights=st.lists(
        st.integers(min_value=1, max_value=6), min_size=3, max_size=3
    ),
    successful_pairs=st.sets(
        st.tuples(st.integers(min_value=0, max_value=2), st.integers(0, 2)),
        min_size=1,
    ),
)
@settings(max_examples=40, deadline=None)
def test_linear_and_binary_search_agree_on_random_multi_root_beams(
    left_weights: list[int],
    right_weights: list[int],
    successful_pairs: set[tuple[int, int]],
) -> None:
    left = tuple(
        (f"L{position}", -weight / 10) for position, weight in enumerate(left_weights)
    )
    right = tuple(
        (f"R{position}", -weight / 10) for position, weight in enumerate(right_weights)
    )
    source = _beam(left, right)
    table = {
        (f"L{left_position}", f"R{right_position}"): (
            f"S{left_position}{right_position}",
        )
        for left_position, right_position in successful_pairs
    }
    targets = tuple(output for outputs in table.values() for output in outputs)
    context = _context(table, targets)
    before = source.beams
    linear = astar_parse(("left", "right"), source, context, _options())
    binary = astar_parse(("left", "right"), source, context, _options("binary"))

    assert linear.parsed is binary.parsed is True
    assert linear.source.frontier == binary.source.frontier
    assert tuple(root.reading for root in linear.roots) == tuple(
        root.reading for root in binary.roots
    )
    assert tuple(edge.output for edge in linear.root_edges) == tuple(
        edge.output for edge in binary.root_edges
    )
    assert source.beams is before


def test_strategy_thresholds_use_global_scores_for_equal_candidate_weights() -> None:
    source = _beam(
        (("A", -0.1), ("X", -0.3)),
        (("B", -0.2), ("Y", -0.3)),
    )
    context = _context({("A", "Y"): ("R1",), ("X", "B"): ("R2",)}, ("R1", "R2"))
    linear = astar_parse(("left", "right"), source, context, _options())
    binary = astar_parse(("left", "right"), source, context, _options("binary"))

    assert linear.source.frontier == binary.source.frontier == (1, 2)
    assert tuple(_label(root.reading) for root in linear.roots) == ("R1",)
    assert linear.roots == binary.roots


def test_strategies_admit_an_entire_tied_global_score_level() -> None:
    source = _beam(
        (("A", -0.1), ("X", -0.3)),
        (("B", -0.1), ("Y", -0.3)),
    )
    context = _context({("A", "Y"): ("R1",), ("X", "B"): ("R2",)}, ("R1", "R2"))
    linear = astar_parse(("left", "right"), source, context, _options())
    binary = astar_parse(("left", "right"), source, context, _options("binary"))

    assert linear.source.frontier == binary.source.frontier == (2, 2)
    assert {_label(root.reading) for root in linear.roots} == {"R1", "R2"}
    assert linear.roots == binary.roots


@pytest.mark.parametrize("strategy", ["linear", "binary"])
def test_free_root_returns_every_spanning_edge_without_a_target_match(
    strategy: str,
) -> None:
    source = _beam((("A", -0.1),), (("B", -0.2),))
    context = _context({("A", "B"): ("X", "Y")})
    targeted = astar_parse(
        ("left", "right"),
        source,
        context,
        _options(),
    )
    free = astar_parse(
        ("left", "right"),
        source,
        context,
        _options(strategy, free=True),
    )

    assert not targeted.parsed
    assert {_label(root.reading) for root in free.roots} == {"X", "Y"}
    assert {_label(edge.output) for edge in free.root_edges} == {"X", "Y"}
    assert all(root.span == (0, 2) for root in free.roots)


def test_binary_threshold_admits_tied_candidates_together() -> None:
    source = _beam(
        (("X", -0.1), ("NP", -0.3)),
        (("Y", -0.2), ("S\\NP", -0.3)),
    )
    result = _parse(source, "binary")

    assert result.parsed
    assert result.source.frontier == (2, 2)
    assert result.widenings == 2


def test_an_exhausted_search_returns_its_full_chart() -> None:
    source = _beam((("A", -0.1), ("B", -0.2)), (("C", -0.3),))
    result = _parse(source, "linear")

    assert not result.parsed
    assert result.source.frontier == (2, 1)
    assert result.widenings == 1
    assert result.attempts == 2


def test_search_result_is_frozen() -> None:
    result = _parse(_beam((("NP", -0.1),), (("S\\NP", -0.2),)), "linear")

    with pytest.raises(AttributeError):
        result.__setattr__("roots", ())


def test_search_rejects_sentence_beam_and_budget_mismatches() -> None:
    source = _beam((("NP", -0.1),), (("S\\NP", -0.2),))
    context = _context({("NP", "S\\NP"): ("S",)})

    with pytest.raises(ValueError, match="sentence has 1 tokens"):
        astar_parse(("left",), source, context, _options())

    with pytest.raises(ValueError, match="unary_depth"):
        astar_parse(
            ("left", "right"),
            source,
            context._replace(unary_depth=-1),
            _options(),
        )


def test_zero_and_supertag_outside_heuristics_find_the_same_parse() -> None:
    source = _beam(
        (("A", -0.1), ("NP", -0.4)),
        (("B", -0.2), ("S\\NP", -0.3)),
    )
    context = _context({("NP", "S\\NP"): ("S",)})
    informed = astar_parse(("left", "right"), source, context, _options())
    zero = astar_parse(("left", "right"), source, context, _options(zero=True))

    assert informed.parsed is zero.parsed is True
    assert tuple(root.reading for root in informed.roots) == tuple(
        root.reading for root in zero.roots
    )
    assert zero.outside == (0.0, 0.0)


def test_zero_heuristic_keeps_one_candidate_at_every_position() -> None:
    source = _beam(
        (("A", -0.1), ("NP", -0.2)),
        (("S\\NP", -1.0), ("B", -2.0)),
    )
    result = astar_parse(
        ("left", "right"),
        source,
        _context({("NP", "S\\NP"): ("S",)}),
        _options(zero=True),
    )

    assert result.parsed
    assert result.source.frontier == (2, 1)


def test_search_rejects_empty_beams_and_positive_log_weights() -> None:
    context = _context({("NP", "S\\NP"): ("S",)})

    with pytest.raises(ValueError, match="position 0 is empty"):
        astar_parse(
            ("left",),
            BeamSource(((),)),
            context,
            _options(),
        )

    with pytest.raises(ValueError, match="positive log weight"):
        astar_parse(
            ("left",),
            BeamSource(((Entry("left", _category("NP"), weight=0.1),),)),
            context,
            _options(),
        )


def test_every_probed_frontier_is_parsed_with_the_declared_chart_settings() -> None:
    source = _beam((("NP", -0.1),), (("S\\NP", -0.2),))
    context = _context({("NP", "S\\NP"): ("S",)})
    exhaustive = astar_parse(("left", "right"), source, context, _options())
    beamed = astar_parse(
        ("left", "right"),
        source,
        context._replace(settings=VITERBI_BEAM),
        _options(),
    )

    assert isinstance(exhaustive.chart.semiring, LogInside)
    assert isinstance(beamed.chart.semiring, Viterbi)
    assert beamed.parsed
    assert beamed.roots == exhaustive.roots


def test_a_measured_beam_reports_its_pruned_mass_on_every_probed_chart() -> None:
    source = _beam((("NP", -0.1),), (("S\\NP", -0.2),))
    context = _context({("NP", "S\\NP"): ("S",)})
    measured = ParseSettings(
        ParsePolicy(Retain.FOREST, Reachability.ALL),
        CKY,
        CONTIGUOUS_SPANS,
        NATIVE_CHARTS,
        TopKPruner[object](1),
        LogInside(),
        prune_lexical=True,
        measure_pruned_mass=True,
    )
    result = astar_parse(
        ("left", "right"), source, context._replace(settings=measured), _options()
    )
    unmeasured = astar_parse(("left", "right"), source, context, _options())
    mass = result.chart.pruned_mass

    assert result.parsed
    assert mass is not None
    assert mass.lost == 0.0
    assert unmeasured.chart.pruned_mass is None


def test_every_probed_frontier_is_parsed_with_the_context_s_threads() -> None:
    source = _beam(
        (("A", -0.1), ("NP", -0.2)),
        (("S\\NP", -1.0), ("B", -2.0)),
    )
    context = _context({("NP", "S\\NP"): ("S",)})._replace(threads=3)

    with patch("lofi.deduce._astar.parse", wraps=parse) as parser:
        result = astar_parse(("left", "right"), source, context, _options())

    assert result.parsed
    assert parser.call_args_list
    assert {call.kwargs["threads"] for call in parser.call_args_list} == {3}
