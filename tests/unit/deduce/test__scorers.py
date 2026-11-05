"""Tests for split posteriors and local-attention scoring."""

import copy
import pickle
from math import exp, fsum, log, log1p
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import SpanFree
from lofi.deduce import (
    CoverGoal,
    Edge,
    EdgeScorer,
    ItemKey,
    LocalAttendScorer,
    LogInside,
    MarginalScorer,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    SpanIndex,
    SplitLogWeights,
    SplitProbabilities,
    outside_from_edges,
    parse,
)
from lofi.judge import NormalFormRole
from lofi.lexicon import Entry
from lofi.spec import ConstantPatternSpec, PatternRuleSpec, RuleSetSpec
from lofi.terms import TargetSet, Term

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import Cell, Chart, Goal
    from lofi.judge import (
        Direction,
        Rule,
        RuleProbe,
        RuleRestriction,
        Slot,
    )
    from lofi.lexicon import Clamp
    from lofi.terms import Matching


class Exact:
    """Direct category-string comparison."""

    @property
    def name(self) -> str:
        """Return the relation name."""
        return "exact"

    @property
    def has_key(self) -> bool:
        """Return whether every class has a representative."""
        return True

    def key(self, term: str, /) -> str:
        """Return the category itself."""
        return term

    def equivalent(self, left: str, right: str, /) -> bool:
        """Return whether two categories are equal."""
        return left == right


class RejectingGoal:
    """Spanning root goal that rejects one rule."""

    def __init__(self, rejected: str) -> None:
        self._base = RootGoal(TargetSet(("P", "Q"), EXACT))
        self._rejected = rejected

    @property
    def settings(self) -> Mapping[str, JsonValue]:
        """Return the base goal's stable settings."""
        return self._base.settings

    def accepts[W](self, chart: Chart[str, W], key: ItemKey[str], /) -> bool:
        """Return whether the spanning goal accepts an item."""
        return self._base.accepts(chart, key)

    def roots[W](self, chart: Chart[str, W], /) -> Iterator[ItemKey[str]]:
        """Enumerate accepted spanning items."""
        return self._base.roots(chart)

    def root_groups[W](
        self,
        chart: Chart[str, W],
        /,
    ) -> tuple[tuple[ItemKey[str], ...], ...]:
        """Return one group of spanning alternatives."""
        return self._base.root_groups(chart)

    def root_weight(self, root: ItemKey[str], /) -> float:
        """Return the spanning root weight."""
        return self._base.root_weight(root)

    def combination_weight(self, group_count: int, /) -> float:
        """Return the spanning combination weight."""
        return self._base.combination_weight(group_count)

    def admits_edge[W](
        self,
        chart: Chart[str, W],
        parent: ItemKey[str],
        edge: Edge[str, W],
        /,
    ) -> bool:
        """Return whether an edge avoids the rejected rule."""
        del chart, parent

        return edge.rule_name != self._rejected


class BaseScorer:
    """Fixed local weights with call recording."""

    lowering = SpanFree()

    def __init__(self, axiom: float = -0.5, edge: float = -0.25) -> None:
        self._axiom = axiom
        self._edge = edge
        self.edge_calls = 0

    def axiom_weight(self, position: int, entry: Entry[str], /) -> float:
        """Return the fixed lexical weight."""
        del position, entry

        return self._axiom

    def edge_weight(
        self,
        rule: Rule[str],
        inputs: tuple[str, ...],
        output: str,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the fixed rule weight."""
        del rule, inputs, output, span, split
        self.edge_calls += 1

        return self._edge


class ConcatRule:
    """Concatenation of two adjacent item categories."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        return "concat"

    @property
    def arity(self) -> int:
        """Return the binary arity."""
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the supported direction."""
        return frozenset({"forward"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return an empty set of residual slots."""
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        """Return whether forward enumeration is complete."""
        return True

    @property
    def matching(self) -> Matching[str]:
        """Return the rule's matching relation."""
        return EXACT

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return no rule-instance restriction."""
        return None

    @property
    def probe(self) -> RuleProbe | None:
        """Return no conservative head probe."""
        return None

    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        """Yield the concatenated category."""
        if len(inputs) == 2:
            yield "".join(inputs)

    def admits(self, inputs: tuple[str, ...], output: str, /) -> bool:
        """Return whether the output concatenates both inputs."""
        return len(inputs) == 2 and output == "".join(inputs)

    def residuate(self, output: str, known: tuple[Slot[str], ...], /) -> Iterator[str]:
        """Yield no residual candidates."""
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        return {"rule": self.name, "arity": self.arity}


class Source:
    """One unit-weight category per token."""

    @property
    def is_re_callable(self) -> bool:
        """Return whether the source can be called repeatedly."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        """Return no lexical clamps."""
        return ()

    def axioms(
        self,
        tokens: Sequence[str],
        /,
    ) -> Iterable[tuple[int, Entry[str], float]]:
        """Return one unit-weight entry per token."""
        return tuple(
            (position, Entry(token, token), 0.0)
            for position, token in enumerate(tokens)
        )


class TermSource:
    """Yield the declared category corresponding to each token."""

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    def axioms(
        self,
        tokens: Sequence[str],
        /,
    ) -> Iterable[tuple[int, Entry[Term], float]]:
        return tuple(
            (position, Entry(token, _term(token)), 0.0)
            for position, token in enumerate(tokens)
        )


class TermBaseScorer:
    """Assign zero local weight to the finite term relation."""

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


EXACT: Final[Matching[str]] = Exact()

_CONCATENATIONS: Final = (
    ("a", "b", "ab"),
    ("b", "c", "bc"),
    ("ab", "c", "abc"),
    ("a", "bc", "abc"),
)
_CONCAT: Final = declared_rules(
    ccg_signature("a", "b", "c", "ab", "bc", "abc", name="local-attention-categories"),
    RuleSetSpec(
        name="concat",
        rules=tuple(
            PatternRuleSpec(
                name=f"concat-{left}-{right}",
                premises=(
                    ConstantPatternSpec(name=left),
                    ConstantPatternSpec(name=right),
                ),
                conclusion=ConstantPatternSpec(name=output),
                head_policy="non-absorbing",
            )
            for left, right, output in _CONCATENATIONS
        ),
    ),
)
_TERM_STORE: Final = _CONCAT.context.store
_TERM_MATCHING: Final[Matching[Term]] = _CONCAT.rule_set.matching


def _term(name: str) -> Term:
    """Return one declared local-attention test category."""
    return _CONCAT.context.syntax.parse(_TERM_STORE, name)


A = ItemKey((0, 1), "A", "lex")
B = ItemKey((1, 2), "B", "lex")
C = ItemKey((2, 3), "C", "lex")
AB = ItemKey((0, 2), "AB", "binary")
BC = ItemKey((1, 3), "BC", "binary")
WHOLE = ItemKey((0, 3), "W", "axiom")
P = ItemKey((0, 3), "P", "root")
Q = ItemKey((0, 3), "Q", "root")


def posterior_forest() -> PackedForest[str, float]:
    """Build a forest with two parents and several binary alternatives."""
    chart = PackedForest[str, float](SpanIndex(3), LogInside())
    chart.add(A, Edge("a", (), "A", log(0.8)))
    chart.add(B, Edge("b", (), "B", log(0.7)))
    chart.add(C, Edge("c", (), "C", log(0.6)))
    chart.add(AB, Edge("ab", (A, B), "AB", log(0.5), split=1))
    chart.add(BC, Edge("bc", (B, C), "BC", log(0.4), split=2))
    chart.add(WHOLE, Edge("whole", (), "W", log(0.2)))
    chart.add(P, Edge("p-left-a", (A, BC), "P", log(0.3), split=1))
    chart.add(P, Edge("p-left-b", (A, BC), "P", log(0.1), split=1))
    chart.add(P, Edge("p-right", (AB, C), "P", log(0.5), split=2))
    chart.add(P, Edge("p-unary", (WHOLE,), "P", log(0.4)))
    chart.add(Q, Edge("q-left", (A, BC), "Q", log(0.2), split=1))
    chart.add(Q, Edge("q-right", (AB, C), "Q", log(0.6), split=2))

    return chart


def test_split_values_copy_validate_and_reconstruct_rows() -> None:
    raw = {(0, 3): {1: 0.25, 2: 0.75}}
    probabilities = SplitProbabilities(raw)
    log_weights = SplitLogWeights({(0, 3): {1: log(0.25), 2: log(0.75)}})
    raw[(0, 3)][1] = 1.0

    assert probabilities.probability((0, 3), 1) == 0.25
    assert probabilities.spans == ((0, 3),)
    assert log_weights.weight((0, 3), 2) == pytest.approx(log(0.75))
    assert repr(probabilities) == "SplitProbabilities(rows=1)"
    assert repr(log_weights) == "SplitLogWeights(rows=1)"

    for value in (probabilities, log_weights):
        assert copy.copy(value) == value
        assert copy.deepcopy(value) == value
        assert pickle.loads(pickle.dumps(value)) == value

        with pytest.raises(TypeError):
            cast("dict[int, float]", value.at((0, 3)))[1] = 0.5

        with pytest.raises(AttributeError, match="immutable"):
            value.__delattr__("_rows")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({(0, 1): {0: 1.0}}, "not binary"),
        ({(0, 3): {0: 1.0}}, "not internal"),
        ({(0, 3): {}}, "empty"),
        ({(0, 3): {1: 0.4, 2: 0.4}}, "not 1.0"),
        ({(0, 3): {1: -0.1, 2: 1.1}}, "outside"),
    ],
)
def test_split_probabilities_refuse_invalid_rows(
    rows: Mapping[Cell, Mapping[int, float]],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SplitProbabilities(rows)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({(0, 3): {1: log(0.4), 2: log(0.4)}}, "not 1.0"),
        ({(0, 3): {1: float("nan"), 2: 0.0}}, "invalid"),
        ({(0, 3): {1: float("inf"), 2: -float("inf")}}, "invalid"),
        ({(0, 3): {1: 1_000.0, 2: -float("inf")}}, "invalid"),
        ({(0, 3): {1: -float("inf"), 2: -float("inf")}}, "not 1.0"),
    ],
)
def test_split_log_weights_refuse_invalid_rows(
    rows: Mapping[Cell, Mapping[int, float]],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SplitLogWeights(rows)


def test_a_split_row_is_normalized_to_its_declared_tolerance() -> None:
    near = {(0, 3): {1: 0.5, 2: 0.5 + 1e-6}}
    log_near = {(0, 3): {1: log(0.5), 2: log(0.5 + 1e-6)}}

    with pytest.raises(ValueError, match=r"not 1\.0"):
        SplitProbabilities(near)

    with pytest.raises(ValueError, match=r"not 1\.0"):
        SplitLogWeights(log_near)

    probabilities = SplitProbabilities(near, tolerance=1e-5)
    weights = SplitLogWeights(log_near, tolerance=1e-5)

    assert probabilities.tolerance == 1e-5
    assert weights.tolerance == 1e-5
    assert SplitProbabilities({(0, 2): {1: 1.0}}).tolerance == 1e-9
    assert pickle.loads(pickle.dumps(probabilities)) == probabilities
    assert copy.deepcopy(weights) == weights
    assert LocalAttendScorer(BaseScorer(), weights).split_weights.tolerance == 1e-5


@pytest.mark.parametrize("tolerance", [-1e-9, float("nan"), float("inf")])
def test_a_split_tolerance_is_a_finite_nonnegative_number(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        SplitProbabilities({(0, 2): {1: 1.0}}, tolerance=tolerance)


def test_marginal_scorer_aggregates_every_parent_and_binary_edge() -> None:
    chart = posterior_forest()
    outside = outside_from_edges(chart, RootGoal(TargetSet(("P", "Q"), EXACT)))
    before = pickle.dumps(chart)
    scorer = MarginalScorer(chart, outside)
    row = scorer.probabilities.at((0, 3))
    oracle: dict[int, list[float]] = {1: [], 2: []}

    for parent in (P, Q):
        for edge, marginal in outside.rule_marginals(parent):
            if edge.arity == 2:
                oracle[cast("int", edge.split)].append(marginal)

    masses = {split: fsum(values) for split, values in oracle.items()}
    total = fsum(masses.values())

    assert row == pytest.approx({split: mass / total for split, mass in masses.items()})
    assert fsum(row.values()) == pytest.approx(1.0, abs=1e-9)
    assert pickle.dumps(chart) == before


def test_marginal_scorer_uses_only_goal_admitted_binary_edges() -> None:
    chart = posterior_forest()
    goal: Goal[str] = RejectingGoal("p-right")
    outside = outside_from_edges(chart, goal)
    row = MarginalScorer(chart, outside).probabilities.at((0, 3))
    admitted = tuple(
        (edge, marginal)
        for parent in (P, Q)
        for edge, marginal in outside.rule_marginals(parent)
        if edge.arity == 2
    )
    expected_right = fsum(
        marginal for edge, marginal in admitted if edge.split == 2
    ) / fsum(marginal for _, marginal in admitted)

    assert row[2] == pytest.approx(expected_right)
    assert all(edge.rule_name != "p-right" for edge, _ in admitted)


def test_marginal_scorer_handles_grouped_cover_roots() -> None:
    chart = posterior_forest()
    tail = ItemKey((3, 4), "D", "lex")
    covered = PackedForest[str, float](SpanIndex(4), LogInside())

    for key in (A, B, C, AB, BC, WHOLE, P, Q):
        for edge in chart.edges(key):
            covered.add(key, edge)

    covered.add(tail, Edge("d", (), "D", log(0.9)))
    outside = outside_from_edges(covered, CoverGoal(2, penalty=0.5))
    row = MarginalScorer(covered, outside).probabilities.at((0, 3))

    assert fsum(row.values()) == pytest.approx(1.0, abs=1e-9)
    assert outside.constituent_marginal(tail) == pytest.approx(1.0)


def test_marginal_scorer_is_frozen() -> None:
    chart = posterior_forest()
    scorer = MarginalScorer(
        chart,
        outside_from_edges(chart, RootGoal(TargetSet(("P", "Q"), EXACT))),
    )

    with pytest.raises(AttributeError, match="immutable"):
        scorer.__setattr__("_probabilities", SplitProbabilities({}))

    with pytest.raises(AttributeError, match="immutable"):
        scorer.__delattr__("_probabilities")


def test_marginal_scorer_refuses_an_outside_result_from_another_chart() -> None:
    first = posterior_forest()
    second = posterior_forest()
    outside = outside_from_edges(first, RootGoal(TargetSet(("P", "Q"), EXACT)))

    with pytest.raises(ValueError, match="different chart"):
        MarginalScorer(second, outside)


def test_zero_partitions_and_nonbinary_cells_produce_no_split_rows() -> None:
    chart = posterior_forest()
    absent = outside_from_edges(chart, RootGoal(TargetSet(("absent",), EXACT)))
    only_axiom = PackedForest[str, float](SpanIndex(3), LogInside())
    only_axiom.add(WHOLE, Edge("whole", (), "W", 0.0))
    nonbinary = outside_from_edges(only_axiom, RootGoal(TargetSet(("W",), EXACT)))

    assert MarginalScorer(chart, absent).probabilities.spans == ()
    assert MarginalScorer(only_axiom, nonbinary).probabilities.spans == ()


def test_local_attention_scores_binary_edges_and_passes_other_calls_through() -> None:
    base = BaseScorer()
    weights = SplitLogWeights({(0, 3): {1: log(0.25), 2: log(0.75)}})
    scorer = LocalAttendScorer(base, weights)
    rule = cast("Rule[str]", ConcatRule())
    entry = Entry("a", "A")

    assert isinstance(scorer, EdgeScorer)
    assert scorer.base is base
    assert scorer.split_weights == weights
    assert scorer.split_weights is not weights
    assert scorer.axiom_weight(0, entry) == -0.5
    assert scorer.edge_weight(rule, ("A",), "A", (0, 1), None) == -0.25
    assert scorer.edge_weight(rule, ("A", "BC"), "P", (0, 3), 1) == (
        pytest.approx(-0.25 + log(0.25))
    )
    assert base.edge_calls == 2


@settings(max_examples=200)
@given(
    left=st.floats(min_value=1e-6, max_value=1.0 - 1e-6),
    base_weight=st.floats(
        min_value=-50.0,
        max_value=50.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_local_attention_adds_exactly_one_normalized_split_weight(
    left: float,
    base_weight: float,
) -> None:
    weights = SplitLogWeights({(0, 3): {1: log(left), 2: log1p(-left)}})
    scorer = LocalAttendScorer(BaseScorer(edge=base_weight), weights)
    rule = cast("Rule[str]", ConcatRule())

    for split in (1, 2):
        scored = scorer.edge_weight(rule, ("A", "B"), "AB", (0, 3), split)

        assert scored - base_weight == pytest.approx(weights.weight((0, 3), split))


@pytest.mark.parametrize(
    ("span", "split"),
    [((0, 3), None), ((0, 3), 0), ((0, 3), 3), ((0, 4), 2)],
)
def test_local_attention_refuses_invalid_or_missing_binary_splits_before_scoring(
    span: Cell,
    split: int | None,
) -> None:
    base = BaseScorer()
    scorer = LocalAttendScorer(
        base,
        SplitLogWeights({(0, 3): {1: log(0.25), 2: log(0.75)}}),
    )

    with pytest.raises(ValueError, match=r"split|weight"):
        scorer.edge_weight(
            cast("Rule[str]", ConcatRule()),
            ("A", "B"),
            "AB",
            span,
            split,
        )

    assert base.edge_calls == 0


def test_local_attention_reconstructs_with_the_shared_base() -> None:
    base = BaseScorer()
    scorer = LocalAttendScorer(
        base,
        SplitLogWeights({(0, 3): {1: log(0.25), 2: log(0.75)}}),
    )

    for rebuilt in (
        copy.copy(scorer),
        copy.deepcopy(scorer),
        pickle.loads(pickle.dumps(scorer)),
    ):
        assert rebuilt.split_weights == scorer.split_weights

    assert copy.copy(scorer).base is base
    assert copy.deepcopy(scorer).base is base


def test_parse_multiplies_supplied_split_weights_into_binary_edges() -> None:
    base = TermBaseScorer()
    scorer = LocalAttendScorer(
        base,
        SplitLogWeights(
            {
                (0, 2): {1: 0.0},
                (1, 3): {2: 0.0},
                (0, 3): {1: log(0.25), 2: log(0.75)},
            }
        ),
    )
    rules = _CONCAT.rule_set
    chart = parse(
        ("a", "b", "c"),
        rules=rules,
        source=TermSource(),
        semiring=LogInside(),
        scorer=scorer,
        goal=RootGoal(TargetSet((_term("abc"),), _TERM_MATCHING)),
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    root = next(key for key in chart.items_at((0, 3)) if key.reading is _term("abc"))

    assert exp(chart.inside(root)) == pytest.approx(1.0)
    assert {edge.split: edge.weight for edge in chart.edges(root)} == pytest.approx(
        {1: log(0.25), 2: log(0.75)}
    )
