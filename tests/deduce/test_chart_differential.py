"""Differential checks against explicit derivation-tree enumeration."""

from math import isclose
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from tests.support.chart_reference import reference_chart
from tests.support.grammar_strategies import (
    CHART_CASES,
    ChartCase,
    category,
    compiled,
    exact_pattern,
)

from lofi.deduce import (
    Boolean,
    Counting,
    Expectation,
    KBest,
    LogInside,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    UnitEdges,
    Viterbi,
    parse,
)
from lofi.judge import NoNormalForm, RuleSet
from lofi.lexicon import Entry
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    PatternRuleSpec,
    SubstitutionSpec,
)
from lofi.spec.grammar import FeatureErasureMatchingSpec
from lofi.terms import Identity, TargetSet, Term, TermRuleKernel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.deduce import Cell, EdgeScorer, Semiring
    from lofi.judge import NormalFormDecisionTable, Rule
    from lofi.lexicon import Clamp


class UnitSource[W]:
    """Serve generated lexical rows at one semiring's multiplicative unit."""

    __slots__ = ("_one", "_rows")

    def __init__(self, rows: tuple[tuple[int, Term], ...], one: W) -> None:
        self._rows = rows
        self._one = one

    @property
    def is_re_callable(self) -> bool:
        """Return whether the rows can be read repeatedly."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        """Return no lexical clamps."""
        return ()

    def axioms(self, tokens: Sequence[str], /) -> Iterable[tuple[int, Entry[Term], W]]:
        """Return every generated lexical row."""
        del tokens

        return tuple(
            (position, Entry(f"w{position}", reading), self._one)
            for position, reading in self._rows
        )


class UnitScorer[W]:
    """Assign a semiring's multiplicative unit to every local factor."""

    __slots__ = ("_one",)
    lowering = UnitEdges()

    def __init__(self, one: W) -> None:
        self._one = one

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> W:
        """Return the unit lexical factor."""
        del position, entry

        return self._one

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> W:
        """Return the unit rule factor."""
        del rule, inputs, output, span, split

        return self._one


class _CountingNormalForm:
    """Count result-tag decisions while preserving unrestricted behavior."""

    __slots__ = ("_inner", "calls")

    def __init__(self) -> None:
        self._inner = NoNormalForm[Term]()
        self.calls = 0

    @property
    def name(self) -> str:
        """Return the wrapped normal form's name."""
        return self._inner.name

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the wrapped normal form's alphabet."""
        return self._inner.alphabet

    def lexical_tag(self) -> str:
        """Return the wrapped lexical tag."""
        return self._inner.lexical_tag()

    def decisions(self, rules: Sequence[Rule[Term]], /) -> NormalFormDecisionTable:
        """Compile the wrapped child-tag decision table."""
        return self._inner.decisions(rules)

    def tag(self, rule: Rule[Term], children: tuple[str, ...], /) -> str:
        """Count one compiled result-tag decision."""
        self.calls += 1
        return self._inner.tag(rule, children)

    def admits(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Delegate exact-premise admission to the wrapped normal form."""
        return self._inner.admits(rule, children, inputs, output)


class _NativeUnitScorer[W]:
    """Declare unit rule edges while rejecting Python edge callbacks."""

    __slots__ = ("_one",)
    lowering = UnitEdges()

    def __init__(self, one: W) -> None:
        self._one = one

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> W:
        """Return the declared unit for a lexical axiom."""
        del position, entry

        return self._one

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> W:
        """Reject a Python callback for a natively declared unit edge."""
        del rule, inputs, output, span, split
        message = "the parser requested a declared unit rule edge"
        raise AssertionError(message)


def _weights_equal(left: object, right: object) -> bool:
    """Compare exact carriers and tolerate floating accumulation roundoff."""
    if isinstance(left, float) and isinstance(right, float):
        return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)

    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _weights_equal(x, y) for x, y in zip(left, right, strict=True)
        )

    return left == right


def _assert_rows_equal[W](
    expected: tuple[tuple[Term, W], ...],
    observed: tuple[tuple[Term, W], ...],
) -> None:
    """Assert equality of one span's exact-category rows."""
    expected_by_category = dict(expected)
    observed_by_category = dict(observed)

    assert observed_by_category.keys() == expected_by_category.keys()
    assert all(
        _weights_equal(observed_by_category[term], weight)
        for term, weight in expected_by_category.items()
    )


def _assert_case[W](
    case: ChartCase,
    semiring: Semiring[W],
    *,
    threads: tuple[int, ...] = (1, 2, 8),
) -> None:
    """Compare both engine representations with the tree reference."""
    source = UnitSource(case.rows, semiring.one)
    scorer: EdgeScorer[Term, W] = UnitScorer(semiring.one)
    goal = RootGoal(TargetSet((category("a"),), case.rules.matching))
    expected = reference_chart(
        case.sentence,
        rules=case.rules,
        source=source,
        semiring=semiring,
        scorer=scorer,
        unary_depth=case.unary_depth,
    )
    forest_baseline: object | None = None
    inside_baseline: object | None = None

    for thread_count in threads:
        forest = parse(
            case.sentence,
            rules=case.rules,
            source=source,
            semiring=semiring,
            scorer=scorer,
            goal=goal,
            unary_depth=case.unary_depth,
            threads=thread_count,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )
        inside = parse(
            case.sentence,
            rules=case.rules,
            source=source,
            semiring=semiring,
            scorer=scorer,
            goal=goal,
            unary_depth=case.unary_depth,
            threads=thread_count,
            policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
        )

        for span in expected.spans():
            rows = expected.categories_at(span)
            _assert_rows_equal(rows, forest.categories_at(span))
            _assert_rows_equal(rows, inside.categories_at(span))

        forest_snapshot = tuple(
            (
                span,
                tuple(
                    (key, forest.inside(key), forest.edges(key))
                    for key in forest.items_at(span)
                ),
            )
            for span in expected.spans()
        )
        inside_snapshot = tuple(
            (span, inside.tagged_at(span)) for span in expected.spans()
        )

        if forest_baseline is None:
            forest_baseline = forest_snapshot
            inside_baseline = inside_snapshot
        else:
            assert forest_snapshot == forest_baseline
            assert inside_snapshot == inside_baseline


@given(CHART_CASES)
def test_every_exact_chart_value_equals_explicit_derivation_tree_evaluation(
    case: ChartCase,
) -> None:
    """Compare every span and exact category under all required semirings."""
    _assert_case(case, Counting())
    _assert_case(case, Viterbi())
    _assert_case(case, LogInside())
    _assert_case(case, Boolean())
    _assert_case(case, KBest(3))
    _assert_case(case, Expectation(2))


def _ground_rule(name: str, premises: tuple[Term, ...], output: Term) -> Rule[Term]:
    """Compile one exact ground rule under identity matching."""
    (rule,) = compiled(
        (
            PatternRuleSpec(
                name=name,
                premises=tuple(exact_pattern(premise) for premise in premises),
                conclusion=exact_pattern(output),
                head_policy="pass-through" if len(premises) == 1 else "non-absorbing",
            ),
        ),
        name=name,
    ).rule_set.rules

    return rule


def _callback_elimination_cases() -> tuple[
    tuple[Rule[Term], tuple[tuple[int, Term], ...], Term], ...
]:
    """Build witnesses for every shipped native rule-kernel family."""
    argument = category("a")
    result = category("b")
    carried = category("c")
    principal = category("/", result, argument)
    container = category("/", argument, carried)
    composed = category("/", result, carried)
    lifted_principal = category("/", principal, carried)
    application, composition, substitution = compiled(
        (
            ApplicationSpec(view="/"),
            CompositionSpec(view="/", order=1, direction="unrestricted"),
            SubstitutionSpec(view="/", order=1),
        ),
        name="native-families",
    ).rule_set.rules

    return (
        (application, ((0, principal), (1, argument)), result),
        (composition, ((0, principal), (1, container)), composed),
        (substitution, ((0, lifted_principal), (1, container)), composed),
        (
            _ground_rule("coercion", (argument,), result),
            ((0, argument),),
            result,
        ),
    )


@pytest.mark.parametrize(
    ("rule", "rows", "expected"),
    _callback_elimination_cases(),
    ids=("application", "composition", "substitution", "pattern"),
)
def test_shipped_rules_resolve_without_python_application_callbacks(
    rule: Rule[Term],
    rows: tuple[tuple[int, Term], ...],
    expected: Term,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep shipped rule matching and construction inside the native kernel."""
    matching = rule.matching
    rules = RuleSet(
        "native-rule-kernel",
        (rule,),
        matching,
        NoNormalForm[Term](),
    )

    def forbid_python_apply(*values: object) -> None:
        del values
        message = "the parser called a Python rule application"
        raise AssertionError(message)

    monkeypatch.setattr(type(rule), "apply", forbid_python_apply)
    monkeypatch.setattr(TermRuleKernel, "apply", forbid_python_apply)
    chart = parse(
        tuple(f"w{position}" for position in range(len(rows))),
        rules=rules,
        source=UnitSource(rows, 1),
        semiring=Counting(),
        scorer=_NativeUnitScorer(1),
        goal=RootGoal(TargetSet((expected,), matching)),
        unary_depth=1,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )

    assert dict(chart.categories_at((0, len(rows))))[expected] == 1


def test_result_tags_are_compiled_once_outside_the_chart_inner_loop() -> None:
    """Resolve repeated premise tuples through one native normal-form table."""
    atom = category("a")
    combine = _ground_rule("combine", (atom, atom), atom)
    normal_form = _CountingNormalForm()
    rules = RuleSet(
        "compiled-normal-form",
        (combine,),
        Identity[Term](),
        normal_form,
    )
    chart = parse(
        ("w0", "w1", "w2", "w3"),
        rules=rules,
        source=UnitSource(tuple((position, atom) for position in range(4)), 1),
        semiring=Counting(),
        scorer=UnitScorer(1),
        goal=RootGoal(TargetSet((atom,), Identity[Term]())),
        unary_depth=0,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )

    assert dict(chart.categories_at((0, 4)))[atom] == 5
    assert normal_form.calls == 1


def test_one_rule_program_uses_declared_native_representatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not invoke Python representative callbacks from the chart loop."""
    argument = category("a")
    first = category("b")
    second = category("c")
    adopted = compiled(
        tuple(
            PatternRuleSpec(
                name=name,
                premises=(exact_pattern(argument), exact_pattern(argument)),
                conclusion=exact_pattern(output),
                head_policy="non-absorbing",
            )
            for name, output in (("first", first), ("second", second))
        ),
        name="shared-native-matching",
    )
    rules = adopted.rule_set
    goal = RootGoal(TargetSet((first,), rules.matching))
    calls: list[Term] = []
    key = rules.matching.key

    def counted_key(self: object, term: Term, /) -> Term:
        del self
        calls.append(term)
        return key(term)

    monkeypatch.setattr(type(rules.matching), "key", counted_key)

    chart = parse(
        ("w0", "w1"),
        rules=rules,
        source=UnitSource(((0, argument), (1, argument)), 1),
        semiring=Counting(),
        scorer=UnitScorer(1),
        goal=goal,
        unary_depth=0,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )

    assert dict(chart.categories_at((0, 2)))[first] == 1
    assert calls == []


def test_the_reference_and_engine_sum_derivations_from_every_split() -> None:
    """Require both binary splits of a three-token span."""
    atom = category("a")
    combine = _ground_rule("combine", (atom, atom), atom)
    case = ChartCase(
        ("w0", "w1", "w2"),
        ((0, atom), (1, atom), (2, atom)),
        RuleSet(
            "every-split",
            (combine,),
            Identity[Term](),
            NoNormalForm[Term](),
        ),
        0,
    )

    _assert_case(case, Counting())
    source = UnitSource(case.rows, 1)
    reference = reference_chart(
        case.sentence,
        rules=case.rules,
        source=source,
        semiring=Counting(),
        scorer=UnitScorer(1),
        unary_depth=0,
    )

    assert reference.inside((0, 3), atom) == 2


def test_the_reference_and_engine_take_each_requested_unary_round() -> None:
    """Require a conclusion that appears only after the second unary round."""
    first = category("a")
    second = category("b")
    third = category("c")
    case = ChartCase(
        ("w0",),
        ((0, first),),
        RuleSet(
            "unary-rounds",
            (
                _ground_rule("first", (first,), second),
                _ground_rule("second", (second,), third),
            ),
            Identity[Term](),
            NoNormalForm[Term](),
        ),
        2,
    )

    _assert_case(case, Counting())
    source = UnitSource(case.rows, 1)
    reference = reference_chart(
        case.sentence,
        rules=case.rules,
        source=source,
        semiring=Counting(),
        scorer=UnitScorer(1),
        unary_depth=2,
    )

    assert reference.inside((0, 1), third) == 1


def test_feature_erased_matching_never_erases_the_exact_memo_key() -> None:
    """Require repeated erased premises to keep their feature-bearing outputs."""
    (application,) = compiled(
        (ApplicationSpec(view="\\"),),
        matching=FeatureErasureMatchingSpec(),
        name="exact-memo",
    ).rule_set.rules
    erasure = application.matching
    terminal = category("t")
    first = category("a[f1]")
    second = category("a[f2]")
    case = ChartCase(
        ("w0", "w1", "w2", "w3"),
        (
            (0, terminal),
            (1, category("\\", first, terminal)),
            (2, terminal),
            (3, category("\\", second, terminal)),
        ),
        RuleSet(
            "exact-memo",
            (application,),
            erasure,
            NoNormalForm[Term](),
        ),
        0,
    )

    _assert_case(case, Counting())
    source = UnitSource(case.rows, 1)
    reference = reference_chart(
        case.sentence,
        rules=case.rules,
        source=source,
        semiring=Counting(),
        scorer=UnitScorer(1),
        unary_depth=0,
    )

    assert dict(reference.categories_at((0, 2))) == {first: 1}
    assert dict(reference.categories_at((2, 4))) == {second: 1}
