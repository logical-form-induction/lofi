from math import exp, fsum, inf, isclose, log
from typing import TYPE_CHECKING, cast, override

import pytest

from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    Counting,
    Edge,
    ItemKey,
    KBest,
    LogInside,
    PackedForest,
    SpanIndex,
    Viterbi,
)
from lofi.interpret._derive import (
    BestOfRanked,
    ConstituentUtility,
    LexicalTable,
    LogProbability,
    StepContext,
    axiom_entries,
    constituent_gain,
    constituents,
    derivation_entropy,
    is_axiom,
    k_best_scores,
    mbr,
    read,
    step_context,
    unfold,
)
from lofi.interpret._errors import CyclicForestError
from lofi.lexicon import Entry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi.deduce import Cell, Chart, Derivation, Ranked, Semiring
    from lofi.judge import Rule


# edge fixture fields: span, rule, child spans, and log probability.
type ForestStep = tuple[Cell, str, tuple[Cell, ...], float]

# three-token ambiguity between left- and right-branching analyses.
THREE: tuple[ForestStep, ...] = (
    ((0, 1), "lex", (), log(0.5)),
    ((1, 2), "lex", (), log(0.4)),
    ((2, 3), "lex", (), log(0.3)),
    ((0, 2), "fa", ((0, 1), (1, 2)), log(0.9)),
    ((1, 3), "ba", ((1, 2), (2, 3)), log(0.7)),
    ((0, 3), "fa", ((0, 1), (1, 3)), log(0.6)),
    ((0, 3), "ba", ((0, 2), (2, 3)), log(0.2)),
)


class _NamedRule:
    """Rule name used by the fold fixture."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


PAIR = cast("Rule[str]", _NamedRule("pair"))
FORWARD = cast("Rule[str]", _NamedRule("fa"))
BACKWARD = cast("Rule[str]", _NamedRule("ba"))


class _Join:
    """String-valued derivation interpretation."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "join"

    @property
    def refines_items(self) -> bool:
        """Return whether each packed item has one string value."""
        return False

    def supports(self, rule: Rule[str], /) -> bool:
        """Return whether the pair rule is defined.

        Parameters
        ----------
        rule : Rule[str]
        """
        return rule.name == "pair"

    def axiom(self, entry: Entry[str], /) -> str:
        """Return the lexical form.

        Parameters
        ----------
        entry : Entry[str]
        """
        return entry.form

    def step(
        self,
        context: StepContext[str],
        children: tuple[str, ...],
        /,
    ) -> str | None:
        """Join the children of a supported rule.

        Parameters
        ----------
        context : StepContext[str]
        children : tuple[str, ...]
        """
        if not self.supports(context.rule):
            return None

        return f"{' '.join(children)}@{context.order}"


def test_an_interpretation_folds_axioms_and_steps() -> None:
    chart = PackedForest[str, float](SpanIndex(2), LogInside())
    chart.add(key((0, 1)), Edge(f"{LEXICAL_TAG_PREFIX}red", (), "N", 0.0))
    chart.add(key((1, 2)), Edge(f"{LEXICAL_TAG_PREFIX}fox", (), "N", 0.0))
    chart.add(
        key((0, 2)),
        Edge("pair", (key((0, 1)), key((1, 2))), "N", 0.7, split=1, order=2),
    )
    derivation = next(chart.derivations(key((0, 2))))
    table = LexicalTable({"pair": PAIR}, ((Entry("red", "N"),), (Entry("fox", "N"),)))

    assert not is_axiom(derivation)
    assert is_axiom(derivation.children[0])
    assert step_context(derivation, table) == StepContext(PAIR, "N", 0.7, 2, ("N", "N"))
    assert read(derivation, _Join(), table) == ("red fox@2",)


class Spy[T, W]:
    """A chart that counts derivation enumerations.

    Parameters
    ----------
    inner : PackedForest[T, W]
        The forest accessed through the spy.
    """

    def __init__(self, inner: PackedForest[T, W]) -> None:
        self._inner = inner
        self.drawn = 0

    @property
    def index(self) -> SpanIndex:
        """Return the index the inner forest is addressed by."""
        return self._inner.index

    @property
    def semiring(self) -> Semiring[W]:
        """Return the inner forest's weight algebra."""
        return self._inner.semiring

    def add(self, item: ItemKey[T], edge: Edge[T, W], /) -> None:
        """Record one step in the inner forest.

        Parameters
        ----------
        item : ItemKey[T]
        edge : Edge[T, W]
        """
        self._inner.add(item, edge)

    def items_at(self, span: Cell, /) -> tuple[ItemKey[T], ...]:
        """Return the items covering one span.

        Parameters
        ----------
        span : Cell
        """
        return self._inner.items_at(span)

    def inside(self, item: ItemKey[T], /) -> W:
        """Return an item's inside weight.

        Parameters
        ----------
        item : ItemKey[T]
        """
        return self._inner.inside(item)

    def edges(self, item: ItemKey[T], /) -> tuple[Edge[T, W], ...]:
        """Return every step that concluded an item.

        Parameters
        ----------
        item : ItemKey[T]
        """
        return self._inner.edges(item)

    def term(self, item: ItemKey[T], /) -> T:
        """Return an item's term.

        Parameters
        ----------
        item : ItemKey[T]
        """
        return self._inner.term(item)

    def derivations(self, item: ItemKey[T], /) -> Iterator[Derivation[T, W]]:
        """Enumerate an item's derivations, counting the trees handed over.

        Parameters
        ----------
        item : ItemKey[T]
        """
        for tree in self._inner.derivations(item):
            self.drawn += 1

            yield tree


def build[W](
    steps: Sequence[ForestStep],
    semiring: Semiring[W],
    lift: Callable[[float], W],
    n: int = 3,
    category: str = "N",
) -> PackedForest[str, W]:
    chart = PackedForest[str, W](SpanIndex(n), semiring)

    for span, rule, children, weight in steps:
        chart.add(
            key(span, category),
            Edge(
                rule,
                tuple(key(child, category) for child in children),
                category,
                lift(weight),
                children[0][1] if len(children) == 2 else None,
            ),
        )

    return chart


def key(span: Cell, category: str = "N") -> ItemKey[str]:
    return ItemKey(span, category)


def log_weights(weight: float) -> float:
    return weight


def ranked_weights(weight: float) -> Ranked:
    return (weight,)


def unit_weights(weight: float) -> int:
    del weight

    return 1


def score[T](derivation: Derivation[T, float]) -> float:
    return derivation.edge.weight + fsum(score(child) for child in derivation.children)


def catalan_steps(n: int) -> tuple[ForestStep, ...]:
    steps: list[ForestStep] = [
        ((at, at + 1), "lex", (), -0.01 * (at + 1)) for at in range(n)
    ]

    for width in range(2, n + 1):
        for first in range(n - width + 1):
            last = first + width

            steps.extend(
                (
                    (first, last),
                    "fa",
                    ((first, split), (split, last)),
                    -0.1 * split - 0.007 * first - 0.003 * last,
                )
                for split in range(first + 1, last)
            )

    return tuple(steps)


def test_the_unordered_unfold_draws_no_more_trees_than_its_bound() -> None:
    spy = Spy(build(THREE, LogInside(), log_weights))
    chart: Chart[str, float] = spy

    taken = list(unfold(chart, key((0, 3)), limit=1))

    assert len(taken) == 1
    assert spy.drawn < 1 + 2


def test_the_unordered_unfold_hands_over_every_tree_within_its_bound() -> None:
    chart = build(THREE, LogInside(), log_weights)

    taken = list(unfold(chart, key((0, 3)), limit=10))
    rules = sorted(tree.edge.rule_name for tree in taken)

    assert rules == ["ba", "fa"]


def test_a_bound_of_zero_reads_nothing_at_all() -> None:
    spy = Spy(build(THREE, LogInside(), log_weights))

    assert list(unfold(spy, key((0, 3)), limit=0)) == []
    assert spy.drawn == 0


def test_a_negative_bound_is_refused_before_anything_is_read() -> None:
    spy = Spy(build(THREE, LogInside(), log_weights))

    with pytest.raises(ValueError, match="limit is negative"):
        unfold(spy, key((0, 3)), limit=-1)

    assert spy.drawn == 0


def test_an_item_the_forest_does_not_hold_has_no_derivation() -> None:
    chart = build(THREE, LogInside(), log_weights)

    assert list(unfold(chart, key((0, 3), "S"), limit=4)) == []
    assert list(unfold(chart, key((0, 3), "S"), limit=4, order=LogProbability())) == []


def test_the_ordered_unfold_hands_the_trees_over_best_first() -> None:
    chart = build(THREE, LogInside(), log_weights)

    ordered = [
        score(tree)
        for tree in unfold(chart, key((0, 3)), limit=9, order=LogProbability())
    ]
    enumerated = sorted(
        (score(tree) for tree in unfold(chart, key((0, 3)), limit=9)), reverse=True
    )

    assert ordered == enumerated
    assert ordered == sorted(ordered, reverse=True)


def test_the_ordered_unfold_draws_nothing_from_the_forest_enumeration() -> None:
    spy = Spy(build(THREE, LogInside(), log_weights))

    taken = list(unfold(spy, key((0, 3)), limit=2, order=LogProbability()))

    assert len(taken) == 2
    assert spy.drawn == 0


def test_the_best_tree_scores_what_the_viterbi_algebra_says_it_does() -> None:
    inside = build(THREE, LogInside(), log_weights)
    viterbi = build(THREE, Viterbi(), log_weights)

    best = next(unfold(inside, key((0, 3)), limit=1, order=LogProbability()))

    assert isclose(score(best), viterbi.inside(key((0, 3))))


def test_the_ordered_unfold_reaches_a_forest_too_large_to_enumerate() -> None:
    steps = catalan_steps(15)
    counting = build(steps, Counting(), unit_weights, n=15)
    inside = build(steps, LogInside(), log_weights, n=15)
    ranked = build(steps, KBest(3), ranked_weights, n=15)

    taken = [
        score(tree)
        for tree in unfold(inside, key((0, 15)), limit=3, order=LogProbability())
    ]

    assert counting.inside(key((0, 15))) > 2_000_000
    assert len(taken) == 3
    assert all(
        isclose(a, b) for a, b in zip(taken, ranked.inside(key((0, 15))), strict=True)
    )


def test_k_best_scores_agree_with_the_scores_of_the_ordered_unfold() -> None:
    chart = build(THREE, LogInside(), log_weights)

    taken = [
        score(tree)
        for tree in unfold(chart, key((0, 3)), limit=2, order=LogProbability())
    ]
    folded = k_best_scores(chart, key((0, 3)), order=LogProbability(), k=2)

    assert len(folded) == 2
    assert all(isclose(a, b) for a, b in zip(taken, folded, strict=True))


def test_k_best_scores_agree_with_a_sweep_run_under_the_k_best_algebra() -> None:
    inside = build(THREE, LogInside(), log_weights)
    ranked = build(THREE, KBest(2), ranked_weights)

    folded = k_best_scores(inside, key((0, 3)), order=LogProbability(), k=2)

    assert all(
        isclose(a, b) for a, b in zip(folded, ranked.inside(key((0, 3))), strict=True)
    )


def test_k_best_scores_keep_no_more_than_the_width_they_were_given() -> None:
    chart = build(catalan_steps(8), LogInside(), log_weights, n=8)

    assert len(k_best_scores(chart, key((0, 8)), order=LogProbability(), k=4)) == 4


def test_k_best_scores_are_empty_where_the_item_has_no_derivation() -> None:
    chart = build(THREE, LogInside(), log_weights)

    assert k_best_scores(chart, key((0, 3), "S"), order=LogProbability(), k=3) == ()


def test_a_zero_width_k_best_is_refused_by_the_algebra() -> None:
    chart = build(THREE, LogInside(), log_weights)

    with pytest.raises(ValueError, match="k-best width"):
        k_best_scores(chart, key((0, 3)), order=LogProbability(), k=0)


def test_the_ranked_ordering_reads_a_k_best_chart_the_log_ordering_reads() -> None:
    inside = build(THREE, LogInside(), log_weights)
    ranked = build(THREE, KBest(2), ranked_weights)

    over_logs = [
        tree.edge.rule_name
        for tree in unfold(inside, key((0, 3)), limit=2, order=LogProbability())
    ]
    over_ranks = [
        tree.edge.rule_name
        for tree in unfold(ranked, key((0, 3)), limit=2, order=BestOfRanked())
    ]

    assert over_logs == over_ranks


def test_the_ranked_ordering_scores_an_empty_list_as_no_derivation() -> None:
    assert BestOfRanked().rank(()) == -inf
    assert BestOfRanked().rank((-1.0, -2.0)) == -1.0


def test_the_log_probability_ordering_ranks_a_weight_as_itself() -> None:
    assert LogProbability().rank(-1.5) == -1.5


def test_derivation_entropy_agrees_with_the_enumerated_distribution() -> None:
    chart = build(THREE, LogInside(), log_weights)

    total = chart.inside(key((0, 3)))
    shares = [exp(score(tree) - total) for tree in unfold(chart, key((0, 3)), limit=9)]
    enumerated = -fsum(share * log(share) for share in shares)

    assert isclose(fsum(shares), 1.0)
    assert isclose(derivation_entropy(chart, key((0, 3))), enumerated)


def test_derivation_entropy_agrees_over_a_deeper_forest() -> None:
    chart = build(catalan_steps(7), LogInside(), log_weights, n=7)

    total = chart.inside(key((0, 7)))
    shares = [
        exp(score(tree) - total) for tree in unfold(chart, key((0, 7)), limit=10_000)
    ]
    enumerated = -fsum(share * log(share) for share in shares)

    assert len(shares) == 132
    assert isclose(fsum(shares), 1.0)
    assert isclose(derivation_entropy(chart, key((0, 7))), enumerated)


def test_derivation_entropy_is_zero_where_the_item_has_no_derivation() -> None:
    chart = build(THREE, LogInside(), log_weights)

    assert derivation_entropy(chart, key((0, 3), "S")) == 0.0


def test_a_normal_form_leaves_an_unambiguous_sentence_at_zero_entropy() -> None:
    # compare the ambiguous forest with its right-branching normalization.
    spurious = build(THREE, LogInside(), log_weights)
    filtered = build(
        tuple(step for step in THREE if step[1] != "ba" or step[0] != (0, 3)),
        LogInside(),
        log_weights,
    )

    assert spurious.term(key((0, 3))) == filtered.term(key((0, 3)))
    assert derivation_entropy(spurious, key((0, 3))) > 0.5
    assert derivation_entropy(filtered, key((0, 3))) == 0.0
    assert len(k_best_scores(filtered, key((0, 3)), order=LogProbability(), k=4)) == 1


def test_constituents_names_every_item_the_tree_concludes() -> None:
    chart = build(THREE, LogInside(), log_weights)

    best = next(unfold(chart, key((0, 3)), limit=1, order=LogProbability()))
    spans = sorted(item.span for item in constituents(best))

    assert spans == [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_constituent_gain_counts_only_the_items_the_marginals_name() -> None:
    chart = build(THREE, LogInside(), log_weights)

    best = next(unfold(chart, key((0, 3)), limit=1, order=LogProbability()))
    marginals: Mapping[ItemKey[str], float] = {key((0, 1)): 0.25, key((1, 3)): 0.75}

    assert isclose(constituent_gain(best, marginals=marginals), 1.0)
    assert constituent_gain(best, marginals={}) == 0.0


def test_mbr_prefers_the_analysis_whose_constituents_carry_more_mass() -> None:
    chart = build(THREE, LogInside(), log_weights)
    total = chart.inside(key((0, 3)))

    marginals: dict[ItemKey[str], float] = {}

    for tree in unfold(chart, key((0, 3)), limit=9):
        share = exp(score(tree) - total)

        for item in constituents(tree):
            marginals[item] = marginals.get(item, 0.0) + share

    chosen = mbr(
        chart,
        key((0, 3)),
        marginals=marginals,
        utility=ConstituentUtility(),
        candidates=9,
        order=LogProbability(),
    )
    forced = mbr(
        chart,
        key((0, 3)),
        marginals=marginals,
        utility=ConstituentUtility(),
        candidates=1,
    )

    assert chosen is not None
    assert forced is not None
    assert constituent_gain(chosen, marginals=marginals) >= constituent_gain(
        forced, marginals=marginals
    )
    assert key((1, 3)) in constituents(chosen)


def test_mbr_can_disagree_with_the_best_scoring_tree() -> None:
    chart = build(THREE, LogInside(), log_weights)

    best = next(unfold(chart, key((0, 3)), limit=1, order=LogProbability()))
    against = {key((0, 2)): 1.0}

    chosen = mbr(
        chart,
        key((0, 3)),
        marginals=against,
        utility=ConstituentUtility(),
        candidates=9,
        order=LogProbability(),
    )

    assert chosen is not None
    assert best.edge.rule_name == "fa"
    assert chosen.edge.rule_name == "ba"


def test_mbr_returns_nothing_where_the_item_has_no_derivation() -> None:
    chart = build(THREE, LogInside(), log_weights)

    assert (
        mbr(
            chart,
            key((0, 3), "S"),
            marginals={},
            utility=ConstituentUtility(),
            candidates=4,
        )
        is None
    )


class Looping:
    """A one-item cyclic forest."""

    def __init__(self) -> None:
        self._index = SpanIndex(1)
        self._semiring = LogInside()
        self._key = key((0, 1))
        self._edge = Edge("up", (self._key,), "N", -1.0)

    @property
    def index(self) -> SpanIndex:
        """Return the index the forest is addressed by."""
        return self._index

    @property
    def semiring(self) -> LogInside:
        """Return the forest's log-inside algebra."""
        return self._semiring

    def add(self, item: ItemKey[str], edge: Edge[str, float], /) -> None:
        """Reject additions to the fixed forest.

        Parameters
        ----------
        item : ItemKey[str]
        edge : Edge[str, float]

        Raises
        ------
        NotImplementedError
            Always.
        """
        del item, edge

        msg = "the looping forest is fixed"

        raise NotImplementedError(msg)

    def items_at(self, span: Cell, /) -> tuple[ItemKey[str], ...]:
        """Return the one item, where the span is the one it covers.

        Parameters
        ----------
        span : Cell
        """
        return (self._key,) if span == (0, 1) else ()

    def inside(self, item: ItemKey[str], /) -> float:
        """Return the item's inside weight, which its own step never settles.

        Parameters
        ----------
        item : ItemKey[str]
        """
        return -1.0 if item == self._key else -inf

    def edges(self, item: ItemKey[str], /) -> tuple[Edge[str, float], ...]:
        """Return the one step, which consumes the item it concludes.

        Parameters
        ----------
        item : ItemKey[str]
        """
        return (self._edge,) if item == self._key else ()

    def term(self, item: ItemKey[str], /) -> str:
        """Return an item's category.

        Parameters
        ----------
        item : ItemKey[str]
        """
        del item

        return "N"

    def derivations(self, item: ItemKey[str], /) -> Iterator[Derivation[str, float]]:
        """Yield no derivations.

        Parameters
        ----------
        item : ItemKey[str]
        """
        del item

        return iter(())

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the shape of the forest."""
        return "Looping()"


def test_an_ordered_unfold_refuses_a_forest_that_loops() -> None:
    with pytest.raises(CyclicForestError, match="reached from itself"):
        list(unfold(Looping(), key((0, 1)), limit=1, order=LogProbability()))


def test_k_best_scores_refuse_a_forest_that_loops() -> None:
    with pytest.raises(CyclicForestError, match="reached from itself"):
        k_best_scores(Looping(), key((0, 1)), order=LogProbability(), k=2)


def test_the_entropy_refuses_a_forest_that_loops() -> None:
    with pytest.raises(CyclicForestError, match="reached from itself"):
        derivation_entropy(Looping(), key((0, 1)))


def test_the_unordered_unfold_survives_a_forest_that_loops() -> None:
    assert list(unfold(Looping(), key((0, 1)), limit=3)) == []


def tagged(forms: Sequence[str]) -> tuple[ForestStep, ...]:
    """Return the three-token forest with its axioms tagged by form.

    Parameters
    ----------
    forms : Sequence[str]
    """
    return tuple(
        (
            span,
            f"{LEXICAL_TAG_PREFIX}{forms[span[0]]}" if not children else rule,
            children,
            weight,
        )
        for span, rule, children, weight in THREE
    )


class _Meanings:
    """Collect the meanings of a derivation's axioms, left to right."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "meanings"

    @property
    def refines_items(self) -> bool:
        """Return whether meanings refine packed items."""
        return True

    def supports(self, rule: Rule[str], /) -> bool:
        """Return that every rule is defined.

        Parameters
        ----------
        rule : Rule[str]
        """
        del rule
        return True

    def axiom(self, entry: Entry[str, str], /) -> tuple[str, ...]:
        """Return the entry's meaning.

        Parameters
        ----------
        entry : Entry[str, str]
        """
        return () if entry.meaning is None else (entry.meaning,)

    def step(
        self,
        context: StepContext[str],
        children: tuple[tuple[str, ...], ...],
        /,
    ) -> tuple[str, ...]:
        """Concatenate the children's meanings.

        Parameters
        ----------
        context : StepContext[str]
        children : tuple[tuple[str, ...], ...]
        """
        del context
        return tuple(meaning for child in children for meaning in child)


def test_reading_a_forest_tree_finds_its_rules_and_entries() -> None:
    chart = build(tagged(("the", "red", "fox")), LogInside(), log_weights)
    lexicon = (
        (Entry("the", "N", "the'"),),
        (Entry("red", "N", "red'"),),
        (Entry("fox", "N", "fox'"),),
    )
    table = LexicalTable({"fa": FORWARD, "ba": BACKWARD}, lexicon)
    (tree,) = [
        tree for tree in chart.derivations(key((0, 3))) if tree.edge.rule_name == "ba"
    ]

    assert step_context(tree, table) == StepContext(
        BACKWARD, "N", log(0.2), 0, ("N", "N")
    )
    left, right = tree.children
    assert step_context(left, table).rule is FORWARD
    assert axiom_entries(right, table) == (lexicon[2][0],)
    assert read(tree, _Meanings(), table) == (("the'", "red'", "fox'"),)
    assert read(tree, _Join(), table) == (None,)


def test_reading_yields_one_value_per_lexical_choice_in_premise_order() -> None:
    chart = build(tagged(("a", "b", "c")), LogInside(), log_weights)
    lexicon = (
        (Entry("a", "N", "a1"), Entry("a", "N", "a2")),
        (Entry("b", "N", "b1"),),
        (Entry("c", "N", "c1"), Entry("c", "N", "c2"), Entry("c", "S", "c3")),
    )
    table = LexicalTable({"fa": FORWARD, "ba": BACKWARD}, lexicon)
    (tree,) = [
        tree for tree in chart.derivations(key((0, 3))) if tree.edge.rule_name == "ba"
    ]

    assert read(tree, _Meanings(), table) == (
        ("a1", "b1", "c1"),
        ("a1", "b1", "c2"),
        ("a2", "b1", "c1"),
        ("a2", "b1", "c2"),
    )


def test_an_axiom_stands_only_for_entries_with_the_form_it_is_tagged_by() -> None:
    chart = build(tagged(("a", "b", "c")), LogInside(), log_weights)
    lexicon = (
        (Entry("b", "N", "b-at-0"), Entry("a", "N", "a-at-0")),
        (Entry("b", "N", "b1"),),
        (Entry("c", "N", "c1"),),
    )
    tree = next(chart.derivations(key((0, 1))))

    assert axiom_entries(tree, LexicalTable({}, lexicon)) == (lexicon[0][1],)

    with pytest.raises(ValueError, match="has the form 'a'"):
        axiom_entries(tree, LexicalTable({}, ((Entry("b", "N", "b-at-0"),),)))


def test_an_untagged_nullary_edge_is_read_as_a_rule_step() -> None:
    chart = PackedForest[str, float](SpanIndex(1), LogInside())
    chart.add(key((0, 1)), Edge("empty", (), "N", log(0.5)))
    tree = next(chart.derivations(key((0, 1))))
    lexicon = ((Entry("a", "N"),),)

    assert not is_axiom(tree)
    assert step_context(tree, LexicalTable({"empty": PAIR}, lexicon)) == StepContext(
        PAIR, "N", log(0.5), 0, ()
    )

    with pytest.raises(ValueError, match="names no rule 'empty'"):
        read(tree, _Meanings(), LexicalTable({}, lexicon))


def test_reading_refuses_a_lexical_edge_with_premises() -> None:
    chart = PackedForest[str, float](SpanIndex(2), LogInside())
    chart.add(key((0, 1)), Edge(f"{LEXICAL_TAG_PREFIX}a", (), "N", log(0.5)))
    chart.add(
        key((0, 2)),
        Edge(f"{LEXICAL_TAG_PREFIX}b", (key((0, 1)),), "N", log(0.5)),
    )
    tree = next(chart.derivations(key((0, 2))))
    table = LexicalTable({}, ((Entry("a", "N"),), (Entry("b", "N"),)))

    with pytest.raises(ValueError, match="lexical edge 'lex:b' has premises"):
        read(tree, _Meanings(), table)


def test_reading_refuses_a_rule_the_table_lacks() -> None:
    chart = build(tagged(("a", "b", "c")), LogInside(), log_weights)
    lexicon = tuple((Entry(form, "N"),) for form in ("a", "b", "c"))
    tree = next(chart.derivations(key((0, 2))))

    with pytest.raises(ValueError, match="names no rule 'fa'"):
        read(tree, _Meanings(), LexicalTable({"ba": BACKWARD}, lexicon))


def test_reading_refuses_an_axiom_no_entry_at_its_position_licenses() -> None:
    chart = build(tagged(("a", "b", "c")), LogInside(), log_weights)
    tree = next(chart.derivations(key((1, 2))))

    with pytest.raises(ValueError, match="no entry at position 1"):
        read(
            tree,
            _Meanings(),
            LexicalTable({}, ((Entry("a", "N"),), (Entry("b", "S"),))),
        )

    with pytest.raises(ValueError, match="beyond the 1 lexicon positions"):
        read(tree, _Meanings(), LexicalTable({}, ((Entry("a", "N"),),)))
