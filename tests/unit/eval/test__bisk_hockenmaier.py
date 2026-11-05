"""Tests for the Bisk--Hockenmaier induction core."""

from typing import Final, cast

import pytest
from tests.support.term_reference import categories, infix_syntax

from lofi._types import HeadSelection, ItemId
from lofi.data import Token
from lofi.data.deps import BiskHockenmaierDependencyProjector, CCGCategoryAlgebra
from lofi.deduce import Edge, ItemKey, LogInside, PackedForest, RootGoal, SpanIndex
from lofi.eval._bisk_hockenmaier import (
    BiskEvent,
    BiskHockenmaierInducer,
    BiskHockenmaierSettings,
    BiskItem,
    _copy_forest,
    _normalized,
    bisk_candidates,
)
from lofi.terms import Functors, Identity, TargetSet, Term, TermStore

_SIGNATURE: Final = categories()
_SYNTAX: Final = infix_syntax(_SIGNATURE, ("/", "fwd"), ("\\", "bwd"))


class _Functors:
    """Functor operations over the small category fixture."""

    __slots__ = ("_store",)

    def __init__(self, store: TermStore) -> None:
        self._store = store

    @property
    def slashes(self) -> frozenset[str]:
        return frozenset({"fwd", "bwd"})

    @property
    def forward_slash(self) -> str:
        return "fwd"

    @property
    def backward_slash(self) -> str:
        return "bwd"

    @property
    def result_path(self) -> tuple[int, ...]:
        return (0,)

    @property
    def argument_path(self) -> tuple[int, ...]:
        return (1,)

    @property
    def category_sort(self) -> str:
        """Return the sort the functor constructors build."""
        return "Cat"

    def is_functor(self, term: Term, /) -> bool:
        return term.op.name in self.slashes

    def slash(self, term: Term, /) -> str:
        if not self.is_functor(term):
            msg = f"not a functor: {term!r}"
            raise ValueError(msg)
        return term.op.name

    def make(self, slash: str, result: Term, argument: Term, /) -> Term:
        return self._store.op(_SIGNATURE.by_name(slash), result, argument)

    def opposite(self, slash: str, /) -> str:
        return "bwd" if slash == "fwd" else "fwd"


class _DependencyAlgebra:
    """Atomic category algebra for the packed-forest fixtures."""

    __slots__ = ()

    def is_functor(self, category: Term, /) -> bool:
        del category
        return False

    def result(self, category: Term, /) -> Term:
        raise ValueError(category)

    def argument(self, category: Term, /) -> Term:
        raise ValueError(category)

    def equivalent(self, left: Term, right: Term, /) -> bool:
        return left is right

    def shares_head(self, category: Term, /) -> bool:
        del category
        return False

    def coindices(self, category: Term, /) -> tuple[()]:
        del category
        return ()


def _fixture() -> tuple[TermStore, Functors[Term], Term, Term, Term]:
    store = TermStore(_SIGNATURE)
    n = store.op(_SIGNATURE.by_name("n"))
    s = store.op(_SIGNATURE.by_name("s"))
    conj = store.op(_SIGNATURE.by_name("np"))
    return store, cast("Functors[Term]", _Functors(store)), n, s, conj


def _settings(
    *, k: int = 1, steps: int = 1, derived: bool = False
) -> BiskHockenmaierSettings:
    return BiskHockenmaierSettings(
        nominal_tags=frozenset({"NN"}),
        verbal_tags=frozenset({"VB", "VBG"}),
        coordination_tags=frozenset({"CC"}),
        other_tags=frozenset({"JJ"}),
        gerund_tag="VBG",
        max_arity=3,
        category_induction_steps=steps,
        derived_constituent_pass=derived,
        second_order_modifiers=True,
        iterations=2,
        k=k,
        probability_floor=1e-6,
        tolerance=0.0,
        conventions={"formalism": "ccg"},
    )


def _projector() -> BiskHockenmaierDependencyProjector[Term]:
    return BiskHockenmaierDependencyProjector(
        cast("CCGCategoryAlgebra[Term]", _DependencyAlgebra()),
        frozenset(),
    )


def _item(
    item_id: str,
    forms: tuple[str, ...],
    tags: tuple[str, ...],
    chart: PackedForest[Term, float],
    roots: tuple[Term, ...],
    *,
    store: TermStore,
    head_selections: dict[str, HeadSelection] | None = None,
    raising_rules: frozenset[str] = frozenset(),
    raising_targets: dict[tuple[str, Term], Term] | None = None,
    root_feeders: frozenset[Term] | None = None,
) -> BiskItem:
    return BiskItem(
        ItemId(item_id),
        forms,
        tags,
        tuple(Token(position, form) for position, form in enumerate(forms)),
        chart,
        RootGoal(TargetSet(roots, Identity[Term]())),
        _SYNTAX,
        store,
        {} if head_selections is None else head_selections,
        raising_rules,
        {} if raising_targets is None else raising_targets,
        _projector(),
        root_feeders=root_feeders,
    )


def _inducer(
    settings: BiskHockenmaierSettings,
    functors: Functors[Term],
    n: Term,
    s: Term,
    conj: Term,
) -> BiskHockenmaierInducer:
    return BiskHockenmaierInducer(
        settings,
        functors,
        nominal=n,
        sentence=s,
        coordination=conj,
        headed_arguments=(n, s),
        sentence_arguments=(n,),
    )


def test_one_packed_lexical_edge_becomes_one_candidate() -> None:
    store, functors, n, _, _ = _fixture()
    key = ItemKey((0, 1), n, "lexical")
    dead = functors.make(functors.forward_slash, n, n)
    chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    chart.add(key, Edge("lex:NN", (), n, 0.0))
    chart.add(ItemKey((0, 1), dead, "lexical"), Edge("lex:NN:dead", (), dead, 0.0))
    item = _item("one", ("dogs",), ("NN",), chart, (n,), store=store)
    probabilities = {
        BiskEvent("expansion", ("n",), "lexical"): 1.0,
        BiskEvent("word", ("n", "lexical"), "NN"): 1.0,
    }

    candidates = bisk_candidates(item, {"NN": (n, dead)}, probabilities, limit=1)

    assert len(candidates) == 1
    assert candidates[0].categories == (n,)
    assert candidates[0].dependencies.heads == (0,)
    assert candidates[0].events == tuple(probabilities)

    second = _item("two", ("cats",), ("NN",), chart, (n,), store=store)
    assert (
        bisk_candidates(second, {"NN": (n,)}, probabilities, limit=1)[0].events
        == candidates[0].events
    )


def test_nominal_tags_may_acquire_complex_categories() -> None:
    _, functors, n, s, conj = _fixture()
    modifier = functors.make(functors.forward_slash, n, n)
    inducer = _inducer(_settings(steps=2), functors, n, s, conj)

    assert inducer.lexical_categories("NN", (n, modifier)) == (n, modifier)


def test_the_gerund_tag_starts_with_the_sentence_category() -> None:
    _, functors, n, s, conj = _fixture()
    inducer = _inducer(_settings(), functors, n, s, conj)

    assert inducer.lexical_categories("VBG", (n, s)) == (s,)


def test_sentence_and_second_order_modifiers_are_structurally_admitted() -> None:
    _, functors, n, s, conj = _fixture()
    backward = functors.make(functors.backward_slash, s, s)
    second_order = functors.make(functors.forward_slash, backward, backward)
    inducer = _inducer(_settings(steps=2), functors, n, s, conj)

    assert inducer.lexical_categories("JJ", (backward, second_order)) == (
        backward,
        second_order,
    )


def test_the_derived_constituent_pass_adds_a_distinct_lexical_category() -> None:
    store, functors, n, s, conj = _fixture()
    modifier = functors.make(functors.forward_slash, n, n)
    derived_result = functors.make(functors.forward_slash, modifier, n)
    left = ItemKey((0, 1), n, "lexical")
    right = ItemKey((1, 2), n, "lexical")
    derived_root = ItemKey((0, 2), modifier, "other")
    constituent_chart = PackedForest[Term, float](SpanIndex(2), LogInside())
    constituent_chart.add(left, Edge("lex:NN:left", (), n, 0.0))
    constituent_chart.add(right, Edge("lex:NN:right", (), n, 0.0))
    constituent_chart.add(
        derived_root,
        Edge("derive-modifier", (left, right), modifier, 0.0, 1),
    )
    constituent = _item(
        "constituent",
        ("dogs", "bark"),
        ("NN", "NN"),
        constituent_chart,
        (modifier,),
        store=store,
        head_selections={"derive-modifier": HeadSelection("functor", 0)},
    )
    lexical_root = ItemKey((0, 1), derived_result, "lexical")
    lexical_chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    lexical_chart.add(
        lexical_root,
        Edge("lex:JJ:derived", (), derived_result, 0.0),
    )
    lexical = _item(
        "lexical",
        ("old",),
        ("JJ",),
        lexical_chart,
        (derived_result,),
        store=store,
    )

    ordinary = _inducer(_settings(steps=2, derived=False), functors, n, s, conj).fit(
        (constituent, lexical)
    )
    with_derived = _inducer(_settings(steps=2, derived=True), functors, n, s, conj).fit(
        (constituent, lexical)
    )

    assert ordinary.categories("JJ") == ()
    assert with_derived.categories("JJ") == (derived_result,)


def test_only_the_dominated_transitive_slash_order_is_rejected() -> None:
    _, functors, n, s, conj = _fixture()
    forward = functors.forward_slash
    backward = functors.backward_slash
    backward_result = functors.make(backward, s, n)
    forward_result = functors.make(forward, s, n)
    canonical = functors.make(forward, backward_result, n)
    dominated = functors.make(backward, forward_result, n)
    both_backward = functors.make(backward, backward_result, n)
    both_forward = functors.make(forward, forward_result, n)
    inducer = _inducer(_settings(steps=2), functors, n, s, conj)

    admitted = inducer.lexical_categories(
        "JJ",
        (
            backward_result,
            forward_result,
            canonical,
            dominated,
            both_backward,
            both_forward,
        ),
    )

    assert canonical in admitted
    assert both_backward in admitted
    assert both_forward in admitted
    assert dominated not in admitted


def test_filtered_forest_drops_a_parent_with_a_missing_child_path() -> None:
    store, functors, n, _, _ = _fixture()
    modifier = functors.make(functors.forward_slash, n, n)
    left = ItemKey((0, 1), modifier, "lexical")
    right = ItemKey((1, 2), n, "lexical")
    root = ItemKey((0, 2), n, "other")
    chart = PackedForest[Term, float](SpanIndex(2), LogInside())
    chart.add(left, Edge("lex:JJ", (), modifier, 0.0))
    chart.add(right, Edge("lex:NN", (), n, 0.0))
    chart.add(root, Edge("apply", (left, right), n, 0.0, 1))
    item = _item(
        "dangling",
        ("old", "dogs"),
        ("JJ", "NN"),
        chart,
        (n,),
        store=store,
        head_selections={"apply": HeadSelection("functor", 0)},
    )

    filtered = _copy_forest(item, {"JJ": (modifier,)}, None)

    assert filtered.edges(left) == ()
    assert filtered.edges(right) == ()
    assert filtered.edges(root) == ()
    assert item.goal.root_groups(filtered) == ()


def test_declared_root_feeders_drop_every_other_unary_edge_into_a_root() -> None:
    store, _, n, s, top = _fixture()
    nominal = ItemKey((0, 1), n, "lexical")
    sentence = ItemKey((0, 1), s, "lexical")
    root = ItemKey((0, 1), top, "top")
    chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    chart.add(nominal, Edge("lex:VB", (), n, 0.0))
    chart.add(sentence, Edge("lex:VB", (), s, 0.0))
    chart.add(root, Edge("top-from-nominal", (nominal,), top, 0.0))
    chart.add(root, Edge("top-from-sentence", (sentence,), top, 0.0))

    def edges(feeders: frozenset[Term] | None) -> tuple[str, ...]:
        item = _item(
            "rooted",
            ("go",),
            ("VB",),
            chart,
            (top,),
            store=store,
            root_feeders=feeders,
        )
        filtered = _copy_forest(item, {"VB": (n, s)}, None)
        return tuple(sorted(edge.rule_name for edge in filtered.edges(root)))

    assert edges(None) == ("top-from-nominal", "top-from-sentence")
    assert edges(frozenset({s})) == ("top-from-sentence",)
    assert edges(frozenset()) == ()


def test_type_raising_requires_its_result_in_the_current_global_inventory() -> None:
    store, functors, n, s, _ = _fixture()
    raised = functors.make(
        functors.forward_slash,
        s,
        functors.make(functors.backward_slash, s, n),
    )
    child = ItemKey((0, 1), n, "lexical")
    root = ItemKey((0, 1), raised, "raised")
    chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    chart.add(child, Edge("lex:NN", (), n, 0.0))
    chart.add(root, Edge("raise-forward", (child,), raised, 0.0))
    item = _item(
        "raised",
        ("dogs",),
        ("NN",),
        chart,
        (raised,),
        store=store,
        raising_rules=frozenset({"raise-forward"}),
        raising_targets={("raise-forward", raised): s},
    )

    unattested = _copy_forest(item, {"NN": (n,)}, None)
    attested = _copy_forest(item, {"NN": (n, s)}, None)

    assert unattested.edges(root) == ()
    assert attested.edges(root)


def test_a_type_raising_edge_without_exact_target_metadata_is_refused() -> None:
    store, functors, n, s, _ = _fixture()
    raised = functors.make(functors.forward_slash, s, n)
    child = ItemKey((0, 1), n, "lexical")
    root = ItemKey((0, 1), raised, "raised")
    chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    chart.add(child, Edge("lex:NN", (), n, 0.0))
    chart.add(root, Edge("raise-forward", (child,), raised, 0.0))

    with pytest.raises(ValueError, match="has no declared target"):
        _item(
            "raised",
            ("dogs",),
            ("NN",),
            chart,
            (raised,),
            store=store,
            raising_rules=frozenset({"raise-forward"}),
        )


def test_probability_floor_projects_one_subfloor_outcome_to_the_boundary() -> None:
    frequent = BiskEvent("word", ("n", "lexical"), "NN")
    rare = BiskEvent("word", ("n", "lexical"), "NNS")
    support = {frequent.context: (frequent, rare)}

    projected = _normalized({frequent: 99.0, rare: 1.0}, support, 0.1)

    assert projected == pytest.approx({frequent: 0.9, rare: 0.1})
    assert sum(projected.values()) == pytest.approx(1.0)


def test_probability_floor_leaves_an_already_feasible_distribution_unchanged() -> None:
    first = BiskEvent("word", ("n", "lexical"), "NN")
    second = BiskEvent("word", ("n", "lexical"), "NNS")
    support = {first.context: (first, second)}

    assert _normalized({first: 6.0, second: 4.0}, support, 0.1) == {
        first: 0.6,
        second: 0.4,
    }


def test_probability_floor_refuses_an_infeasible_support() -> None:
    events = tuple(
        BiskEvent("word", ("n", "lexical"), tag) for tag in ("NN", "NNS", "NNP")
    )

    with pytest.raises(ValueError, match="infeasible"):
        _normalized({}, {events[0].context: events}, 0.4)


def test_smoothed_k_best_uses_exact_full_forest_counts_beyond_k() -> None:
    store, functors, n, s, conj = _fixture()
    categories_ = (
        functors.make(functors.forward_slash, n, n),
        functors.make(functors.backward_slash, n, n),
        functors.make(functors.forward_slash, s, n),
        functors.make(functors.backward_slash, s, s),
    )
    chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    root = ItemKey((0, 1), n, "top")
    for position, category in enumerate(categories_):
        lexical = ItemKey((0, 1), category, "lexical")
        chart.add(
            lexical,
            Edge(f"lex:JJ:{position}", (), category, 0.0),
        )
        chart.add(root, Edge("top", (lexical,), n, 0.0))
    item = _item(
        "ambiguous",
        ("old",),
        ("JJ",),
        chart,
        (n,),
        store=store,
        head_selections={"top": HeadSelection("pass-through", 0)},
    )

    model = _inducer(_settings(k=1), functors, n, s, conj).fit((item,))

    probabilities = tuple(
        model.event_probability(
            BiskEvent("head", ("n", "unary"), _SYNTAX.render(store, category))
        )
        for category in categories_
    )
    assert sum(probability > 0.05 for probability in probabilities) == 4
    assert sum(probabilities) == pytest.approx(1.0)
    assert min(probabilities) >= _settings().probability_floor


def test_training_refuses_a_pos_tag_outside_the_declared_partition() -> None:
    store, functors, n, s, conj = _fixture()
    chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    chart.add(
        ItemKey((0, 1), n, "lexical"),
        Edge("lex:UNKNOWN", (), n, 0.0),
    )
    item = _item("unknown", ("word",), ("UNKNOWN",), chart, (n,), store=store)

    with pytest.raises(ValueError, match="outside the declared groups"):
        _inducer(_settings(), functors, n, s, conj).fit((item,))


def test_a_novel_held_out_binary_event_is_pruned_as_zero_probability() -> None:
    store, functors, n, s, conj = _fixture()
    training_chart = PackedForest[Term, float](SpanIndex(1), LogInside())
    training_root = ItemKey((0, 1), n, "lexical")
    training_chart.add(training_root, Edge("lex:NN", (), n, 0.0))
    training = _item("training", ("dogs",), ("NN",), training_chart, (n,), store=store)
    model = _inducer(_settings(), functors, n, s, conj).fit((training,))

    left = ItemKey((0, 1), n, "lexical")
    right = ItemKey((1, 2), n, "lexical")
    root = ItemKey((0, 2), n, "other")
    held_out_chart = PackedForest[Term, float](SpanIndex(2), LogInside())
    held_out_chart.add(left, Edge("lex:NN:left", (), n, 0.0))
    held_out_chart.add(right, Edge("lex:NN:right", (), n, 0.0))
    held_out_chart.add(root, Edge("novel-binary", (left, right), n, 0.0, 1))
    held_out = _item(
        "held-out",
        ("dogs", "dogs"),
        ("NN", "NN"),
        held_out_chart,
        (n,),
        store=store,
        head_selections={"novel-binary": HeadSelection("functor", 0)},
    )

    prediction = model.predict(held_out)

    assert prediction.dependencies is None
    assert prediction.supertags.parser_categories is None
