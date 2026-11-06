from __future__ import annotations

from itertools import combinations, permutations
from random import Random
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
from hypothesis import strategies as st
from tests.language import language

import lofi.eval._logical_form as logical_form_module
from lofi._types import ItemId
from lofi.eval import (
    LOGICAL_FORM_MEASURES,
    ConventionVector,
    LogicalFormMetric,
    LogicalFormScores,
    Metric,
    SmatchScore,
    alpha_match,
    exact_match,
    logical_form_scores,
    normal_form_match,
    normalized_form,
    normalized_tree_distance,
    smatch_score,
)
from lofi.eval._logical_form import _flatten
from lofi.eval.laws import MetricFixture, check_metric_laws
from lofi.terms import (
    BindingSite,
    Calculus,
    Nominal,
    Normalization,
    NotLocallyClosedError,
    Operator,
    Signature,
    Term,
    TermStore,
    TreeEditDistance,
    UnknownOperatorError,
    to_core,
    to_nominal,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.eval._logical_form import LogicalFormMeasure


LANGUAGE: Final = language(("a", "b", "c", "fast", "run", "slow"))
CALCULUS: Final = LANGUAGE.calculus
LAM: Final = LANGUAGE.store.signature.by_name(CALCULUS.lam)
APP: Final = LANGUAGE.store.signature.by_name(CALCULUS.app)
CONST: Final = LANGUAGE.store.signature.constant(LANGUAGE.term_sort)
VARIABLE: Final = LANGUAGE.store.signature.free(LANGUAGE.term_sort)
NAMES: Final = ("x", "y", "z", "u", "v", "w")
SIGNATURE: Final = LANGUAGE.store.signature
RENAMED: Final = tuple(f"{name}1" for name in NAMES)
NORMALIZATION_FUEL: Final = 1_000
NORMALIZATION: Final = Normalization(
    fuel=NORMALIZATION_FUEL,
    order="leftmost-outermost",
    eta="contract",
    associative_commutative=(),
)
COMPARISON_STORE_NODES: Final = LANGUAGE.store.max_nodes


def const(name: str) -> Nominal:
    return Nominal(CONST, payload=name)


def var(name: str) -> Nominal:
    return Nominal(VARIABLE, payload=name)


def app(function: Nominal, argument: Nominal) -> Nominal:
    return Nominal(APP, (function, argument))


def lam(name: str, body: Nominal) -> Nominal:
    return Nominal(LAM, (body,), binders=(name,))


def exists(name: str, body: Nominal) -> Nominal:
    return app(const("exists"), lam(name, body))


def conj(left: Nominal, right: Nominal) -> Nominal:
    return app(app(const("and"), left), right)


def negation(body: Nominal) -> Nominal:
    return app(const("not"), body)


def fresh_store() -> TermStore:
    return TermStore(SIGNATURE, max_nodes=COMPARISON_STORE_NODES)


def _metric(
    measure: LogicalFormMeasure,
    /,
    *,
    calculus: Calculus = CALCULUS,
    seed: int = 0,
    restarts: int = 4,
    normalization: Normalization = NORMALIZATION,
    store_nodes: int = COMPARISON_STORE_NODES,
    conventions: ConventionVector | None = None,
) -> LogicalFormMetric:
    """Build the metric under this fixture's explicit resource bounds."""
    return LogicalFormMetric(
        SIGNATURE,
        calculus,
        measure,
        normalization=normalization,
        store_nodes=store_nodes,
        conventions=conventions,
        seed=seed,
        restarts=restarts,
    )


GOLD: Final = lam("x", app(const("run"), var("x")))
RENAMED_PAIR: Final = (lam("y", app(const("run"), var("y"))), GOLD)
REDEX_PAIR: Final = (
    exists(
        "e",
        app(
            lam("z", conj(app(const("run"), var("z")), app(const("fast"), var("z")))),
            var("e"),
        ),
    ),
    exists("e", conj(app(const("run"), var("e")), app(const("fast"), var("e")))),
)
ETA_PAIR: Final = (const("run"), GOLD)
OVERLAP_PAIR: Final = (
    exists("e", conj(app(const("run"), var("e")), app(const("fast"), var("e")))),
    exists("e", conj(app(const("run"), var("e")), app(const("slow"), var("e")))),
)
FIVE_TRIPLE_PAIR: Final = (conj(const("a"), const("c")), conj(const("a"), const("b")))


def scores(pair: tuple[Nominal, Nominal]) -> LogicalFormScores:
    predicted, gold = pair

    return logical_form_scores(
        fresh_store(),
        CALCULUS,
        predicted,
        gold,
        normalization=NORMALIZATION,
    )


def test_an_identical_pair_matches_at_every_level() -> None:
    result = scores((GOLD, GOLD))

    assert result == LogicalFormScores(
        exact=True,
        alpha=True,
        normal_form=True,
        smatch=SmatchScore(1, 1, 1),
        nted=0.0,
    )


def test_a_renamed_pair_separates_alpha_from_exact() -> None:
    predicted, gold = RENAMED_PAIR
    result = scores(RENAMED_PAIR)

    assert not exact_match(predicted, gold)
    assert alpha_match(fresh_store(), predicted, gold)
    assert (result.exact, result.alpha) == (False, True)
    assert result.nted == 0.0


def test_exact_match_reads_the_canonical_rendering() -> None:
    assert exact_match(GOLD, lam("x", app(const("run"), var("x"))))
    assert not exact_match(GOLD, app(const("run"), var("x")))
    assert repr(GOLD) == "(lam [x]. (eval #const[Tm](run) #fvar[Tm](x)))"


def test_a_redex_pair_separates_normal_form_from_alpha() -> None:
    predicted, gold = REDEX_PAIR
    store = fresh_store()
    result = scores(REDEX_PAIR)

    assert not alpha_match(store, predicted, gold)
    assert normal_form_match(
        store,
        CALCULUS,
        predicted,
        gold,
        normalization=NORMALIZATION,
    )
    assert (result.exact, result.alpha, result.normal_form) == (False, False, True)
    assert result.smatch.f1 == 1.0
    assert result.nted > 0.0


def test_an_eta_expanded_pair_matches_only_when_eta_contraction_is_declared() -> None:
    result = scores(ETA_PAIR)
    without_eta = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )

    assert (result.exact, result.alpha, result.normal_form) == (False, False, True)
    assert not normal_form_match(
        fresh_store(),
        CALCULUS,
        *ETA_PAIR,
        normalization=without_eta,
    )
    assert result.smatch.f1 == 1.0
    assert result.nted > 0.0


def test_a_partial_overlap_pair_separates_smatch_from_normal_form() -> None:
    result = scores(OVERLAP_PAIR)

    assert not result.normal_form
    assert 0.0 < result.smatch.f1 < 1.0


def test_the_leniency_order_holds_on_the_fixture_pairs() -> None:
    for pair in (
        (GOLD, GOLD),
        RENAMED_PAIR,
        REDEX_PAIR,
        ETA_PAIR,
        OVERLAP_PAIR,
        FIVE_TRIPLE_PAIR,
    ):
        result = scores(pair)

        assert result.alpha or not result.exact
        assert result.normal_form or not result.alpha
        assert result.smatch.f1 == 1.0 or not result.normal_form
        assert result.nted == 0.0 or not result.alpha


def test_the_smatch_value_matches_a_worked_nine_triple_case() -> None:
    store = fresh_store()
    predicted, gold = FIVE_TRIPLE_PAIR
    predicted_core = to_core(store, predicted)
    gold_core = to_core(store, gold)
    gold_triples = _flatten(gold_core)
    expected_labels = (
        ("eval", None),
        ("eval", None),
        ("#const[Tm]", store.signature.vocabulary("Tm").id("and")),
        ("#const[Tm]", store.signature.vocabulary("Tm").id("a")),
        ("#const[Tm]", store.signature.vocabulary("Tm").id("b")),
    )

    assert gold_triples.labels == expected_labels
    assert gold_triples.edges == (
        ("arg0", 0, 1),
        ("arg0", 1, 2),
        ("arg1", 1, 3),
        ("arg1", 0, 4),
    )
    assert gold_triples.triples == 9

    score = smatch_score(predicted_core, gold_core)
    matched = 8

    assert score == SmatchScore(matched, 9, 9)
    assert score.precision == matched / 9
    assert score.recall == matched / 9
    assert score.f1 == 2 * matched / (9 + 9)
    assert scores(FIVE_TRIPLE_PAIR).smatch == score


def test_smatch_measures_surface_overlap_rather_than_equivalence() -> None:
    swapped = (conj(const("b"), const("a")), conj(const("a"), const("b")))
    result = scores(swapped)

    assert not result.normal_form
    assert result.smatch.f1 < 1.0


def test_binder_variables_flatten_to_binding_and_occurrence_relations() -> None:
    store = fresh_store()
    core = to_core(store, exists("e", conj(var("e"), app(var("e"), const("a")))))
    flattening = _flatten(core)

    assert flattening.labels == (
        ("eval", None),
        ("#const[Tm]", store.signature.vocabulary("Tm").id("exists")),
        ("lam", None),
        ("", None),
        ("eval", None),
        ("eval", None),
        ("#const[Tm]", store.signature.vocabulary("Tm").id("and")),
        ("eval", None),
        ("#const[Tm]", store.signature.vocabulary("Tm").id("a")),
    )
    assert flattening.edges == (
        ("arg0", 0, 1),
        ("arg1", 0, 2),
        ("binds0", 2, 3),
        ("arg0", 2, 4),
        ("arg0", 4, 5),
        ("arg0", 5, 6),
        ("arg1", 5, 3),
        ("arg1", 4, 7),
        ("arg0", 7, 3),
        ("arg1", 7, 8),
    )
    assert flattening.variables == 1
    assert flattening.triples == 8 + 10


def test_multiple_variables_of_one_binder_resolve_by_index() -> None:
    pair = Operator("pair", "F", arg_sorts=("F", "F"), family="connective")
    both = Operator(
        "both",
        "F",
        arg_sorts=("F",),
        binding=(BindingSite((0,), "F"), BindingSite((0,), "F")),
        family="binder",
    )
    signature = Signature("dual", ("F",), (pair, both))
    store = TermStore(signature)
    left = Nominal(signature.free("F"), payload="p")
    right = Nominal(signature.free("F"), payload="q")
    body = Nominal(pair, (right, Nominal(pair, (left, right))))
    term = to_core(store, Nominal(both, (body,), binders=("p", "q")))
    flattening = _flatten(term)

    assert flattening.edges == (
        ("binds0", 0, 1),
        ("binds1", 0, 2),
        ("arg0", 0, 3),
        ("arg0", 3, 2),
        ("arg1", 3, 4),
        ("arg0", 4, 1),
        ("arg1", 4, 2),
    )


def test_an_open_term_cannot_be_flattened() -> None:
    store = fresh_store()

    with pytest.raises(NotLocallyClosedError):
        _flatten(store.bvar(LANGUAGE.term_sort, 0))

    with pytest.raises(NotLocallyClosedError):
        smatch_score(
            store.bvar(LANGUAGE.term_sort, 0),
            store.const(LANGUAGE.term_sort, "a"),
        )


def _brute_force(predicted: Term, gold: Term) -> int:
    left = _flatten(predicted)
    right = _flatten(gold)
    gold_edges = set(right.edges)
    size = min(len(left.labels), len(right.labels))
    best = 0

    for sources in combinations(range(len(left.labels)), size):
        for targets in permutations(range(len(right.labels)), size):
            mapping = dict(zip(sources, targets, strict=True))
            matched = sum(
                1
                for node, label in enumerate(left.labels)
                if label != ("", None)
                and node in mapping
                and right.labels[mapping[node]] == label
            )
            matched += sum(
                1
                for relation, source, target in left.edges
                if source in mapping
                and target in mapping
                and (relation, mapping[source], mapping[target]) in gold_edges
            )
            best = max(best, matched)

    return best


def _generated(rng: Random, depth: int, scope: tuple[str, ...]) -> Nominal:
    choices = ["const", "not", "and", "eval", "lam", "exists"]

    if scope:
        choices.append("var")

    kind = rng.choice(choices) if depth > 0 else rng.choice(["const", "var"])

    if kind == "var" and scope:
        return var(rng.choice(scope))

    if kind in ("const", "var"):
        return const(rng.choice(("a", "b", "run", "fast")))

    if kind == "not":
        return negation(_generated(rng, depth - 1, scope))

    if kind == "and":
        return conj(
            _generated(rng, depth - 1, scope), _generated(rng, depth - 1, scope)
        )

    if kind == "eval":
        return app(_generated(rng, depth - 1, scope), _generated(rng, depth - 1, scope))

    name = rng.choice(tuple(held for held in NAMES if held not in scope))
    body = _generated(rng, depth - 1, (*scope, name))

    return lam(name, body) if kind == "lam" else exists(name, body)


def _renamed(store: TermStore, form: Nominal) -> Nominal:
    return to_nominal(store, to_core(store, form), RENAMED)


def _pool(count: int, *, depth: int, seed: int) -> tuple[Nominal, ...]:
    rng = Random(seed)
    store = fresh_store()
    forms: list[Nominal] = []

    for _ in range(count):
        form = _generated(rng, depth, ())
        forms.extend((form, _renamed(store, form)))

    return tuple(forms)


def test_hill_climbing_finds_the_optimal_alignment_on_small_pairs() -> None:
    store = fresh_store()
    small = tuple(
        core
        for core in (to_core(store, form) for form in _pool(40, depth=2, seed=3))
        if len(_flatten(core).labels) <= 6
    )
    rng = Random(5)

    assert len(small) >= 20

    for _ in range(60):
        predicted = rng.choice(small)
        gold = rng.choice(small)
        score = smatch_score(predicted, gold, restarts=2, seed=11)

        assert score.matched == _brute_force(predicted, gold)


def test_smatch_is_deterministic_under_a_seed_and_symmetric_in_counts() -> None:
    store = fresh_store()
    predicted, gold = (to_core(store, form) for form in OVERLAP_PAIR)
    first = smatch_score(predicted, gold, restarts=3, seed=7)
    second = smatch_score(predicted, gold, restarts=3, seed=7)
    without = smatch_score(predicted, gold, restarts=0)

    assert first == second
    assert without.matched <= first.matched
    assert smatch_score(gold, predicted).matched == first.matched


def test_smatch_rejects_a_negative_restart_count_or_seed() -> None:
    store = fresh_store()
    term = store.const(LANGUAGE.term_sort, "a")

    with pytest.raises(ValueError, match="restart count is negative"):
        smatch_score(term, term, restarts=-1)

    with pytest.raises(ValueError, match="seed is negative"):
        smatch_score(term, term, seed=-1)


def _classes(
    forms: Sequence[Nominal],
) -> dict[str, tuple[Nominal, ...]]:
    store = fresh_store()
    grouped: dict[str, list[Nominal]] = {}

    for form in forms:
        grouped.setdefault(repr(to_core(store, form)), []).append(form)

    return {key: tuple(members) for key, members in grouped.items()}


def test_alpha_match_is_an_equivalence_relation_over_generated_terms() -> None:
    forms = tuple(dict.fromkeys(_pool(160, depth=4, seed=17)))
    store = fresh_store()
    rng = Random(23)

    assert len(forms) >= 200
    assert len({repr(form) for form in forms}) == len(forms)

    for form in forms:
        assert alpha_match(store, form, form)

    sampled = [(rng.choice(forms), rng.choice(forms)) for _ in range(500)]

    for left, right in sampled:
        assert alpha_match(store, left, right) is alpha_match(store, right, left)

    classes = _classes(forms)
    equivalent_triples = [
        (first, second, third)
        for members in classes.values()
        for first in members
        for second in members
        for third in members
    ]

    assert any(len(members) > 1 for members in classes.values())

    for first, second, third in equivalent_triples[:400]:
        assert alpha_match(store, first, second)
        assert alpha_match(store, second, third)
        assert alpha_match(store, first, third)

    for _ in range(300):
        first, second, third = (rng.choice(forms) for _ in range(3))

        if alpha_match(store, first, second) and alpha_match(store, second, third):
            assert alpha_match(store, first, third)


def test_alpha_match_agrees_with_renaming_and_disagrees_with_structure() -> None:
    forms = _pool(50, depth=3, seed=29)
    store = fresh_store()

    for original, renamed in zip(forms[::2], forms[1::2], strict=True):
        assert alpha_match(store, original, renamed)

    assert not alpha_match(store, GOLD, negation(GOLD))


def test_the_leniency_order_holds_on_generated_pairs() -> None:
    forms = _pool(30, depth=3, seed=31)
    store = fresh_store()
    rng = Random(37)
    distance: TreeEditDistance = TreeEditDistance()

    for _ in range(80):
        predicted, gold = rng.choice(forms), rng.choice(forms)
        result = logical_form_scores(
            store,
            CALCULUS,
            predicted,
            gold,
            distance=distance,
            restarts=1,
            normalization=NORMALIZATION,
        )

        assert result.alpha or not result.exact
        assert result.normal_form or not result.alpha
        assert result.smatch.f1 == 1.0 or not result.normal_form
        assert result.nted == normalized_tree_distance(
            distance, to_core(store, predicted), to_core(store, gold)
        )


def test_normalized_form_is_a_fixed_point() -> None:
    store = fresh_store()
    redex = to_core(
        store, lam("y", app(lam("z", app(const("run"), var("z"))), var("y")))
    )
    normal = normalized_form(store, redex, CALCULUS, normalization=NORMALIZATION)

    assert normal is to_core(store, const("run"))
    assert (
        normalized_form(store, normal, CALCULUS, normalization=NORMALIZATION) is normal
    )


def test_normalized_distance_is_reported_beside_the_match_measures() -> None:
    metric = _metric("alpha")
    metric.update(*ETA_PAIR, key=ItemId("eta"))
    metric.update(*RENAMED_PAIR, key=ItemId("renamed"))
    result = metric.compute()
    per_item = metric.per_item()
    store = fresh_store()
    eta_distance = normalized_tree_distance(
        TreeEditDistance(), to_core(store, ETA_PAIR[0]), to_core(store, ETA_PAIR[1])
    )

    assert result.value == 0.5
    assert result.n == 2
    assert result.components["nted"] == eta_distance / 2
    assert per_item[ItemId("eta")].components["nted"] == eta_distance
    assert per_item[ItemId("renamed")].components["nted"] == 0.0
    assert set(result.components) == {
        *LOGICAL_FORM_MEASURES,
        "smatch-precision",
        "smatch-recall",
    }


@pytest.mark.parametrize("measure", LOGICAL_FORM_MEASURES)
def test_each_measure_names_its_metric_and_reports_its_own_mean(
    measure: LogicalFormMeasure,
) -> None:
    metric = _metric(measure)
    metric.update(*OVERLAP_PAIR, key=ItemId("overlap"))
    metric.update(*REDEX_PAIR, key=ItemId("redex"))
    expected = [
        scores(OVERLAP_PAIR).measure(measure),
        scores(REDEX_PAIR).measure(measure),
    ]
    result = metric.compute()

    assert metric.name == f"logical-form-{measure}"
    assert metric.measure == measure
    assert result.value == pytest.approx(sum(expected) / 2)
    assert result.components[measure] == result.value
    assert metric.conventions is None
    assert metric.order_invariant


def test_an_unknown_measure_or_calculus_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="logical-form measure"):
        _metric(cast("LogicalFormMeasure", "beta"))

    with pytest.raises(UnknownOperatorError):
        _metric("exact", calculus=Calculus(lam="lam", app="app"))

    with pytest.raises(ValueError, match="seed is negative"):
        _metric("exact", seed=-1)

    with pytest.raises(ValueError, match="restart count is negative"):
        _metric("exact", restarts=-1)


def test_update_rejects_unnamed_forms_and_conflicting_rescoring() -> None:
    metric = _metric("exact")
    store = fresh_store()

    unnamed = cast("Nominal", to_core(store, GOLD))

    with pytest.raises(TypeError, match="predicted logical form"):
        metric.update(unnamed, GOLD, key=ItemId("a"))

    with pytest.raises(TypeError, match="gold logical form"):
        metric.update(GOLD, unnamed, key=ItemId("a"))

    metric.update(GOLD, GOLD, key=ItemId("a"))
    metric.update(GOLD, GOLD, key=ItemId("a"))

    assert metric.compute().n == 1

    with pytest.raises(ValueError, match="different form"):
        metric.update(*RENAMED_PAIR, key=ItemId("a"))


def test_a_form_outside_the_signature_is_rejected_at_update() -> None:
    metric = _metric("exact")
    foreign = Nominal(Operator("ite", "F", family="connective"))

    with pytest.raises(UnknownOperatorError):
        metric.update(foreign, GOLD, key=ItemId("a"))


def test_reset_forgets_items_and_releases_the_comparison_store() -> None:
    metric = _metric("smatch")
    metric.update(*OVERLAP_PAIR, key=ItemId("a"))
    fresh = _metric("smatch")
    metric.reset()

    assert metric.compute() == fresh.compute()
    assert dict(metric.per_item()) == {}


def test_per_item_results_count_one_item_each() -> None:
    metric = _metric("smatch", seed=3)
    metric.update(*FIVE_TRIPLE_PAIR, key=ItemId("five"))
    item = metric.per_item()[ItemId("five")]

    assert item.n == 1
    assert item.value == scores(FIVE_TRIPLE_PAIR).smatch.f1
    assert item.resampling.seed == 3
    assert item.resampling.unit == "item"


@pytest.mark.parametrize("coverage", ["zero", "ignore"])
def test_a_missing_prediction_obeys_the_declared_coverage(
    coverage: Literal["zero", "ignore"],
) -> None:
    conventions = ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="sentence",
        coverage=coverage,
        self_arcs="allow",
        gold_derivation="canonical",
    )
    metric = _metric("exact", conventions=conventions)
    item_id = ItemId("missing")
    metric.update(None, GOLD, key=item_id)

    result = metric.compute()
    per_item = metric.per_item()

    if coverage == "zero":
        assert (result.value, result.n) == (0.0, 1)
        assert per_item[item_id].value == 0.0
    else:
        assert (result.value, result.n) == (None, 0)
        assert item_id not in per_item


def test_the_kernel_is_shared_across_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[TreeEditDistance] = []
    kernel = TreeEditDistance

    def counting() -> TreeEditDistance:
        instance: TreeEditDistance = kernel()
        built.append(instance)

        return instance

    monkeypatch.setattr(logical_form_module, "TreeEditDistance", counting)
    metric = _metric("nted")
    metric.update(*RENAMED_PAIR, key=ItemId("a"))
    metric.update(*ETA_PAIR, key=ItemId("b"))

    assert len(built) == 1


def _build() -> Metric[Nominal, Nominal]:
    return _metric("smatch", restarts=1)


type _Example = tuple[str, Nominal, Nominal]

_POOL: Final[tuple[_Example, ...]] = (
    ("same", GOLD, GOLD),
    ("renamed", *RENAMED_PAIR),
    ("redex", *REDEX_PAIR),
    ("eta", *ETA_PAIR),
    ("overlap", *OVERLAP_PAIR),
    ("five", *FIVE_TRIPLE_PAIR),
)


def test_the_metric_passes_every_metric_law() -> None:
    check_metric_laws(MetricFixture(_build, st.sampled_from(_POOL), max_examples=20))
