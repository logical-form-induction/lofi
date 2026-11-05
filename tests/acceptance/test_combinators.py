"""Combinators and their meanings are declared in documents.

A rule set declares the built-in families (application, composition, type
raising, coercion sets) and arbitrary pattern rules over the declared
constructors. An interpretation derives each rule's meaning as a proof term
or takes the meaning the document declares. The cases cover application,
generalized and crossed composition, raising, a declared coercion meaning,
currying, uncurrying, projections, injections, modal introduction and
elimination, and a three-premise case rule, and check that the rule-level
choices a normal form or chart needs are declared rather than inferred from
names.
"""

from typing import TYPE_CHECKING, Literal, cast

import pytest
from tests.acceptance.driver import (
    PointFailedError,
    composed,
    interpretation,
    keys,
    leaf,
    train,
)
from tests.support.derivations import axiom, step

from lofi.data.synthetic import SyntheticGenerator
from lofi.deduce import ItemKey, PackedForest, ParsePolicy, Reachability, Retain
from lofi.interpret import CategoryWitnessRequiredError
from lofi.run import plan_training

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.deduce import Goal
    from lofi.terms import Term


def _forest_policy(
    policy: ParsePolicy[Retain],
) -> ParsePolicy[Literal[Retain.FOREST]]:
    """Return a document policy after requiring packed-forest retention."""
    if policy.retain is not Retain.FOREST:
        msg = f"the reachability law needs a forest policy, not {policy!r}"
        raise AssertionError(msg)

    return cast("ParsePolicy[Literal[Retain.FOREST]]", policy)


def _root_reachable(
    forest: PackedForest[Term, float], goal: Goal[Term]
) -> frozenset[ItemKey[Term]]:
    """Return every item used by at least one derivation of a declared root."""
    pending = list(goal.roots(forest))
    found: set[ItemKey[Term]] = set()

    while pending:
        key = pending.pop()

        if key in found:
            continue

        found.add(key)

        for edge in forest.edges(key):
            pending.extend(edge.children)

    return frozenset(found)


def _meanings(document: str) -> dict[str, str]:
    """Return each rule's meaning, written over its premises p0, p1, ..."""
    interpreted = interpretation(document)
    rendered: dict[str, str] = {}

    for name in interpreted.actions.rules:
        selected = interpreted.actions.resolve(name, None)

        if len(selected) != 1:
            msg = f"the rule has no unique meaning action: {name!r}"
            raise AssertionError(msg)

        rendered[name] = interpreted.render(selected[0])

    return rendered


def test_application_meanings_are_derived_as_proof_terms(tmp_path: Path) -> None:
    assert _meanings("application-meanings.yaml") == {
        "application:/": "p0(p1)",
        "application:\\": "p1(p0)",
    }
    assert train("application-meanings.yaml", tmp_path).rows


def test_generalized_and_crossed_composition_meanings_are_derived(
    tmp_path: Path,
) -> None:
    meanings = _meanings("composition-meanings.yaml")

    assert meanings["composition:/:1:harmonic:unrestricted"] == "\\x0. p0(p1(x0))"
    assert meanings["composition:/:2:harmonic:unrestricted"] == (
        "\\x0. \\x1. p0(p1(x0)(x1))"
    )
    assert meanings["composition:\\:1:crossed:unrestricted"] == "\\x0. p1(p0(x0))"
    assert train("composition-meanings.yaml", tmp_path).rows


def test_unbounded_composition_interprets_a_third_order_step() -> None:
    interpreted = interpretation("unbounded-composition-meanings.yaml")
    name = "composition:/:any:harmonic:unrestricted"
    predicate_category = next(
        category for form, category in interpreted.entries if form == "predicate"
    )
    container_category = next(
        category for form, category in interpreted.entries if form == "container"
    )
    predicate = axiom(interpreted.primary("predicate", predicate_category))
    container = axiom(interpreted.primary("container", container_category))
    rule = interpreted.rule(name)
    (conclusion,) = tuple(rule.apply((predicate_category, container_category)))

    (meaning,) = interpreted.read(
        step(rule, (predicate, container), conclusion, 0.0), interpreted.primaries
    )

    assert interpreted.render(meaning) == "\\x0. \\x1. \\x2. sleep(x0)"
    with pytest.raises(CategoryWitnessRequiredError, match="concrete derivation-step"):
        interpreted.actions.resolve(name, None)


def test_a_type_raising_meaning_is_derived() -> None:
    assert _meanings("raising-meanings.yaml")[
        "type-raising:/:targets=S:arguments=NP"
    ] == ("\\x0. x0(p0)")


def test_a_coercion_whose_meaning_is_not_derivable_takes_the_declared_one(
    tmp_path: Path,
) -> None:
    interpreted = interpretation("coercion-meanings.yaml")
    coercion = "bare-plural:0:N=>NP"

    selected = interpreted.actions.resolve(coercion, None)

    assert len(selected) == 1
    assert interpreted.render(selected[0]) == "iota x0. p0(x0)"
    assert coercion not in interpreted.actions.uncovered
    assert train("coercion-meanings.yaml", tmp_path).rows


def test_a_curry_combinator_over_products_trains(tmp_path: Path) -> None:
    point = train("arrow-product-point.yaml", tmp_path)
    names = [
        leaf(
            point.resolved, f"data.datasets.train.dataset.grammar.rules.rules.{at}.name"
        )
        for at in range(3)
    ]

    assert names == ["apply", "pair", "curry"]


def test_uncurrying_and_both_projections_train(tmp_path: Path) -> None:
    point = train("arrow-uncurry-point.yaml", tmp_path)

    assert keys(point.resolved, "model.spaces.categories.combinators.decoders") == {
        "apply",
    }


def test_modal_introduction_and_elimination_train(tmp_path: Path) -> None:
    point = train("modal-point.yaml", tmp_path)

    assert point.rows


def test_injections_into_any_sum_train(tmp_path: Path) -> None:
    assert train("arrow-injection-point.yaml", tmp_path).rows


def test_a_three_premise_sum_elimination_is_refused_by_its_action_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        PointFailedError, match=r"composing action 'case' applies arity 3"
    ):
        train("arrow-sum-point.yaml", tmp_path)


def test_a_curry_combinator_carries_its_declared_meaning() -> None:
    meanings = _meanings("curry-meanings.yaml")

    assert meanings["curry"] == "\\x0. \\x1. p0(pair(x1)(x0))"


def test_a_coercion_matches_its_source_exactly(tmp_path: Path) -> None:
    point = train("exact-coercion.yaml", tmp_path)
    assert leaf(
        point.resolved, "data.datasets.train.dataset.grammar.rules.rules.2.matching"
    ) == {"kind": "identity"}


def test_a_normal_form_constrains_a_composition_declared_as_a_pattern(
    tmp_path: Path,
) -> None:
    point = train("normal-form-role.yaml", tmp_path)
    assert leaf(
        point.resolved,
        "data.datasets.train.dataset.grammar.rules.rules.2.normal_form_role",
    ) == {
        "kind": "composition",
        "direction": "forward",
        "principal_slot": 0,
        "order": 1,
        "crossed": False,
    }


def test_rules_declare_the_categories_they_introduce_and_consume(
    tmp_path: Path,
) -> None:
    point = train("rule-categories.yaml", tmp_path)
    assert leaf(
        point.resolved, "data.datasets.train.dataset.grammar.rules.rules.0.introduces"
    ) == [
        "S",
        "S\\NP",
    ]
    assert leaf(
        point.resolved, "data.datasets.train.dataset.grammar.rules.rules.2.consumes"
    ) == ["(S\\NP)/(S\\NP)"]


def test_a_document_prunes_only_items_outside_its_root_derivations() -> None:
    rooted = plan_training(composed("rule-categories.yaml"))
    exhaustive = plan_training(
        composed("rule-categories.yaml", "parse.reachability=all")
    )
    assembly = rooted.learner.grammar
    tokens = ("john", "might", "sleeps")
    rooted_forest = SyntheticGenerator(
        assembly, policy=_forest_policy(rooted.parse_policy)
    ).parse(tokens)
    exhaustive_forest = SyntheticGenerator(
        assembly, policy=_forest_policy(exhaustive.parse_policy)
    ).parse(tokens)
    rooted_items = _root_reachable(rooted_forest, assembly.grammar.goal)
    exhaustive_items = _root_reachable(exhaustive_forest, assembly.grammar.goal)

    assert rooted.parse_policy == ParsePolicy(Retain.FOREST, Reachability.ROOTS)
    assert exhaustive.parse_policy == ParsePolicy(Retain.FOREST, Reachability.ALL)
    assert rooted_items == exhaustive_items
    assert {key: rooted_forest.inside(key) for key in rooted_items} == {
        key: exhaustive_forest.inside(key) for key in exhaustive_items
    }
    assert rooted_forest.num_items < exhaustive_forest.num_items
