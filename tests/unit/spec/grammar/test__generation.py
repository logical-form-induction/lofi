from random import Random
from typing import TYPE_CHECKING, Final

import pytest
from tests.support.rules import ccg_signature, declared_rules

from lofi.spec import (
    ApplicationSpec,
    InventoryExhaustedError,
    RuleSetSpec,
)
from lofi.spec.grammar import (
    CategoryInventory,
    CategoryMeasure,
    EisnerNormalFormSpec,
    FeatureErasureMatchingSpec,
    InventoryDerivation,
    category_depth,
    derives,
    generate_inventory,
)
from lofi.spec.grammar._generation import _Grower

if TYPE_CHECKING:
    from lofi.judge import RuleSet
    from lofi.spec.grammar import NormalFormSpec
    from lofi.terms import Term


APPLICATION: Final = RuleSetSpec(
    name="application",
    rules=(
        ApplicationSpec(view="/"),
        ApplicationSpec(view="\\"),
    ),
)


class _Grammar:
    """A directed CCG over S and K - 1 further primitives, with its rules."""

    def __init__(
        self,
        primitives: int,
        normal_form: NormalFormSpec | None = None,
    ) -> None:
        names = ("S", *(f"P{index}" for index in range(1, primitives)))
        adopted = declared_rules(
            ccg_signature(*names, name="generated"),
            APPLICATION,
            matching=FeatureErasureMatchingSpec(),
            normal_form=normal_form,
        )
        self.store = adopted.context.store
        self.syntax = adopted.context.syntax
        self.rules: RuleSet[Term] = adopted.rule_set
        self.primitives = tuple(self.syntax.parse(self.store, name) for name in names)
        self.goals = (self.primitives[0],)

    def generate(
        self,
        *,
        categories: int,
        seed: int = 0,
        max_depth: int = 2,
        derivation_depth: int = 3,
        expansion: float = 0.5,
        patience: int = 200,
        max_arguments: int | None = None,
        max_order: int | None = None,
    ) -> CategoryInventory[Term]:
        return generate_inventory(
            self.rules,
            self.goals,
            self.primitives,
            category_measure=CategoryMeasure((("/", 0), ("\\", 0))),
            categories=categories,
            max_depth=max_depth,
            max_arguments=max_arguments,
            max_order=max_order,
            derivation_depth=derivation_depth,
            expansion=expansion,
            patience=patience,
            random=Random(seed),
        )

    def render(self, inventory: CategoryInventory[Term]) -> tuple[str, ...]:
        return tuple(
            self.syntax.render(self.store, category)
            for category in inventory.categories
        )


def _height(derivation: InventoryDerivation[Term]) -> int:
    return 1 + max(map(_height, derivation.premises)) if derivation.premises else 0


def _forward(
    derivation: InventoryDerivation[Term],
    rules: RuleSet[Term],
) -> bool:
    """Check a derivation by forward application alone."""
    rule = derivation.rule

    if rule is None:
        return not derivation.premises

    inputs = tuple(premise.category for premise in derivation.premises)

    return (
        rule in rules.rules
        and derivation.category in tuple(rule.apply(inputs))
        and all(_forward(premise, rules) for premise in derivation.premises)
    )


def test_category_depth_counts_constructor_nesting() -> None:
    grammar = _Grammar(3)

    def depth(text: str) -> int:
        return category_depth(grammar.syntax.parse(grammar.store, text))

    assert depth("S") == 0
    assert depth("S\\P1") == 1
    assert depth("(S\\P1)/P2") == 2
    assert depth("P1/((S\\P1)/P2)") == 3


@pytest.mark.parametrize("primitives", [2, 4, 8, 16])
def test_every_category_is_a_leaf_of_a_goal_derivation_over_the_inventory(
    primitives: int,
) -> None:
    grammar = _Grammar(primitives)
    inventory = grammar.generate(categories=10, derivation_depth=4)
    held = set(inventory.categories)

    assert len(inventory.categories) == len(held) == 10
    assert len(inventory.witnesses) == 10
    assert inventory.draws >= 1

    for category, witness in zip(
        inventory.categories, inventory.witnesses, strict=True
    ):
        leaves = tuple(witness.leaves())

        assert witness.category in grammar.goals
        assert category in leaves
        assert set(leaves) <= held
        assert _forward(witness, grammar.rules)
        assert derives(witness, grammar.rules)
        assert category_depth(category) <= 2
        assert _height(witness) <= 4


def test_generation_is_deterministic_in_its_seed() -> None:
    grammar = _Grammar(4)
    first = grammar.render(grammar.generate(categories=8, seed=5))

    assert grammar.render(grammar.generate(categories=8, seed=5)) == first
    other = _Grammar(4)

    assert other.render(other.generate(categories=8, seed=5)) == first
    assert grammar.render(grammar.generate(categories=8, seed=6)) != first


def test_the_inventory_draws_on_more_primitives_as_more_are_declared() -> None:
    used: dict[int, set[str]] = {}

    for primitives in (2, 4, 8, 16):
        grammar = _Grammar(primitives)
        inventory = grammar.generate(categories=24, max_depth=3, derivation_depth=4)
        declared = {grammar.syntax.render(grammar.store, p) for p in grammar.primitives}
        atoms: set[str] = set()
        pending = list(inventory.categories)

        while pending:
            term = pending.pop()

            if not term.args:
                atoms.add(grammar.syntax.render(grammar.store, term))

            pending.extend(term.args)

        assert atoms <= declared
        used[primitives] = atoms

    assert len(used[2]) == 2
    assert len(used[16]) > len(used[4]) > len(used[2])


def test_without_expansion_below_the_root_every_witness_is_one_step() -> None:
    grammar = _Grammar(4)
    inventory = grammar.generate(categories=6, expansion=0.0)

    assert all(_height(witness) == 1 for witness in inventory.witnesses)


def test_the_normal_form_admits_every_witness_it_was_generated_under() -> None:
    grammar = _Grammar(4, EisnerNormalFormSpec())
    inventory = grammar.generate(categories=8)

    assert all(derives(witness, grammar.rules) for witness in inventory.witnesses)


def test_a_derivation_with_a_step_no_rule_licenses_is_not_derived() -> None:
    grammar = _Grammar(3)
    s, p1, _ = grammar.primitives
    forward = grammar.rules.rules[0]
    over = grammar.syntax.parse(grammar.store, "S/P1")
    good = InventoryDerivation(
        s, forward, (InventoryDerivation(over), InventoryDerivation(p1))
    )
    swapped = InventoryDerivation(
        s, forward, (InventoryDerivation(p1), InventoryDerivation(over))
    )

    assert tuple(good.leaves()) == (over, p1)
    assert derives(good, grammar.rules)
    assert derives(InventoryDerivation(s), grammar.rules)
    assert not derives(swapped, grammar.rules)
    assert not derives(InventoryDerivation(p1, forward, (good,)), grammar.rules)


def test_an_inventory_the_bounds_cannot_hold_is_refused() -> None:
    grammar = _Grammar(2)

    with pytest.raises(InventoryExhaustedError) as caught:
        grammar.generate(categories=3, max_depth=0, patience=50)

    assert caught.value.requested == 3
    assert caught.value.reached == 1
    assert caught.value.patience == 50

    reachable = grammar.render(
        grammar.generate(categories=6, max_depth=1, derivation_depth=1)
    )

    assert sorted(reachable) == sorted(("S", "P1", "S/S", "S/P1", "S\\S", "S\\P1"))

    with pytest.raises(InventoryExhaustedError, match="reached 6 of 7"):
        grammar.generate(categories=7, max_depth=1, derivation_depth=1)


def _grower(
    grammar: _Grammar, pool: tuple[str, ...], patience: int, seed: int
) -> _Grower:
    """Return a grower over the grammar's rules whose pool is the named categories."""
    return _Grower(
        grammar.rules,
        tuple(grammar.syntax.parse(grammar.store, text) for text in pool),
        category_measure=CategoryMeasure((("/", 0), ("\\", 0))),
        max_depth=1,
        max_arguments=None,
        max_order=None,
        derivation_depth=3,
        expansion=0.0,
        patience=patience,
        random=Random(seed),
    )


def test_the_root_is_expanded_even_when_most_premise_draws_fail() -> None:
    grammar = _Grammar(3)
    complex_premises = tuple(
        f"P{result}{slash}{argument}"
        for result in (1, 2)
        for slash in ("/", "\\")
        for argument in ("S", "P1", "P2")
    )
    pool = ("P1", *complex_premises)
    goal = grammar.primitives[0]

    for seed in range(10):
        grown = _grower(grammar, pool, 200, seed).grow(goal, 0)

        assert grown is not None
        assert grown.rule is not None
        assert derives(grown, grammar.rules)


def test_a_root_no_draw_can_expand_is_left_a_leaf_after_its_patience() -> None:
    grammar = _Grammar(3)
    goal = grammar.primitives[0]
    grown = _grower(grammar, ("P1/P2", "P2\\P1"), 50, 0).grow(goal, 0)

    assert grown is not None
    assert grown.category == goal
    assert grown.rule is None
    assert grown.premises == ()
