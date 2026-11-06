from itertools import product
from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests._controllers import mlp_controller

from lofi.judge import Action, ActionSet, NormalFormRole, RuleSet, recover
from lofi.nn._values import Encoded
from lofi.nn.sites import ComponentId
from lofi.nn.supervision import (
    ControllerExample,
    action_columns,
    negatives,
    positives,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge import (
        Direction,
        NormalFormDecisionTable,
        Rule,
        RuleProbe,
        RuleRestriction,
        Slot,
    )

type Cat = str | tuple[str, Cat, Cat]

FORWARD_SLASH: Final = "/"
BACKWARD_SLASH: Final = "\\"

_BOTH: Final[frozenset[Direction]] = frozenset({"forward", "check"})
_CHECKING: Final[frozenset[Direction]] = frozenset({"check"})
_ENUMERATING: Final[frozenset[Direction]] = frozenset({"forward"})
_PREMISES: Final = 2


def fwd(result: Cat, argument: Cat) -> Cat:
    return (FORWARD_SLASH, result, argument)


def bwd(result: Cat, argument: Cat) -> Cat:
    return (BACKWARD_SLASH, result, argument)


class Structural:
    @property
    def name(self) -> str:
        return "structural"

    @property
    def has_key(self) -> bool:
        return True

    def key(self, term: Cat, /) -> Cat:
        return term

    def equivalent(self, left: Cat, right: Cat, /) -> bool:
        return left == right


MATCHING: Final = Structural()


class Anything:
    @property
    def name(self) -> str:
        return "anything"

    @property
    def alphabet(self) -> frozenset[str]:
        return frozenset({"item"})

    def lexical_tag(self) -> str:
        return "item"

    def decisions(self, rules: Sequence[Rule[Cat]], /) -> NormalFormDecisionTable:
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "premises")
            for rule in rules
        )

    def tag(self, rule: Rule[Cat], children: tuple[str, ...], /) -> str:
        del rule, children

        return "item"

    def admits(
        self,
        rule: Rule[Cat],
        children: tuple[str, ...],
        inputs: tuple[Cat, ...],
        output: Cat,
        /,
    ) -> bool:
        del rule, children, inputs, output

        return True


class Combinator:
    def __init__(
        self, name: str, conclusions: Callable[[Cat, Cat], tuple[Cat, ...]]
    ) -> None:
        self._name = name
        self._conclusions = conclusions

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return _PREMISES

    @property
    def directions(self) -> frozenset[Direction]:
        return _BOTH

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        return True

    @property
    def matching(self) -> Structural:
        return MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    def apply(self, inputs: tuple[Cat, ...], /) -> Iterator[Cat]:
        if len(inputs) != _PREMISES:
            return iter(())

        return iter(self._conclusions(inputs[0], inputs[1]))

    def admits(self, inputs: tuple[Cat, ...], output: Cat, /) -> bool:
        if len(inputs) != _PREMISES:
            return False

        return output in self._conclusions(inputs[0], inputs[1])

    def residuate(self, output: Cat, known: tuple[Slot[Cat], ...], /) -> Iterator[Cat]:
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": self._name}


class Licensing(Combinator):
    @override
    def admits(self, inputs: tuple[Cat, ...], output: Cat, /) -> bool:
        del inputs, output

        return False


class Enumerating(Combinator):
    @property
    @override
    def directions(self) -> frozenset[Direction]:
        return _ENUMERATING


class Checking(Combinator):
    @property
    @override
    def directions(self) -> frozenset[Direction]:
        return _CHECKING

    @property
    @override
    def forward_is_complete(self) -> bool:
        return False


class Ternary(Combinator):
    @property
    @override
    def arity(self) -> int:
        return 3


def _forward_application(left: Cat, right: Cat) -> tuple[Cat, ...]:
    if isinstance(left, tuple) and left[0] == FORWARD_SLASH and left[2] == right:
        return (left[1],)

    return ()


def _backward_application(left: Cat, right: Cat) -> tuple[Cat, ...]:
    if isinstance(right, tuple) and right[0] == BACKWARD_SLASH and right[2] == left:
        return (right[1],)

    return ()


def _forward_composition(left: Cat, right: Cat) -> tuple[Cat, ...]:
    if (
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and left[0] == FORWARD_SLASH
        and right[0] == FORWARD_SLASH
        and left[2] == right[1]
    ):
        return (fwd(left[1], right[2]),)

    return ()


def _backward_composition(left: Cat, right: Cat) -> tuple[Cat, ...]:
    if (
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and left[0] == BACKWARD_SLASH
        and right[0] == BACKWARD_SLASH
        and right[2] == left[1]
    ):
        return (bwd(right[1], left[2]),)

    return ()


FA: Final = Combinator("forward-application", _forward_application)
BA: Final = Combinator("backward-application", _backward_application)
FC: Final = Combinator("forward-composition", _forward_composition)
BC: Final = Combinator("backward-composition", _backward_composition)

INVENTORY: Final = (FA, BA, FC, BC)


def rule_set(*rules: Combinator) -> RuleSet[Cat]:
    inventory = tuple(cast("Rule[Cat]", rule) for rule in rules)

    return RuleSet("ab", inventory, MATCHING, Anything())


RULES: Final = rule_set(*INVENTORY)

APPLY_FORWARD: Final = Action("apply-forward", "forward-application")
APPLY_BACKWARD: Final = Action("apply-backward", "backward-application")
COMPOSE_FORWARD: Final = Action("compose-forward", "forward-composition")
COMPOSE_BACKWARD: Final = Action("compose-backward", "backward-composition")
KEEP: Final = Action("keep")

# identity is second, so rule columns differ from inventory positions.
ACTIONS: Final = ActionSet(
    (APPLY_FORWARD, KEEP, APPLY_BACKWARD, COMPOSE_FORWARD, COMPOSE_BACKWARD), KEEP
)

ATOMS: Final[tuple[Cat, ...]] = ("s", "np", "n", "pp")


def _pool() -> tuple[Cat, ...]:
    slashed = tuple(
        slashing(result, argument)
        for result in ATOMS
        for argument in ATOMS
        for slashing in (fwd, bwd)
    )

    return ATOMS + slashed


CELLS: Final = _pool()

WIRED: Final = ComponentId("chart-controller")
WIDTH: Final = 4
HUNDREDS: Final = 100
PAIRS: Final = 8


@pytest.fixture(scope="module")
def accepted() -> tuple[ControllerExample[Cat], ...]:
    return positives(RULES, action_set=ACTIONS, cells=CELLS)


@pytest.fixture(scope="module")
def refused() -> tuple[ControllerExample[Cat], ...]:
    return negatives(RULES, action_set=ACTIONS, cells=CELLS)


def admitting(pair: tuple[Cat, Cat], output: Cat) -> tuple[str, ...]:
    return tuple(sorted(rule.name for rule in INVENTORY if rule.admits(pair, output)))


def named(example: ControllerExample[Cat], actions: ActionSet) -> tuple[str, ...]:
    return tuple(
        sorted(
            rule
            for rule in (actions.actions[column].rule for column in example.columns)
            if rule is not None
        )
    )


def test_the_generated_pools_reach_hundreds_of_each(
    accepted: tuple[ControllerExample[Cat], ...],
    refused: tuple[ControllerExample[Cat], ...],
) -> None:
    assert len(accepted) >= HUNDREDS
    assert len(refused) >= HUNDREDS


def test_every_positive_is_a_step_some_rule_actually_licenses(
    accepted: tuple[ControllerExample[Cat], ...],
) -> None:
    for example in accepted:
        licensed = {
            output
            for rule in INVENTORY
            for output in rule.apply((example.left, example.right))
        }

        assert example.output in licensed


def test_a_positives_columns_are_every_rule_that_admits_it(
    accepted: tuple[ControllerExample[Cat], ...],
) -> None:
    for example in accepted:
        assert example.is_positive
        assert named(example, ACTIONS) == admitting(
            (example.left, example.right), example.output
        )


def test_a_column_is_the_index_the_action_set_gives_that_action(
    accepted: tuple[ControllerExample[Cat], ...],
) -> None:
    for example in accepted:
        for column in example.columns:
            action = ACTIONS.actions[column]

            assert action.rule is not None
            assert ACTIONS.index(action.name) == column


def test_reordering_the_action_set_moves_every_column_with_it() -> None:
    reordered = ActionSet(
        (COMPOSE_BACKWARD, COMPOSE_FORWARD, APPLY_BACKWARD, APPLY_FORWARD, KEEP), KEEP
    )
    first = positives(RULES, action_set=ACTIONS, cells=CELLS)
    second = positives(RULES, action_set=reordered, cells=CELLS)
    moved = 0

    assert len(first) == len(second)

    for one, other in zip(first, second, strict=True):
        assert one.left == other.left
        assert one.right == other.right
        assert one.output == other.output
        assert named(one, ACTIONS) == named(other, reordered)
        moved += int(one.columns != other.columns)

    assert moved > 0


def test_an_ambiguous_step_carries_both_columns_rather_than_the_first() -> None:
    restricted = Combinator("restricted-application", _forward_application)
    both = rule_set(FA, restricted)
    actions = ActionSet(
        (
            Action("apply", "forward-application"),
            KEEP,
            Action("apply-here", "restricted-application"),
        ),
        KEEP,
    )
    drawn = positives(both, action_set=actions, cells=(fwd("s", "np"), "np"))

    assert len(drawn) == 1
    assert drawn[0].columns == (0, 2)
    assert drawn[0].output == "s"


def test_every_negative_is_refused_by_the_whole_inventory(
    refused: tuple[ControllerExample[Cat], ...],
) -> None:
    for example in refused:
        assert not example.is_positive
        assert example.columns == ()
        assert recover(RULES, (example.left, example.right), example.output) == ()


def test_every_negative_keeps_one_daughter_and_the_output_of_a_positive(
    accepted: tuple[ControllerExample[Cat], ...],
    refused: tuple[ControllerExample[Cat], ...],
) -> None:
    keeping_right = {(example.right, example.output) for example in accepted}
    keeping_left = {(example.left, example.output) for example in accepted}

    for example in refused:
        kept_right = (example.right, example.output) in keeping_right
        kept_left = (example.left, example.output) in keeping_left

        assert kept_right or kept_left
        assert example.left in CELLS
        assert example.right in CELLS


def test_the_accepted_pool_holds_no_step_twice(
    accepted: tuple[ControllerExample[Cat], ...],
) -> None:
    steps = [(example.left, example.right, example.output) for example in accepted]

    assert len(set(steps)) == len(steps)


def test_the_refused_pool_holds_no_step_twice(
    refused: tuple[ControllerExample[Cat], ...],
) -> None:
    steps = [(example.left, example.right, example.output) for example in refused]

    assert len(set(steps)) == len(steps)


def test_no_step_is_both_accepted_and_refused(
    accepted: tuple[ControllerExample[Cat], ...],
    refused: tuple[ControllerExample[Cat], ...],
) -> None:
    licensed = {(example.left, example.right, example.output) for example in accepted}
    denied = {(example.left, example.right, example.output) for example in refused}

    assert licensed & denied == set()


def test_the_columns_gather_out_of_a_controllers_own_action_axis(
    accepted: tuple[ControllerExample[Cat], ...],
) -> None:
    torch.manual_seed(5)
    controller = mlp_controller(ACTIONS, width=WIDTH, component=WIRED, hidden=64)
    pairs = accepted[:PAIRS]
    gold = torch.tensor([example.columns[0] for example in pairs])
    left = Encoded(torch.randn(len(pairs), WIDTH), ("cell", "width"))
    right = Encoded(torch.randn(len(pairs), WIDTH), ("cell", "width"))

    weights = controller.weigh(left, right=right)
    chosen = weights.tensor.gather(1, gold.unsqueeze(1)).squeeze(1)

    assert weights.size("action") == len(ACTIONS.actions)
    assert chosen.shape == (len(pairs),)
    assert bool(torch.isfinite(chosen).all())


def test_the_identity_action_labels_nothing() -> None:
    columns = action_columns(ACTIONS, RULES)

    assert KEEP.name not in columns
    assert set(columns) == {rule.name for rule in INVENTORY}
    assert columns[FA.name] == (0,)
    assert columns[BA.name] == (_PREMISES,)


def test_every_action_applying_one_rule_is_returned_in_column_order() -> None:
    doubled = ActionSet(
        (APPLY_FORWARD, Action("again", "forward-application"), KEEP), KEEP
    )

    assert action_columns(doubled, rule_set(FA))[FA.name] == (0, 1)


def test_every_product_column_of_an_accepting_rule_is_supervised() -> None:
    doubled = ActionSet(
        (APPLY_FORWARD, Action("again", "forward-application"), KEEP), KEEP
    )
    drawn = positives(
        rule_set(FA),
        action_set=doubled,
        cells=(fwd("s", "np"), "np"),
    )

    assert len(drawn) == 1
    assert drawn[0].columns == (0, 1)


def test_an_action_applying_an_undeclared_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        action_columns(ACTIONS, rule_set(FA, BA))


def test_a_binary_rule_no_action_applies_is_refused() -> None:
    narrow = ActionSet((APPLY_FORWARD, KEEP), KEEP)

    with pytest.raises(ValueError, match="no action applies"):
        action_columns(narrow, rule_set(FA, BA))


def test_a_ternary_rule_takes_its_declared_action_column() -> None:
    ternary = Ternary("ternary", _forward_application)
    actions = ActionSet((Action("apply-ternary", "ternary"), KEEP), KEEP)

    assert action_columns(actions, rule_set(ternary))["ternary"] == (0,)


def test_an_inventory_that_licenses_what_it_will_not_admit_is_refused() -> None:
    liar = Licensing("liar", lambda left, right: (fwd(left, right),))
    actions = ActionSet((Action("liar-column", "liar"), KEEP), KEEP)

    with pytest.raises(ValueError, match="admits it under no rule"):
        positives(rule_set(liar), action_set=actions, cells=("s", "np"))


def test_a_rule_that_only_checks_enumerates_nothing_and_still_takes_a_column() -> None:
    watching = Checking("watching", _forward_application)
    both = rule_set(FA, watching)
    actions = ActionSet(
        (
            Action("apply", "forward-application"),
            KEEP,
            Action("watch", "watching"),
        ),
        KEEP,
    )
    drawn = positives(both, action_set=actions, cells=(fwd("s", "np"), "np"))

    assert "forward" not in watching.directions
    assert len(drawn) == 1
    assert drawn[0].columns == (0, 2)


def test_a_rule_declaring_no_forward_direction_enumerates_nothing_here() -> None:
    watching = Checking("watching", _backward_application)
    both = rule_set(FA, watching)
    actions = ActionSet(
        (Action("apply", "forward-application"), KEEP, Action("watch", "watching")),
        KEEP,
    )
    backward = ("np", bwd("s", "np"))

    assert watching.admits(backward, "s")
    assert positives(both, action_set=actions, cells=backward) == ()


def test_a_rule_that_does_not_check_is_refused_rather_than_left_unlabeled() -> None:
    blind = Enumerating("blind", _forward_application)
    actions = ActionSet((Action("blind-column", "blind"), KEEP), KEEP)

    assert "check" not in blind.directions

    with pytest.raises(ValueError, match="do not check"):
        positives(rule_set(blind), action_set=actions, cells=(fwd("s", "np"), "np"))


def test_an_example_refuses_columns_that_repeat_or_descend() -> None:
    with pytest.raises(ValueError, match="ascend"):
        ControllerExample("s", right="np", output="s", columns=(1, 1))

    with pytest.raises(ValueError, match="ascend"):
        ControllerExample("s", right="np", output="s", columns=(_PREMISES, 0))


def test_an_example_refuses_a_column_no_set_has() -> None:
    with pytest.raises(ValueError, match="column"):
        ControllerExample("s", right="np", output="s", columns=(-1,))


def test_an_example_says_what_it_holds() -> None:
    example = ControllerExample(fwd("s", "np"), right="np", output="s", columns=(0,))

    assert example.left == fwd("s", "np")
    assert example.right == "np"
    assert example.output == "s"
    assert example.is_positive
    assert "columns=(0,)" in repr(example)


def test_a_refused_example_is_not_positive() -> None:
    example = ControllerExample("np", right="np", output="s", columns=())

    assert not example.is_positive
    assert example.columns == ()


@settings(max_examples=30)
@given(indices=st.lists(st.integers(0, len(CELLS) - 1), min_size=2, max_size=5))
def test_any_pool_labels_a_positive_by_what_the_inventory_admits(
    indices: list[int],
) -> None:
    pool: Sequence[Cat] = tuple(CELLS[index] for index in sorted(set(indices)))

    for example in positives(RULES, action_set=ACTIONS, cells=pool):
        assert named(example, ACTIONS) == admitting(
            (example.left, example.right), example.output
        )
        assert example.columns == tuple(sorted(set(example.columns)))


@settings(max_examples=30)
@given(indices=st.lists(st.integers(0, len(CELLS) - 1), min_size=2, max_size=5))
def test_any_pools_negatives_are_refused_by_every_rule(indices: list[int]) -> None:
    pool: Sequence[Cat] = tuple(CELLS[index] for index in sorted(set(indices)))

    for example in negatives(RULES, action_set=ACTIONS, cells=pool):
        assert recover(RULES, (example.left, example.right), example.output) == ()
        assert all(
            not rule.admits((example.left, example.right), example.output)
            for rule in INVENTORY
        )
