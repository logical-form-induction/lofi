from typing import TYPE_CHECKING, Final, cast, override

import pytest

from lofi.judge import NoNormalForm, NormalFormRole, RuleSet
from lofi.nn.parsers._states import (
    category_index,
    checked_categories,
    checked_state,
    distinct_outputs,
    rules_by_arity,
)
from lofi.spec import ApplicationSpec
from lofi.spec._rule_laws import LawGrammar

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.judge import (
        Direction,
        GeneralizedComposition,
        RuleProbe,
        RuleRestriction,
        Slot,
    )
    from lofi.terms import TermRuleKernel

FORWARD: Final = "/"
BACKWARD: Final = "\\"


class CaseFolding:
    """A matching under which two categories agree up to letter case."""

    @property
    def name(self) -> str:
        return "case-folding"

    @property
    def has_key(self) -> bool:
        return True

    def key(self, term: str, /) -> str:
        return term.lower()

    def equivalent(self, left: str, right: str, /) -> bool:
        return left.lower() == right.lower()


class Functors:
    @property
    def forward_slash(self) -> str:
        return FORWARD

    @property
    def backward_slash(self) -> str:
        return BACKWARD

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


class ToyRule:
    """A binary rule joining two agreeing categories."""

    def __init__(self, name: str, matching: CaseFolding, /) -> None:
        self._name = name
        self._matching = matching
        self._functors = Functors()

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset({"forward", "check"})

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
    def matching(self) -> CaseFolding:
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        return UNUSED_LOWERING

    @property
    def functors(self) -> Functors:
        return self._functors

    @property
    def order(self) -> int:
        return 1

    @property
    def slash(self) -> str:
        return FORWARD

    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        if len(inputs) == 2 and self._matching.equivalent(inputs[0], inputs[1]):
            yield inputs[0]

    def admits(self, inputs: tuple[str, ...], output: str, /) -> bool:
        return any(self._matching.equivalent(found, output) for found in inputs)

    def residuate(self, output: str, known: tuple[Slot[str], ...], /) -> Iterator[str]:
        del output, known
        return
        yield

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "composition", "direction": "unrestricted"}


class UnaryRule(ToyRule):
    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        if len(inputs) == 1:
            yield inputs[0].upper()

    @override
    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "type-raising"}


class TernaryRule(ToyRule):
    @property
    @override
    def arity(self) -> int:
        return 3

    @override
    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        del inputs
        return
        yield


class ResidualOnlyRule(ToyRule):
    @property
    @override
    def directions(self) -> frozenset[Direction]:
        return frozenset({"check", "residual"})


class PartialRule(ToyRule):
    @property
    @override
    def forward_is_complete(self) -> bool:
        return False


class UndecidableRule(PartialRule):
    @property
    @override
    def directions(self) -> frozenset[Direction]:
        return frozenset({"residual"})


class WideningRule(ToyRule):
    """A rule whose conclusions repeat a class and leave the inventory."""

    @override
    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        yield inputs[0]
        yield inputs[0].upper()
        yield "outside"


MATCHING: Final = CaseFolding()
UNUSED_LOWERING: Final = cast(
    "GeneralizedComposition", LawGrammar("direct").rule(ApplicationSpec(view="fwd"))
).lowering
"""A compiled kernel the string-valued toy rules carry but never run."""


def ruleset(rules: tuple[ToyRule, ...]) -> RuleSet[str]:
    return RuleSet("toy", rules, MATCHING, NoNormalForm[str]())


def test_the_toy_rule_serializes_its_composition_direction() -> None:
    assert ToyRule("binary", MATCHING).to_spec() == {
        "rule": "composition",
        "direction": "unrestricted",
    }


def test_a_state_comes_back_as_the_pair_it_was_given() -> None:
    assert checked_state("left state", (2, 3)) == (2, 3)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ([0, 1], r"left state is not a category and tag pair"),
        ((0, 1, 2), r"left state is not a category and tag pair"),
        ((0,), r"left state is not a category and tag pair"),
        ((0.0, 1), r"left state category is not an integer"),
        ((0, True), r"left state tag is not an integer"),
    ],
)
def test_a_state_that_is_not_a_pair_of_indices_is_refused(
    state: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        checked_state("left state", cast("tuple[int, int]", state))


@pytest.mark.parametrize("state", [(-1, 0), (0, -1)])
def test_a_negative_index_is_refused(state: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=r"left state (category|tag) is negative"):
        checked_state("left state", state)


def test_a_category_is_found_by_the_rule_sets_matching_and_not_by_identity() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING),))

    assert category_index(("np", "s"), rule_set, "NP") == 0
    assert category_index(("np", "s"), rule_set, "S") == 1


def test_a_category_outside_the_inventory_has_no_column() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING),))

    assert category_index(("np", "s"), rule_set, "pp") is None


def test_the_inventory_comes_back_in_the_order_it_was_given() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING),))

    assert checked_categories(["s", "np"], rule_set) == ("s", "np")


def test_an_empty_inventory_is_refused() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING),))

    with pytest.raises(ValueError, match=r"at least one category"):
        checked_categories([], rule_set)


def test_two_categories_addressing_one_column_are_refused() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING),))

    with pytest.raises(
        ValueError, match=r"positions 0 and 2 are equivalent under 'case-folding'"
    ):
        checked_categories(["np", "s", "NP"], rule_set)


def test_one_conclusion_per_class_reaches_the_transition_table() -> None:
    rule_set = ruleset((WideningRule("widen", MATCHING),))
    rule = rule_set.of_arity(2)[0]

    found = distinct_outputs(rule, ("np", "np"), ("np", "s"), rule_set)

    assert found == ((0, "np"),)


def test_a_conclusion_outside_the_inventory_reaches_no_column() -> None:
    rule_set = ruleset((WideningRule("widen", MATCHING),))
    rule = rule_set.of_arity(2)[0]

    found = distinct_outputs(rule, ("pp", "pp"), ("np", "s"), rule_set)

    assert found == ()


@pytest.mark.parametrize("rule", [ResidualOnlyRule, PartialRule])
def test_a_rule_without_complete_forward_enumeration_checks_finite_candidates(
    rule: type[ToyRule],
) -> None:
    rule_set = ruleset((rule("join", MATCHING),))
    checked = rule_set.of_arity(2)[0]

    found = distinct_outputs(checked, ("np", "NP"), ("np", "s"), rule_set)

    assert found == ((0, "np"),)


def test_the_rules_come_back_split_into_unary_and_combining_order() -> None:
    first = ToyRule("first", MATCHING)
    second = ToyRule("second", MATCHING)
    raising = UnaryRule("raising", MATCHING)
    rule_set = ruleset((first, raising, second))

    unary, combining = rules_by_arity(rule_set)

    assert [rule.name for rule in unary] == ["raising"]
    assert [rule.name for rule in combining] == ["first", "second"]


def test_a_ternary_rule_is_one_of_the_combining_rules() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING), TernaryRule("coordinate", MATCHING)))

    unary, combining = rules_by_arity(rule_set)

    assert unary == ()
    assert [rule.name for rule in combining] == ["join", "coordinate"]


def test_a_rule_that_can_neither_enumerate_nor_check_is_refused() -> None:
    rule_set = ruleset((UndecidableRule("join", MATCHING),))

    with pytest.raises(ValueError, match=r"'join' cannot decide every conclusion"):
        rules_by_arity(rule_set)


def test_a_checking_unary_rule_is_retained_beside_the_binary_ones() -> None:
    class ResidualRaising(UnaryRule):
        @property
        @override
        def directions(self) -> frozenset[Direction]:
            return frozenset({"check", "residual"})

        @property
        @override
        def forward_is_complete(self) -> bool:
            return False

    raising = ResidualRaising("raise", MATCHING)
    rule_set = ruleset((ToyRule("join", MATCHING), raising))

    unary, binary = rules_by_arity(rule_set)

    assert unary == (raising,)
    assert [rule.name for rule in binary] == ["join"]


def test_a_rule_set_of_one_arity_returns_an_empty_list_for_the_other() -> None:
    rule_set = ruleset((ToyRule("join", MATCHING),))

    unary, binary = rules_by_arity(rule_set)

    assert unary == ()
    assert len(binary) == 1


def test_the_shared_rule_objects_come_back_rather_than_copies() -> None:
    raising = UnaryRule("raising", MATCHING)
    rule_set = ruleset((raising,))

    unary, _ = rules_by_arity(rule_set)

    assert unary[0] is raising
