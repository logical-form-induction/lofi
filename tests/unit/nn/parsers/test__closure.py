from itertools import product
from math import inf, log
from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch

from lofi.judge import NoNormalForm, NormalFormRole, RuleSet
from lofi.nn.parsers._closure import UnaryClosure, UnaryTransition
from lofi.nn.parsers._reduction import LogSumExpReduction
from lofi.spec import ApplicationSpec
from lofi.spec._rule_laws import LawGrammar

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge import (
        Direction,
        GeneralizedComposition,
        NormalForm,
        NormalFormDecisionTable,
        Rule,
        RuleProbe,
        RuleRestriction,
        Slot,
    )
    from lofi.terms import TermRuleKernel

FORWARD: Final = "/"
BACKWARD: Final = "\\"

#: Absolute tolerance for log weights.
TOLERANCE = 1e-9


class Identity:
    @property
    def name(self) -> str:
        return "identity"

    @property
    def has_key(self) -> bool:
        return True

    def key(self, term: str, /) -> str:
        return term

    def equivalent(self, left: str, right: str, /) -> bool:
        return left == right


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


class Step:
    """A unary rule lifting one named category to another."""

    def __init__(self, name: str, source: str, output: str, /) -> None:
        self._name = name
        self._source = source
        self._output = output
        self._functors = Functors()

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return 1

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
        return frozenset({0})

    @property
    def forward_is_complete(self) -> bool:
        return True

    @property
    def matching(self) -> Identity:
        return MATCHING

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
        return 0

    @property
    def slash(self) -> str:
        return FORWARD

    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        if inputs == (self._source,):
            yield self._output

    def admits(self, inputs: tuple[str, ...], output: str, /) -> bool:
        return inputs == (self._source,) and output == self._output

    def residuate(self, output: str, known: tuple[Slot[str], ...], /) -> Iterator[str]:
        del output, known
        return
        yield

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "type-raising"}


class RaisedOnce:
    """A filter tagging every unary conclusion and refusing a second one."""

    @property
    def name(self) -> str:
        return "raised-once"

    @property
    def alphabet(self) -> frozenset[str]:
        return frozenset({"OTHER", "RAISED"})

    def lexical_tag(self) -> str:
        return "OTHER"

    def decisions(self, rules: Sequence[Rule[str]], /) -> NormalFormDecisionTable:
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "premises")
            for rule in rules
        )

    def tag(self, rule: Rule[str], children: tuple[str, ...], /) -> str:
        del rule, children
        return "RAISED"

    def admits(
        self,
        rule: Rule[str],
        children: tuple[str, ...],
        inputs: tuple[str, ...],
        output: str,
        /,
    ) -> bool:
        del rule, inputs, output
        return children[0] != "RAISED"


class Unknown(RaisedOnce):
    """A filter naming a tag its own alphabet does not hold."""

    @override
    def tag(self, rule: Rule[str], children: tuple[str, ...], /) -> str:
        del rule, children
        return "UNLISTED"


MATCHING: Final = Identity()
UNUSED_LOWERING: Final = cast(
    "GeneralizedComposition", LawGrammar("direct").rule(ApplicationSpec(view="fwd"))
).lowering
"""A compiled kernel the string-valued toy rules carry but never run."""
PLAIN: Final = NoNormalForm[str]()
SINGLE: Final = ("OTHER",)
TAGS: Final = ("OTHER", "RAISED")


def ruleset(rules: tuple[Step, ...], normal_form: NormalForm[str]) -> RuleSet[str]:
    return RuleSet("toy", rules, MATCHING, normal_form)


def chain(depth: int) -> UnaryClosure[str]:
    rules = (Step("lift", "a", "b"), Step("raise", "b", "c"))

    return UnaryClosure(
        ruleset(rules, PLAIN), rules, ("a", "b", "c"), SINGLE, depth=depth
    )


def diamond(depth: int) -> UnaryClosure[str]:
    rules = (
        Step("left", "a", "b"),
        Step("right", "a", "c"),
        Step("join", "b", "c"),
        Step("top", "c", "d"),
    )

    return UnaryClosure(
        ruleset(rules, PLAIN), rules, ("a", "b", "c", "d"), SINGLE, depth=depth
    )


def slab(values: list[float]) -> torch.Tensor:
    return torch.tensor([[values]], dtype=torch.float64).unsqueeze(-1)


def scores(values: list[float]) -> torch.Tensor:
    return torch.tensor([[values]], dtype=torch.float64)


def test_a_transition_reports_the_rule_and_the_states_it_joins() -> None:
    transition = UnaryTransition(2, (3, 4), (5, 6), admitted=True)

    assert transition.rule == 2
    assert transition.source_category == 3
    assert transition.source_tag == 4
    assert transition.output_category == 5
    assert transition.output_tag == 6
    assert transition.admitted


def test_two_transitions_over_the_same_fields_are_one_value() -> None:
    transition = UnaryTransition(1, (2, 3), (4, 5), admitted=False)
    same = UnaryTransition(1, (2, 3), (4, 5), admitted=False)
    different = UnaryTransition(1, (2, 3), (4, 5), admitted=True)

    assert transition == same
    assert hash(transition) == hash(same)
    assert transition != different
    assert transition != (1, (2, 3), (4, 5), False)
    assert repr(transition) == "UnaryTransition(1, (2, 3), (4, 5), admitted=False)"


def test_a_transition_refuses_reassignment_and_deletion() -> None:
    transition = UnaryTransition(0, (0, 0), (0, 0), admitted=True)
    field = "_rule"

    with pytest.raises(AttributeError, match=r"unary transition is immutable"):
        setattr(transition, field, 9)

    with pytest.raises(AttributeError, match=r"unary transition is immutable"):
        delattr(transition, field)


def test_an_admission_that_is_not_boolean_is_refused() -> None:
    with pytest.raises(TypeError, match=r"admission is not Boolean: 1"):
        UnaryTransition(0, (0, 0), (0, 0), admitted=cast("bool", 1))


def test_a_negative_rule_index_is_refused() -> None:
    with pytest.raises(ValueError, match=r"rule index"):
        UnaryTransition(-1, (0, 0), (0, 0), admitted=True)


def test_every_rule_and_premise_tag_reaches_the_compiled_table() -> None:
    rules = (Step("lift", "a", "b"),)
    closure = UnaryClosure(
        ruleset(rules, RaisedOnce()), rules, ("a", "b"), TAGS, depth=1
    )

    assert closure.transitions == (
        UnaryTransition(0, (0, 0), (1, 1), admitted=True),
        UnaryTransition(0, (0, 1), (1, 1), admitted=False),
    )


def test_a_transition_the_normal_form_refuses_is_never_applied() -> None:
    rules = (Step("lift", "a", "b"),)
    closure = UnaryClosure(
        ruleset(rules, RaisedOnce()), rules, ("a", "b"), TAGS, depth=2
    )

    assert closure.retained == (UnaryTransition(0, (0, 0), (1, 1), admitted=True),)
    assert not closure.truncated


def test_a_tag_outside_the_alphabet_is_refused_by_name() -> None:
    rules = (Step("lift", "a", "b"),)

    with pytest.raises(
        ValueError, match=r"'raised-once' assigns unknown tag 'UNLISTED'"
    ):
        UnaryClosure(ruleset(rules, Unknown()), rules, ("a", "b"), TAGS, depth=1)


def test_a_chain_within_the_depth_keeps_every_step() -> None:
    closure = chain(2)

    assert len(closure.retained) == 2
    assert not closure.truncated


def test_a_chain_beyond_the_depth_is_cut_and_says_so() -> None:
    closure = chain(1)
    held = torch.tensor([[[True, False, False]]])

    assert closure.truncated
    assert closure.reachable(held).tolist() == [[[True, False, False]]]


def test_a_depth_of_zero_applies_no_rule_at_all() -> None:
    closure = chain(0)

    closed = closure.close(
        slab([0.0, -inf, -inf]), scores([log(2.0), log(3.0)]), LogSumExpReduction()
    )

    assert closure.truncated
    assert closed.exp().reshape(-1).tolist() == pytest.approx(
        [1.0, 0.0, 0.0], abs=TOLERANCE
    )


def test_a_state_two_ways_from_the_cell_is_reached_by_the_shorter_one() -> None:
    closure = diamond(2)
    held = torch.tensor([[[True, False, False, False]]])

    assert [transition.rule for transition in closure.retained] == [0, 1, 2, 3]
    assert closure.reachable(held).tolist() == [[[True, True, True, False]]]
    assert not closure.truncated


def test_a_state_no_round_within_the_depth_reaches_is_cut() -> None:
    closure = diamond(1)

    closed = closure.close(
        slab([0.0, -inf, -inf, -inf]),
        scores([0.0, 0.0, 0.0, 0.0]),
        LogSumExpReduction(),
    )

    assert closure.truncated
    assert closed.exp().reshape(-1).tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 0.0], abs=TOLERANCE
    )


def test_a_cell_already_holding_a_premise_applies_its_rule_within_one_round() -> None:
    closed = diamond(1).close(
        slab([-inf, -inf, 0.0, -inf]),
        scores([0.0, 0.0, 0.0, 0.0]),
        LogSumExpReduction(),
    )

    assert closed.exp().reshape(-1).tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 1.0], abs=TOLERANCE
    )


def test_a_diamond_folds_both_of_its_paths_into_the_state_they_meet_at() -> None:
    closed = diamond(2).close(
        slab([0.0, -inf, -inf, -inf]),
        scores([0.0, 0.0, 0.0, 0.0]),
        LogSumExpReduction(),
    )

    assert closed.exp().reshape(-1).tolist() == pytest.approx(
        [1.0, 1.0, 2.0, 2.0], abs=TOLERANCE
    )


def test_a_rule_set_with_no_unary_rule_is_never_truncated() -> None:
    closure = UnaryClosure(ruleset((), PLAIN), (), ("a",), SINGLE, depth=3)

    assert closure.retained == ()
    assert not closure.truncated


def test_a_cycle_is_cut_rather_than_followed() -> None:
    rules = (Step("lift", "a", "b"), Step("drop", "b", "a"))
    closure = UnaryClosure(ruleset(rules, PLAIN), rules, ("a", "b"), SINGLE, depth=4)

    assert closure.retained == (UnaryTransition(0, (0, 0), (1, 0), admitted=True),)
    assert closure.truncated


def test_the_schedule_reaches_a_premise_before_its_conclusion() -> None:
    closure = chain(2)
    order = [state for state, _ in closure.schedule]

    assert order == [closure.state(1, 0), closure.state(2, 0)]


def test_a_state_addresses_one_column_per_category_and_tag() -> None:
    rules = (Step("lift", "a", "b"),)
    closure = UnaryClosure(
        ruleset(rules, RaisedOnce()), rules, ("a", "b"), TAGS, depth=1
    )

    assert [closure.state(category, tag) for category in (0, 1) for tag in (0, 1)] == [
        0,
        1,
        2,
        3,
    ]


def test_a_negative_depth_is_refused() -> None:
    with pytest.raises(ValueError, match=r"unary_depth"):
        chain(-1)


def test_a_depth_that_is_not_a_whole_number_is_refused() -> None:
    with pytest.raises(TypeError, match=r"unary_depth"):
        chain(cast("int", 1.0))


def test_the_rules_come_back_in_the_order_the_score_channel_indexes_them() -> None:
    rules = (Step("lift", "a", "b"), Step("raise", "b", "c"))
    closure = UnaryClosure(
        ruleset(rules, PLAIN), rules, ("a", "b", "c"), SINGLE, depth=2
    )

    assert [rule.name for rule in closure.rules] == ["lift", "raise"]
    assert [transition.rule for transition in closure.retained] == [0, 1]


def test_a_chain_folds_each_step_into_the_cell_it_concludes_in() -> None:
    closure = chain(2)
    values = slab([0.0, -inf, -inf])

    closed = closure.close(values, scores([log(2.0), log(3.0)]), LogSumExpReduction())

    assert closed.exp().reshape(-1).tolist() == pytest.approx(
        [1.0, 2.0, 6.0], abs=TOLERANCE
    )


def test_a_lifted_derivation_joins_whatever_the_cell_already_held() -> None:
    closure = chain(1)
    values = slab([0.0, log(5.0), -inf])

    closed = closure.close(values, scores([log(2.0), log(3.0)]), LogSumExpReduction())

    assert closed.exp().reshape(-1).tolist() == pytest.approx(
        [1.0, 7.0, 21.0], abs=TOLERANCE
    )


def test_a_cut_step_contributes_nothing_to_the_cell_it_would_have_filled() -> None:
    values = slab([0.0, -inf, -inf])
    steps = scores([log(2.0), log(3.0)])

    cut = chain(1).close(values, steps, LogSumExpReduction())
    whole = chain(2).close(values, steps, LogSumExpReduction())

    assert cut[..., 2, 0].item() == -inf
    assert whole[..., 2, 0].exp().item() == pytest.approx(6.0, abs=TOLERANCE)


def test_a_closure_with_nothing_to_apply_returns_the_cell_it_was_given() -> None:
    closure = UnaryClosure(ruleset((), PLAIN), (), ("a",), SINGLE, depth=3)
    values = slab([0.0])

    assert closure.close(values, scores([]), LogSumExpReduction()) is values


def test_every_span_of_a_slab_is_closed_at_once() -> None:
    closure = chain(2)
    values = torch.tensor(
        [[[[0.0], [-inf], [-inf]], [[log(4.0)], [-inf], [-inf]]]], dtype=torch.float64
    )
    steps = torch.tensor([[[log(2.0), log(3.0)], [0.0, 0.0]]], dtype=torch.float64)

    closed = closure.close(values, steps, LogSumExpReduction())

    assert closed.exp()[0, 0].reshape(-1).tolist() == pytest.approx(
        [1.0, 2.0, 6.0], abs=TOLERANCE
    )
    assert closed.exp()[0, 1].reshape(-1).tolist() == pytest.approx(
        [4.0, 4.0, 4.0], abs=TOLERANCE
    )
