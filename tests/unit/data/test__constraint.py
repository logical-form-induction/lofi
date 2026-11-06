import pickle
import re
from typing import Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.data._constraint import (
    SpanConstraint,
    SpanConstraints,
    TokenScope,
    TypeScope,
)
from lofi.data._constraint import occurrences as form_occurrences
from lofi.data._derivation import Derivation, DerivationNode
from lofi.data._values import Item, Token, TokenSpan

MAX_EXAMPLES = 50

QUANTIFIER: Final = frozenset({"S/(S\\NP)", "(S/(S\\NP))/N"})


def item(identifier: str, *surfaces: str) -> Item:
    return Item(
        ItemId(f"at://example.invalid/expression/{identifier}"),
        tuple(Token(position, surface) for position, surface in enumerate(surfaces)),
    )


EVERY_PERSON_LEFT: Final = item("1", "every", "person", "left")

EVERY_PERSON_SAW_EVERY_PERSON: Final = item(
    "2", "every", "person", "saw", "every", "person"
)

NOBODY_LEFT: Final = item("3", "nobody", "left")

FIXTURE: Final = (EVERY_PERSON_LEFT, EVERY_PERSON_SAW_EVERY_PERSON, NOBODY_LEFT)

EVERY_PERSON: Final = SpanConstraint(
    TokenSpan(0, 2), QUANTIFIER, TypeScope(("every", "person"))
)

DERIVATION: Final = Derivation(
    0,
    {
        0: DerivationNode(TokenSpan(0, 3), "S", rule="ba", children=(1, 4)),
        1: DerivationNode(TokenSpan(0, 2), "NP", rule="fa", children=(2, 3)),
        2: DerivationNode(TokenSpan(0, 1), "NP/N"),
        3: DerivationNode(TokenSpan(1, 2), "N"),
        4: DerivationNode(TokenSpan(2, 3), "S\\NP"),
    },
)


def test_a_token_level_scope_names_no_form() -> None:
    assert TokenScope().form is None
    assert TypeScope(("every", "person")).form == ("every", "person")


def test_two_type_level_scopes_are_equal_only_where_they_bind_one_form() -> None:
    one = TypeScope(("every", "person"))
    same = TypeScope(("every", "person"))
    other = TypeScope(("some", "person"))
    shorter = TypeScope(("every",))

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert one != shorter
    assert len({one, same, other, shorter}) == 3


def test_a_type_level_scope_names_at_least_one_word() -> None:
    with pytest.raises(ValueError, match="form of no words"):
        TypeScope(())


def test_a_constraint_defaults_to_binding_one_occurrence() -> None:
    constraint = SpanConstraint(TokenSpan(2, 3), frozenset({"S\\NP"}))

    assert constraint.scope == TokenScope()
    assert constraint.span == TokenSpan(2, 3)
    assert constraint.permitted == frozenset({"S\\NP"})


def test_a_constraint_covers_at_least_one_token() -> None:
    with pytest.raises(ValueError, match="covers no token"):
        SpanConstraint(TokenSpan(1, 1), frozenset({"N"}))


def test_a_constraint_permits_at_least_one_label() -> None:
    with pytest.raises(ValueError, match="permits no label"):
        SpanConstraint(TokenSpan(0, 1), frozenset())


def test_a_type_level_anchor_is_as_wide_as_the_form_it_binds() -> None:
    with pytest.raises(ValueError, match="2 words wide"):
        SpanConstraint(TokenSpan(0, 1), QUANTIFIER, TypeScope(("every", "person")))

    with pytest.raises(ValueError, match="1 words wide"):
        SpanConstraint(TokenSpan(0, 2), QUANTIFIER, TypeScope(("person",)))


def test_a_span_outside_the_item_is_refused_at_construction() -> None:
    constraint = SpanConstraint(TokenSpan(2, 4), frozenset({"S\\NP"}))

    with pytest.raises(ValueError, match="runs past the 3 tokens"):
        SpanConstraints((constraint,), 3)

    assert SpanConstraints((constraint,), 4).constraints == (constraint,)


def test_an_arm_over_a_negative_token_count_is_refused() -> None:
    with pytest.raises(ValueError, match="is negative"):
        SpanConstraints((), -1)

    with pytest.raises(TypeError, match="not an integer"):
        SpanConstraints((), cast("int", 3.0))


def test_an_unconstrained_item_is_an_empty_arm_rather_than_an_absence() -> None:
    arm: SpanConstraints[str] = SpanConstraints((), 3)

    assert arm.constraints == ()
    assert arm.token_count == 3
    assert arm.instances(EVERY_PERSON_LEFT.tokens) == ()


def test_a_type_level_constraint_expands_once_per_occurrence() -> None:
    arm = SpanConstraints((EVERY_PERSON,), len(EVERY_PERSON_LEFT.tokens))
    expanded = {gold.item_id: arm.instances(gold.tokens) for gold in FIXTURE}

    assert expanded[EVERY_PERSON_LEFT.item_id] == (
        SpanConstraint(TokenSpan(0, 2), QUANTIFIER),
    )
    assert expanded[EVERY_PERSON_SAW_EVERY_PERSON.item_id] == (
        SpanConstraint(TokenSpan(0, 2), QUANTIFIER),
        SpanConstraint(TokenSpan(3, 5), QUANTIFIER),
    )
    assert expanded[NOBODY_LEFT.item_id] == ()


def test_every_instance_a_type_level_constraint_expands_to_is_token_level() -> None:
    arm = SpanConstraints((EVERY_PERSON,), len(EVERY_PERSON_LEFT.tokens))
    expanded = arm.instances(EVERY_PERSON_SAW_EVERY_PERSON.tokens)

    assert expanded
    assert all(instance.scope == TokenScope() for instance in expanded)
    assert all(instance.permitted == QUANTIFIER for instance in expanded)


def test_an_overlapping_repeat_is_counted_at_every_position_it_starts_at() -> None:
    doubled = item("4", "very", "very", "very")
    arm = SpanConstraints(
        (
            SpanConstraint(
                TokenSpan(0, 2), frozenset({"X"}), TypeScope(("very", "very"))
            ),
        ),
        3,
    )

    assert arm.instances(doubled.tokens) == (
        SpanConstraint(TokenSpan(0, 2), frozenset({"X"})),
        SpanConstraint(TokenSpan(1, 3), frozenset({"X"})),
    )


def test_a_token_level_constraint_expands_to_itself() -> None:
    constraint = SpanConstraint(TokenSpan(2, 3), frozenset({"S\\NP"}))
    arm = SpanConstraints((constraint,), 3)

    assert arm.instances(EVERY_PERSON_LEFT.tokens) == (constraint,)


def test_a_token_level_constraint_binds_no_item_that_cannot_hold_its_span() -> None:
    arm = SpanConstraints((SpanConstraint(TokenSpan(2, 3), frozenset({"S\\NP"})),), 3)

    assert arm.instances(NOBODY_LEFT.tokens) == ()


def test_a_phrase_constraint_reaches_what_a_per_position_arm_cannot() -> None:
    arm = SpanConstraints((EVERY_PERSON,), 3)
    phrasal = arm.instances(EVERY_PERSON_LEFT.tokens)

    assert [instance.span.length for instance in phrasal] == [2]


def test_the_width_one_total_coverage_constructor_constrains_every_position() -> None:
    arm = SpanConstraints.from_positions(
        (frozenset({"NP/N"}), frozenset({"N"}), frozenset({"S\\NP", "S/NP"}))
    )

    assert arm.token_count == 3
    assert tuple(constraint.span for constraint in arm.constraints) == (
        TokenSpan(0, 1),
        TokenSpan(1, 2),
        TokenSpan(2, 3),
    )
    assert arm.instances(EVERY_PERSON_LEFT.tokens) == arm.constraints
    assert arm.constraints[2].permitted == frozenset({"S\\NP", "S/NP"})


def test_the_width_one_constructor_refuses_a_position_permitting_nothing() -> None:
    with pytest.raises(ValueError, match="permits no label"):
        SpanConstraints.from_positions((frozenset({"NP"}), frozenset()))


PAIR: Final = Derivation(
    0,
    {
        0: DerivationNode(TokenSpan(0, 2), "S", rule="fa", children=(1, 2)),
        1: DerivationNode(TokenSpan(0, 1), "S/NP"),
        2: DerivationNode(TokenSpan(1, 2), "NP"),
    },
)


@pytest.mark.parametrize("derivation", [DERIVATION, PAIR])
def test_a_gold_analysis_spans_exactly_the_tokens_its_root_covers(
    derivation: Derivation[str],
) -> None:
    arm = SpanConstraints.from_derivation(derivation)

    assert arm.token_count == len(derivation.leaves())
    assert max(constraint.span.end for constraint in arm.constraints) == arm.token_count

    with pytest.raises(ValueError, match="runs past"):
        SpanConstraints(arm.constraints, arm.token_count - 1)


def test_a_gold_analysis_is_the_total_singleton_arm() -> None:
    arm = SpanConstraints.from_derivation(DERIVATION)

    assert all(len(constraint.permitted) == 1 for constraint in arm.constraints)
    assert {
        (constraint.span, next(iter(constraint.permitted)))
        for constraint in arm.constraints
    } == {
        (TokenSpan(0, 3), "S"),
        (TokenSpan(0, 2), "NP"),
        (TokenSpan(0, 1), "NP/N"),
        (TokenSpan(1, 2), "N"),
        (TokenSpan(2, 3), "S\\NP"),
    }


def test_the_supervised_limit_covers_every_token_and_every_constituent() -> None:
    arm = SpanConstraints.from_derivation(DERIVATION)
    covered = {
        constraint.span for constraint in arm.instances(EVERY_PERSON_LEFT.tokens)
    }

    assert covered == {
        TokenSpan(0, 3),
        TokenSpan(0, 2),
        TokenSpan(0, 1),
        TokenSpan(1, 2),
        TokenSpan(2, 3),
    }
    assert all(TokenSpan(position, position + 1) in covered for position in range(3))


def test_a_constraint_is_equal_on_its_span_its_targets_and_its_scope() -> None:
    one = SpanConstraint(TokenSpan(0, 2), QUANTIFIER, TypeScope(("every", "person")))
    same = SpanConstraint(TokenSpan(0, 2), QUANTIFIER, TypeScope(("every", "person")))
    narrower = SpanConstraint(
        TokenSpan(0, 2), frozenset({"S/(S\\NP)"}), TypeScope(("every", "person"))
    )
    token_level = SpanConstraint(TokenSpan(0, 2), QUANTIFIER)
    other_form = SpanConstraint(
        TokenSpan(0, 2), QUANTIFIER, TypeScope(("some", "person"))
    )

    assert one == same
    assert hash(one) == hash(same)
    assert one != narrower
    assert one != token_level
    assert one != other_form
    assert len({one, same, narrower, token_level, other_form}) == 4


def test_two_arms_over_the_same_constraints_and_item_are_one_arm() -> None:
    one = SpanConstraints((EVERY_PERSON,), 3)
    same = SpanConstraints((EVERY_PERSON,), 3)
    wider = SpanConstraints((EVERY_PERSON,), 5)

    assert one == same
    assert hash(one) == hash(same)
    assert one != wider
    assert len({one, same, wider}) == 2


def test_an_arm_is_not_equal_to_a_value_of_another_type() -> None:
    assert SpanConstraints((), 0) != TokenScope()
    assert TokenScope() != TypeScope(("a",))
    assert SpanConstraint(TokenSpan(0, 1), frozenset({"N"})) != TokenScope()


@pytest.mark.parametrize(
    ("value", "field"),
    [
        (TypeScope(("a",)), "form"),
        (EVERY_PERSON, "permitted"),
        (SpanConstraints((EVERY_PERSON,), 3), "constraints"),
    ],
)
def test_the_values_here_refuse_an_attribute_write(
    value: TypeScope | SpanConstraint[str] | SpanConstraints[str],
    field: str,
) -> None:
    with pytest.raises(AttributeError):
        setattr(value, field, None)


def test_the_values_here_survive_pickling() -> None:
    for value in (
        TokenScope(),
        TypeScope(("every", "person")),
        EVERY_PERSON,
        SpanConstraints((EVERY_PERSON,), 3),
    ):
        restored = pickle.loads(pickle.dumps(value))

        assert restored == value
        assert hash(restored) == hash(value)


LETTERS: Final = {"every": "e", "person": "p", "left": "l"}


def occurrences(surfaces: list[str], form: list[str]) -> list[int]:
    text = "".join(LETTERS[surface] for surface in surfaces)
    needle = "".join(LETTERS[word] for word in form)

    return [match.start() for match in re.finditer(f"(?={re.escape(needle)})", text)]


@given(
    surfaces=st.lists(st.sampled_from(tuple(LETTERS)), max_size=8),
    form=st.lists(st.sampled_from(("every", "person")), min_size=1, max_size=2),
)
@settings(max_examples=MAX_EXAMPLES)
def test_expansion_finds_exactly_the_occurrences_of_the_form(
    surfaces: list[str],
    form: list[str],
) -> None:
    tokens = tuple(
        Token(position, surface) for position, surface in enumerate(surfaces)
    )
    width = len(form)
    arm = SpanConstraints(
        (
            SpanConstraint(
                TokenSpan(0, width), frozenset({"X"}), TypeScope(tuple(form))
            ),
        ),
        max(width, len(surfaces)),
    )
    instances = arm.instances(tokens)

    assert all(
        [token.surface for token in tokens[instance.span.start : instance.span.end]]
        == form
        for instance in instances
    )
    assert [instance.span.start for instance in instances] == occurrences(
        surfaces, form
    )
    assert all(instance.span.length == width for instance in instances)
    assert all(instance.scope == TokenScope() for instance in instances)


def test_occurrences_are_every_start_of_a_form_overlaps_included() -> None:
    tokens = item("7", "a", "a", "a", "b").tokens

    assert form_occurrences(("a", "a"), tokens) == (0, 1)
    assert form_occurrences(("a", "b"), tokens) == (2,)
    assert form_occurrences(("b", "a"), tokens) == ()
    assert form_occurrences(("a", "a", "a", "b", "c"), tokens) == ()
    assert form_occurrences(("a",), ()) == ()
