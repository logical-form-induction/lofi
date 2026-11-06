import pickle
from typing import Final, cast, get_args, get_origin

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.bounded_rating import (
    BoundedRatingSubject,
    BoundedRatingSuite,
)

from lofi._types import ItemId
from lofi.data._constraint import SpanConstraint, SpanConstraints, TypeScope
from lofi.data._derivation import (
    Dependency,
    Derivation,
    DerivationNode,
    GraphEdge,
    GraphNode,
)
from lofi.data._supervision import (
    RESPONSE_MISSING,
    SCALE_KINDS,
    AnnotatorId,
    Categorical,
    Example,
    ForcedChoice,
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    Relational,
    RelationalScalar,
    Scalar,
    Scale,
    ScaleKind,
    Supervision,
    SupervisionArm,
    Unlabeled,
    Weighted,
)
from lofi.data._values import Item, Token, TokenSpan

MAX_EXAMPLES = 50

RATING: Final = Scale("ordinal", 1.0, 7.0)

BINARY: Final = Scale("ordinal", 0.0, 1.0)

SLIDER: Final = Scale("continuous", 0.0, 100.0)

MAGNITUDE: Final = Scale("ratio", 0.0, modulus=20.0)

PREMISE: Final = ItemId("at://example.invalid/expression/1")

HYPOTHESIS: Final = ItemId("at://example.invalid/expression/2")

DERIVATION: Final = Derivation(
    0,
    {
        0: DerivationNode(TokenSpan(0, 2), "S", rule="fwd_app", children=(1, 2)),
        1: DerivationNode(TokenSpan(0, 1), "NP"),
        2: DerivationNode(TokenSpan(1, 2), "S\\NP"),
    },
)

ITEM: Final = Item(ItemId("at://example.invalid/expression/1"), (Token(0, "rain"),))

ARMS: Final = tuple(get_origin(arm) or arm for arm in get_args(Supervision.__value__))

EXAMPLES: Final[tuple[Supervision[str], ...]] = (
    Unlabeled(),
    Scalar(6.0, RATING, AnnotatorId("annotator-3")),
    Categorical("entailed", frozenset({"entailed", "contradicted"})),
    ForcedChoice((PREMISE, HYPOTHESIS), frozenset({HYPOTHESIS})),
    SpanConstraints(
        (
            SpanConstraint(
                TokenSpan(0, 2), frozenset({"S"}), TypeScope(("it", "rains"))
            ),
        ),
        2,
    ),
    GoldDerivation(DERIVATION),
    GoldDependencies((Dependency(1, 0, "nsubj"),)),
    GoldGraph(
        (GraphNode("b1", "box"), GraphNode("x1", "referent")),
        (GraphEdge("b1", "x1", "in"),),
    ),
    Relational(HYPOTHESIS, "entails"),
    RelationalScalar(HYPOTHESIS, "entails", Scalar(80.0, SLIDER)),
)

ARM_NAMES: Final = tuple(type(example).__name__ for example in EXAMPLES)

WRITTEN_FIELDS: Final = (
    "supervision",
    "value",
    "label",
    "chosen",
    "constraints",
    "derivation",
    "dependencies",
    "nodes",
    "relation",
    "value",
)


def test_the_examples_here_cover_every_arm_of_the_union() -> None:
    assert {type(example) for example in EXAMPLES} == set(ARMS)


def test_scalar_ratings_expose_their_typed_related_items() -> None:
    scalar = Scalar(0.73, SLIDER)
    relational = RelationalScalar(HYPOTHESIS, "entails", scalar)

    assert scalar.related_items == ()
    assert relational.related_items == (HYPOTHESIS,)


class TestScalarBoundedRatingConformance(BoundedRatingSuite):
    @pytest.fixture
    def bounded_rating_subject(self) -> BoundedRatingSubject:
        return BoundedRatingSubject(
            Scalar(0.73, SLIDER, AnnotatorId("annotator-5")),
            value=0.73,
            kind="continuous",
            minimum=0.0,
            maximum=100.0,
            annotator="annotator-5",
            related_items=(),
        )


class TestRelationalScalarBoundedRatingConformance(BoundedRatingSuite):
    @pytest.fixture
    def bounded_rating_subject(self) -> BoundedRatingSubject:
        return BoundedRatingSubject(
            RelationalScalar(
                HYPOTHESIS,
                "entails",
                Scalar(0.73, SLIDER, AnnotatorId("annotator-5")),
            ),
            value=0.73,
            kind="continuous",
            minimum=0.0,
            maximum=100.0,
            annotator="annotator-5",
            related_items=(HYPOTHESIS,),
        )


@pytest.mark.parametrize("example", EXAMPLES, ids=ARM_NAMES)
def test_an_arm_survives_pickling_equal_and_equally_hashed(
    example: Supervision[str],
) -> None:
    restored = pickle.loads(pickle.dumps(example))

    assert restored is not example
    assert restored == example
    assert hash(restored) == hash(example)
    assert {example, restored} == {example}


@pytest.mark.parametrize(
    ("example", "field"),
    tuple(zip(EXAMPLES, WRITTEN_FIELDS, strict=True)),
    ids=ARM_NAMES,
)
def test_an_arm_refuses_an_attribute_write(
    example: Supervision[str],
    field: str,
) -> None:
    with pytest.raises(AttributeError):
        setattr(example, field, None)


@pytest.mark.parametrize("example", EXAMPLES, ids=ARM_NAMES)
def test_no_two_arms_are_equal(example: Supervision[str]) -> None:
    others = [other for other in EXAMPLES if other is not example]

    assert all(other != example for other in others)


@given(
    weight=st.floats(min_value=-8.0, max_value=0.0),
    other=st.floats(min_value=-8.0, max_value=0.0),
)
@settings(max_examples=MAX_EXAMPLES)
def test_a_weight_sits_beside_a_value_and_never_inside_it(
    weight: float,
    other: float,
) -> None:
    label = "S/NP"
    one, two = Weighted(label, weight), Weighted(label, other)

    assert one.value == two.value == label
    assert hash(one.value) == hash(label)
    assert (one == two) == (weight == other)
    assert len({one, two}) == (1 if weight == other else 2)


def test_a_weight_is_a_number_a_report_can_print() -> None:
    with pytest.raises(ValueError, match="not finite"):
        Weighted("NP", float("nan"))

    with pytest.raises(ValueError, match="not finite"):
        Weighted("NP", float("inf"))


def test_a_scale_admits_the_responses_between_its_bounds() -> None:
    assert RATING.covers(1.0)
    assert RATING.covers(7.0)
    assert not RATING.covers(0.5)
    assert not RATING.covers(7.5)


def test_a_scale_declares_how_its_distances_read() -> None:
    declared = {scale.kind for scale in (RATING, SLIDER, MAGNITUDE)}

    assert declared == set(SCALE_KINDS)

    # a reader may supply an undeclared enum value.
    with pytest.raises(ValueError, match="no declared kind"):
        Scale(cast("ScaleKind", "interval"), 0.0, 1.0)


def test_a_scale_runs_upward() -> None:
    with pytest.raises(ValueError, match="admits no response"):
        Scale("continuous", 1.0, 0.0)


def test_a_rating_falls_on_its_scale() -> None:
    with pytest.raises(ValueError, match="not a response on an ordinal scale"):
        Scalar(8.0, RATING)

    with pytest.raises(ValueError, match="not a response on an ordinal scale"):
        Scalar(0.0, RATING)


def test_an_ordinal_scale_admits_its_levels_and_nothing_between_them() -> None:
    assert RATING.covers(6.0)
    assert not RATING.covers(6.5)

    with pytest.raises(ValueError, match=r"a rating of 6\.5"):
        Scalar(6.5, RATING)


def test_an_ordinal_scale_counts_the_levels_a_cutpoint_fit_needs() -> None:
    assert RATING.levels == 7
    assert Scale("ordinal", 0.0, 4.0).levels == 5
    assert Scale("ordinal", -2.0, 2.0).levels == 5

    # ordered logits use one fewer cutpoint than levels.
    assert RATING.levels is not None
    assert RATING.levels - 1 == 6


def test_only_an_ordinal_scale_has_levels() -> None:
    assert SLIDER.levels is None
    assert MAGNITUDE.levels is None


def test_an_ordinal_scale_is_bounded_at_a_button() -> None:
    with pytest.raises(ValueError, match=r"bounded at 7\.5"):
        Scale("ordinal", 1.0, 7.5)


def test_an_ordinal_scale_orders_at_least_two_levels() -> None:
    with pytest.raises(ValueError, match="has 1 level"):
        Scale("ordinal", 3.0, 3.0)


def test_a_level_is_counted_from_zero_the_way_a_response_family_indexes_it() -> None:
    assert RATING.level_of(1.0) == 0
    assert RATING.level_of(7.0) == 6
    assert Scale("ordinal", -2.0, 2.0).level_of(0.0) == 2


def test_a_response_on_a_scale_without_levels_is_at_no_level() -> None:
    with pytest.raises(ValueError, match="has no level"):
        SLIDER.level_of(37.0)

    with pytest.raises(ValueError, match="is not a level of"):
        RATING.level_of(6.5)


def test_a_yes_or_no_judgment_keeps_an_order_a_two_label_inventory_drops() -> None:
    yes, no = Scalar(1.0, BINARY), Scalar(0.0, BINARY)

    assert BINARY.levels == 2
    assert yes.value is not None
    assert no.value is not None
    assert BINARY.level_of(yes.value) > BINARY.level_of(no.value)

    # a label inventory does not order its labels.
    labeled = Categorical("yes", frozenset({"yes", "no"}))

    assert labeled.label_set == frozenset({"no", "yes"})


def test_a_slider_response_falls_anywhere_between_the_ends_of_its_scale() -> None:
    assert SLIDER.covers(37.5)
    assert Scalar(37.5, SLIDER).value == 37.5
    assert not SLIDER.covers(100.5)


def test_a_missing_scalar_response_keeps_a_null_and_its_marker() -> None:
    response = Scalar(None, RATING, features=frozenset({RESPONSE_MISSING}))

    assert response.value is None
    assert response.features == frozenset({"response_missing"})


def test_a_scalar_response_rejects_inconsistent_missing_state() -> None:
    with pytest.raises(ValueError, match="null without"):
        Scalar(None, RATING)

    with pytest.raises(ValueError, match="has a value"):
        Scalar(6.0, RATING, features=frozenset({RESPONSE_MISSING}))


def test_a_magnitude_estimation_response_reads_against_its_modulus() -> None:
    response = Scalar(40.0, MAGNITUDE)

    assert MAGNITUDE.modulus == 20.0
    assert response.scale.modulus == 20.0
    assert response.value is not None
    assert response.value / 20.0 == 2.0


def test_a_ratio_scale_has_no_greatest_response() -> None:
    assert MAGNITUDE.maximum is None
    assert MAGNITUDE.covers(1_000_000.0)
    assert Scalar(1_000_000.0, MAGNITUDE).value == 1_000_000.0


def test_a_ratio_scale_refuses_a_response_at_zero() -> None:
    assert not MAGNITUDE.covers(0.0)

    with pytest.raises(ValueError, match=r"a rating of 0\.0"):
        Scalar(0.0, MAGNITUDE)


def test_a_ratio_scale_states_what_its_responses_read_against() -> None:
    with pytest.raises(ValueError, match="states no modulus"):
        Scale("ratio", 0.0)

    with pytest.raises(ValueError, match="modulus is not positive"):
        Scale("ratio", 0.0, modulus=0.0)

    with pytest.raises(ValueError, match=r"a modulus of 5\.0"):
        Scale("ratio", 10.0, 100.0, modulus=5.0)


def test_a_ratio_scale_runs_upward_from_a_response_that_is_not_negative() -> None:
    with pytest.raises(ValueError, match="least response is negative"):
        Scale("ratio", -1.0, modulus=20.0)


def test_only_a_ratio_scale_leaves_its_greatest_response_unstated() -> None:
    with pytest.raises(ValueError, match="continuous scale states no greatest"):
        Scale("continuous", 0.0)

    with pytest.raises(ValueError, match="ordinal scale states no greatest"):
        Scale("ordinal", 1.0)


def test_only_a_ratio_scale_states_a_modulus() -> None:
    with pytest.raises(ValueError, match="continuous scale states a modulus"):
        Scale("continuous", 0.0, 1.0, modulus=0.5)


def test_a_scale_says_how_a_message_names_it() -> None:
    assert RATING.description == "an ordinal scale of 7 levels from 1.0 to 7.0"
    assert SLIDER.description == "a continuous scale from 0.0 to 100.0"
    assert MAGNITUDE.description == "a ratio scale above 0.0 with a modulus of 20.0"


def test_two_scales_differing_only_in_modulus_are_two_scales() -> None:
    other = Scale("ratio", 0.0, modulus=10.0)

    assert other != MAGNITUDE
    assert len({other, MAGNITUDE}) == 2
    assert Scale("ratio", 0.0, modulus=20.0) == MAGNITUDE
    assert hash(Scale("ratio", 0.0, modulus=20.0)) == hash(MAGNITUDE)


def test_two_annotators_rating_one_item_are_two_responses() -> None:
    first = Scalar(6.0, RATING, AnnotatorId("annotator-1"))
    second = Scalar(6.0, RATING, AnnotatorId("annotator-2"))

    assert first != second
    assert len({first, second}) == 2
    assert first.annotator == "annotator-1"
    assert Scalar(6.0, RATING).annotator is None


def test_an_annotator_is_named_or_absent_and_never_empty() -> None:
    with pytest.raises(ValueError, match="annotator id is empty"):
        Scalar(6.0, RATING, AnnotatorId(""))

    with pytest.raises(ValueError, match="annotator id is empty"):
        Categorical("yes", frozenset({"yes"}), AnnotatorId(""))


def test_a_label_comes_from_the_inventory_it_was_chosen_over() -> None:
    inventory = frozenset({"entailed", "contradicted"})

    assert Categorical("entailed", inventory).label_set == inventory

    with pytest.raises(ValueError, match="not in the inventory"):
        Categorical("neutral", inventory)

    with pytest.raises(ValueError, match="empty inventory"):
        Categorical("neutral", frozenset[str]())


def test_a_missing_categorical_response_keeps_a_null_and_its_marker() -> None:
    inventory = frozenset({"yes", "no"})
    response = Categorical(
        None,
        inventory,
        features=frozenset({RESPONSE_MISSING}),
    )

    assert response.label is None
    assert response.label_set == inventory
    assert response.features == frozenset({"response_missing"})


def test_a_categorical_response_rejects_inconsistent_missing_state() -> None:
    inventory = frozenset({"yes", "no"})

    with pytest.raises(ValueError, match="null without"):
        Categorical(None, inventory)

    with pytest.raises(ValueError, match="has a value"):
        Categorical(
            "yes",
            inventory,
            features=frozenset({RESPONSE_MISSING}),
        )


def test_a_meaning_graph_holds_the_nodes_its_edges_name() -> None:
    nodes = (GraphNode("b1", "box"), GraphNode("x1", "referent"))

    assert GoldGraph(nodes, (GraphEdge("b1", "x1", "in"),)).nodes == nodes

    with pytest.raises(ValueError, match="names no node it holds"):
        GoldGraph(nodes, (GraphEdge("b1", "x2", "in"),))

    with pytest.raises(ValueError, match="two nodes under one identifier"):
        GoldGraph((GraphNode("b1", "box"), GraphNode("b1", "referent")))


def test_a_relation_names_the_other_item_by_its_id() -> None:
    other = ItemId("at://example.invalid/expression/2")

    assert Relational(other, "entails").other == other

    with pytest.raises(ValueError, match="a relation is empty"):
        Relational(other, "")

    with pytest.raises(ValueError, match="related item's id is empty"):
        Relational(ItemId(""), "entails")


def test_a_forced_choice_holds_the_set_that_was_shown_and_what_was_picked() -> None:
    choice = ForcedChoice(
        (PREMISE, HYPOTHESIS),
        frozenset({HYPOTHESIS}),
        AnnotatorId("annotator-4"),
    )

    assert choice.alternatives == (PREMISE, HYPOTHESIS)
    assert choice.chosen == frozenset({HYPOTHESIS})
    assert choice.annotator == "annotator-4"
    assert not choice.is_tie


def test_a_forced_choice_keeps_the_order_the_items_were_shown_in() -> None:
    shown = ForcedChoice((PREMISE, HYPOTHESIS), frozenset({PREMISE}))
    reversed_ = ForcedChoice((HYPOTHESIS, PREMISE), frozenset({PREMISE}))

    assert shown.alternatives != reversed_.alternatives
    assert shown != reversed_
    assert len({shown, reversed_}) == 2


def test_a_forced_choice_runs_over_more_than_two_alternatives() -> None:
    third = ItemId("at://example.invalid/expression/3")
    choice = ForcedChoice((PREMISE, HYPOTHESIS, third), frozenset({third}))

    assert len(choice.alternatives) == 3
    assert choice.chosen == frozenset({third})


def test_a_forced_choice_records_a_tie_as_a_pick_of_more_than_one() -> None:
    tie = ForcedChoice((PREMISE, HYPOTHESIS), frozenset({PREMISE, HYPOTHESIS}))

    assert tie.is_tie
    assert tie.chosen == frozenset(tie.alternatives)


def test_a_forced_choice_puts_at_least_two_items_up() -> None:
    with pytest.raises(ValueError, match="fewer than 2"):
        ForcedChoice((PREMISE,), frozenset({PREMISE}))


def test_a_forced_choice_shows_no_item_twice() -> None:
    with pytest.raises(ValueError, match="shown twice"):
        ForcedChoice((PREMISE, PREMISE), frozenset({PREMISE}))


def test_a_forced_choice_picks_from_what_was_shown_and_picks_something() -> None:
    third = ItemId("at://example.invalid/expression/3")

    with pytest.raises(ValueError, match="nothing was picked"):
        ForcedChoice((PREMISE, HYPOTHESIS), frozenset[ItemId]())

    with pytest.raises(ValueError, match="was picked out of a trial"):
        ForcedChoice((PREMISE, HYPOTHESIS), frozenset({third}))


def test_a_forced_choice_names_its_alternatives_and_its_annotator() -> None:
    with pytest.raises(ValueError, match="alternative's id is empty"):
        ForcedChoice((PREMISE, ItemId("")), frozenset({PREMISE}))

    with pytest.raises(ValueError, match="annotator id is empty"):
        ForcedChoice((PREMISE, HYPOTHESIS), frozenset({PREMISE}), AnnotatorId(""))


def test_a_graded_relation_rates_how_strongly_it_holds() -> None:
    unit = Scale("continuous", 0.0, 1.0)
    judgment = RelationalScalar(
        HYPOTHESIS,
        "entails",
        Scalar(0.8, unit, AnnotatorId("annotator-5")),
    )

    assert judgment.other == HYPOTHESIS
    assert judgment.relation == "entails"
    assert judgment.value == 0.8
    assert judgment.scale == unit
    assert judgment.annotator == "annotator-5"


def test_a_graded_relation_on_an_ordinal_scale_is_the_veridicality_shape() -> None:
    three_way = Scale("ordinal", 1.0, 3.0)
    judgment = RelationalScalar(HYPOTHESIS, "veridical", Scalar(3.0, three_way))

    assert three_way.levels == 3
    assert judgment.value is not None
    assert three_way.level_of(judgment.value) == 2

    with pytest.raises(ValueError, match="not a response on an ordinal scale"):
        RelationalScalar(HYPOTHESIS, "veridical", Scalar(4.0, three_way))


def test_a_graded_relation_names_the_other_item_and_the_relation() -> None:
    rating = Scalar(0.8, Scale("continuous", 0.0, 1.0))

    with pytest.raises(ValueError, match="a relation is empty"):
        RelationalScalar(HYPOTHESIS, "", rating)

    with pytest.raises(ValueError, match="related item's id is empty"):
        RelationalScalar(ItemId(""), "entails", rating)


def test_a_graded_relation_and_a_labeled_one_are_two_responses() -> None:
    graded = RelationalScalar(HYPOTHESIS, "entails", Scalar(80.0, SLIDER))
    labeled = Relational(HYPOTHESIS, "entails")

    assert graded != labeled
    assert len({graded, labeled}) == 2


@pytest.mark.parametrize("supervision", EXAMPLES, ids=ARM_NAMES)
def test_an_example_pairs_one_item_with_what_is_said_about_it(
    supervision: Supervision[str],
) -> None:
    example = Example(ITEM, supervision)

    assert example.item is ITEM
    assert example.supervision is supervision
    assert example == Example(ITEM, supervision)
    assert hash(example) == hash(Example(ITEM, supervision))


def test_two_examples_over_one_item_differ_by_what_is_said() -> None:
    unlabeled = Example(ITEM, Unlabeled())
    rated = Example(ITEM, Scalar(6.0, RATING, AnnotatorId("annotator-1")))

    assert unlabeled != rated
    assert len({unlabeled, rated}) == 2


def test_one_sentence_rated_twice_is_two_examples_over_two_items() -> None:
    tokens = (Token(0, "rain"),)
    first = Item(ItemId("at://example.invalid/expression/1"), tokens)
    second = Item(ItemId("at://example.invalid/expression/2"), tokens)
    responses = (
        Example(first, Scalar(6.0, RATING, AnnotatorId("annotator-1"))),
        Example(second, Scalar(2.0, RATING, AnnotatorId("annotator-1"))),
    )

    assert len({response.item.item_id for response in responses}) == 2
    assert len(set(responses)) == 2


def test_an_arm_of_the_union_is_named_by_its_own_type() -> None:
    emitted: frozenset[SupervisionArm] = frozenset({Unlabeled, GoldDerivation})
    read: frozenset[SupervisionArm] = frozenset({Scalar, Unlabeled})

    assert emitted & read == frozenset({Unlabeled})
    assert GoldDerivation not in read
