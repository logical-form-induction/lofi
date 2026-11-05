import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._types import ItemId
from lofi.data import Item, Token
from lofi.data._eval import Annotation, EvalDataset, Join, Pairing

MAX_EXAMPLES = 50

RATED = ("who", "did", "you", "see", "the", "picture", "of")

OTHER = ("the", "child", "slept")


def sentence(surface: tuple[str, ...]) -> tuple[Token, ...]:
    return tuple(Token(index, form) for index, form in enumerate(surface))


def item(name: str, surface: tuple[str, ...] = RATED) -> Item:
    return Item(ItemId(name), sentence(surface))


def rating(name: str, value: int, annotator: str) -> Annotation[int]:
    return Annotation(ItemId(name), value, annotator)


@pytest.fixture
def three_ratings() -> EvalDataset[int]:
    return EvalDataset(
        [item("rating/1"), item("rating/2"), item("rating/3")],
        [
            rating("rating/1", 1, "annotator-a"),
            rating("rating/2", 4, "annotator-b"),
            rating("rating/3", 7, "annotator-c"),
        ],
    )


def test_one_sentence_rated_three_times_is_three_items(
    three_ratings: EvalDataset[int],
) -> None:
    surfaces = {
        tuple(token.surface for token in one.tokens) for one in three_ratings.items
    }

    assert len(three_ratings) == 3
    assert surfaces == {RATED}


def test_three_ratings_of_one_sentence_are_all_scored(
    three_ratings: EvalDataset[int],
) -> None:
    predictions = {
        ItemId("rating/1"): 2,
        ItemId("rating/2"): 4,
        ItemId("rating/3"): 5,
    }

    joined = three_ratings.join(predictions)
    errors = [abs(pair.gold.value - pair.prediction) for pair in joined.pairs]

    assert len(joined.pairs) == 3
    assert errors == [1, 0, 2]
    assert sum(errors) / len(errors) == 1.0


def test_a_join_keeps_each_rating_with_its_own_annotator_and_id(
    three_ratings: EvalDataset[int],
) -> None:
    predictions = {
        ItemId("rating/1"): 2,
        ItemId("rating/2"): 4,
        ItemId("rating/3"): 5,
    }

    joined = three_ratings.join(predictions)

    assert [pair.item_id for pair in joined.pairs] == [
        ItemId("rating/1"),
        ItemId("rating/2"),
        ItemId("rating/3"),
    ]
    assert [pair.gold.value for pair in joined.pairs] == [1, 4, 7]
    assert [pair.gold.annotator for pair in joined.pairs] == [
        "annotator-a",
        "annotator-b",
        "annotator-c",
    ]
    assert [pair.prediction for pair in joined.pairs] == [2, 4, 5]


def test_ids_built_from_the_surface_collide_and_are_refused() -> None:
    shared = "".join(RATED)

    with pytest.raises(ValueError, match="two items share the id"):
        EvalDataset(
            [item(shared), item(shared), item(shared)],
            [
                rating(shared, 1, "annotator-a"),
                rating(shared, 4, "annotator-b"),
                rating(shared, 7, "annotator-c"),
            ],
        )


def test_two_sentences_alike_but_for_their_ids_are_scored_apart(
    three_ratings: EvalDataset[int],
) -> None:
    joined = three_ratings.join({ItemId("rating/1"): 1, ItemId("rating/3"): 1})
    scores = {pair.item_id: pair.gold.value == pair.prediction for pair in joined.pairs}

    assert scores == {ItemId("rating/1"): True, ItemId("rating/3"): False}


def test_an_item_no_prediction_covered_is_reported(
    three_ratings: EvalDataset[int],
) -> None:
    joined = three_ratings.join({ItemId("rating/2"): 4})

    assert [pair.item_id for pair in joined.pairs] == [ItemId("rating/2")]
    assert joined.unpredicted == (ItemId("rating/1"), ItemId("rating/3"))
    assert joined.unmatched == ()


def test_a_prediction_the_set_does_not_contain_is_reported(
    three_ratings: EvalDataset[int],
) -> None:
    joined = three_ratings.join(
        {ItemId("rating/2"): 4, ItemId("elsewhere/9"): 3, ItemId("elsewhere/8"): 2}
    )

    assert joined.unmatched == (ItemId("elsewhere/9"), ItemId("elsewhere/8"))
    assert len(joined.pairs) == 1


def test_pairs_follow_the_set_and_not_the_predictions(
    three_ratings: EvalDataset[int],
) -> None:
    forward = three_ratings.join(
        {ItemId("rating/1"): 1, ItemId("rating/2"): 4, ItemId("rating/3"): 7}
    )
    backward = three_ratings.join(
        {ItemId("rating/3"): 7, ItemId("rating/2"): 4, ItemId("rating/1"): 1}
    )

    assert forward.pairs == backward.pairs


def test_the_set_reads_back_by_id(three_ratings: EvalDataset[int]) -> None:
    assert three_ratings.item(ItemId("rating/2")).item_id == ItemId("rating/2")
    assert three_ratings.annotation(ItemId("rating/2")).value == 4
    assert ItemId("rating/2") in three_ratings
    assert ItemId("rating/4") not in three_ratings


def test_the_set_refuses_an_id_it_does_not_contain(
    three_ratings: EvalDataset[int],
) -> None:
    with pytest.raises(KeyError):
        three_ratings.item(ItemId("rating/4"))

    with pytest.raises(KeyError):
        three_ratings.annotation(ItemId("rating/4"))


def test_the_set_lines_its_items_up_with_its_annotations(
    three_ratings: EvalDataset[int],
) -> None:
    paired = zip(three_ratings.items, three_ratings.annotations, strict=True)

    assert three_ratings.ids == tuple(one.item_id for one in three_ratings.items)
    assert all(one.item_id == gold.item_id for one, gold in paired)


def test_an_annotation_of_an_absent_item_is_refused() -> None:
    with pytest.raises(ValueError, match="which the set does not contain"):
        EvalDataset([item("rating/1")], [rating("rating/9", 1, "annotator-a")])


def test_two_annotations_of_one_item_are_refused() -> None:
    with pytest.raises(ValueError, match="has two annotations"):
        EvalDataset(
            [item("rating/1")],
            [
                rating("rating/1", 1, "annotator-a"),
                rating("rating/1", 7, "annotator-b"),
            ],
        )


def test_an_unannotated_item_is_refused() -> None:
    with pytest.raises(ValueError, match="has no annotation"):
        EvalDataset([item("rating/1"), item("rating/2")], [rating("rating/1", 1, "a")])


def test_an_empty_set_joins_to_nothing() -> None:
    empty: EvalDataset[int] = EvalDataset([], [])
    joined = empty.join({ItemId("rating/1"): 1})

    assert len(empty) == 0
    assert joined.pairs == ()
    assert joined.unpredicted == ()
    assert joined.unmatched == (ItemId("rating/1"),)


def test_an_annotation_names_an_item() -> None:
    with pytest.raises(ValueError, match="no item id"):
        Annotation(ItemId(""), 1, "annotator-a")


def test_an_annotator_is_named_or_absent() -> None:
    with pytest.raises(ValueError, match="empty annotator"):
        rating("rating/1", 1, "")

    assert Annotation(ItemId("rating/1"), 1).annotator is None


def test_a_pairing_refuses_gold_from_another_item() -> None:
    with pytest.raises(ValueError, match="is paired with an annotation of"):
        Pairing(item("rating/1"), rating("rating/2", 4, "annotator-b"), 4)


def test_annotations_differing_only_in_annotator_are_different() -> None:
    first = rating("rating/1", 4, "annotator-a")
    second = rating("rating/1", 4, "annotator-b")

    assert first != second
    assert first == rating("rating/1", 4, "annotator-a")
    assert hash(first) == hash(rating("rating/1", 4, "annotator-a"))


def test_a_pairing_reports_the_id_it_was_joined_on() -> None:
    pair = Pairing(item("rating/1"), rating("rating/1", 4, "annotator-a"), 5)

    assert pair.item_id == ItemId("rating/1")
    assert pair.item.tokens == sentence(RATED)
    assert pair.gold.value == 4
    assert pair.prediction == 5


def test_a_pairing_is_not_equal_to_an_unrelated_object() -> None:
    pair = Pairing(item("rating/1"), rating("rating/1", 4, "annotator-a"), 5)

    assert pair != 5
    assert rating("rating/1", 4, "annotator-a") != 4


def test_a_join_reports_what_it_holds() -> None:
    joined: Join[int, int] = Join((), (ItemId("rating/1"),), ())

    assert joined.pairs == ()
    assert joined.unpredicted == (ItemId("rating/1"),)
    assert joined.unmatched == ()


@given(
    item_numbers=st.lists(
        st.integers(min_value=0, max_value=12), min_size=0, max_size=8, unique=True
    ),
    predicted=st.sets(st.integers(min_value=0, max_value=12), max_size=8),
    surfaces=st.lists(st.sampled_from([RATED, OTHER]), min_size=8, max_size=8),
)
@settings(max_examples=MAX_EXAMPLES)
def test_a_join_partitions_the_set_whatever_the_surfaces_are(
    item_numbers: list[int], predicted: set[int], surfaces: list[tuple[str, ...]]
) -> None:
    dataset = EvalDataset(
        [
            item(f"rating/{n}", surfaces[position])
            for position, n in enumerate(item_numbers)
        ],
        [rating(f"rating/{n}", n % 7 + 1, f"annotator-{n}") for n in item_numbers],
    )
    predictions = {ItemId(f"rating/{n}"): n for n in predicted}

    joined = dataset.join(predictions)
    paired = tuple(pair.item_id for pair in joined.pairs)

    assert len(dataset) == len(item_numbers)
    assert len(paired) + len(joined.unpredicted) == len(dataset)
    assert set(paired) | set(joined.unpredicted) == set(dataset.ids)
    assert set(paired).isdisjoint(joined.unpredicted)
    assert set(joined.unmatched) == set(predictions) - set(dataset.ids)
    assert all(pair.gold.item_id == pair.item_id for pair in joined.pairs)
    assert all(pair.prediction == predictions[pair.item_id] for pair in joined.pairs)
