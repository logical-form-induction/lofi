from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.judgment import (
    DesignCodec,
    GroupedJudgmentSet,
    JudgmentDesign,
    JudgmentRow,
    RowColumns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def complete_set() -> GroupedJudgmentSet:
    rows = (
        JudgmentRow("p-b", "i-b", 2, RowColumns({"frame": "z"}, {})),
        JudgmentRow("p-a", "i-a", 0, RowColumns({"frame": "a"}, {})),
        JudgmentRow("p-b", "i-a", 1, RowColumns({"frame": "a"}, {})),
        JudgmentRow("p-a", "i-b", 2, RowColumns({"frame": "z"}, {})),
    )
    design = DesignCodec(3, factors=("frame",)).encode(rows)

    return GroupedJudgmentSet(
        design,
        ("i-b", "i-a"),
        (("second", "item"), ("first",)),
    )


def test_a_group_view_contains_every_and_only_selected_group_row() -> None:
    judgments = complete_set()
    batch = judgments.select((1,))

    assert batch.active_groups == (1,)
    assert batch.design.participants.tolist() == [1, 1]
    assert batch.design.responses.tolist() == [2, 1]
    assert batch.design.participant_levels == judgments.design.participant_levels
    assert batch.design.item_levels == judgments.design.item_levels
    assert batch.design.factor_levels == judgments.design.factor_levels
    assert batch.item_ids == ("i-b", "i-a")
    assert batch.words == (("second", "item"), ("first",))
    assert batch.lengths == (2, 1)
    assert batch.response_items.dtype is torch.int64
    assert batch.response_items.tolist() == [0, 1]


def test_active_groups_and_neural_items_have_canonical_batch_order() -> None:
    batch = complete_set().select((1, 0))

    assert batch.active_groups == (0, 1)
    assert batch.item_ids == ("i-b", "i-a")
    assert batch.design.participants.tolist() == [1, 0, 1, 0]
    assert batch.response_items.tolist() == [0, 1, 1, 0]


def test_response_rows_follow_the_exact_neural_item_mapping() -> None:
    batch = complete_set().select((0,))
    values = torch.tensor([[2.0, 20.0], [1.0, 10.0]])

    assert batch.item_ids == ("i-b", "i-a")
    assert batch.response_items.tolist() == [1, 0]
    assert torch.equal(
        batch.response_rows(values),
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
    )


def test_response_item_indices_are_not_caller_mutable() -> None:
    batch = complete_set().select((0,))
    exposed = batch.response_items
    exposed.fill_(0)

    assert batch.response_items.tolist() == [1, 0]
    assert batch.response_rows(torch.tensor([20.0, 10.0])).tolist() == [10.0, 20.0]


@pytest.mark.parametrize(
    ("item_ids", "words", "message"),
    [
        (("i-a", "i-a"), (("a",), ("b",)), "unique"),
        (("i-a",), (("a",),), "missing"),
        (("i-a", "i-b", "i-c"), (("a",), ("b",), ("c",)), "extra"),
    ],
)
def test_inexact_response_to_neural_item_mappings_are_refused(
    item_ids: tuple[str, ...],
    words: tuple[tuple[str, ...], ...],
    message: str,
) -> None:
    design = complete_set().design

    with pytest.raises(ValueError, match=message):
        GroupedJudgmentSet(design, item_ids, words)


@pytest.mark.parametrize("groups", [(), (0, 0), (-1,), (2,)])
def test_invalid_active_groups_are_refused(groups: tuple[int, ...]) -> None:
    with pytest.raises((IndexError, ValueError)):
        complete_set().select(groups)


def test_a_grouped_set_refuses_an_empty_neural_item() -> None:
    with pytest.raises(ValueError, match="has no word"):
        GroupedJudgmentSet(
            complete_set().design,
            ("i-a", "i-b"),
            ((), ("word",)),
        )


def test_a_grouped_set_refuses_a_non_design_before_reading_it() -> None:
    with pytest.raises(TypeError, match="not a judgment design"):
        GroupedJudgmentSet(
            cast("JudgmentDesign", object()),
            ("i-a",),
            (("word",),),
        )


@pytest.mark.parametrize(
    ("item_ids", "words", "message"),
    [
        ("i-a", (("word",),), "identifiers are not a sequence"),
        (("i-a",), "word", "words are not a sequence of sequences"),
        (("i-a",), ("word",), "item's words are not a sequence"),
        (("i-a",), (cast("Sequence[str]", object()),), "not a sequence"),
    ],
)
def test_a_grouped_set_refuses_text_and_nonsequences_at_sequence_boundaries(
    item_ids: Sequence[str],
    words: Sequence[Sequence[str]],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        GroupedJudgmentSet(complete_set().design, item_ids, words)
