"""Boundary tests for the documents a publication of Layers records reads."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.run._boundary.publishing import (
    PublishSelection,
    TokenizationMapping,
    TokenizationRecord,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_a_selection_names_evaluator_roles_a_split_and_a_formalism() -> None:
    selection = PublishSelection.model_validate(
        {"evaluators": ["decoded", "beams"], "split": "test", "formalism": "ccg"}
    )

    assert selection.evaluators == ("decoded", "beams")
    assert selection.split == "test"
    assert selection.formalism == "ccg"


@pytest.mark.parametrize(
    ("document", "named"),
    [
        ({"evaluators": [], "split": "test", "formalism": "ccg"}, "no evaluator"),
        (
            {"evaluators": ["decoded", "decoded"], "split": "test", "formalism": "ccg"},
            "'decoded'",
        ),
        ({"evaluators": [""], "split": "test", "formalism": "ccg"}, "role"),
        ({"evaluators": ["decoded"], "split": "", "formalism": "ccg"}, "split"),
        ({"evaluators": ["decoded"], "split": "test", "formalism": ""}, "formalism"),
        (
            {
                "evaluators": ["decoded"],
                "split": "test",
                "formalism": "ccg",
                "items": ["a"],
            },
            "unknown field 'items'",
        ),
    ],
)
def test_a_selection_that_names_nothing_or_a_role_twice_is_refused(
    document: dict[str, JsonValue], named: str
) -> None:
    with pytest.raises(dx.ValidationError, match=named):
        PublishSelection.model_validate(document)


def test_a_mapping_joins_each_item_to_one_tokenization() -> None:
    mapping = TokenizationMapping.model_validate(
        {
            "tokenizations": [
                {"item": "item-1", "tokenization": "tokenization-1"},
                {"item": "item-2", "tokenization": "tokenization-2"},
            ]
        }
    )

    assert mapping.tokenizations == (
        TokenizationRecord(item="item-1", tokenization="tokenization-1"),
        TokenizationRecord(item="item-2", tokenization="tokenization-2"),
    )


@pytest.mark.parametrize(
    ("records", "named"),
    [
        (
            [
                {"item": "item-1", "tokenization": "tokenization-1"},
                {"item": "item-1", "tokenization": "tokenization-2"},
            ],
            "'item-1'",
        ),
        ([{"item": "", "tokenization": "tokenization-1"}], "item"),
        ([{"item": "item-1", "tokenization": ""}], "tokenization"),
    ],
)
def test_a_mapping_naming_an_item_twice_or_an_empty_identifier_is_refused(
    records: list[JsonValue], named: str
) -> None:
    with pytest.raises(dx.ValidationError, match=named):
        TokenizationMapping.model_validate({"tokenizations": records})
