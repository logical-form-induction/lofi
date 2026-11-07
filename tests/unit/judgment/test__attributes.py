"""Tests for bead property expressions over item attributes."""

from uuid import UUID

import pytest
from bead.items.item import Item

from lofi.judgment._attributes import (
    NO_FACTORS,
    AttributedItem,
    AttributeExpression,
    ItemFactors,
    item_key,
    refuse_uncarried,
)
from lofi.judgment._design import DesignError


def _item(**metadata: str | float | bool) -> Item:
    return Item(
        id=UUID(int=7), item_template_id=UUID(int=0), item_metadata=dict(metadata)
    )


def test_an_expression_reads_an_item_attribute_through_its_metadata() -> None:
    expression = AttributeExpression("item['group']", role="a stratum")

    assert expression.level(_item(group="g")) == "g"
    assert expression.levels(_item(group="g")) == ("g",)
    assert item_key(_item(group="g")) == str(UUID(int=7))


def test_an_expression_declared_to_yield_several_levels_reads_a_list() -> None:
    expression = AttributeExpression(
        "[item['left'], item['right']]", role="a constraint", many=True
    )

    assert expression.levels(_item(left="a", right="b")) == ("a", "b")


def test_a_single_level_expression_refuses_a_list() -> None:
    expression = AttributeExpression("[item['left']]", role="a stratum")

    with pytest.raises(ValueError, match="not a level name"):
        expression.levels(_item(left="a"))


def test_an_unparsable_expression_is_refused_when_declared() -> None:
    with pytest.raises(ValueError, match="not a bead expression"):
        AttributeExpression("item[", role="a stratum")


def test_an_attribute_no_item_carries_is_refused_before_any_rule_reads_it() -> None:
    expression = AttributeExpression("item['group']", role="a stratum")

    with pytest.raises(ValueError, match="carries other"):
        refuse_uncarried((expression,), (_item(group="g"), _item(other="x")))


def test_a_corpus_item_record_is_read_as_a_bead_item_is() -> None:
    record = AttributedItem("s-1", {"group": "g"}, {"stimulus": "A text."})

    assert item_key(record) == "s-1"
    assert AttributeExpression("item['group']", role="a rule").level(record) == "g"
    assert record.rendered_elements == {"stimulus": "A text."}


def test_factors_read_each_level_from_the_item_a_row_names() -> None:
    factors = ItemFactors(
        {"kind": "item['k']", "size": "item['s']"},
        (AttributedItem("a", {"k": "x", "s": "1"}, {}), _item(k="y", s="2")),
    )

    assert factors.names == ("kind", "size")
    assert factors.levels("a") == {"kind": "x", "size": "1"}
    assert factors.levels(str(UUID(int=7))) == {"kind": "y", "size": "2"}
    with pytest.raises(DesignError, match="no item has the identity 'b'"):
        factors.levels("b")


def test_factors_refuse_an_uncarried_attribute_when_declared() -> None:
    with pytest.raises(ValueError, match="factor 'kind'"):
        ItemFactors({"kind": "item['k']"}, (AttributedItem("a", {"s": "1"}, {}),))


def test_factors_refuse_two_items_with_one_identity() -> None:
    with pytest.raises(ValueError, match="share the identity"):
        ItemFactors({}, (AttributedItem("a", {}, {}), AttributedItem("a", {}, {})))


def test_no_factors_read_no_level_and_need_no_item() -> None:
    assert NO_FACTORS.names == ()
    assert NO_FACTORS.levels("anything") == {}


def test_a_numeric_expression_reads_a_finite_number() -> None:
    expression = AttributeExpression("item['x']", role="a numeric axis", numeric=True)

    assert expression.number(_item(x=2)) == 2.0
    expression.check(_item(x=0.5))
    for refused in (_item(x="two"), _item(x=True), _item(x=float("inf"))):
        with pytest.raises(ValueError, match="not a finite number"):
            expression.check(refused)
