"""Type and category constructors are whatever the document's signature declares.

A signature names each category constructor with its arity and, for a functor,
its direction. The cases declare the slash pair, a single undirected arrow, a
product, a unary modality, a ternary constructor and several of these at
once. Each supported grammar compiles, trains, and can be interpreted into a
typed language whose type constructors mirror the categories'. A grammar
outside the neural chart's declared rank bound is refused at that boundary.
"""

from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import (
    PointFailedError,
    interpretation,
    leaf,
    resolved,
    train,
)

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


def _records(document: JsonValue, path: str) -> tuple[dict[str, JsonValue], ...]:
    """Return a resolved list of declaration records."""
    value = leaf(document, path)

    assert isinstance(value, list)
    records: list[dict[str, JsonValue]] = []

    for record in value:
        assert isinstance(record, dict)
        records.append(record)

    return tuple(records)


def _named_record(document: JsonValue, path: str, name: str) -> dict[str, JsonValue]:
    """Return one named declaration record from a resolved list."""
    for record in _records(document, path):
        if record.get("name") == name:
            return record

    msg = f"{path} has no declaration named {name!r}"
    raise AssertionError(msg)


def _constructor_names(document: JsonValue) -> frozenset[str]:
    """Return the names of the document's declared category constructors."""
    names: set[str] = set()

    for record in _records(
        document, "data.datasets.train.dataset.grammar.signature.constructors"
    ):
        name = record["name"]

        assert isinstance(name, str)
        names.add(name)

    return frozenset(names)


def test_the_slash_pair_trains_a_tiny_point(tmp_path: Path) -> None:
    point = train("slash-point.yaml", tmp_path)

    assert _constructor_names(point.resolved) == {
        "/",
        "\\",
    }


def test_a_single_undirected_arrow_trains_a_tiny_point(tmp_path: Path) -> None:
    point = train("arrow-point.yaml", tmp_path)

    assert _named_record(
        point.resolved,
        "data.datasets.train.dataset.grammar.signature.functors",
        "arrow",
    ) == {
        "name": "arrow",
        "constructor": "arrow",
        "argument_side": "either",
        "opposite": "arrow",
        "result": 0,
        "argument": 1,
    }


def test_a_product_beside_the_slash_pair_trains(tmp_path: Path) -> None:
    point = train("slash-product-point.yaml", tmp_path)

    assert "*" not in {
        record["constructor"]
        for record in _records(
            point.resolved, "data.datasets.train.dataset.grammar.signature.functors"
        )
    }


def test_a_unary_modality_is_read_in_s_expression_categories() -> None:
    document = resolved("modal-point.yaml")

    assert _named_record(
        document, "data.datasets.train.dataset.grammar.signature.constructors", "<>"
    )["kind"] == ["Cat", "Cat"]
    assert (
        leaf(document, "data.datasets.train.dataset.grammar.signature.syntax")
        == "s-expression"
    )


def test_a_slash_pair_a_product_and_a_modality_train_together(
    tmp_path: Path,
) -> None:
    point = train("views-point.yaml", tmp_path)

    assert _constructor_names(point.resolved) == {
        "/",
        "back",
        "*",
        "<>",
    }


def test_a_ternary_rule_is_refused_by_its_action_name(tmp_path: Path) -> None:
    with pytest.raises(
        PointFailedError, match=r"composing action 'apply-two' applies arity 3"
    ):
        train("ternary-point.yaml", tmp_path)


def test_two_slash_pairs_train_as_a_multimodal_grammar(tmp_path: Path) -> None:
    assert train("multimodal-point.yaml", tmp_path).rows


def test_the_ccg_slash_pair_trains_beside_a_unary_modality(tmp_path: Path) -> None:
    assert train("backslash-modal-point.yaml", tmp_path).rows


def test_the_arrow_grammar_is_interpreted_into_typed_meanings() -> None:
    meanings = interpretation("arrow-meanings.yaml")

    assert meanings.actions.rules


def test_a_product_category_erases_to_a_product_type() -> None:
    meanings = interpretation("product-meanings.yaml")
    terms = meanings.actions.resolve("first", None)

    assert len(terms) == 1
    assert meanings.render(terms[0]) == "fst(p0)"
