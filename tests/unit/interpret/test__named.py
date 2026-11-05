"""Tests for grammar-declared named interpretations."""

from typing import TYPE_CHECKING, cast

import pytest

from lofi.interpret import (
    CategoryInterpretation,
    CategoryReading,
    DependencyInterpretation,
    HeadDependencies,
    MeaningInterpretation,
    MeaningReading,
    StringInterpretation,
    StringReading,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.terms import Term


def _meaning() -> MeaningReading[Term]:
    """Return an opaque meaning reading for declaration binding tests."""
    return cast("MeaningReading[Term]", object())


def test_each_named_declaration_binds_its_runtime_interpretation() -> None:
    meaning = _meaning()

    assert type(CategoryInterpretation("syntax").bind(meaning)) is CategoryReading
    assert type(StringInterpretation("surface").bind(meaning)) is StringReading
    assert type(DependencyInterpretation("deps", {"fa": 0}).bind(meaning)) is (
        HeadDependencies
    )
    assert MeaningInterpretation("logical-form").bind(meaning) is meaning


def test_a_named_declaration_exposes_its_fingerprint_settings() -> None:
    declaration = DependencyInterpretation("deps", {"fa": 0, "ba": 1})

    assert declaration.settings == {
        "name": "deps",
        "kind": "dependency",
        "heads": {"fa": 0, "ba": 1},
    }

    with pytest.raises(TypeError):
        cast("dict[str, JsonValue]", declaration.settings)["name"] = "other"


def test_a_named_declaration_needs_a_name() -> None:
    with pytest.raises(ValueError, match="interpretation"):
        StringInterpretation("")
