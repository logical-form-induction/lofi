"""Tests for named interpretation declarations."""

import json
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest
from tests.unit.spec.grammar.test__grammar import _declaration

from lofi._types import HeadSelection
from lofi.interpret import (
    CategoryInterpretation,
    DependencyInterpretation,
    MeaningInterpretation,
    StringInterpretation,
)
from lofi.spec.grammar import (
    CategoryInterpretationSpec,
    DependencyInterpretationSpec,
    MeaningInterpretationSpec,
    NamedInterpretationSpec,
    StringInterpretationSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"name": "syntax", "reading": {"kind": "category"}}, "category"),
        ({"name": "surface", "reading": {"kind": "string"}}, "string"),
        ({"name": "deps", "reading": {"kind": "dependency"}}, "dependency"),
        ({"name": "logical-form", "reading": {"kind": "meaning"}}, "meaning"),
    ],
)
def test_a_document_declares_each_named_interpretation(
    document: dict[str, JsonValue], expected: str
) -> None:
    declaration = NamedInterpretationSpec.model_validate(document)

    assert declaration.name == document["name"]
    assert declaration.reading.kind == expected


def test_a_named_interpretation_needs_a_nonempty_name() -> None:
    with pytest.raises(dx.ValidationError, match="name"):
        NamedInterpretationSpec(name="", reading=StringInterpretationSpec())


def test_the_four_builtin_readings_round_trip_through_json() -> None:
    declarations = (
        NamedInterpretationSpec(name="syntax", reading=CategoryInterpretationSpec()),
        NamedInterpretationSpec(name="surface", reading=StringInterpretationSpec()),
        NamedInterpretationSpec(name="deps", reading=DependencyInterpretationSpec()),
        NamedInterpretationSpec(
            name="logical-form", reading=MeaningInterpretationSpec()
        ),
    )

    for declaration in declarations:
        assert (
            NamedInterpretationSpec.model_validate_json(declaration.model_dump_json())
            == declaration
        )


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        (
            NamedInterpretationSpec(
                name="syntax", reading=CategoryInterpretationSpec()
            ),
            CategoryInterpretation,
        ),
        (
            NamedInterpretationSpec(name="surface", reading=StringInterpretationSpec()),
            StringInterpretation,
        ),
        (
            NamedInterpretationSpec(
                name="deps", reading=DependencyInterpretationSpec()
            ),
            DependencyInterpretation,
        ),
        (
            NamedInterpretationSpec(
                name="logical-form", reading=MeaningInterpretationSpec()
            ),
            MeaningInterpretation,
        ),
    ],
)
def test_each_installed_interpretation_builds_from_a_document(
    declaration: NamedInterpretationSpec,
    expected: type[
        CategoryInterpretation
        | StringInterpretation
        | DependencyInterpretation
        | MeaningInterpretation
    ],
) -> None:
    built = declaration.build(
        {
            "combine": HeadSelection("functor", 1),
            "unary": HeadSelection("pass-through", None),
        }
    )

    assert type(built) is expected

    if type(built) is DependencyInterpretation:
        assert built.settings["heads"] == {"combine": 1}


def test_a_grammar_refuses_two_interpretations_under_one_name() -> None:
    document = json.loads(_declaration().model_dump_json())
    document["interpretations"] = [
        {"name": "selected", "reading": {"kind": "string"}},
        {"name": "selected", "reading": {"kind": "meaning"}},
    ]

    with pytest.raises(dx.ValidationError, match="one interpretation per name"):
        type(_declaration()).model_validate(document)
