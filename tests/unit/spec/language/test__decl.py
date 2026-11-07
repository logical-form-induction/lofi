import json
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.registry import _registry as registry_module
from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec._compile import canonical_declaration_bytes
from lofi.spec.language import (
    AtomicTypeSpec,
    BinderSpec,
    ConstantSpec,
    LanguageDecl,
    SortBoundSpec,
)
from lofi.spec.language import _decl as declaration_module

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_language_declaration_rejects_duplicate_constant_names() -> None:
    with pytest.raises(dx.ValidationError, match="declared twice"):
        LanguageDecl(
            name="duplicate",
            atomic_types=(AtomicTypeSpec(name="t"),),
            constructors=(
                ConstructorSpec(
                    name="->",
                    kind=("Ty", "Ty", "Ty"),
                    notation=InfixConstructorNotationSpec(
                        spelling="->", precedence=1, associativity="right"
                    ),
                ),
            ),
            constants=(
                ConstantSpec(name="p", type="t"),
                ConstantSpec(name="p", type="t"),
            ),
            binders=BinderSpec(style="curry"),
        )


def test_each_sort_declares_its_one_neural_representation_bound() -> None:
    declaration = LanguageDecl.model_validate(_declared())

    assert declaration.bounds == {
        "Ty": SortBoundSpec(layout="complete-tree", depth=4, nodes=None),
        "Tm": SortBoundSpec(layout="preorder-arity", depth=8, nodes=24),
        "Cat": SortBoundSpec(layout="complete-tree", depth=4, nodes=None),
    }

    with pytest.raises(dx.ValidationError, match="bounds match the language sorts"):
        LanguageDecl.model_validate(
            _declared(bounds={"Ty": {"layout": "complete-tree", "depth": 3}})
        )


def test_the_compile_envelope_is_derived_eagerly_once_per_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    canonical = declaration_module.canonical_declaration_bytes

    def counted(source: dx.Model, /) -> bytes:
        nonlocal calls
        calls += 1
        return canonical(source)

    monkeypatch.setattr(declaration_module, "canonical_declaration_bytes", counted)

    declaration = LanguageDecl.model_validate(_declared(name="compile-envelope-test"))
    envelope = declaration.compile_envelope

    assert calls == 1
    assert declaration.compile_envelope is envelope

    reparsed = LanguageDecl.model_validate_json(declaration.model_dump_json())

    assert reparsed.compile_envelope is envelope
    assert calls == 1

    changed = declaration.with_(name="changed")

    assert calls == 2
    assert changed.compile_envelope is changed.compile_envelope
    assert changed.compile_envelope is not envelope
    assert changed.compile_envelope.canonical_declaration != (
        envelope.canonical_declaration
    )


def test_language_plugin_discovery_is_shared_across_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [0]

    def discover() -> tuple[()]:
        calls[0] += 1
        return ()

    monkeypatch.setattr(registry_module, "entry_points", discover)

    LanguageDecl.model_validate(_declared(name="first"))
    LanguageDecl.model_validate(_declared(name="second"))

    assert calls == [1]


def test_the_language_compile_envelope_is_frozen_and_not_declared() -> None:
    declaration = LanguageDecl.model_validate(_declared())
    envelope = declaration.compile_envelope

    assert envelope.source_version == "lofi-language-decl"
    assert envelope.target_version == "lofi-language-native"
    assert set(envelope.model_dump()) == {
        "source_version",
        "target_version",
        "canonical_declaration",
    }
    assert "compile_envelope" not in declaration.model_dump()
    assert declaration.model_dump(include={"compile_envelope"}) == {}
    assert "compile_envelope" not in json.loads(declaration.model_dump_json())

    with pytest.raises(AttributeError, match="immutable"):
        envelope.source_version = "other"

    assert canonical_declaration_bytes(
        LanguageDecl.model_validate_json(envelope.canonical_declaration)
    ) == canonical_declaration_bytes(declaration)


def _declared(**changes: JsonValue) -> dict[str, JsonValue]:
    return {
        "name": "pairs",
        "atomic_types": [{"name": "e"}, {"name": "t"}],
        "constructors": [
            {
                "name": "*",
                "kind": ["Ty", "Ty", "Ty"],
                "notation": {
                    "style": "infix",
                    "spelling": "*",
                    "precedence": 2,
                    "associativity": "left",
                },
            },
            {
                "name": "->",
                "kind": ["Ty", "Ty", "Ty"],
                "notation": {
                    "style": "infix",
                    "spelling": "->",
                    "precedence": 1,
                    "associativity": "right",
                },
            },
        ],
        "constants": [{"name": "p", "type": "e -> t"}],
        "binders": {"style": "curry"},
        **changes,
    }


def test_a_constant_class_is_an_open_name() -> None:
    language = LanguageDecl.model_validate(
        _declared(
            constants=[{"name": "p", "type": "e -> t", "classes": ["evidential"]}]
        )
    )

    assert language.constants[0].classes == ("evidential",)


def test_a_logical_constant_selects_one_semantics_route() -> None:
    defined = ConstantSpec(name="both", type="t -> t -> t", definition=r"\p. p")
    postulated = ConstantSpec(
        name="most",
        type="(e -> t) -> (e -> t) -> t",
        semantics="most",
    )

    assert defined.definition == r"\p. p"
    assert postulated.semantics == "most"

    with pytest.raises(dx.ValidationError, match="cannot also declare"):
        ConstantSpec(
            name="both",
            type="t",
            definition="p",
            semantics="most",
        )


def test_the_function_type_names_a_declared_binary_constructor() -> None:
    assert LanguageDecl.model_validate(_declared()).function_type == "->"

    with pytest.raises(dx.ValidationError, match="names no declared binary"):
        LanguageDecl.model_validate(_declared(function_type="=>"))

    with pytest.raises(dx.ValidationError, match="names no declared binary"):
        LanguageDecl.model_validate(
            _declared(
                constructors=[
                    {
                        "name": "->",
                        "kind": ["Ty", "Ty"],
                        "notation": {"style": "prefix", "spelling": "->"},
                    },
                    {
                        "name": "*",
                        "kind": ["Ty", "Ty", "Ty"],
                        "notation": {
                            "style": "infix",
                            "spelling": "*",
                            "precedence": 2,
                            "associativity": "left",
                        },
                    },
                ]
            )
        )
