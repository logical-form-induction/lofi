"""Logical-form typing against category erasure."""

from typing import TYPE_CHECKING

import pytest

from lofi.spec import ConstructorSpec, InfixConstructorNotationSpec
from lofi.spec.grammar import (
    FunctorSpec,
    SignatureSpec,
)
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language
from lofi.typecheck import (
    CategoryTypeMismatchError,
    Erasure,
    check_logical_form_category,
)

if TYPE_CHECKING:
    from lofi.spec.language import LanguageAssembly


def _case() -> tuple[LanguageAssembly, Erasure]:
    """Build a document-shaped language, grammar and erasure."""
    base = registered_language("propositional")
    language = base.with_(
        constants=(
            *base.constants,
            ConstantSpec(name="john", type="e"),
            ConstantSpec(name="sleep", type="e -> t"),
        )
    )
    signature = SignatureSpec(
        primitives=("S", "NP"),
        constructors=tuple(
            ConstructorSpec(
                name=name,
                kind=("Cat", "Cat", "Cat"),
                notation=InfixConstructorNotationSpec(
                    spelling=name, precedence=10, associativity="left"
                ),
            )
            for name in ("/", "\\")
        ),
        functors=tuple(
            FunctorSpec(
                name=name,
                constructor=name,
                argument_side=side,
                opposite=opposite,
            )
            for name, side, opposite in (
                ("/", "right", "\\"),
                ("\\", "left", "/"),
            )
        ),
    )
    assembly = LanguageIso().extend(language, signature.extend)
    store = assembly.store
    a0 = store.fvar(assembly.type_sort, "a0")
    a1 = store.fvar(assembly.type_sort, "a1")
    function = store.signature.by_name(assembly.function_type)
    constructor_image = store.op(function, a1, a0)
    erasure = Erasure(
        store,
        {"S": assembly.read_type("t"), "NP": assembly.read_type("e")},
        {"/": constructor_image, "\\": constructor_image},
        category_sort=assembly.category_sort,
        function_type=assembly.function_type,
    )
    return assembly, erasure


def test_the_bridge_accepts_a_form_at_its_category_erasure() -> None:
    language, erasure = _case()
    category = language.store.const(language.category_sort, "S")
    logical_form = language.read("sleep(john)")

    assert (
        check_logical_form_category(language, erasure, logical_form, category)
        is category
    )


def test_a_mismatch_names_the_logical_form_and_category_types() -> None:
    language, erasure = _case()
    category = language.store.const(language.category_sort, "NP")
    logical_form = language.read("sleep(john)")

    with pytest.raises(CategoryTypeMismatchError) as raised:
        check_logical_form_category(language, erasure, logical_form, category)

    assert raised.value.logical_form_type is language.read_type("t")
    assert raised.value.category_type is language.read_type("e")
    assert "logical-form type 't'" in str(raised.value)
    assert "category type 'e'" in str(raised.value)
