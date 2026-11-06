"""Tests for declared category-to-type signature morphisms."""

from typing import TYPE_CHECKING

import pytest

from lofi.spec import (
    ConstructorSpec,
    DeclarationError,
    PrefixConstructorNotationSpec,
    RuleSetSpec,
)
from lofi.spec.grammar import (
    FunctorSpec,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureMorphismErasureSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from lofi.typecheck import Erasure


def _assembly() -> GrammarAssembly:
    """Build a grammar exposing functor, unary, binary and ternary constructors."""
    declaration = GrammarDecl(
        name="erasure-cases",
        signature=SignatureSpec(
            primitives=("S", "NP"),
            constructors=tuple(
                ConstructorSpec(
                    name=name,
                    kind=("Cat",) * (arity + 1),
                    notation=PrefixConstructorNotationSpec(spelling=name),
                )
                for name, arity in (
                    ("arrow", 2),
                    ("product", 2),
                    ("box", 1),
                    ("triple", 3),
                )
            ),
            functors=(
                FunctorSpec(
                    name="arrow",
                    constructor="arrow",
                    argument_side="either",
                    opposite="arrow",
                ),
            ),
            syntax="s-expression",
        ),
        rules=RuleSetSpec(name="empty"),
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="sentence", category='("#const[Cat]" "S")'),)
        ),
        goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
    )

    language = registered_language("propositional")

    return GrammarIso().forward(
        declaration,
        language.with_(
            constructors=(
                *language.constructors,
                ConstructorSpec(
                    name="pair",
                    kind=("Ty", "Ty", "Ty"),
                    notation=PrefixConstructorNotationSpec(spelling="pair"),
                ),
            )
        ),
    )


def _erasure() -> tuple[GrammarAssembly, Erasure]:
    """Build the representative arbitrary-arity signature morphism."""
    assembly = _assembly()
    declaration = SignatureMorphismErasureSpec(
        atoms={"S": "t", "NP": "e"},
        constructors={
            "product": "pair(a0, a1)",
            "box": "a0",
            "triple": "pair(a2, pair(a0, a1))",
        },
    )

    return (
        assembly,
        declaration.build(assembly, type_syntax="angle"),
    )


def test_a_declared_morphism_maps_product_to_pair() -> None:
    """Instantiate a non-functor binary constructor with its own image."""
    assembly, erasure = _erasure()
    categories = assembly.store
    types = assembly.language.store
    atom = categories.const(assembly.language.category_sort, "NP")
    category = categories.op(categories.signature.by_name("product"), atom, atom)
    entity = assembly.language.read_type("e")

    assert categories is types
    assert erasure(category) is types.op(
        types.signature.by_name("pair"), entity, entity
    )


def test_a_declared_morphism_maps_a_unary_modality_to_identity() -> None:
    """Instantiate a semantically transparent unary constructor."""
    assembly, erasure = _erasure()
    categories = assembly.store
    sentence = categories.const(assembly.language.category_sort, "S")

    assert erasure(categories.op(categories.signature.by_name("box"), sentence)) is (
        assembly.language.read_type("t")
    )


def test_a_declared_morphism_maps_arbitrary_arity_and_reorders_arguments() -> None:
    """Instantiate a nested target term over three positional metavariables."""
    assembly, erasure = _erasure()
    categories = assembly.store
    types = assembly.language.store
    sentence = categories.const(assembly.language.category_sort, "S")
    entity_category = categories.const(assembly.language.category_sort, "NP")
    category = categories.op(
        categories.signature.by_name("triple"),
        entity_category,
        sentence,
        entity_category,
    )
    entity = assembly.language.read_type("e")
    truth = assembly.language.read_type("t")
    pair = types.signature.by_name("pair")

    assert erasure(category) is types.op(pair, entity, types.op(pair, entity, truth))


def test_functor_images_are_derived_from_their_declared_roles() -> None:
    """Map a functor's argument and result through the language function type."""
    assembly, erasure = _erasure()
    categories = assembly.store
    sentence = categories.const(assembly.language.category_sort, "S")
    entity_category = categories.const(assembly.language.category_sort, "NP")
    category = categories.op(
        categories.signature.by_name("arrow"), sentence, entity_category
    )

    assert erasure(category) is assembly.language.read_type("e -> t")


def test_a_declaration_cannot_restate_a_functor_image() -> None:
    """Keep functor erasure at its grammar-owned declaration site."""
    assembly = _assembly()
    declaration = SignatureMorphismErasureSpec(
        atoms={"S": "t", "NP": "e"},
        constructors={
            "arrow": "a1 -> a0",
            "product": "pair(a0, a1)",
            "box": "a0",
            "triple": "pair(a2, pair(a0, a1))",
        },
    )

    with pytest.raises(DeclarationError, match=r"restates.*\('arrow',\)"):
        declaration.build(assembly, type_syntax="angle")
