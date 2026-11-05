from copy import copy, deepcopy
from pickle import dumps
from types import MappingProxyType
from typing import Final

import pytest

from lofi.terms import Operator, Signature, Term, TermStore, Vocabulary
from lofi.typecheck import Erasure

SIGNATURE: Final = Signature(
    "signature-morphism",
    ("Cat", "Ty"),
    (
        Operator("product", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("unary", "Cat", arg_sorts=("Cat",)),
        Operator("triple", "Cat", arg_sorts=("Cat", "Cat", "Cat")),
        Operator("->", "Ty", arg_sorts=("Ty", "Ty")),
        Operator("pair", "Ty", arg_sorts=("Ty", "Ty")),
        Operator("triple-image", "Ty", arg_sorts=("Ty", "Ty", "Ty")),
        Operator("unary-type", "Ty", arg_sorts=("Ty",)),
    ),
    vocabularies=(
        Vocabulary("Cat", ("S", "S[dcl]", "NP", "NP[conj]")),
        Vocabulary("Ty", ("s", "np")),
    ),
    constant_cores={"Cat": {"S[dcl]": "S", "NP[conj]": "NP"}},
    open_sorts=("Ty",),
)
STORE: Final = TermStore(SIGNATURE)
S: Final = STORE.const("Cat", "S")
S_DCL: Final = STORE.const("Cat", "S[dcl]")
NP: Final = STORE.const("Cat", "NP")
NP_CONJ: Final = STORE.const("Cat", "NP[conj]")
S_TYPE: Final = STORE.const("Ty", "s")
NP_TYPE: Final = STORE.const("Ty", "np")
A0: Final = STORE.fvar("Ty", "a0")
A1: Final = STORE.fvar("Ty", "a1")
A2: Final = STORE.fvar("Ty", "a2")
PAIR: Final = SIGNATURE.by_name("pair")
TRIPLE_IMAGE: Final = SIGNATURE.by_name("triple-image")
PRODUCT: Final = SIGNATURE.by_name("product")
UNARY: Final = SIGNATURE.by_name("unary")
TRIPLE: Final = SIGNATURE.by_name("triple")
ATOMS: Final = {"S": S_TYPE, "NP": NP_TYPE}
CONSTRUCTORS: Final = {
    "product": STORE.op(PAIR, A1, A0),
    "unary": A0,
    "triple": STORE.op(TRIPLE_IMAGE, A2, A1, A0),
}


def erasure() -> Erasure:
    return Erasure(
        STORE,
        ATOMS,
        CONSTRUCTORS,
        category_sort="Cat",
        function_type="->",
    )


def test_a_signature_morphism_exposes_one_store_and_frozen_declarations() -> None:
    held = erasure()

    assert held.store is STORE
    assert held.category_sort == "Cat"
    assert held.target_sort == "Ty"
    assert held.function_type == "->"
    assert dict(held.atoms) == ATOMS
    assert dict(held.constructors) == CONSTRUCTORS
    assert type(held.atoms) is MappingProxyType
    assert type(held.constructors) is MappingProxyType
    assert repr(held) == (
        "Erasure(store='signature-morphism', category_sort='Cat', atoms=2, "
        "constructors=3, function_type='->')"
    )
    assert copy(held) is held
    assert deepcopy(held) is held

    with pytest.raises(AttributeError, match="immutable"):
        held.__setattr__("_store", STORE)

    with pytest.raises(AttributeError, match="immutable"):
        held.__delattr__("_store")

    with pytest.raises(TypeError, match="cannot be pickled"):
        dumps(held)


def test_atomic_features_erase_through_their_declared_constant_cores() -> None:
    held = erasure()

    assert held(S) is S_TYPE
    assert held(S_DCL) is S_TYPE
    assert held(NP) is NP_TYPE
    assert held(NP_CONJ) is NP_TYPE


def test_constructor_images_support_product_unary_and_arbitrary_arity_maps() -> None:
    held = erasure()
    product = STORE.op(PRODUCT, S_DCL, NP_CONJ)
    unary = STORE.op(UNARY, product)
    triple = STORE.op(TRIPLE, S, NP, S_DCL)
    paired = STORE.op(PAIR, NP_TYPE, S_TYPE)

    assert held(product) is paired
    assert held(unary) is paired
    assert held(triple) is STORE.op(TRIPLE_IMAGE, S_TYPE, NP_TYPE, S_TYPE)
    assert held.image("product", (S_TYPE, NP_TYPE)) is STORE.op(PAIR, NP_TYPE, S_TYPE)


@pytest.mark.parametrize(
    ("atoms", "constructors", "fault"),
    [
        ({"S": S_TYPE}, CONSTRUCTORS, "atom map coverage"),
        ({**ATOMS, "extra": S_TYPE}, CONSTRUCTORS, "atom map coverage"),
        (ATOMS, {"product": CONSTRUCTORS["product"]}, "constructor map coverage"),
        (
            ATOMS,
            {**CONSTRUCTORS, "extra": CONSTRUCTORS["unary"]},
            "constructor map coverage",
        ),
    ],
)
def test_a_signature_morphism_requires_exact_source_coverage(
    atoms: dict[str, Term], constructors: dict[str, Term], fault: str
) -> None:
    with pytest.raises(ValueError, match=fault):
        Erasure(
            STORE,
            atoms,
            constructors,
            category_sort="Cat",
            function_type="->",
        )


def test_atom_images_must_be_closed_target_terms_in_the_one_store() -> None:
    foreign = TermStore(SIGNATURE)

    for image, fault in (
        (foreign.const("Ty", "s"), "outside the language store lineage"),
        (S, "sort 'Cat', not target sort 'Ty'"),
        (A0, "not closed"),
    ):
        with pytest.raises(ValueError, match=fault):
            Erasure(
                STORE,
                {**ATOMS, "S": image},
                CONSTRUCTORS,
                category_sort="Cat",
                function_type="->",
            )


def test_constructor_images_must_use_only_their_declared_parameters() -> None:
    undeclared = STORE.fvar("Ty", "outside")

    with pytest.raises(ValueError, match="undeclared metavariables"):
        Erasure(
            STORE,
            ATOMS,
            {**CONSTRUCTORS, "unary": undeclared},
            category_sort="Cat",
            function_type="->",
        )


def test_the_function_type_must_be_a_pure_binary_target_endofunctor() -> None:
    with pytest.raises(ValueError, match="not a pure binary endofunctor"):
        Erasure(
            STORE,
            ATOMS,
            CONSTRUCTORS,
            category_sort="Cat",
            function_type="unary-type",
        )


def test_erasing_and_instantiating_reject_foreign_sorts_stores_and_arities() -> None:
    held = erasure()
    foreign = TermStore(SIGNATURE)

    with pytest.raises(ValueError, match="outside the language store lineage"):
        held(foreign.const("Cat", "S"))

    with pytest.raises(ValueError, match="not category sort"):
        held(S_TYPE)

    with pytest.raises(ValueError, match="not declared"):
        held.image("undeclared", ())

    with pytest.raises(ValueError, match="expects 2 arguments, not 1"):
        held.image("product", (S_TYPE,))

    with pytest.raises(ValueError, match="outside the target store lineage"):
        held.image("product", (S_TYPE, foreign.const("Ty", "np")))

    with pytest.raises(ValueError, match="sort 'Cat', not target sort 'Ty'"):
        held.image("product", (S_TYPE, S))
