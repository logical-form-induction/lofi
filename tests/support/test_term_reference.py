"""Tests for the shared reference signatures used by tests."""

from typing import TYPE_CHECKING, Final

import pytest
from tests.support.term_reference import CALCULUS, categories, formulas

from lofi.terms import Calculus, Operator, Signature

if TYPE_CHECKING:
    from collections.abc import Callable

BUILDERS: Final = (categories, formulas)

NAMES: Final = tuple(build().name for build in BUILDERS)


@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
def test_each_reference_signature_builds_afresh_on_every_call(
    build: Callable[[], Signature],
) -> None:
    here, there = build(), build()

    assert isinstance(here, Signature)
    assert here is not there
    assert here.operators is not there.operators
    assert here.digest == there.digest


def test_the_two_reference_signatures_are_two_tables() -> None:
    assert categories().digest != formulas().digest
    assert categories().name != formulas().name


def test_the_categorial_signature_is_the_first_order_case() -> None:
    signature = categories()

    assert signature.sorts == ("Cat",)
    assert not signature.has_binders
    assert signature.max_arity == 2
    assert {op.payload_sort for op in signature.operators} == {"none"}


def test_the_categorial_signature_carries_three_featured_sentence_types() -> None:
    signature = categories()
    bare = signature.by_name("s")
    featured = [signature.featured("s", each) for each in signature.features["s"]]

    assert signature.features == {"s": ("dcl", "b", "ng")}
    assert len({id(each) for each in featured}) == len(featured)
    assert [signature.erase(each) for each in featured] == [bare] * len(featured)
    assert signature.erase(bare) is bare
    assert {(each.arity, each.sort) for each in featured} == {(bare.arity, bare.sort)}


def test_erasing_the_categorial_signature_leaves_the_primitives_it_declared() -> None:
    signature = categories()
    erased = {signature.erase(op).name for op in signature.operators}

    assert erased == {"np", "n", "s", "fwd", "bwd"}
    assert len(erased) < len(signature.operators)


def test_the_signature_with_binding_covers_arity_zero_through_three() -> None:
    signature = formulas()

    assert sorted({op.arity for op in signature.operators}) == [0, 1, 2, 3]
    assert signature.by_name(CALCULUS.lam).arity == 1
    assert signature.by_name(CALCULUS.app).arity == 2
    assert [op.name for op in signature.by_arity(3)] == ["ite"]


def test_the_codes_group_the_derived_families_then_the_declared_ones() -> None:
    signature = formulas()

    assert [op.name for op in signature.operators] == [
        "#const[Lf]",
        "#bvar[Lf]",
        "#fvar[Lf]",
        CALCULUS.app,
        CALCULUS.lam,
        "ite",
    ]
    assert signature.vocabulary("Lf").names == ("p", "q", "r", "s")


def test_the_calculus_names_the_two_operators_the_kernel_rewrites_with() -> None:
    signature = formulas()

    assert isinstance(CALCULUS, Calculus)
    assert signature.has_binders
    assert signature.by_name(CALCULUS.lam).is_binder
    assert not signature.by_name(CALCULUS.app).is_binder


def test_the_de_bruijn_payload_belongs_to_one_operator_of_the_signature() -> None:
    signature = formulas()
    indexed = [op.name for op in signature.operators if op.payload_sort == "index"]

    assert indexed == ["#bvar[Lf]"]
    assert {op.payload_sort for op in signature.operators} == {
        "none",
        "index",
        "free",
        "constant",
    }


@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
def test_every_operator_of_a_reference_signature_reads_back_by_name_and_code(
    build: Callable[[], Signature],
) -> None:
    signature = build()

    for op in signature.operators:
        assert isinstance(op, Operator)
        assert signature.by_name(op.name) is op
        assert signature.operator(signature.code(op)) is op
