"""Tests for named term presentation."""

from typing import Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.term_strategies import terms

from lofi.terms import (
    ArityError,
    BindingAtomCountError,
    BindingSite,
    ForeignTermError,
    Nominal,
    NotLocallyClosedError,
    Operator,
    Signature,
    Term,
    TermStore,
    UnknownConstantError,
    Vocabulary,
    render,
    to_core,
    to_nominal,
)

EXPRESSION: Final = "Expression"


def scoping(*positions: tuple[int, ...]) -> tuple[BindingSite, ...]:
    """Return one binding site of the expression sort per position tuple."""
    return tuple(BindingSite(each, EXPRESSION) for each in positions)


UNIT: Final = Operator("unit", EXPRESSION)
PAIR: Final = Operator(
    "pair",
    EXPRESSION,
    arg_sorts=(EXPRESSION, EXPRESSION),
)
SCOPE: Final = Operator(
    "scope",
    EXPRESSION,
    arg_sorts=(EXPRESSION,),
    binding=scoping((0,)),
)
SCOPE_TWO: Final = Operator(
    "scope-two",
    EXPRESSION,
    arg_sorts=(EXPRESSION,),
    binding=scoping((0,), (0,)),
)
SCOPE_FOUR: Final = Operator(
    "scope-four",
    EXPRESSION,
    arg_sorts=(EXPRESSION,),
    binding=scoping((0,), (0,), (0,), (0,)),
)
BRANCH: Final = Operator(
    "branch",
    EXPRESSION,
    arg_sorts=(EXPRESSION, EXPRESSION),
    binding=scoping((0,), (1,)),
)
MIXED: Final = Operator(
    "mixed",
    EXPRESSION,
    arg_sorts=(EXPRESSION, EXPRESSION, EXPRESSION),
    binding=scoping((0,), (0,), (2,)),
)
SHARED: Final = Operator(
    "shared",
    EXPRESSION,
    arg_sorts=(EXPRESSION, EXPRESSION),
    binding=scoping((0, 1)),
)
SIGNATURE: Final = Signature(
    "presentation",
    (EXPRESSION,),
    (UNIT, PAIR, SCOPE, SCOPE_TWO, SCOPE_FOUR, BRANCH, MIXED, SHARED),
    vocabularies=(Vocabulary(EXPRESSION, ("free0", "free1", "free2")),),
)
BOUND: Final = SIGNATURE.bound(EXPRESSION)
FREE: Final = SIGNATURE.free(EXPRESSION)
SYMBOL: Final = SIGNATURE.constant(EXPRESSION)


NAMES: Final = (
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
)
ALPHA_NAMES: Final = tuple(f"alpha{index}" for index in range(len(NAMES)))


STORE: Final = TermStore(SIGNATURE)


def bound(index: int) -> Term:
    """Build one bound variable."""
    return STORE.op(BOUND, payload=index)


def free(name: str) -> Term:
    """Build one free variable."""
    return STORE.fvar(EXPRESSION, name)


def symbol(name: str) -> Term:
    """Build one symbol."""
    return STORE.const(EXPRESSION, name)


def paired(nodes: tuple[Term, ...]) -> Term:
    """Build a right-branching pair from one or more nodes."""
    head, *tail = nodes

    if not tail:
        return head

    return STORE.op(PAIR, head, paired(tuple(tail)))


@given(terms(STORE, sort=EXPRESSION, max_size=12, closed=True))
@settings(max_examples=200, deadline=None)
def test_core_round_trip_preserves_interned_identity(generated: Term) -> None:
    nominal = to_nominal(STORE, generated, NAMES)

    assert to_core(STORE, nominal) is generated


@given(terms(STORE, sort=EXPRESSION, max_size=12, closed=True))
@settings(max_examples=200, deadline=None)
def test_nominal_round_trip_is_alpha_canonical_and_idempotent(
    generated: Term,
) -> None:
    canonical = to_nominal(STORE, generated, NAMES)
    alpha_variant = to_nominal(STORE, generated, ALPHA_NAMES)
    once = to_nominal(STORE, to_core(STORE, alpha_variant), NAMES)
    twice = to_nominal(STORE, to_core(STORE, once), NAMES)

    assert once == canonical
    assert twice == once


@given(
    binder=st.sampled_from((SCOPE, SCOPE_TWO, SCOPE_FOUR)),
    generated=terms(STORE, sort=EXPRESSION, max_size=8, closed=True),
)
@settings(max_examples=200, deadline=None)
def test_generated_round_trips_cover_nonvacuous_binders(
    binder: Operator,
    generated: Term,
) -> None:
    body = paired(
        (*tuple(bound(index) for index in range(len(binder.binding))), generated)
    )
    term = STORE.op(binder, body)
    nominal = to_nominal(STORE, term, NAMES)

    assert to_core(STORE, nominal) is term
    assert len(nominal.binders) == len(binder.binding)


@given(
    outer_name=st.sampled_from(ALPHA_NAMES),
    inner_name=st.sampled_from(ALPHA_NAMES),
    free_name=st.sampled_from(ALPHA_NAMES),
)
@settings(max_examples=200, deadline=None)
def test_independent_nested_nominal_terms_canonicalize(
    outer_name: str,
    inner_name: str,
    free_name: str,
) -> None:
    inner_body = Nominal(
        PAIR,
        (
            Nominal(FREE, payload=inner_name),
            Nominal(FREE, payload=outer_name),
        ),
    )
    inner = Nominal(SCOPE, (inner_body,), binders=(inner_name,))
    outer_body = Nominal(
        PAIR,
        (Nominal(FREE, payload=outer_name), inner),
    )
    scoped = Nominal(SCOPE, (outer_body,), binders=(outer_name,))
    independent = Nominal(PAIR, (scoped, Nominal(FREE, payload=free_name)))
    core = to_core(STORE, independent)
    canonical = to_nominal(STORE, core, NAMES)

    assert to_core(STORE, canonical) is core


def test_alpha_variants_collapse_only_on_the_core_side() -> None:
    x = Nominal(FREE, payload="x")
    y = Nominal(FREE, payload="y")
    first = Nominal(SCOPE, (x,), binders=("x",))
    second = Nominal(SCOPE, (y,), binders=("y",))

    assert first != second
    assert to_core(STORE, first) is to_core(STORE, second)
    assert to_nominal(STORE, to_core(STORE, first), NAMES) == to_nominal(
        STORE,
        to_core(STORE, second),
        NAMES,
    )


def test_nominal_nodes_are_frozen_slotted_structural_values() -> None:
    first = Nominal(UNIT)
    second = Nominal(UNIT)

    assert first == second
    assert hash(first) == hash(second)
    assert not hasattr(first, "__dict__")

    with pytest.raises(AttributeError, match="immutable"):
        first.payload = "changed"


def test_each_argument_has_its_declared_binder_count() -> None:
    leaf = Nominal(UNIT)

    with pytest.raises(ArityError):
        Nominal(PAIR, (leaf,))

    with pytest.raises(BindingAtomCountError):
        Nominal(SCOPE, (leaf,), binders=("x", "y"))

    with pytest.raises(BindingAtomCountError):
        Nominal(SCOPE, (leaf,), binders=())

    with pytest.raises(ValueError, match="share a name"):
        Nominal(SCOPE_TWO, (leaf,), binders=("x", "x"))

    assert Nominal(BRANCH, (leaf, leaf), binders=("x", "x")).binders == ("x", "x")


def test_nominal_nodes_refuse_de_bruijn_payloads() -> None:
    with pytest.raises(ValueError, match="cannot hold a de Bruijn index"):
        Nominal(BOUND, payload="x")


def test_names_follow_argument_binder_counts_and_reuse_across_siblings() -> None:
    first = STORE.op(PAIR, bound(0), bound(1))
    middle = STORE.op(UNIT)
    last = bound(0)
    term = STORE.op(MIXED, first, middle, last)
    nominal = to_nominal(STORE, term, NAMES)

    assert nominal.binders == ("x", "y", "x")
    assert (nominal.scoped(0), nominal.scoped(1), nominal.scoped(2)) == (
        ("x", "y"),
        (),
        ("x",),
    )
    assert nominal.args[0].args[0].payload == "x"
    assert nominal.args[0].args[1].payload == "y"
    assert nominal.args[2].payload == "x"
    assert to_core(STORE, nominal) is term


def test_nested_sibling_scopes_reuse_depth_first_names() -> None:
    body = STORE.op(SCOPE, STORE.op(PAIR, bound(1), bound(0)))
    term = STORE.op(BRANCH, body, body)
    nominal = to_nominal(STORE, term, NAMES)

    assert nominal.binders == ("x", "x")
    assert nominal.args[0].binders == ("y",)
    assert nominal.args[1].binders == ("y",)
    assert nominal.args[0].args[0].args[0].payload == "x"
    assert nominal.args[0].args[0].args[1].payload == "y"


def test_canonical_binders_avoid_free_variable_names() -> None:
    term = STORE.op(SCOPE, STORE.op(PAIR, bound(0), free("x")))
    nominal = to_nominal(STORE, term, ("x", "y"))

    assert nominal.binders == ("y",)
    assert nominal.args[0].args[0].payload == "y"
    assert nominal.args[0].args[1].payload == "x"
    assert to_core(STORE, nominal) is term


def test_binders_may_reuse_a_free_name_from_an_unrelated_sibling() -> None:
    scoped = STORE.op(SCOPE, bound(0))
    term = STORE.op(PAIR, scoped, free("x"))
    nominal = to_nominal(STORE, term, ("x",))

    assert nominal.args[0].binders == ("x",)
    assert nominal.args[0].args[0].payload == "x"
    assert nominal.args[1].payload == "x"
    assert to_core(STORE, nominal) is term


def test_atom_and_symbol_payloads_use_store_names() -> None:
    term = STORE.op(PAIR, free("free0"), symbol("free1"))
    nominal = to_nominal(STORE, term, NAMES)

    assert nominal.args[0].payload == "free0"
    assert nominal.args[1].payload == "free1"
    assert to_core(STORE, nominal) is term


def test_name_inventory_bounds_presentable_binder_nesting() -> None:
    term = STORE.op(SCOPE, STORE.op(SCOPE, bound(0)))
    message = "name inventory size 1 bounds presentable binder nesting"

    with pytest.raises(ValueError, match=message):
        to_nominal(STORE, term, ("x",))


def test_presentation_refuses_foreign_and_open_core_terms() -> None:
    foreign_store = TermStore(SIGNATURE)
    foreign = foreign_store.op(UNIT)

    with pytest.raises(ForeignTermError):
        to_nominal(STORE, foreign, NAMES)

    with pytest.raises(NotLocallyClosedError):
        to_nominal(STORE, bound(0), NAMES)


def test_the_human_renderer_uses_names_and_generic_prefix_notation() -> None:
    term = STORE.op(SCOPE, STORE.op(PAIR, bound(0), symbol("free1")))

    assert render(STORE, term, NAMES) == (
        "(scope [x]. (pair #fvar[Expression](x) #const[Expression](free1)))"
    )


def test_name_inventories_are_nonempty_and_unique_strings() -> None:
    term = STORE.op(SCOPE, bound(0))

    with pytest.raises(ValueError, match="size 0"):
        to_nominal(STORE, term, ())

    with pytest.raises(ValueError, match="repeats"):
        to_nominal(STORE, term, ("x", "x"))

    with pytest.raises(TypeError, match="not a string"):
        to_nominal(STORE, term, cast("tuple[str, ...]", ("x", 1)))


def test_one_site_over_two_arguments_names_one_variable_in_both() -> None:
    term = STORE.op(SHARED, bound(0), STORE.op(PAIR, bound(0), free("x")))
    nominal = to_nominal(STORE, term, ("x", "y"))

    assert nominal.binders == ("y",)
    assert (nominal.scoped(0), nominal.scoped(1)) == (("y",), ("y",))
    assert nominal.args[0].payload == "y"
    assert nominal.args[1].args[0].payload == "y"
    assert to_core(STORE, nominal) is term
    assert render(STORE, term, ("x", "y")) == (
        "(shared [y]. #fvar[Expression](y) "
        "[y]. (pair #fvar[Expression](y) #fvar[Expression](x)))"
    )


def test_a_constant_is_named_by_its_vocabulary_and_refused_outside_it() -> None:
    nominal = Nominal(SYMBOL, payload="free2")

    assert to_core(STORE, nominal) is STORE.const(EXPRESSION, "free2")
    assert to_core(STORE, nominal).payload == 2

    with pytest.raises(UnknownConstantError):
        to_core(STORE, Nominal(SYMBOL, payload="snore"))
