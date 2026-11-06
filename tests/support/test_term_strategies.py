"""Tests for the shared property-test strategies."""

from typing import Final, Literal

import didactic.api as dx
import pytest
from hypothesis import find, given, settings
from tests.support.term_strategies import spec_of, terms

from lofi.terms import BindingSite, Operator, Signature, Term, TermStore, Vocabulary

EXPRESSION: Final = "Expression"
LABEL: Final = "Label"
BOUND: Final = "#bvar[Expression]"
SCOPE: Final = "scope"

SIGNATURE: Final = Signature(
    "external",
    (EXPRESSION, LABEL),
    (
        Operator("unit", EXPRESSION),
        Operator(
            "branch",
            EXPRESSION,
            arg_sorts=(EXPRESSION, LABEL, EXPRESSION),
        ),
        Operator(
            SCOPE,
            EXPRESSION,
            arg_sorts=(EXPRESSION, EXPRESSION, LABEL),
            binding=(
                BindingSite((1,), EXPRESSION),
                BindingSite((1,), EXPRESSION),
            ),
        ),
        Operator(
            "typed",
            LABEL,
            arg_sorts=(LABEL,),
            binding=(BindingSite((0,), LABEL),),
        ),
    ),
    vocabularies=(Vocabulary(LABEL, ("first", "second")),),
)
STORE: Final = TermStore(SIGNATURE)


def walk(term: Term) -> tuple[Term, ...]:
    """Return a term and its descendants in preorder."""
    return (term, *(node for child in term.args for node in walk(child)))


@given(terms(STORE, sort=EXPRESSION, max_size=12, closed=True))
@settings(max_examples=200)
def test_closed_terms_respect_the_signature_and_size(term: Term) -> None:
    assert term.size <= 12
    assert term.is_closed

    for node in walk(term):
        assert node.op in SIGNATURE.operators
        assert tuple(child.op.sort for child in node.args) == node.op.arg_sorts
        assert STORE.op(node.op, *node.args, payload=node.payload) is node


def test_a_ternary_binder_declared_outside_the_package_is_reachable() -> None:
    term = find(
        terms(STORE, sort=EXPRESSION, max_size=8, closed=True),
        lambda candidate: (
            candidate.op.name == SCOPE and candidate.args[1].op.name == BOUND
        ),
    )

    assert term.op.arity == 3
    assert [len(term.op.scope(position)) for position in range(3)] == [0, 2, 0]
    assert term.args[1].payload in {0, 1}
    assert term.is_closed


def test_open_generation_reaches_a_dangling_index() -> None:
    term = find(
        terms(STORE, sort=EXPRESSION, max_size=8, closed=False),
        lambda candidate: not candidate.is_closed,
    )

    assert term.maxidx > 0


def test_a_broken_closedness_law_shrinks_below_four_nodes() -> None:
    counterexample = find(
        terms(STORE, sort=EXPRESSION, max_size=12, closed=False),
        lambda candidate: not candidate.is_closed,
    )

    assert counterexample.size <= 3


@pytest.mark.parametrize("sort", SIGNATURE.sorts)
def test_each_declared_sort_is_generated(sort: str) -> None:
    generated = find(
        terms(STORE, sort=sort, max_size=8, closed=True),
        lambda _candidate: True,
    )

    assert generated.op.sort == sort


def test_unknown_sorts_and_nonpositive_budgets_are_refused() -> None:
    with pytest.raises(ValueError, match="not declared"):
        terms(STORE, sort="Missing", max_size=4, closed=True)

    with pytest.raises(ValueError, match="must be positive"):
        terms(STORE, sort=EXPRESSION, max_size=0, closed=True)


class Choice(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A test specification family."""


class Count(Choice, extra="forbid"):
    """An integer choice."""

    kind: Literal["count"] = "count"
    value: int


class Toggle(Choice, extra="forbid"):
    """A Boolean choice."""

    kind: Literal["toggle"] = "toggle"
    enabled: bool


@given(spec_of(Choice))
def test_specifications_are_built_from_registered_variants(
    specification: Choice,
) -> None:
    assert specification.kind in Choice.__variants__


@pytest.mark.parametrize("tag", Choice.__variants__)
def test_each_registered_variant_is_reachable(tag: str) -> None:
    specification = find(
        spec_of(Choice),
        lambda candidate: candidate.kind == tag,
    )

    assert specification.kind == tag


@given(terms(STORE, sort=EXPRESSION, max_size=12, closed=True))
@settings(max_examples=200)
def test_closed_terms_count_binding_sites_per_sort(term: Term) -> None:
    assert term.is_closed
    assert term.unbound == ()


def test_a_label_binder_does_not_close_an_expression_index() -> None:
    label = find(
        terms(STORE, sort=LABEL, max_size=6, closed=True),
        lambda candidate: (
            candidate.op.name == "typed"
            and candidate.args[0].op.payload_sort == "index"
        ),
    )

    assert label.args[0].op.sort == LABEL
    assert label.is_closed


def test_constants_are_drawn_from_the_vocabulary_and_free_variables_by_name() -> None:
    constant = find(
        terms(STORE, sort=LABEL, max_size=1, closed=True),
        lambda candidate: candidate.op.payload_sort == "constant",
    )
    free = find(
        terms(STORE, sort=LABEL, max_size=1, closed=True),
        lambda candidate: candidate.op.payload_sort == "free",
    )

    assert STORE.payload_name(constant) in {"first", "second"}
    assert STORE.payload_name(free).startswith("x")


def test_an_empty_vocabulary_generates_no_constant() -> None:
    signature = Signature("empty", ("A",), (), vocabularies=(Vocabulary("A", ()),))

    assert terms(TermStore(signature), sort="A", max_size=3, closed=True).is_empty
