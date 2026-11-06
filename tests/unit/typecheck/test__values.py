from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.terms import MVar, Node, Operator, Signature, Term, TermStore
from lofi.typecheck import (
    Constraint,
    Scheme,
    TypingDirection,
    TypingResult,
    TypingRule,
)

if TYPE_CHECKING:
    from lofi.typecheck import TypingFailureKind


def store() -> TermStore:
    """Return a small semantic-type store."""
    signature = Signature(
        "types",
        ("Ty",),
        (
            Operator("t", "Ty"),
            Operator("e", "Ty"),
            Operator("arrow", "Ty", arg_sorts=("Ty", "Ty")),
        ),
        open_sorts=("Ty",),
    )

    return TermStore(signature)


def test_a_typing_rule_reads_back_its_frozen_fields() -> None:
    variable = MVar("A")
    truth = Node("t")
    rule = TypingRule(
        "forall",
        (variable,),
        (truth,),
        truth,
        enforced=(0,),
        direction="down",
    )

    assert rule.op == "forall"
    assert rule.binds == (variable,)
    assert rule.premises == (truth,)
    assert rule.result == truth
    assert rule.enforced == (0,)
    assert rule.direction == "down"
    assert rule == TypingRule("forall", (variable,), (truth,), truth, (0,), "down")
    assert hash(rule) == hash(
        TypingRule("forall", (variable,), (truth,), truth, (0,), "down")
    )

    with pytest.raises(AttributeError, match="immutable"):
        rule.__setattr__("op", "exists")


def test_a_typing_rule_refuses_inconsistent_premise_declarations() -> None:
    truth = Node("t")

    with pytest.raises(ValueError, match="binder entries"):
        TypingRule("not", (), (truth,), truth)

    with pytest.raises(ValueError, match="more than once"):
        TypingRule("and", (None, None), (truth, truth), truth, (0, 0))

    with pytest.raises(ValueError, match="outside"):
        TypingRule("not", (None,), (truth,), truth, (1,))

    with pytest.raises(ValueError, match="unknown direction"):
        TypingRule(
            "not",
            (None,),
            (truth,),
            truth,
            direction=cast("TypingDirection", "left"),
        )


def test_a_constraint_carries_its_node_path_as_integer_positions() -> None:
    terms = store()
    truth = terms.op(terms.signature.by_name("t"))
    entity = terms.op(terms.signature.by_name("e"))
    constraint = Constraint("unification", truth, entity, (1, 0))

    assert (constraint.kind, constraint.left, constraint.right, constraint.path) == (
        "unification",
        truth,
        entity,
        (1, 0),
    )
    assert constraint == Constraint("unification", truth, entity, [1, 0])
    assert hash(constraint) == hash(Constraint("unification", truth, entity, (1, 0)))
    assert constraint != Constraint("annotation", truth, entity, (1, 0))

    with pytest.raises(ValueError, match="unknown typing failure kind"):
        Constraint(cast("TypingFailureKind", "other"), truth, entity, ())

    with pytest.raises(ValueError, match="negative"):
        Constraint("unification", truth, entity, (0, -1))


def test_a_scheme_binds_declared_free_type_atoms() -> None:
    terms = store()
    variable = terms.fvar("Ty", "A")
    atom = variable.payload

    assert atom is not None
    scheme = Scheme(variable, (atom,))

    assert scheme.body is variable
    assert scheme.quantified == (atom,)
    assert scheme == Scheme(variable, (atom,))
    assert hash(scheme) == hash(Scheme(variable, (atom,)))

    with pytest.raises(ValueError, match="more than once"):
        Scheme(variable, (atom, atom))

    truth = terms.op(terms.signature.by_name("t"))

    with pytest.raises(ValueError, match="absent"):
        Scheme(truth, (atom,))


def test_well_typed_is_exactly_an_empty_residual() -> None:
    terms = store()
    truth = terms.op(terms.signature.by_name("t"))
    successful = TypingResult({(): truth}, (), Scheme(truth))
    failed = TypingResult(
        {(): truth},
        (Constraint("unification", truth, truth, ()),),
        None,
    )

    assert successful.is_well_typed
    assert not failed.is_well_typed
    assert failed.scheme is None


def test_assignment_is_keyed_by_paths_and_is_not_live() -> None:
    terms = store()
    truth = terms.op(terms.signature.by_name("t"))
    source: dict[tuple[int, ...], Term] = {(): truth}
    result = TypingResult(source, (), Scheme(truth))
    source[(0,)] = truth

    assert dict(result.assignment) == {(): truth}

    with pytest.raises(TypeError):
        cast("dict[tuple[int, ...], Term]", result.assignment)[(0,)] = truth


def test_defaulting_rebuilds_nested_types_through_the_supplied_store() -> None:
    terms = store()
    variable = terms.fvar("Ty", "A")
    atom = variable.payload

    assert atom is not None
    truth = terms.op(terms.signature.by_name("t"))
    entity = terms.op(terms.signature.by_name("e"))
    arrow = terms.op(terms.signature.by_name("arrow"), variable, truth)
    residual = Constraint("infinite_type", arrow, variable, (1,))
    result = TypingResult(
        {(): arrow, (0,): variable},
        (residual,),
        Scheme(arrow, (atom,)),
    )

    defaulted = result.defaulted(terms, entity)
    expected = terms.op(terms.signature.by_name("arrow"), entity, truth)

    assert defaulted.assignment == {(): expected, (0,): entity}
    assert defaulted.residual is result.residual
    assert defaulted.scheme == Scheme(expected)
    assert result.scheme == Scheme(arrow, (atom,))


def test_defaulting_an_unquantified_result_returns_the_same_value() -> None:
    terms = store()
    truth = terms.op(terms.signature.by_name("t"))
    result = TypingResult({(): truth}, (), Scheme(truth))

    assert result.defaulted(terms, truth) is result


def test_defaulting_reinterns_a_foreign_nullary_ground_type() -> None:
    terms = store()
    foreign = store()
    variable = terms.fvar("Ty", "A")
    atom = variable.payload

    assert atom is not None
    entity = foreign.op(foreign.signature.by_name("e"))
    result = TypingResult({(): variable}, (), Scheme(variable, (atom,)))
    defaulted = result.defaulted(terms, entity)
    expected = terms.op(terms.signature.by_name("e"))
    scheme = defaulted.scheme

    assert defaulted.assignment == {(): expected}
    assert scheme == Scheme(expected)
    assert scheme is not None
    assert scheme.body is not entity


def test_defaulting_preserves_featured_operators_in_foreign_compounds() -> None:
    signature = Signature(
        "featured-types",
        ("Ty",),
        (
            Operator("e", "Ty"),
            Operator("arrow", "Ty", arg_sorts=("Ty", "Ty")),
        ),
        features={"e": ("marked",)},
        open_sorts=("Ty",),
    )
    terms = TermStore(signature)
    foreign = TermStore(signature)
    variable = terms.fvar("Ty", "A")
    atom = variable.payload

    assert atom is not None
    ground = foreign.op(
        signature.by_name("arrow"),
        foreign.op(signature.featured("e", "marked")),
        foreign.op(signature.by_name("e")),
    )
    expected = terms.op(
        signature.by_name("arrow"),
        terms.op(signature.featured("e", "marked")),
        terms.op(signature.by_name("e")),
    )
    result = TypingResult({(): variable}, (), Scheme(variable, (atom,)))

    assert result.defaulted(terms, ground).assignment[()] is expected


@given(primitives=st.lists(st.booleans(), min_size=1, max_size=8))
@settings(max_examples=200, deadline=None)
def test_defaulting_reinterns_foreign_compound_ground_types(
    primitives: list[bool],
) -> None:
    terms = store()
    foreign = store()
    variable = terms.fvar("Ty", "A")
    atom = variable.payload

    assert atom is not None
    ground = foreign.op(foreign.signature.by_name("e"))
    expected = terms.op(terms.signature.by_name("e"))

    for use_entity in primitives:
        foreign_primitive = foreign.op(
            foreign.signature.by_name("e" if use_entity else "t")
        )
        primitive = terms.op(terms.signature.by_name("e" if use_entity else "t"))
        ground = foreign.op(
            foreign.signature.by_name("arrow"), ground, foreign_primitive
        )
        expected = terms.op(terms.signature.by_name("arrow"), expected, primitive)

    result = TypingResult({(): variable}, (), Scheme(variable, (atom,)))
    defaulted = result.defaulted(terms, ground)

    assert defaulted.assignment[()] is expected
    assert result.assignment[()] is variable


def test_defaulting_refuses_a_type_that_is_not_ground() -> None:
    terms = store()
    variable = terms.fvar("Ty", "A")
    atom = variable.payload

    assert atom is not None
    result = TypingResult({(): variable}, (), Scheme(variable, (atom,)))

    with pytest.raises(ValueError, match="no free or open variables"):
        result.defaulted(terms, terms.fvar("Ty", "B"))


def test_typing_values_have_no_instance_dictionary() -> None:
    terms = store()
    truth = terms.op(terms.signature.by_name("t"))
    values = (
        TypingRule("not", (None,), (Node("t"),), Node("t")),
        Constraint("unification", truth, truth, ()),
        Scheme(truth),
        TypingResult({(): truth}, (), Scheme(truth)),
    )

    for value in values:
        with pytest.raises(TypeError):
            vars(value)


def test_typing_values_refuse_assignment_and_deletion() -> None:
    terms = store()
    truth = terms.op(terms.signature.by_name("t"))
    values = (
        TypingRule("not", (None,), (Node("t"),), Node("t")),
        Constraint("unification", truth, truth, ()),
        Scheme(truth),
        TypingResult({(): truth}, (), Scheme(truth)),
    )

    for value in values:
        for slot in value.__slots__:
            with pytest.raises(AttributeError, match="immutable"):
                value.__setattr__(slot, None)

            with pytest.raises(AttributeError, match="immutable"):
                value.__delattr__(slot)
