"""Properties of the canonical logical-form fixture inventory."""

from typing import Final, Literal, NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.lf import fixtures

from lofi.terms import Path, Term

__all__: list[str] = []

pytestmark = pytest.mark.conformance

_NAMES: Final = (
    "jo",
    "sleep",
    "eat",
    "x0_e",
    "x1_e",
    "f0_et",
    "f0_eet",
    "appl_et_const",
    "appl_t_const",
    "appl2_t_const",
    "quant_t_const",
    "quant2_t_const",
    "abs_et_const",
    "abs2_et_const",
    "abs_t_const",
    "prop_appl_et_const",
    "nested_abstractions",
    "nested_quantifiers",
    "appl_et_distinct_vars",
    "appl_t_distinct_vars",
)

type _VariableIdentity = tuple[Literal["bound"], Path] | tuple[Literal["free"], int]


class _Visit(NamedTuple):
    """A subterm with its path and nearest-first binder owners."""

    path: Path
    node: Term
    owners: tuple[Path, ...]


def _inventory() -> tuple[tuple[str, Term], ...]:
    """Return every fixture in canonical order."""
    return tuple((name, getattr(fixtures, name)) for name in _NAMES)


def _visits(term: Term) -> tuple[_Visit, ...]:
    """Traverse a term with explicit binder ownership."""
    found: list[_Visit] = []
    pending: list[tuple[Path, Term, tuple[Path, ...]]] = [((), term, ())]

    while pending:
        path, node, owners = pending.pop()
        found.append(_Visit(path, node, owners))

        for position, child in reversed(tuple(enumerate(node.args))):
            child_owners = (path,) * len(node.op.scope(position)) + owners
            pending.append(((*path, position), child, child_owners))

    return tuple(found)


def _identity(visit: _Visit) -> _VariableIdentity | None:
    """Resolve a variable occurrence to its free atom or binder owner."""
    payload = visit.node.payload

    if visit.node.op.name == "#fvar[Tm]" and payload is not None:
        return "free", payload

    if visit.node.op.name != "#bvar[Tm]" or payload is None:
        return None

    if payload >= len(visit.owners):
        return None

    return "bound", visit.owners[payload]


def _variable_applications_are_distinct(term: Term) -> bool:
    """Return whether each variable-headed application uses distinct variables."""
    by_path = {visit.path: visit for visit in _visits(term)}

    for visit in by_path.values():
        if visit.node.op.name != "eval":
            continue

        head = by_path[(*visit.path, 0)]
        argument = by_path[(*visit.path, 1)]
        head_identity = _identity(head)
        argument_identity = _identity(argument)

        if (
            head_identity is not None
            and argument_identity is not None
            and head_identity == argument_identity
        ):
            return False

    return True


def test_the_public_inventory_is_exactly_the_twenty_named_fixtures() -> None:
    """Keep the public fixture inventory exact."""
    assert tuple(fixtures.__all__) == tuple(sorted(_NAMES))
    assert {name for name in vars(fixtures) if not name.startswith("_")} == set(_NAMES)
    assert len(_inventory()) == 20
    assert len({name for name, _ in _inventory()}) == 20


def test_atomic_and_application_fixtures_have_the_declared_structure() -> None:
    """Preserve the atomic and application port structure."""
    assert all(
        term.op.name == "#const[Tm]"
        for term in (fixtures.jo, fixtures.sleep, fixtures.eat)
    )
    assert len({fixtures.jo.payload, fixtures.sleep.payload, fixtures.eat.payload}) == 3
    variables = (fixtures.x0_e, fixtures.x1_e, fixtures.f0_et, fixtures.f0_eet)
    assert all(term.op.name == "#fvar[Tm]" for term in variables)
    assert len({term.payload for term in variables}) == 4
    applications = (
        fixtures.appl_et_const,
        fixtures.appl_t_const,
        fixtures.appl2_t_const,
        fixtures.prop_appl_et_const,
        fixtures.appl_et_distinct_vars,
        fixtures.appl_t_distinct_vars,
    )
    assert all(term.op.name == "eval" for term in applications)

    assert fixtures.appl_et_const.args == (fixtures.eat, fixtures.x0_e)
    assert fixtures.appl_t_const.args == (fixtures.sleep, fixtures.x0_e)
    assert fixtures.appl2_t_const.args == (
        fixtures.appl_et_const,
        fixtures.x1_e,
    )
    assert fixtures.appl_et_distinct_vars.args == (fixtures.f0_eet, fixtures.x0_e)
    assert fixtures.appl_t_distinct_vars.args == (fixtures.f0_et, fixtures.x0_e)


def test_binding_fixtures_have_the_declared_structure() -> None:
    """Preserve the abstraction and quantifier port structure."""
    quantifier, abstraction = fixtures.quant_t_const.args
    (quantifier_body,) = abstraction.args
    assert fixtures.quant_t_const.op.name == "eval"
    assert quantifier.op.name == "#const[Tm]"
    assert abstraction.op.name == "lam"
    assert quantifier_body.args[0] is fixtures.sleep
    assert quantifier_body.args[1].op.name == "#bvar[Tm]"
    assert quantifier_body.args[1].payload == 0

    quantifier, abstraction = fixtures.quant2_t_const.args
    (quantifier_body,) = abstraction.args
    assert fixtures.quant2_t_const.op.name == "eval"
    assert quantifier.op.name == "#const[Tm]"
    assert abstraction.op.name == "lam"
    assert quantifier_body.args[0].args[0] is fixtures.eat
    assert quantifier_body.args[0].args[1].payload == 0
    assert quantifier_body.args[1] is fixtures.x1_e

    (abstraction_body,) = fixtures.abs_et_const.args
    assert fixtures.abs_et_const.op.name == "lam"
    assert abstraction_body.args[0].args[0] is fixtures.eat
    assert abstraction_body.args[0].args[1].payload == 0
    assert abstraction_body.args[1] is fixtures.x1_e

    (inner_abstraction,) = fixtures.abs2_et_const.args
    (abstraction_body,) = inner_abstraction.args
    assert fixtures.abs2_et_const.op.name == "lam"
    assert inner_abstraction.op.name == "lam"
    assert abstraction_body.args[0].args[1].payload == 0
    assert abstraction_body.args[1].payload == 1

    (abstraction_body,) = fixtures.abs_t_const.args
    assert fixtures.abs_t_const.op.name == "lam"
    assert abstraction_body.args[0] is fixtures.sleep
    assert abstraction_body.args[1].payload == 0
    assert fixtures.prop_appl_et_const.args == (fixtures.abs2_et_const, fixtures.jo)

    (inner_abstraction,) = fixtures.nested_abstractions.args
    (application,) = inner_abstraction.args
    assert fixtures.nested_abstractions.op.name == "lam"
    assert inner_abstraction.op.name == "lam"
    assert application.op.name == "eval"
    assert application.args[0].payload == 1
    assert application.args[1].payload == 0

    _, outer_abstraction = fixtures.nested_quantifiers.args
    (existential,) = outer_abstraction.args
    _, inner_abstraction = existential.args
    (application,) = inner_abstraction.args
    assert fixtures.nested_quantifiers.op.name == "eval"
    assert outer_abstraction.op.name == "lam"
    assert existential.op.name == "eval"
    assert inner_abstraction.op.name == "lam"
    assert application.args[0].args[0].op.name == "#const[Tm]"
    assert application.args[0].args[1].payload == 1
    assert application.args[1].payload == 0


@given(order=st.permutations(_NAMES), data=st.data())
@settings(max_examples=100, deadline=None)
def test_every_bound_occurrence_has_one_owner_under_every_traversal_order(
    order: list[str],
    data: st.DataObject,
) -> None:
    """Resolve every bound occurrence against one enclosing binder."""
    inventory = dict(_inventory())

    for name in order:
        visits = _visits(inventory[name])
        shuffled = data.draw(st.permutations(visits), label=name)

        assert len(shuffled) == len(visits)

        for visit in shuffled:
            assert len(visit.owners) == len(set(visit.owners))

            if visit.node.op.name != "#bvar[Tm]":
                continue

            payload = visit.node.payload
            assert payload is not None
            assert 0 <= payload < len(visit.owners)
            assert _identity(visit) == ("bound", visit.owners[payload])


def test_variable_headed_applications_use_distinct_variable_identities() -> None:
    """Separate every variable application head from its argument."""
    for _, term in _inventory():
        assert _variable_applications_are_distinct(term)

    head = _identity(_visits(fixtures.appl_et_distinct_vars)[1])
    argument = _identity(_visits(fixtures.appl_et_distinct_vars)[2])

    assert head is not None
    assert argument is not None
    assert head != argument
