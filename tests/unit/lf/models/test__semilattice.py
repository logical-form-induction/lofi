"""Validated finite join-semilattices for plural denotations."""

import pytest

from lofi.lf.models import Element, JoinSemilattice


def test_the_canonical_plural_domain_is_a_nonempty_powerset_lattice() -> None:
    john, mary, group = tuple(Element("e", at) for at in range(3))
    lattice = JoinSemilattice("e", (john, mary, group))

    assert lattice.join(john, mary) == group
    assert lattice.join(mary, john) == group
    assert lattice.part_of(john, group)
    assert not lattice.part_of(group, john)
    assert lattice.atom(john)
    assert not lattice.atom(group)


def test_a_plural_domain_must_be_closed_under_nonempty_sum() -> None:
    domain = tuple(Element("e", at) for at in range(2))

    with pytest.raises(ValueError, match="nonempty powerset"):
        JoinSemilattice("e", domain)
