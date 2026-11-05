"""Reusable laws for bounded type samplers."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.data import (
    InventoriedTypeSampler,
    TypeDistribution,
    TypeInventory,
    TypeSampler,
    term_depth,
)

if TYPE_CHECKING:
    from lofi.terms import Term, TermStore

__all__ = [
    "INVENTORIED_TYPE_SAMPLER_CONFORMANCE_LAWS",
    "TYPE_INVENTORY_CONFORMANCE_LAWS",
    "TYPE_SAMPLER_CONFORMANCE_LAWS",
    "InventoriedTypeSamplerConformance",
    "InventoriedTypeSamplerFixture",
    "TypeInventoryConformance",
    "TypeInventoryFixture",
    "TypeSamplerConformance",
    "TypeSamplerFixture",
    "draws_are_deterministic",
    "inventory_depth_filters_are_exact",
    "inventory_is_stable_and_store_backed",
    "reports_match_draws",
    "sampler_and_inventory_share_ownership",
    "types_are_bounded_and_interned",
]


class TypeSamplerFixture(NamedTuple):
    """A sampler, distribution, and owning store."""

    sampler: TypeSampler
    distribution: TypeDistribution
    store: TermStore


class TypeInventoryFixture(NamedTuple):
    """A finite type inventory and its owning store."""

    inventory: TypeInventory
    store: TermStore


class InventoriedTypeSamplerFixture(NamedTuple):
    """An inventoried sampler and its owning store."""

    sampler: InventoriedTypeSampler
    store: TermStore


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a named law violation."""
    message = f"{law}: {detail}"
    raise LawViolationError(message)


def inventory_is_stable_and_store_backed(
    fixture: TypeInventoryFixture,
) -> None:
    """Check stable order, uniqueness, bounds, and exact store ownership."""
    inventory = fixture.inventory
    candidates = inventory.candidates

    if inventory.store is not fixture.store:
        _fail("the inventory does not retain the fixture store", law="ownership")

    if inventory.signature is not fixture.store.signature:
        _fail("the inventory does not retain the store signature", law="signature")

    if tuple(inventory) != candidates or inventory[:] != candidates:
        _fail("iteration or slicing changed candidate order", law="stable order")

    if len(inventory) != len(candidates):
        _fail(
            f"the inventory reports {len(inventory)} candidates for "
            f"{len(candidates)} stored candidates",
            law="length",
        )

    if len(set(candidates)) != len(candidates):
        _fail("the inventory contains duplicate terms", law="uniqueness")

    for position, term in enumerate(candidates):
        if inventory[position] is not term:
            _fail(
                f"index {position} does not return its stored candidate",
                law="stable order",
            )

        if not fixture.store.contains(term):
            _fail(f"{term!r} is not interned by the fixture store", law="ownership")

        if term.size > inventory.max_size:
            _fail(
                f"{term!r} has size {term.size} above {inventory.max_size}",
                law="size bound",
            )


def inventory_depth_filters_are_exact(
    fixture: TypeInventoryFixture,
) -> None:
    """Check exact-depth and through-depth filters against the stable inventory."""
    inventory = fixture.inventory
    candidates = inventory.candidates
    maximum = max(map(term_depth, candidates), default=1)

    for depth in range(1, maximum + 1):
        exact = tuple(term for term in candidates if term_depth(term) == depth)
        through = tuple(term for term in candidates if term_depth(term) <= depth)

        if inventory.at_depth(depth) != exact:
            _fail(
                f"depth {depth} returned {inventory.at_depth(depth)!r}, "
                f"expected {exact!r}",
                law="exact depth",
            )

        if inventory.through_depth(depth) != through:
            _fail(
                f"through depth {depth} returned {inventory.through_depth(depth)!r}, "
                f"expected {through!r}",
                law="depth bound",
            )


def sampler_and_inventory_share_ownership(
    fixture: InventoriedTypeSamplerFixture,
) -> None:
    """Check that a sampler and its inventory retain one store and signature."""
    sampler = fixture.sampler
    inventory = sampler.inventory

    if inventory.store is not fixture.store:
        _fail(
            "the sampler inventory does not retain the fixture store", law="ownership"
        )

    if inventory.signature is not fixture.store.signature:
        _fail(
            "the sampler inventory does not retain the store signature",
            law="signature",
        )

    if sampler.signature is not inventory.signature:
        _fail(
            "the sampler and inventory publish different signatures",
            law="signature",
        )


def draws_are_deterministic(fixture: TypeSamplerFixture) -> None:
    """Check that a seed determines the complete draw."""
    first, _ = fixture.sampler.sample(40, fixture.distribution, seed=2109)
    second, _ = fixture.sampler.sample(40, fixture.distribution, seed=2109)

    if tuple(term.key() for term in first) != tuple(term.key() for term in second):
        _fail("equal seeds produced different terms", law="seed determinism")


def types_are_bounded_and_interned(fixture: TypeSamplerFixture) -> None:
    """Check ownership, signature, and the hard depth bound."""
    terms, _ = fixture.sampler.sample(60, fixture.distribution, seed=981)

    for term in terms:
        if not fixture.store.contains(term):
            _fail(f"{term!r} is not interned by the fixture store", law="ownership")

        if term.op not in fixture.store.signature.operators:
            _fail(f"{term!r} uses an undeclared root", law="signature")

        depth = term_depth(term)

        if depth > fixture.distribution.max_depth:
            _fail(
                f"{term!r} has depth {depth}",
                law="depth bound",
            )


def reports_match_draws(fixture: TypeSamplerFixture) -> None:
    """Check reported depth and constructor distributions."""
    terms, report = fixture.sampler.sample(50, fixture.distribution, seed=72)
    depths: dict[int, int] = {}
    constructors: dict[str, int] = {}

    def visit(term: Term) -> None:
        constructors[term.op.name] = constructors.get(term.op.name, 0) + 1

        for child in term.args:
            visit(child)

    for term in terms:
        depth = term_depth(term)
        depths[depth] = depths.get(depth, 0) + 1
        visit(term)

    if report.draws != len(terms):
        _fail(f"reported {report.draws} draws for {len(terms)} terms", law="report")

    if dict(report.depths) != depths:
        _fail(f"reported depths {report.depths!r}, expected {depths!r}", law="report")

    if dict(report.constructors) != constructors:
        _fail(
            f"reported constructors {report.constructors!r}, expected {constructors!r}",
            law="report",
        )

    expected_depths = {depth: count / len(terms) for depth, count in depths.items()}
    constructor_total = sum(constructors.values())
    expected_constructors = {
        name: count / constructor_total for name, count in constructors.items()
    }

    if dict(report.depth_distribution) != expected_depths:
        _fail(
            f"reported depth distribution {report.depth_distribution!r}, "
            f"expected {expected_depths!r}",
            law="report",
        )

    if dict(report.constructor_distribution) != expected_constructors:
        _fail(
            f"reported constructor distribution "
            f"{report.constructor_distribution!r}, expected "
            f"{expected_constructors!r}",
            law="report",
        )


class TypeSamplerConformance:
    """Reusable type-sampler conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_draws_are_deterministic(
        self, type_sampler_fixture: TypeSamplerFixture
    ) -> None:
        """Run the seed-determinism law."""
        draws_are_deterministic(type_sampler_fixture)

    def test_types_are_bounded_and_interned(
        self, type_sampler_fixture: TypeSamplerFixture
    ) -> None:
        """Run the bounded-interned-term law."""
        types_are_bounded_and_interned(type_sampler_fixture)

    def test_reports_match_draws(
        self, type_sampler_fixture: TypeSamplerFixture
    ) -> None:
        """Run the report law."""
        reports_match_draws(type_sampler_fixture)


class TypeInventoryConformance:
    """Reusable finite type-inventory conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_inventory_is_stable_and_store_backed(
        self, type_inventory_fixture: TypeInventoryFixture
    ) -> None:
        """Run the stable store-backed inventory law."""
        inventory_is_stable_and_store_backed(type_inventory_fixture)

    def test_inventory_depth_filters_are_exact(
        self, type_inventory_fixture: TypeInventoryFixture
    ) -> None:
        """Run the exact inventory-depth-filter law."""
        inventory_depth_filters_are_exact(type_inventory_fixture)


class InventoriedTypeSamplerConformance:
    """Reusable inventoried type-sampler conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_sampler_and_inventory_share_ownership(
        self,
        inventoried_type_sampler_fixture: InventoriedTypeSamplerFixture,
    ) -> None:
        """Run the shared store-and-signature law."""
        sampler_and_inventory_share_ownership(inventoried_type_sampler_fixture)


TYPE_SAMPLER_CONFORMANCE_LAWS: Final = (
    TypeSamplerConformance.test_draws_are_deterministic,
    TypeSamplerConformance.test_types_are_bounded_and_interned,
    TypeSamplerConformance.test_reports_match_draws,
)

TYPE_INVENTORY_CONFORMANCE_LAWS: Final = (
    TypeInventoryConformance.test_inventory_is_stable_and_store_backed,
    TypeInventoryConformance.test_inventory_depth_filters_are_exact,
)

INVENTORIED_TYPE_SAMPLER_CONFORMANCE_LAWS: Final = (
    InventoriedTypeSamplerConformance.test_sampler_and_inventory_share_ownership,
)
