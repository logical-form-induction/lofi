"""Bind the type-sampler laws to both synthetic implementations."""

from typing import Final

import pytest
from tests.conformance.type_sampler import (
    InventoriedTypeSamplerConformance,
    InventoriedTypeSamplerFixture,
    TypeInventoryConformance,
    TypeInventoryFixture,
    TypeSamplerConformance,
    TypeSamplerFixture,
)

from lofi.data import (
    InventoriedTypeSampler,
    SizeExhaustiveWeighting,
    TypeDistribution,
)
from lofi.data.synthetic import (
    ExhaustiveTypeInventory,
    ExhaustiveTypeSampler,
    RecursiveTypeSampler,
)
from lofi.terms import Operator, Signature, TermStore

SIGNATURE: Final = Signature(
    "type-sampler-conformance",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)


def _sampler(kind: str, store: TermStore) -> InventoriedTypeSampler:
    """Build one inventoried sampler over a store."""
    return (
        RecursiveTypeSampler(store, max_size=7, sort="Cat")
        if kind == "recursive"
        else ExhaustiveTypeSampler(
            store, 7, sort="Cat", weighting=SizeExhaustiveWeighting()
        )
    )


def _distribution() -> TypeDistribution:
    """Return the shared bounded sampling request."""
    return TypeDistribution(
        3,
        {"n": 2.0, "s": 1.0, "fwd": 3.0, "bwd": 1.0},
        max_depth=3,
    )


def _fixture(kind: str) -> TypeSamplerFixture:
    """Build one independent ordinary sampler fixture."""
    store = TermStore(SIGNATURE)

    return TypeSamplerFixture(_sampler(kind, store), _distribution(), store)


def _inventoried_fixture(kind: str) -> InventoriedTypeSamplerFixture:
    """Build one independent inventoried-sampler fixture."""
    store = TermStore(SIGNATURE)

    return InventoriedTypeSamplerFixture(_sampler(kind, store), store)


class TestExhaustiveTypeInventory(TypeInventoryConformance):
    """Conformance tests for the stable exhaustive inventory."""

    @pytest.fixture
    def type_inventory_fixture(self) -> TypeInventoryFixture:
        """Build an exhaustive inventory fixture."""
        store = TermStore(SIGNATURE)

        return TypeInventoryFixture(
            ExhaustiveTypeInventory(store, 7, sort="Cat"), store
        )


class TestRecursiveTypeSampler(
    TypeSamplerConformance, InventoriedTypeSamplerConformance
):
    """Conformance tests for the recursive implementation."""

    @pytest.fixture
    def type_sampler_fixture(self) -> TypeSamplerFixture:
        """Build a recursive fixture."""
        return _fixture("recursive")

    @pytest.fixture
    def inventoried_type_sampler_fixture(
        self,
    ) -> InventoriedTypeSamplerFixture:
        """Build a recursive inventoried-sampler fixture."""
        return _inventoried_fixture("recursive")


class TestExhaustiveTypeSampler(
    TypeSamplerConformance, InventoriedTypeSamplerConformance
):
    """Conformance tests for the exhaustive implementation."""

    @pytest.fixture
    def type_sampler_fixture(self) -> TypeSamplerFixture:
        """Build an exhaustive fixture."""
        return _fixture("exhaustive")

    @pytest.fixture
    def inventoried_type_sampler_fixture(
        self,
    ) -> InventoriedTypeSamplerFixture:
        """Build an exhaustive inventoried-sampler fixture."""
        return _inventoried_fixture("exhaustive")
