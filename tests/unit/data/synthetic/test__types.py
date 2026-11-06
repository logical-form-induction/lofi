"""Tests for synthetic type samplers."""

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)

from lofi.data import (
    ArityBranching,
    PowerChildDepth,
    SizeExhaustiveWeighting,
    TypeDistribution,
    UniformExhaustiveWeighting,
    term_depth,
)
from lofi.data.synthetic import (
    ExhaustiveTypeInventory,
    ExhaustiveTypeSampler,
    RecursiveTypeSampler,
    TypeSamplerContext,
)
from lofi.registry import plugin_for
from lofi.spec.type_samplers.exhaustive import (
    ExhaustiveTypeSamplerSpec,
    SizeExhaustiveWeightingSpec,
)
from lofi.spec.type_samplers.recursive import RecursiveTypeSamplerSpec
from lofi.terms import (
    Operator,
    Signature,
    TermStore,
    TreeEditDistance,
    ball,
)

if TYPE_CHECKING:
    from lofi.data import ExhaustiveWeighting
    from lofi.spec import ConfiguredSpec


SIGNATURE: Final = Signature(
    "synthetic-types",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)

INVENTORY_SIGNATURE: Final = Signature(
    "type-inventory",
    ("Cat",),
    (
        Operator("atom", "Cat"),
        Operator("unary", "Cat", arg_sorts=("Cat",)),
        Operator("binary", "Cat", arg_sorts=("Cat", "Cat")),
    ),
    open_sorts=("Cat",),
)

MULTISORT_SIGNATURE: Final = Signature(
    "multisort-inventory",
    ("Ty", "Cat"),
    (
        Operator("e", "Ty"),
        Operator("->", "Ty", arg_sorts=("Ty", "Ty")),
        Operator("NP", "Cat"),
        Operator("/", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)

MIXED_ARITY_SIGNATURE: Final = Signature(
    "mixed-arity-types",
    ("Cat",),
    (
        Operator("atom", "Cat"),
        Operator("binary", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("ternary", "Cat", arg_sorts=("Cat", "Cat", "Cat")),
    ),
)


def test_inventory_agrees_with_a_hand_enumeration_in_exact_order() -> None:
    store = TermStore(INVENTORY_SIGNATURE)
    inventory = ExhaustiveTypeInventory(store, 3, sort="Cat")
    atom = store.op(INVENTORY_SIGNATURE.by_name("atom"))
    unary = INVENTORY_SIGNATURE.by_name("unary")
    binary = INVENTORY_SIGNATURE.by_name("binary")
    one = store.op(unary, atom)
    expected = (atom, one, store.op(unary, one), store.op(binary, atom, atom))

    assert inventory.candidates == expected
    assert tuple(inventory) == expected
    assert inventory[:] == expected
    assert inventory[0] is atom
    assert len(inventory) == len(expected)
    assert len(set(inventory)) == len(inventory)
    assert all(term.size <= inventory.max_size for term in inventory)
    assert all(
        node.op.payload_sort == "none"
        for term in inventory
        for node in (term, *term.args)
    )


def test_inventory_depth_filters_preserve_the_base_order() -> None:
    store = TermStore(INVENTORY_SIGNATURE)
    inventory = ExhaustiveTypeInventory(store, 3, sort="Cat")

    assert inventory.at_depth(2) == tuple(
        term for term in inventory if term_depth(term) == 2
    )
    assert inventory.through_depth(2) == tuple(
        term for term in inventory if term_depth(term) <= 2
    )


def test_inventory_contains_only_its_declared_root_sort() -> None:
    store = TermStore(MULTISORT_SIGNATURE)
    inventory = ExhaustiveTypeInventory(store, 3, sort="Ty")

    assert {term.op.sort for term in inventory} == {"Ty"}
    assert tuple(map(str, inventory)) == ("e", "(-> e e)")


def test_inventory_is_owned_by_the_exact_store_and_supplies_a_ball_directly() -> None:
    store = TermStore(INVENTORY_SIGNATURE)
    inventory = ExhaustiveTypeInventory(store, 3, sort="Cat")
    center = inventory[0]

    assert inventory.store is store
    assert inventory.signature is store.signature
    assert all(store.contains(term) for term in inventory)
    assert ball(
        center,
        inventory,
        TreeEditDistance(),
        radius=0,
    ) == (center,)


def test_inventory_is_an_immutable_self_copying_value() -> None:
    inventory = ExhaustiveTypeInventory(
        TermStore(INVENTORY_SIGNATURE),
        3,
        sort="Cat",
    )

    with pytest.raises(AttributeError, match="type inventory is immutable"):
        inventory.__setattr__("_max_size", 4)

    with pytest.raises(AttributeError, match="type inventory is immutable"):
        inventory.__delattr__("_store")

    assert copy(inventory) is inventory
    assert deepcopy(inventory) is inventory


@pytest.mark.parametrize(("field", "value"), [("max_size", 0), ("max_size", -1)])
def test_inventory_refuses_a_nonpositive_node_bound(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        ExhaustiveTypeInventory(TermStore(INVENTORY_SIGNATURE), value, sort="Cat")


@pytest.mark.parametrize("kind", ["recursive", "exhaustive"])
def test_each_sampler_publishes_its_exact_store_backed_inventory(kind: str) -> None:
    store = TermStore(SIGNATURE)
    sampler = (
        RecursiveTypeSampler(store, 7, sort="Cat")
        if kind == "recursive"
        else ExhaustiveTypeSampler(
            store, 7, sort="Cat", weighting=UniformExhaustiveWeighting()
        )
    )

    assert isinstance(sampler.inventory, ExhaustiveTypeInventory)
    assert sampler.inventory.store is store
    assert sampler.inventory.signature is store.signature
    assert sampler.signature is sampler.inventory.signature
    assert all(store.contains(term) for term in sampler.inventory)


@pytest.mark.parametrize("kind", ["recursive", "exhaustive"])
def test_samplers_honor_exact_target_and_hard_maximum(kind: str) -> None:
    store = TermStore(SIGNATURE)
    distribution = TypeDistribution(3, {}, max_depth=3)
    sampler = (
        RecursiveTypeSampler(store, max_size=7, sort="Cat")
        if kind == "recursive"
        else ExhaustiveTypeSampler(
            store, 7, sort="Cat", weighting=UniformExhaustiveWeighting()
        )
    )
    terms, report = sampler.sample(80, distribution, seed=65)

    assert {term_depth(term) for term in terms} == {3}
    assert report.depths == {3: 80}


def test_recursive_sampler_reserves_declared_bias_mass() -> None:
    store = TermStore(SIGNATURE)
    n = store.op(SIGNATURE.by_name("n"))
    distribution = TypeDistribution(3, {}, (n,), bias_mass=1.0, max_depth=3)
    terms, report = RecursiveTypeSampler(store, max_size=7, sort="Cat").sample(
        12, distribution, seed=5
    )

    assert terms == (n,) * 12
    assert report.depths == {1: 12}


def test_constructor_weights_can_force_a_root_constructor() -> None:
    store = TermStore(SIGNATURE)
    distribution = TypeDistribution(
        2,
        {"n": 1.0, "s": 1.0, "fwd": 0.0, "bwd": 4.0},
        max_depth=2,
    )
    terms, _ = RecursiveTypeSampler(store, max_size=7, sort="Cat").sample(
        25, distribution, seed=8
    )

    assert {term.op.name for term in terms} == {"bwd"}


def test_exhaustive_sampler_rejects_an_unreachable_target_depth() -> None:
    store = TermStore(SIGNATURE)
    distribution = TypeDistribution(4, {}, max_depth=4)

    with pytest.raises(ValueError, match="no term at target depth"):
        ExhaustiveTypeSampler(
            store, 3, sort="Cat", weighting=UniformExhaustiveWeighting()
        ).sample(1, distribution, seed=0)


def test_per_arity_branching_selects_the_deepest_binary_side() -> None:
    left_store = TermStore(SIGNATURE)
    right_store = TermStore(SIGNATURE)
    left_distribution = TypeDistribution(
        3,
        {"fwd": 1.0, "bwd": 0.0},
        branching=ArityBranching({2: (1.0, 0.0)}),
        max_depth=3,
    )
    right_distribution = TypeDistribution(
        3,
        {"fwd": 1.0, "bwd": 0.0},
        branching=ArityBranching({2: (0.0, 1.0)}),
        max_depth=3,
    )
    left, _ = RecursiveTypeSampler(left_store, max_size=7, sort="Cat").sample(
        120, left_distribution, seed=44
    )
    right, _ = RecursiveTypeSampler(right_store, max_size=7, sort="Cat").sample(
        120, right_distribution, seed=44
    )

    assert all(term_depth(term.args[0]) == 2 for term in left)
    assert all(term_depth(term.args[1]) == 2 for term in right)
    assert any(term_depth(term.args[1]) == 1 for term in left)
    assert any(term_depth(term.args[0]) == 1 for term in right)


def test_branching_controls_each_constructor_arity_independently() -> None:
    first_store = TermStore(MIXED_ARITY_SIGNATURE)
    last_store = TermStore(MIXED_ARITY_SIGNATURE)
    first_distribution = TypeDistribution(
        3,
        {"binary": 0.0, "ternary": 1.0},
        branching=ArityBranching({2: (0.0, 1.0), 3: (1.0, 0.0, 0.0)}),
        max_depth=3,
    )
    last_distribution = TypeDistribution(
        3,
        {"binary": 0.0, "ternary": 1.0},
        branching=ArityBranching({2: (1.0, 0.0), 3: (0.0, 0.0, 1.0)}),
        max_depth=3,
    )
    first, _ = RecursiveTypeSampler(first_store, max_size=7, sort="Cat").sample(
        100, first_distribution, seed=37
    )
    last, _ = RecursiveTypeSampler(last_store, max_size=7, sort="Cat").sample(
        100, last_distribution, seed=37
    )

    assert all(term_depth(term.args[0]) == 2 for term in first)
    assert all(term_depth(term.args[2]) == 2 for term in last)


def test_child_depth_power_moves_nondeepest_children_monotonically() -> None:
    shallow_store = TermStore(SIGNATURE)
    deep_store = TermStore(SIGNATURE)
    shallow_distribution = TypeDistribution(
        4,
        {"fwd": 1.0, "bwd": 0.0},
        branching=ArityBranching({2: (0.0, 1.0)}),
        child_depth=PowerChildDepth(-12.0),
        max_depth=4,
    )
    deep_distribution = TypeDistribution(
        4,
        {"fwd": 1.0, "bwd": 0.0},
        branching=ArityBranching({2: (0.0, 1.0)}),
        child_depth=PowerChildDepth(12.0),
        max_depth=4,
    )
    shallow, _ = RecursiveTypeSampler(shallow_store, max_size=7, sort="Cat").sample(
        600, shallow_distribution, seed=73
    )
    deep, _ = RecursiveTypeSampler(deep_store, max_size=7, sort="Cat").sample(
        600, deep_distribution, seed=73
    )

    shallow_mean = sum(term_depth(term.args[0]) for term in shallow) / len(shallow)
    deep_mean = sum(term_depth(term.args[0]) for term in deep) / len(deep)

    assert deep_mean > shallow_mean


def test_exhaustive_uniform_and_size_weighting_are_distinct() -> None:
    uniform_store = TermStore(SIGNATURE)
    size_store = TermStore(SIGNATURE)
    distribution = TypeDistribution(3, {}, max_depth=3)
    uniform, _ = ExhaustiveTypeSampler(
        uniform_store, 7, sort="Cat", weighting=UniformExhaustiveWeighting()
    ).sample(2_000, distribution, seed=91)
    by_size, _ = ExhaustiveTypeSampler(
        size_store, 7, sort="Cat", weighting=SizeExhaustiveWeighting()
    ).sample(2_000, distribution, seed=91)
    uniform_mean = sum(term.size for term in uniform) / len(uniform)
    size_mean = sum(term.size for term in by_size) / len(by_size)

    assert size_mean > uniform_mean
    assert tuple(term.key() for term in by_size) != tuple(
        term.key() for term in uniform
    )


@pytest.mark.parametrize(
    ("weighting", "expected", "constructors"),
    [
        (
            UniformExhaustiveWeighting(),
            (
                "(fwd (fwd n n) n)",
                "(bwd n (fwd n s))",
                "(bwd (fwd n n) (fwd s s))",
                "(fwd (bwd s s) n)",
                "(bwd (bwd s n) (bwd s n))",
                "(bwd (bwd n n) (fwd s n))",
            ),
            {"fwd": 7, "n": 13, "bwd": 8, "s": 8},
        ),
        (
            SizeExhaustiveWeighting(),
            (
                "(fwd (fwd s n) n)",
                "(bwd s (fwd n s))",
                "(bwd (fwd n s) (fwd n s))",
                "(bwd n (bwd s s))",
                "(bwd (bwd s n) (bwd s s))",
                "(bwd (bwd n n) (bwd n n))",
            ),
            {"fwd": 5, "s": 10, "n": 11, "bwd": 10},
        ),
    ],
)
def test_extraction_preserves_the_existing_seeded_draw_order(
    weighting: ExhaustiveWeighting,
    expected: tuple[str, ...],
    constructors: dict[str, int],
) -> None:
    store = TermStore(SIGNATURE)
    distribution = TypeDistribution(3, {}, max_depth=3)
    sampler = ExhaustiveTypeSampler(store, 7, sort="Cat", weighting=weighting)
    terms, report = sampler.sample(6, distribution, seed=91)

    assert tuple(map(str, terms)) == expected
    assert report.draws == len(expected)
    assert report.depths == {3: len(expected)}
    assert report.constructors == constructors


class TestRecursiveTypeSamplerPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        store = TermStore(SIGNATURE)

        def observe(built: object) -> dict[str, bool]:
            sampler = cast("RecursiveTypeSampler", built)

            return {
                "draws from the context's store": sampler.inventory.store is store,
                "bounds its inventory as declared": sampler.inventory.max_size == 5,
            }

        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]", RecursiveTypeSamplerSpec(max_size=5)
            ),
            context=TypeSamplerContext(store, "Cat"),
            expected_type=RecursiveTypeSampler,
            observe=observe,
            unchanged=lambda: store.signature is SIGNATURE,
        )


class TestExhaustiveTypeSamplerPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        store = TermStore(SIGNATURE)

        def observe(built: object) -> dict[str, bool]:
            sampler = cast("ExhaustiveTypeSampler", built)

            return {
                "interns into the context's store": sampler.inventory.store is store,
                "bounds its inventory as declared": sampler.max_size == 5,
                "weights as declared": sampler.weighting == SizeExhaustiveWeighting(),
            }

        return ConfiguredComponentFixture(
            declaration=cast(
                "ConfiguredSpec[object]",
                ExhaustiveTypeSamplerSpec(
                    max_size=5, weighting=SizeExhaustiveWeightingSpec()
                ),
            ),
            context=TypeSamplerContext(store, "Cat"),
            expected_type=ExhaustiveTypeSampler,
            observe=observe,
            unchanged=lambda: store.signature is SIGNATURE,
        )


def test_the_declared_default_weighting_reaches_the_built_sampler() -> None:
    declaration = cast("ConfiguredSpec[object]", ExhaustiveTypeSamplerSpec(max_size=3))
    sampler = cast(
        "ExhaustiveTypeSampler",
        plugin_for(declaration).construct(
            declaration, TypeSamplerContext(TermStore(SIGNATURE), "Cat")
        ),
    )

    assert sampler.weighting == UniformExhaustiveWeighting()
