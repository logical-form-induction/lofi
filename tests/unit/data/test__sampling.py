"""Tests for shared sampling values."""

from types import MappingProxyType
from typing import Final, cast

import pytest

from lofi.data import (
    ArityBranching,
    PairDistribution,
    PairSampleReport,
    PowerChildDepth,
    SizeExhaustiveWeighting,
    TypeDistribution,
    TypeSampleReport,
    UniformChildDepth,
    UniformExhaustiveWeighting,
)
from lofi.terms import Operator, Signature, TermStore

SIGNATURE: Final = Signature("sampling-values", ("Cat",), (Operator("n", "Cat"),))
STORE: Final = TermStore(SIGNATURE)
N: Final = STORE.op(SIGNATURE.by_name("n"))


def test_type_distribution_copies_mutable_inputs() -> None:
    weights = {"n": 2.0}
    distribution = TypeDistribution(1, weights, (N,), bias_mass=0.25, max_depth=1)
    weights["n"] = 9.0

    assert distribution.constructor_weights == {"n": 2.0}
    assert isinstance(distribution.constructor_weights, MappingProxyType)


def test_sampling_declarations_are_immutable() -> None:
    distribution = PairDistribution(max_depth=2)
    field = "_" + "max_depth"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(distribution, field, 4)


def test_pair_report_records_emissions_and_arity_rates() -> None:
    report = PairSampleReport(7, {1: 2, 2: 5})

    assert report.emitted == 7
    assert report.arity_distribution[1] == 2 / 7
    assert sum(report.arity_distribution.values()) == 1.0


def test_pair_report_refuses_arity_counts_that_do_not_match_emissions() -> None:
    with pytest.raises(ValueError, match="sum to emitted"):
        PairSampleReport(8, {2: 7})


@pytest.mark.parametrize("arity", [-1, True, 1.5])
def test_pair_report_refuses_invalid_rule_arities(arity: object) -> None:
    with pytest.raises((TypeError, ValueError), match="reported rule arity"):
        PairSampleReport(1, cast("dict[int, int]", {arity: 1}))


def test_type_distributions_compare_and_hash_by_every_field() -> None:
    first = TypeDistribution(
        1,
        {"n": 2.0, "other": 1.0},
        (N,),
        bias_mass=0.4,
        branching=ArityBranching({2: (0.25, 0.75)}),
        child_depth=PowerChildDepth(2.0),
        max_depth=2,
    )
    equal = TypeDistribution(
        1,
        {"other": 1.0, "n": 2.0},
        (N,),
        bias_mass=0.4,
        branching=ArityBranching({2: (0.25, 0.75)}),
        child_depth=PowerChildDepth(2.0),
        max_depth=2,
    )
    variants = (
        TypeDistribution(
            2,
            {"n": 2.0, "other": 1.0},
            (N,),
            bias_mass=0.4,
            branching=ArityBranching({2: (0.25, 0.75)}),
            child_depth=PowerChildDepth(2.0),
            max_depth=2,
        ),
        TypeDistribution(
            1,
            {"n": 2.0, "other": 1.0},
            (N,),
            bias_mass=0.4,
            branching=ArityBranching({2: (0.75, 0.25)}),
            child_depth=PowerChildDepth(2.0),
            max_depth=2,
        ),
        TypeDistribution(
            1,
            {"n": 2.0, "other": 1.0},
            (N,),
            bias_mass=0.4,
            branching=ArityBranching({2: (0.25, 0.75)}),
            child_depth=PowerChildDepth(-2.0),
            max_depth=2,
        ),
        TypeDistribution(
            1,
            {"n": 3.0, "other": 1.0},
            (N,),
            bias_mass=0.4,
            branching=ArityBranching({2: (0.25, 0.75)}),
            child_depth=PowerChildDepth(2.0),
            max_depth=2,
        ),
        TypeDistribution(
            1,
            {"n": 2.0, "other": 1.0},
            (),
            branching=ArityBranching({2: (0.25, 0.75)}),
            child_depth=PowerChildDepth(2.0),
            max_depth=2,
        ),
        TypeDistribution(
            1,
            {"n": 2.0, "other": 1.0},
            (N,),
            bias_mass=0.6,
            branching=ArityBranching({2: (0.25, 0.75)}),
            child_depth=PowerChildDepth(2.0),
            max_depth=2,
        ),
        TypeDistribution(
            1,
            {"n": 2.0, "other": 1.0},
            (N,),
            bias_mass=0.4,
            branching=ArityBranching({2: (0.25, 0.75)}),
            child_depth=PowerChildDepth(2.0),
            max_depth=3,
        ),
    )

    assert first == equal
    assert hash(first) == hash(equal)
    assert all(first != variant for variant in variants)
    assert len({first, equal, *variants}) == 1 + len(variants)


def test_child_depth_distributions_supply_distinct_stable_mass() -> None:
    assert UniformChildDepth().weights((1, 2, 4)) == (1.0, 1.0, 1.0)
    assert PowerChildDepth(2.0).weights((1, 2, 4)) == pytest.approx(
        (1 / 16, 1 / 4, 1.0)
    )
    assert PowerChildDepth(-2.0).weights((1, 2, 4)) == pytest.approx(
        (1.0, 1 / 4, 1 / 16)
    )


def test_branching_copies_per_arity_weights_and_defaults_undeclared_arities() -> None:
    source = {2: [3.0, 1.0]}
    branching = ArityBranching(source)
    source[2][0] = 0.0

    assert branching.by_arity == {2: (3.0, 1.0)}
    assert branching.weights(2) == (3.0, 1.0)
    assert branching.weights(3) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "declaration",
    [
        {1: (1.0,)},
        {2: (1.0,)},
        {2: (0.0, 0.0)},
        {2: (-1.0, 1.0)},
        {2: (float("inf"), 1.0)},
    ],
)
def test_branching_refuses_invalid_per_arity_mass(
    declaration: dict[int, tuple[float, ...]],
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"branching|weight|mass"):
        ArityBranching(declaration)


def test_exhaustive_weightings_supply_equal_or_size_power_mass() -> None:
    store = TermStore(
        Signature(
            "weighting-values",
            ("Cat",),
            (Operator("n", "Cat"), Operator("u", "Cat", arg_sorts=("Cat",))),
        )
    )
    n = store.op(store.signature.by_name("n"))
    unary = store.op(store.signature.by_name("u"), n)

    assert UniformExhaustiveWeighting().weights((n, unary)) == (1.0, 1.0)
    assert SizeExhaustiveWeighting(2.0).weights((n, unary)) == pytest.approx(
        (0.25, 1.0)
    )


def test_pair_distributions_compare_and_hash_by_every_field() -> None:
    first = PairDistribution(max_depth=4)
    equal = PairDistribution(max_depth=4)
    variants = (PairDistribution(max_depth=5),)

    assert first == equal
    assert hash(first) == hash(equal)
    assert all(first != variant for variant in variants)


def test_type_reports_compare_independently_of_mapping_order() -> None:
    first = TypeSampleReport(3, {1: 2, 2: 1}, {"n": 4, "fwd": 1})
    equal = TypeSampleReport(3, {2: 1, 1: 2}, {"fwd": 1, "n": 4})
    variants = (
        TypeSampleReport(4, {1: 3, 2: 1}, {"n": 4, "fwd": 1}),
        TypeSampleReport(3, {1: 1, 2: 2}, {"n": 4, "fwd": 1}),
        TypeSampleReport(3, {1: 2, 2: 1}, {"n": 5, "fwd": 1}),
    )

    assert first == equal
    assert hash(first) == hash(equal)
    assert all(first != variant for variant in variants)


def test_pair_reports_compare_independently_of_mapping_order() -> None:
    first = PairSampleReport(4, {1: 2, 2: 2})
    equal = PairSampleReport(4, {2: 2, 1: 2})
    variants = (
        PairSampleReport(3, {1: 1, 2: 2}),
        PairSampleReport(4, {1: 3, 2: 1}),
    )

    assert first == equal
    assert hash(first) == hash(equal)
    assert all(first != variant for variant in variants)
