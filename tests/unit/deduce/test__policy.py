"""Tests for immutable symbolic parse policy."""

import copy
from typing import cast

import pytest

from lofi.deduce import (
    CKY,
    CONTIGUOUS_SPANS,
    DEFAULT_REFINEMENT_PAIRS,
    NATIVE_CHARTS,
    LogInside,
    ParsePolicy,
    ParseSettings,
    Reachability,
    Retain,
    SpanIndex,
    TopKPruner,
)


def test_a_parse_policy_names_retention_and_reachability_once() -> None:
    """Keep both algorithm choices in one immutable interior value."""
    policy = ParsePolicy(Retain.INSIDE, Reachability.ROOTS)

    assert policy.retain is Retain.INSIDE
    assert policy.reachability is Reachability.ROOTS
    assert policy.refinement_pairs == DEFAULT_REFINEMENT_PAIRS
    assert copy.copy(policy) is policy
    assert copy.deepcopy(policy) is policy

    with pytest.raises(AttributeError, match="parse policy is immutable"):
        policy.__setattr__("retain", Retain.FOREST)


def test_the_builtin_parse_capabilities_name_and_build_the_current_algorithm() -> None:
    """Expose the CKY, contiguous-span, and native-chart choices as capabilities."""
    assert CKY.name == "cky"
    assert CONTIGUOUS_SPANS.name == "contiguous"
    assert NATIVE_CHARTS.name == "native"
    assert CONTIGUOUS_SPANS.build(3).num_spans == SpanIndex(3).num_spans


def test_parse_policies_compare_and_hash_by_both_choices() -> None:
    """Make policy identity independent of object allocation."""
    inside_roots = ParsePolicy(Retain.INSIDE, Reachability.ROOTS)

    assert inside_roots == ParsePolicy(Retain.INSIDE, Reachability.ROOTS)
    assert inside_roots != ParsePolicy(Retain.FOREST, Reachability.ROOTS)
    assert inside_roots != ParsePolicy(Retain.INSIDE, Reachability.ALL)
    assert len({inside_roots, ParsePolicy(Retain.INSIDE, Reachability.ROOTS)}) == 1


def test_parse_policies_compare_and_hash_by_their_refinement_cap() -> None:
    """Distinguish two root-reachability plans that refine differently."""
    capped = ParsePolicy(Retain.FOREST, Reachability.ROOTS, refinement_pairs=7)

    assert capped.refinement_pairs == 7
    assert capped == ParsePolicy(Retain.FOREST, Reachability.ROOTS, refinement_pairs=7)
    assert capped != ParsePolicy(Retain.FOREST, Reachability.ROOTS)
    assert len({capped, ParsePolicy(Retain.FOREST, Reachability.ROOTS)}) == 2
    assert "refinement_pairs=7" in repr(capped)


def test_a_parse_policy_refuses_a_negative_refinement_cap() -> None:
    """Refuse a refinement cap that bounds nothing."""
    with pytest.raises(ValueError, match="refinement_pairs"):
        ParsePolicy(Retain.FOREST, Reachability.ROOTS, refinement_pairs=-1)


@pytest.mark.parametrize(
    ("retain", "reachability", "message"),
    [
        ("inside", Reachability.ROOTS, "retention"),
        (Retain.INSIDE, "roots", "reachability"),
    ],
)
def test_a_parse_policy_refuses_unlowered_boundary_names(
    retain: object, reachability: object, message: str
) -> None:
    """Accept only lowered enum values below the declaration boundary."""
    with pytest.raises(TypeError, match=message):
        ParsePolicy(cast("Retain", retain), cast("Reachability", reachability))


def test_parse_settings_measure_pruned_mass_only_under_a_pruner() -> None:
    """Refuse a mass measurement that has no pruner to measure."""
    policy = ParsePolicy(Retain.FOREST, Reachability.ALL)
    measured = ParseSettings(
        policy,
        CKY,
        CONTIGUOUS_SPANS,
        NATIVE_CHARTS,
        TopKPruner[object](2),
        LogInside(),
        prune_lexical=False,
        measure_pruned_mass=True,
    )

    assert measured.measure_pruned_mass

    with pytest.raises(ValueError, match="declare no pruner"):
        ParseSettings(
            policy,
            CKY,
            CONTIGUOUS_SPANS,
            NATIVE_CHARTS,
            None,
            LogInside(),
            prune_lexical=False,
            measure_pruned_mass=True,
        )
