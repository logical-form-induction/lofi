"""Tests for shared logical-form normalization declarations."""

import didactic.api as dx
import pytest

from lofi.spec.normalization import NormalizationSpec


def test_defaults_build_with_owner_supplied_fuel() -> None:
    declared = NormalizationSpec()

    built = declared.build(fuel=37)

    assert built.fuel == 37
    assert built.order == "leftmost-outermost"
    assert built.eta == "off"
    assert built.associative_commutative == ()
    assert declared.conventions() == {
        "order": "leftmost-outermost",
        "policy": "eager",
        "eta": "off",
        "associative_commutative": [],
    }


def test_every_declared_normalization_choice_reaches_the_built_value() -> None:
    declared = NormalizationSpec(
        order="weak-head",
        policy="lazy",
        eta="contract",
        associative_commutative=("and", "or"),
    )

    built = declared.build(fuel=19)

    assert built.fuel == 19
    assert built.order == "weak-head"
    assert built.eta == "contract"
    assert built.associative_commutative == ("and", "or")
    assert declared.conventions() == {
        "order": "weak-head",
        "policy": "lazy",
        "eta": "contract",
        "associative_commutative": ["and", "or"],
    }


def test_a_quotient_symbol_is_declared_once() -> None:
    with pytest.raises(dx.ValidationError, match="symbols are distinct"):
        NormalizationSpec(associative_commutative=("and", "and"))
