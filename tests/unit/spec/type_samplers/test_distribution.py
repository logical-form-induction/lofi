"""Tests for recursively sampled term-shape declarations."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.type_samplers.distribution import (
    ArityBranchingSettings,
    ArityBranchingSpec,
    BranchingAritySettings,
    BranchingAritySpec,
    ChildDepthSpec,
    PowerChildDepthSettings,
    PowerChildDepthSpec,
    UniformChildDepthSettings,
    UniformChildDepthSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_child_depth_is_a_tagged_distribution() -> None:
    uniform = ChildDepthSpec.model_validate({"kind": "uniform"})
    power = ChildDepthSpec.model_validate({"kind": "power", "exponent": 2.5})

    assert type(uniform) is UniformChildDepthSpec
    assert type(power) is PowerChildDepthSpec
    assert uniform.settings() == UniformChildDepthSettings()
    assert power.settings() == PowerChildDepthSettings(2.5)
    assert ChildDepthSpec.__variants__ == {
        "uniform": UniformChildDepthSpec,
        "power": PowerChildDepthSpec,
    }


@pytest.mark.parametrize("exponent", [float("inf"), float("-inf"), float("nan")])
def test_child_depth_refuses_a_nonfinite_power(exponent: float) -> None:
    with pytest.raises(dx.ValidationError, match="exponent"):
        ChildDepthSpec.model_validate({"kind": "power", "exponent": exponent})


def test_branching_resolves_one_row_per_arity_in_declaration_order() -> None:
    declaration = ArityBranchingSpec(
        by_arity=(
            BranchingAritySpec(arity=2, deepest=(0.75, 0.25)),
            BranchingAritySpec(arity=3, deepest=(0.2, 0.3, 0.5)),
        )
    )

    assert declaration.settings() == ArityBranchingSettings(
        (
            BranchingAritySettings(2, (0.75, 0.25)),
            BranchingAritySettings(3, (0.2, 0.3, 0.5)),
        )
    )


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"arity": 1, "deepest": [1.0]}, "arity"),
        ({"arity": 2, "deepest": [1.0]}, "weights"),
        ({"arity": 2, "deepest": [0.0, 0.0]}, "mass"),
        ({"arity": 2, "deepest": [-1.0, 1.0]}, "negative"),
        ({"arity": 2, "deepest": [1.0, float("inf")]}, "finite"),
    ],
)
def test_a_branching_row_refuses_invalid_deepest_child_mass(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        BranchingAritySpec.model_validate(payload)


def test_branching_refuses_duplicate_arities() -> None:
    with pytest.raises(dx.ValidationError, match="more than once"):
        ArityBranchingSpec.model_validate(
            {
                "by_arity": [
                    {"arity": 2, "deepest": [1.0, 1.0]},
                    {"arity": 2, "deepest": [0.25, 0.75]},
                ]
            }
        )


def test_term_shape_declarations_round_trip_through_json() -> None:
    child = ChildDepthSpec.model_validate_json('{"kind": "power", "exponent": -1.0}')
    branching = ArityBranchingSpec.model_validate_json(
        '{"by_arity": [{"arity": 2, "deepest": [0.25, 0.75]}]}'
    )

    assert ChildDepthSpec.model_validate_json(child.model_dump_json()) == child
    assert (
        ArityBranchingSpec.model_validate_json(branching.model_dump_json()) == branching
    )
