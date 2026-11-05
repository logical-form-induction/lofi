"""Tests for the exhaustive type-sampler declaration."""

from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import TypeSamplerSpec
from lofi.spec.type_samplers.exhaustive import (
    ExhaustiveTypeSamplerSpec,
    ExhaustiveWeightingSpec,
    SizeExhaustiveWeightingSettings,
    SizeExhaustiveWeightingSpec,
    UniformExhaustiveWeightingSettings,
    UniformExhaustiveWeightingSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_exhaustive_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = TypeSamplerSpec.model_validate(
        {
            "kind": "exhaustive",
            "max_size": 9,
            "weighting": {"kind": "size", "exponent": 2.0},
        }
    )

    assert type(declared) is ExhaustiveTypeSamplerSpec
    assert declared == ExhaustiveTypeSamplerSpec(
        max_size=9, weighting=SizeExhaustiveWeightingSpec(exponent=2.0)
    )
    assert TypeSamplerSpec.__variants__["exhaustive"] is ExhaustiveTypeSamplerSpec


def test_exhaustive_weighting_defaults_to_uniform() -> None:
    declared = ExhaustiveTypeSamplerSpec(max_size=5)

    assert declared.weighting == UniformExhaustiveWeightingSpec()
    assert declared.settings().weighting == UniformExhaustiveWeightingSettings()


def test_exhaustive_weighting_is_an_open_tagged_union() -> None:
    uniform = ExhaustiveWeightingSpec.model_validate({"kind": "uniform"})
    size = ExhaustiveWeightingSpec.model_validate({"kind": "size", "exponent": -0.5})

    assert type(uniform) is UniformExhaustiveWeightingSpec
    assert type(size) is SizeExhaustiveWeightingSpec
    assert size.settings() == SizeExhaustiveWeightingSettings(-0.5)
    assert (
        ExhaustiveWeightingSpec.__variants__["uniform"]
        is UniformExhaustiveWeightingSpec
    )
    assert ExhaustiveWeightingSpec.__variants__["size"] is SizeExhaustiveWeightingSpec


@pytest.mark.parametrize("max_size", [0, -1, True, 1.5, "7"])
def test_a_nonpositive_or_noninteger_node_bound_is_refused(max_size: int) -> None:
    with pytest.raises(dx.ValidationError, match="max_size"):
        TypeSamplerSpec.model_validate({"kind": "exhaustive", "max_size": max_size})


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "exhaustive"}, "max_size"),
        (
            {
                "kind": "exhaustive",
                "max_size": 7,
                "weighting": {"kind": "depth"},
            },
            "weighting",
        ),
        (
            {"kind": "exhaustive", "max_size": 7, "replacement": False},
            "replacement",
        ),
        ({"kind": "invented", "max_size": 7}, "'invented'"),
    ],
)
def test_a_missing_unknown_or_extra_exhaustive_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TypeSamplerSpec.model_validate(payload)


def test_the_exhaustive_variant_round_trips_through_json() -> None:
    document = (
        '{"kind": "exhaustive", "max_size": 7, '
        '"weighting": {"kind": "size", "exponent": 1.5}, '
        '"declared_output": null}'
    )
    declared = TypeSamplerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TypeSamplerSpec.model_validate_json(declared.model_dump_json()) == declared
