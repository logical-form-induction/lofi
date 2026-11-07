"""Tests for the recursive type-sampler declaration."""

import didactic.api as dx
import pytest

from lofi.spec.components.trainer import TypeSamplerSpec
from lofi.spec.type_samplers.recursive import RecursiveTypeSamplerSpec


def test_the_recursive_tag_reaches_its_variant_through_the_family_root() -> None:
    declared = TypeSamplerSpec.model_validate({"kind": "recursive", "max_size": 9})

    assert type(declared) is RecursiveTypeSamplerSpec
    assert declared == RecursiveTypeSamplerSpec(max_size=9)
    assert TypeSamplerSpec.__variants__["recursive"] is RecursiveTypeSamplerSpec


@pytest.mark.parametrize("max_size", [0, -1, True, 1.5, "7"])
def test_a_nonpositive_or_noninteger_node_bound_is_refused(max_size: int) -> None:
    with pytest.raises(dx.ValidationError, match="max_size"):
        TypeSamplerSpec.model_validate({"kind": "recursive", "max_size": max_size})


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "recursive"}, "max_size"),
        ({"kind": "recursive", "max_size": 7, "replacement": False}, "replacement"),
    ],
)
def test_a_missing_or_extra_recursive_field_is_refused(
    payload: dict[str, int | str], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TypeSamplerSpec.model_validate(payload)


def test_the_recursive_variant_round_trips_through_json() -> None:
    document = '{"kind": "recursive", "max_size": 7, "declared_output": null}'
    declared = TypeSamplerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TypeSamplerSpec.model_validate_json(declared.model_dump_json()) == declared
