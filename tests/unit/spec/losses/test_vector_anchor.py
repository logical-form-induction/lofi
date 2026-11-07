from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.vector_anchor import (
    AnchorAggregateName,
    AnchorDistanceName,
    VectorAnchorLossSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_variant_declares_its_constraint_readout_dependency() -> None:
    declared = LossSpec.model_validate({"kind": "vector-anchor", "readout": "types"})

    assert type(declared) is VectorAnchorLossSpec
    assert LossSpec.__variants__["vector-anchor"] is VectorAnchorLossSpec
    assert declared.dependencies() == (Dependency("types", "constraint-readout"),)
    assert declared.declared_output is None
    assert declared.distance is AnchorDistanceName.COSINE
    assert declared.aggregate is AnchorAggregateName.NEAREST
    assert declared.margin == 1.0


def test_the_variant_consumes_the_target_constraint_route() -> None:
    assert VectorAnchorLossSpec(readout="types").constraint_route() == "target"


def test_distance_aggregation_and_margin_are_declared() -> None:
    declared = LossSpec.model_validate(
        {
            "kind": "vector-anchor",
            "readout": "types",
            "distance": "squared-euclidean",
            "aggregate": "mean",
            "margin": 2.5,
        }
    )

    assert declared == VectorAnchorLossSpec(
        readout="types",
        distance=AnchorDistanceName.SQUARED_EUCLIDEAN,
        aggregate=AnchorAggregateName.MEAN,
        margin=2.5,
    )


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "vector-anchor"}, "readout"),
        ({"kind": "vector-anchor", "readout": ""}, "vector-anchor readout role"),
        ({"kind": "vector-anchor", "readout": True}, "readout"),
        ({"kind": "vector-anchor", "readout": "types", "distance": "dot"}, "distance"),
        (
            {"kind": "vector-anchor", "readout": "types", "aggregate": "sum"},
            "aggregate",
        ),
        ({"kind": "vector-anchor", "readout": "types", "margin": -0.1}, "nonnegative"),
        ({"kind": "vector-anchor", "readout": "types", "unknown": 0.1}, "unknown"),
    ],
)
def test_a_missing_empty_wrong_typed_or_extra_field_is_refused(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        LossSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "vector-anchor", "readout": "types", "distance": "cosine", '
        '"aggregate": "nearest", "margin": 1.0, "declared_output": null}'
    )
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared
