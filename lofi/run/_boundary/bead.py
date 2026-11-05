"""Boundary documents for LoFI predictors served through bead."""

from typing import Literal

import didactic.api as dx
from bead.protocol import ResponseEncoding

__all__ = [
    "BeadPredictionDocument",
    "BeadServingSpec",
    "LoFIPredictorManifest",
    "decode_serving",
]


class BeadServingSpec(dx.Model, extra="forbid"):
    """Declare how bead items are served to one trained point.

    Parameters
    ----------
    response_encoding : ResponseEncoding
        The bead encoding naming the point's response columns.
    inputs : dict[str, str]
        The rendered element of every served item each input role the point
        publishes reads, keyed by role.
    """

    response_encoding: dx.Embed[ResponseEncoding]
    inputs: dict[str, str]

    @dx.validates("inputs")
    @classmethod
    def _inputs_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        for role, element in value.items():
            if not role.strip() or not element.strip():
                msg = f"a served input role and its element are named: {role!r}"
                raise ValueError(msg)
        return value


class LoFIPredictorManifest(dx.Model, extra="forbid"):
    """The strict wrapper around one unchanged LoFI Hub bundle."""

    format: Literal["lofi-bead-predictor"]
    hub_bundle: Literal["hub"]
    serving: BeadServingSpec


class BeadPredictionDocument(dx.Model, extra="forbid"):
    """One bead model prediction written by the serving boundary."""

    item_id: str
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float


def decode_serving(document: bytes, /) -> BeadServingSpec:
    """Validate one serving document at the bead serving boundary."""
    return BeadServingSpec.model_validate_json(document)
