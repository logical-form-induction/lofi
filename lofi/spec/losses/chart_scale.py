"""Chart scale loss declaration."""

import enum
from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.spec.components.loss import LossSpec

__all__ = ["ChartScaleLossSettings", "ChartScaleLossSpec", "ScalePenaltyName"]


class ScalePenaltyName(enum.StrEnum):
    """Names the penalty applied to a chart cell's norm."""

    SQUARED_RATIO = "squared-ratio"
    SQUARED_DISTANCE = "squared-distance"
    ABSOLUTE_DISTANCE = "absolute-distance"


class ChartScaleLossSettings(NamedTuple):
    """The settings a chart scale regularizer is built from.

    Attributes
    ----------
    target : float
        The Euclidean norm each chart cell is pulled toward.
    penalty : ScalePenaltyName
        The form of the departure penalty.
    """

    target: float
    penalty: ScalePenaltyName


class ChartScaleLossSpec(LossSpec):
    """Declares a penalty on chart cells whose norm departs from a target.

    Attributes
    ----------
    kind : Literal["chart-scale"]
        Registry tag.
    target : float
        The Euclidean norm each chart cell is pulled toward.
    penalty : ScalePenaltyName
        The form of the departure penalty.
    """

    kind: Literal["chart-scale"] = "chart-scale"
    target: float = 1.0
    penalty: ScalePenaltyName = ScalePenaltyName.SQUARED_RATIO

    @dx.validates("target")
    @classmethod
    def _target_is_a_length(cls, value: float) -> float:
        """Validate a finite, positive target norm."""
        held = ensure_finite("the target norm", value)

        if held <= 0.0:
            msg = f"the target norm is a positive length, not {value!r}"
            raise ValueError(msg)

        return held

    def settings(self) -> ChartScaleLossSettings:
        """Return the settings the plugin builds this objective from."""
        return ChartScaleLossSettings(target=self.target, penalty=self.penalty)

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the chart cells this regularizer consumes."""
        return frozenset({"cells"})
