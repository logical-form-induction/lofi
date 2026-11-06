"""Vector-anchor loss declaration."""

import enum
from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec._ports import Dependency
from lofi.spec.components.loss import LossSpec

__all__ = [
    "AnchorAggregateName",
    "AnchorDistanceName",
    "VectorAnchorLossSettings",
    "VectorAnchorLossSpec",
]


class AnchorDistanceName(enum.StrEnum):
    """Names the distance between a chart cell and an anchor vector."""

    COSINE = "cosine"
    SQUARED_EUCLIDEAN = "squared-euclidean"


class AnchorAggregateName(enum.StrEnum):
    """Names how distances to several permitted anchors are combined."""

    NEAREST = "nearest"
    MEAN = "mean"


class VectorAnchorLossSettings(NamedTuple):
    """The settings a vector-anchor objective is built from.

    Attributes
    ----------
    readout : str
        The term-readout role whose constraint targets are anchored.
    distance : AnchorDistanceName
        Distance used in the represented space.
    aggregate : AnchorAggregateName
        How distances to several permitted anchors are combined.
    margin : float
        Minimum distance required by a negated constraint.
    """

    readout: str
    distance: AnchorDistanceName
    aggregate: AnchorAggregateName
    margin: float


class VectorAnchorLossSpec(LossSpec):
    """Declares vector-space anchors for constrained chart cells.

    The anchors are the term encoder's vectors for the constraint targets held by one
    installed term-readout role.

    Attributes
    ----------
    kind : Literal["vector-anchor"]
        Registry tag.
    readout : str
        Term-readout role whose constraint targets are anchored.
    distance : AnchorDistanceName
        Distance used in the represented space.
    aggregate : AnchorAggregateName
        How distances to several permitted anchors are combined.
    margin : float
        Finite nonnegative distance required by a negated constraint.
    """

    kind: Literal["vector-anchor"] = "vector-anchor"
    readout: str
    distance: AnchorDistanceName = AnchorDistanceName.COSINE
    aggregate: AnchorAggregateName = AnchorAggregateName.NEAREST
    margin: float = 1.0

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Validate a nonempty term-readout role."""
        return ensure_name("a vector-anchor readout role", value)

    @dx.validates("margin")
    @classmethod
    def _margin_is_nonnegative(cls, value: float) -> float:
        """Validate a finite nonnegative exclusion margin."""
        held = ensure_finite("a vector-anchor margin", value)

        if held < 0.0:
            msg = f"a vector-anchor margin is nonnegative, not {value!r}"
            raise ValueError(msg)

        return held

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared term-readout dependency."""
        return (Dependency(self.readout, "constraint-readout"),)

    def settings(self) -> VectorAnchorLossSettings:
        """Return the settings the plugin builds this objective from."""
        return VectorAnchorLossSettings(
            readout=self.readout,
            distance=self.distance,
            aggregate=self.aggregate,
            margin=self.margin,
        )

    def constraint_route(self) -> Literal["target"]:
        """Return the selected-constraint representation this objective consumes."""
        return "target"
