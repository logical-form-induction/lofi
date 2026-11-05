"""Holographic term encoder declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi.spec.components.model import TermEncoderSpec

__all__ = ["HrrTermEncoderSettings", "HrrTermEncoderSpec"]


class HrrTermEncoderSettings(NamedTuple):
    """The settings a holographic term encoder is built from.

    Attributes
    ----------
    width : int
        The dimension of every atom.
    magnitude : float
        Where each coordinate of an atom starts, before its noise.
    floor : float
        The smallest magnitude any coordinate of an atom is read at.
    """

    width: int
    magnitude: float
    floor: float


class HrrTermEncoderSpec(TermEncoderSpec):
    """Declares an encoder that binds a codec's addresses to its constructor codes.

    Attributes
    ----------
    kind : Literal["hrr"]
        Registry tag.
    width : int
        The dimension of every atom, which is also the encoder's output width.
    magnitude : float
        Where each coordinate of an atom starts, before its noise.
    floor : float
        The smallest magnitude any coordinate of an atom is read at.
    """

    kind: Literal["hrr"] = "hrr"
    width: int
    magnitude: float = 1.0
    floor: float = 1e-3

    __axioms__ = (
        dx.axiom("width > 0", message="an atom of no coordinates holds nothing"),
        dx.axiom("magnitude > 0", message="an atom starts away from the origin"),
        dx.axiom("floor > 0", message="a floor is above zero"),
        dx.axiom(
            "floor <= magnitude",
            message="a floor is no larger than the magnitude it bounds",
        ),
    )

    @override
    def publications(self) -> frozenset[str]:
        """Return the codebook, which a holographic decoder unbinds with."""
        return frozenset({"codebook"})

    def settings(self) -> HrrTermEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return HrrTermEncoderSettings(
            width=self.width,
            magnitude=self.magnitude,
            floor=self.floor,
        )
