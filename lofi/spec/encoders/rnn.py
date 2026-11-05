"""Recurrent term encoder declaration."""

from typing import Final, Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.layers import DEFAULT_RECURRENT_CELL

__all__ = ["TreeRnnEncoderSettings", "TreeRnnEncoderSpec"]

DEFAULT_RECURRENT_LAYERS: Final = 1


class TreeRnnEncoderSettings(NamedTuple):
    """The settings a recurrent term encoder is built from.

    Attributes
    ----------
    width : int
        The width the recursion runs and returns at.
    cell : str
        The installed recurrent cell each layer steps with.
    layers : int
        How many recurrent layers each slot holds.
    norm_cap : float | None
        The largest norm a constructor vector is read at, or None.
    """

    width: int
    cell: str
    layers: int
    norm_cap: float | None
    inventory: Literal["uncached", "shared-dag"]


class TreeRnnEncoderSpec(TermEncoderSpec):
    """Declares an encoder running bottom up over a codec's slots.

    The cell names any installed ``lofi.recurrent_cell`` plugin. A one-layer
    encoder whose cell carries no memory apart from its hidden state also
    publishes exact synthetic-tree continuation from a live term vector. Soft
    controller raising requires this capability; multilayer encoders and encoders
    whose cell carries memory, such as ``lstm``, do not publish it.

    Attributes
    ----------
    kind : Literal["tree-rnn"]
        Registry tag.
    width : int
        The width the recursion runs and returns at.
    cell : str
        The installed recurrent cell each layer steps with.
    layers : int
        How many recurrent layers each slot holds.
    norm_cap : float | None
        The largest norm a constructor vector is read at, or None to read
        constructor vectors as stored.
    """

    kind: Literal["tree-rnn"] = "tree-rnn"
    width: int
    cell: str = DEFAULT_RECURRENT_CELL
    layers: int = DEFAULT_RECURRENT_LAYERS
    norm_cap: float | None = None
    inventory: Literal["uncached", "shared-dag"] = "uncached"

    __axioms__ = (
        dx.axiom("width > 0", message="a state of no coordinates holds nothing"),
        dx.axiom("layers > 0", message="a recurrence has a layer"),
    )

    @dx.validates("cell")
    @classmethod
    def _cell_is_named(cls, value: str) -> str:
        """Return a named recurrent-cell plugin."""
        return ensure_name("a recurrent cell", value)

    @dx.validates("norm_cap")
    @classmethod
    def _is_above_zero(cls, value: float | None) -> float | None:
        """Validate the norm cap."""
        if value is None:
            return None

        held = ensure_finite("a norm cap", value)

        if held <= 0.0:
            msg = f"a norm cap is above zero, not {value!r}"

            raise ValueError(msg)

        return held

    def settings(self) -> TreeRnnEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return TreeRnnEncoderSettings(
            width=self.width,
            cell=self.cell,
            layers=self.layers,
            norm_cap=self.norm_cap,
            inventory=self.inventory,
        )
