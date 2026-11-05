"""Transformer term encoder declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.layers import DEFAULT_ACTIVATION

if TYPE_CHECKING:
    from lofi.spec.layers import Activation, TransformerStructure

__all__ = [
    "TransformerTermEncoderSettings",
    "TransformerTermEncoderSpec",
]


class TransformerTermEncoderSettings(NamedTuple):
    """The settings a Transformer term encoder is built from.

    Attributes
    ----------
    width : int
        The model and output width.
    heads : int
        How many attention heads each layer holds.
    layers : int
        How many layers the stack holds.
    feedforward_width : int
        The hidden width of each feedforward block.
    structure : TransformerStructure
    activation : Activation
        The installed ``lofi.activation`` entry each feedforward block applies.
        The installed ``lofi.transformer_structure`` entry naming which relations
        of the linearization reach the attention.
    norm_cap : float | None
        The largest norm a symbol vector is read at, or None.
    """

    width: int
    heads: int
    layers: int
    feedforward_width: int
    structure: TransformerStructure
    activation: Activation
    norm_cap: float | None


class TransformerTermEncoderSpec(TermEncoderSpec):
    """Declares an encoder attending over a codec's slots.

    Attributes
    ----------
    kind : Literal["transformer"]
        Registry tag.
    width : int
        The model and output width.
    heads : int
        How many attention heads each layer holds.
    layers : int
        How many layers the stack holds.
    feedforward_width : int
        The hidden width of each feedforward block.
    structure : str
    activation : str
        The installed ``lofi.activation`` entry each feedforward block
        applies, ``gelu`` unless declared.
        Which relations of the linearization reach the attention, and how.
    norm_cap : float | None
        The largest norm a symbol vector is read at, or None to read symbol
        vectors as stored.
    """

    kind: Literal["transformer"] = "transformer"
    width: int
    heads: int
    layers: int
    feedforward_width: int
    structure: str = "ancestor-mask"
    activation: str = DEFAULT_ACTIVATION
    norm_cap: float | None = None

    __axioms__ = (
        dx.axiom("width > 0", message="a state of no coordinates holds nothing"),
        dx.axiom("heads > 0", message="an attention layer has a head"),
        dx.axiom("layers > 0", message="a stack has a layer"),
        dx.axiom("feedforward_width > 0", message="a block has a hidden coordinate"),
        dx.axiom(
            "heads > 0 and width % heads == 0",
            message="the heads divide the width they split between them",
        ),
    )

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

    @dx.validates("structure")
    @classmethod
    def _structure_is_named(cls, value: str) -> str:
        """Validate the installed structure plugin name."""
        return ensure_name("a Transformer structure", value)

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Validate the installed activation plugin name."""
        return ensure_name("a feedforward activation", value)

    def settings(self) -> TransformerTermEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return TransformerTermEncoderSettings(
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            feedforward_width=self.feedforward_width,
            activation=self.activation,
            structure=self.structure,
            norm_cap=self.norm_cap,
        )
