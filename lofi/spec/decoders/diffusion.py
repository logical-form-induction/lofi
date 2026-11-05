"""Diffusion combinator decoder declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.layers import (
    DEFAULT_ACTIVATION,
    DEFAULT_CHOICE_HEADS,
    Activation,
    HeadKind,
)

__all__ = ["DiffusionTermDecoderSettings", "DiffusionTermDecoderSpec"]


class DiffusionTermDecoderSettings(NamedTuple):
    """The settings a diffusion term decoder is built from.

    Attributes
    ----------
    width : int
        The stream width.
    layers : int
        How many denoising layers the stack holds.
    feedforward_width : int
        The hidden width of each feedforward block.
    noise_steps : int
        How many steps the masking schedule holds.
    first_masking : float
        The fraction of slots the first step of the schedule masks.
    activation : Activation
        The installed ``lofi.activation`` entry each feedforward block applies.
    """

    width: int
    layers: int
    feedforward_width: int
    noise_steps: int
    first_masking: float
    choice_heads: HeadKind
    activation: Activation


class DiffusionTermDecoderSpec(TermDecoderSpec):
    """Declares a decoder denoising every slot of a term at once.

    Attributes
    ----------
    kind : Literal["diffusion"]
        Registry tag.
    width : int
        The stream width.
    layers : int
        How many denoising layers the stack holds.
    feedforward_width : int
        The hidden width of each feedforward block.
    noise_steps : int
        How many steps the masking schedule holds.
    first_masking : float
        The fraction of slots the first step of the schedule masks, which rises
        evenly to one by the last step.
    choice_heads : HeadKind
        ``flat`` for one categorical over the choices, or ``factored`` for a
        family choice followed by the member.
    activation : str
        The installed ``lofi.activation`` entry each feedforward block
        applies, ``gelu`` unless declared.
    """

    kind: Literal["diffusion"] = "diffusion"
    capabilities = frozenset(
        {DecoderCapability.FRONTIER, DecoderCapability.TEACHER_FORCED}
    )
    width: int
    layers: int
    feedforward_width: int
    noise_steps: int
    first_masking: float
    choice_heads: HeadKind = DEFAULT_CHOICE_HEADS
    activation: str = DEFAULT_ACTIVATION

    __axioms__ = (
        dx.axiom("width > 0", message="a state of no coordinates holds nothing"),
        dx.axiom("layers > 0", message="a stack has a layer"),
        dx.axiom("feedforward_width > 0", message="a block has a hidden coordinate"),
        dx.axiom("noise_steps > 0", message="a schedule has a step"),
    )

    @dx.validates("first_masking")
    @classmethod
    def _is_a_fraction(cls, value: float) -> float:
        """Validate the first masking fraction."""
        held = ensure_finite("a masking fraction", value)

        if not 0.0 < held <= 1.0:
            msg = f"a masking fraction lies in (0, 1], not {value!r}"

            raise ValueError(msg)

        return held

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Validate the installed activation plugin name."""
        return ensure_name("a feedforward activation", value)

    def settings(self) -> DiffusionTermDecoderSettings:
        """Return the settings the plugin builds this component from."""
        return DiffusionTermDecoderSettings(
            width=self.width,
            layers=self.layers,
            feedforward_width=self.feedforward_width,
            noise_steps=self.noise_steps,
            first_masking=self.first_masking,
            choice_heads=self.choice_heads,
            activation=self.activation,
        )
