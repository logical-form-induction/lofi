"""Transformer combinator decoder declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.layers import (
    DEFAULT_ACTIVATION,
    DEFAULT_CHOICE_HEADS,
    Activation,
    HeadKind,
    TransformerStructure,
)

__all__ = [
    "TransformerTermDecoderSettings",
    "TransformerTermDecoderSpec",
]


class TransformerTermDecoderSettings(NamedTuple):
    """The settings a Transformer term decoder is built from.

    Attributes
    ----------
    width : int
        The model and stream width.
    heads : int
        How many attention heads each layer holds.
    layers : int
        How many layers the stack holds.
    feedforward_width : int
        The hidden width of each feedforward block.
    structure : TransformerStructure
        The installed ``lofi.transformer_structure`` entry shaping the attention
        stack.
    activation : Activation
        The installed ``lofi.activation`` entry each feedforward block applies.
    """

    width: int
    heads: int
    layers: int
    feedforward_width: int
    structure: TransformerStructure
    choice_heads: HeadKind
    activation: Activation


class TransformerTermDecoderSpec(TermDecoderSpec):
    """Declares a decoder attending over the slots it fills.

    Attributes
    ----------
    kind : Literal["transformer"]
        Registry tag.
    width : int
        The model and stream width.
    heads : int
        How many attention heads each layer holds.
    layers : int
        How many layers the stack holds.
    feedforward_width : int
        The hidden width of each feedforward block.
    structure : TransformerStructure
        The installed ``lofi.transformer_structure`` entry shaping the attention
        stack.
    choice_heads : HeadKind
        ``flat`` for one categorical over the choices, or ``factored`` for a
        family choice followed by the member.
    activation : str
        The installed ``lofi.activation`` entry each feedforward block
        applies, ``gelu`` unless declared.
    """

    kind: Literal["transformer"] = "transformer"
    capabilities = frozenset(
        {
            DecoderCapability.FRONTIER,
            DecoderCapability.TEACHER_FORCED,
            DecoderCapability.MARGINAL,
        }
    )
    width: int
    heads: int
    layers: int
    feedforward_width: int
    structure: TransformerStructure
    choice_heads: HeadKind = DEFAULT_CHOICE_HEADS
    activation: str = DEFAULT_ACTIVATION

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

    def settings(self) -> TransformerTermDecoderSettings:
        """Return the settings the plugin builds this component from."""
        return TransformerTermDecoderSettings(
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            feedforward_width=self.feedforward_width,
            structure=self.structure,
            choice_heads=self.choice_heads,
            activation=self.activation,
        )
