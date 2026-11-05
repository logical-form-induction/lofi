"""Recursive combinator decoder declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.layers import DEFAULT_CHOICE_HEADS, HeadKind

__all__ = [
    "RnnTermDecoderSettings",
    "RnnTermDecoderSpec",
    "TypeDirectedSettings",
    "TypeDirectedSpec",
]


class TypeDirectedSettings(NamedTuple):
    """Resolved type-directed decoding policy.

    Attributes
    ----------
    prefix : Literal["type-grammar", "none"]
        Whether the type grammar fixes the decoded term's outer type.
    mask : Literal["known", "none"]
        Whether known type declarations restrict decoded terms.
    """

    prefix: Literal["type-grammar", "none"]
    mask: Literal["known", "none"]


class TypeDirectedSpec(dx.Model, extra="forbid"):
    """Declare how typing constrains complete terms from a decoder."""

    prefix: Literal["type-grammar", "none"] = "none"
    mask: Literal["known", "none"] = "none"

    def settings(self) -> TypeDirectedSettings:
        """Return the frozen policy consumed by the decoder."""
        return TypeDirectedSettings(self.prefix, self.mask)


class RnnTermDecoderSettings(NamedTuple):
    """The settings a recursive term decoder is built from.

    Attributes
    ----------
    width : int
        The width of the state the recursion passes between slots.
    type_directed : TypeDirectedSettings
        How declared and inferred types restrict completed terms.
    choice_heads : HeadKind
        ``flat`` for one categorical over the choices, or ``factored`` for a
        family choice followed by the member.
    """

    width: int
    type_directed: TypeDirectedSettings
    choice_heads: HeadKind


class RnnTermDecoderSpec(TermDecoderSpec):
    """Declares a decoder recursing from a term's encoding down to its slots.

    Attributes
    ----------
    kind : Literal["tree-rnn"]
        Registry tag.
    width : int
        The width of the state the recursion passes between slots.
    type_directed : TypeDirectedSpec
        How the type grammar and known declarations constrain decoding.
    choice_heads : HeadKind
        ``flat`` for one categorical over the choices, or ``factored`` for a
        family choice followed by the member.
    """

    kind: Literal["tree-rnn"] = "tree-rnn"
    capabilities = frozenset(
        {
            DecoderCapability.FRONTIER,
            DecoderCapability.TEACHER_FORCED,
            DecoderCapability.MARGINAL,
        }
    )
    width: int
    type_directed: TypeDirectedSpec = dx.field(default_factory=TypeDirectedSpec)
    choice_heads: HeadKind = DEFAULT_CHOICE_HEADS

    __axioms__ = (
        dx.axiom("width > 0", message="a state of no coordinates holds nothing"),
    )

    def settings(self) -> RnnTermDecoderSettings:
        """Return the settings the plugin builds this component from."""
        return RnnTermDecoderSettings(
            width=self.width,
            type_directed=self.type_directed.settings(),
            choice_heads=self.choice_heads,
        )
