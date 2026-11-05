"""Holographic combinator decoder declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, override

from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.layers import DEFAULT_CHOICE_HEADS, HeadKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import TermCodec


__all__ = ["HrrTermDecoderSettings", "HrrTermDecoderSpec"]


class HrrTermDecoderSettings(NamedTuple):
    """The settings of a holographic decoder.

    Attributes
    ----------
    choice_heads : HeadKind
        ``flat`` for one categorical over the choices, or ``factored`` for a
        family choice followed by the member.
    """

    choice_heads: HeadKind


class HrrTermDecoderSpec(TermDecoderSpec):
    """Declares a decoder that unbinds each slot against a shared codebook.

    The codebook it unbinds with is the encoder's, so the decoder takes its
    width and its codes from the component it is wired to.

    Attributes
    ----------
    kind : Literal["hrr"]
        Registry tag.
    choice_heads : HeadKind
        How the unbound slots are read as choices.
    """

    kind: Literal["hrr"] = "hrr"
    capabilities = frozenset({DecoderCapability.TEACHER_FORCED})
    choice_heads: HeadKind = DEFAULT_CHOICE_HEADS

    @override
    def capabilities_for(self, codec: TermCodec, /) -> frozenset[DecoderCapability]:
        """Return the protocols supported under the codec's declared layout."""
        if not codec.layout.path_addressed:
            return self.capabilities
        return self.capabilities | {
            DecoderCapability.FRONTIER,
            DecoderCapability.MARGINAL,
        }

    @override
    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Return the encoder codebook this decoder trains and saves with it."""
        return MappingProxyType(
            {
                "encoder": SubmoduleDeclaration(
                    "codebook",
                    ModuleSharing(in_parameters=True, in_state_dict=True),
                )
            }
        )

    def settings(self) -> HrrTermDecoderSettings:
        """Return the settings the plugin builds this component from."""
        return HrrTermDecoderSettings(self.choice_heads)
