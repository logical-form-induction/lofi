"""Linear-recurrent term decoder declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi._types import ModuleSharing, SubmoduleDeclaration
from lofi.spec.components.model import DecoderCapability, TermDecoderSpec
from lofi.spec.layers import DEFAULT_CHOICE_HEADS, DEFAULT_RECURRENT_CELL, HeadKind

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["LinearRnnDecoderSettings", "LinearRnnDecoderSpec"]


class LinearRnnDecoderSettings(NamedTuple):
    """The resolved linear-recurrent decoder settings."""

    width: int
    cell: str
    layers: int
    choice_heads: HeadKind
    tie: str | None


class LinearRnnDecoderSpec(TermDecoderSpec):
    """Declare an autoregressive recurrence over linearized term choices."""

    kind: Literal["linear-rnn"] = "linear-rnn"
    capabilities = frozenset(
        {DecoderCapability.FRONTIER, DecoderCapability.TEACHER_FORCED}
    )
    width: int
    cell: str = DEFAULT_RECURRENT_CELL
    layers: int = 1
    choice_heads: HeadKind = DEFAULT_CHOICE_HEADS
    tie: str | None = None

    __axioms__ = (
        dx.axiom("width > 0", message="a state of no coordinates holds nothing"),
        dx.axiom("layers > 0", message="a recurrence has a layer"),
    )

    @dx.validates("cell")
    @classmethod
    def _cell_is_named(cls, value: str) -> str:
        """Return a named recurrent-cell plugin."""
        return ensure_name("a recurrent cell", value)

    @dx.validates("tie")
    @classmethod
    def _tie_is_named(cls, value: str | None) -> str | None:
        """Validate the encoder role whose choice embedding is tied."""
        return None if value is None else ensure_name("a choice-embedding tie", value)

    @override
    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Select the represented encoder's choice table when tying is declared."""
        if self.tie is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                "encoder": SubmoduleDeclaration(
                    "choice-embedding",
                    ModuleSharing(in_parameters=True, in_state_dict=True),
                )
            }
        )

    def settings(self) -> LinearRnnDecoderSettings:
        """Return the frozen settings consumed by the decoder plugin."""
        return LinearRnnDecoderSettings(
            self.width,
            self.cell,
            self.layers,
            self.choice_heads,
            self.tie,
        )
