"""Outside-vector word-reconstruction loss declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.loss import LossSpec

if TYPE_CHECKING:
    from lofi.spec.components.model import ModelSection

__all__ = [
    "OutsideReconstructionLossSettings",
    "OutsideReconstructionLossSpec",
]


class OutsideReconstructionLossSettings(NamedTuple):
    """The settings an outside-reconstruction objective is built from.

    Attributes
    ----------
    decoder : str
        The span-decoder role that reconstructs each word.
    """

    decoder: str


class OutsideReconstructionLossSpec(LossSpec):
    """Declare reconstruction of each word from its contextual outside vector.

    This is the DIORA-style leaf objective: it predicts a word from the context
    outside that word's own leaf span. It is distinct from span reconstruction,
    which predicts every span's complete yield from the span's inside vector.

    Attributes
    ----------
    kind : Literal["outside-reconstruction"]
        Registry tag.
    decoder : str
        Span-decoder role supplied by the model graph.
    """

    kind: Literal["outside-reconstruction"] = "outside-reconstruction"
    decoder: str

    @dx.validates("decoder")
    @classmethod
    def _decoder_is_named(cls, value: str) -> str:
        """Validate a nonempty span-decoder role."""
        return ensure_name("a span-decoder role", value)

    @override
    def refuse_model(self, model: ModelSection, path: str, /) -> None:
        """Refuse a span decoder the model does not declare.

        Raises
        ------
        ValueError
            If ``decoder`` is not one of the model's span decoders.
        """
        if self.decoder not in model.span_decoders:
            msg = (
                f"{path}.decoder: the objective names span decoder "
                f"{self.decoder!r}, which is not declared among "
                f"{sorted(model.span_decoders)}"
            )
            raise ValueError(msg)

    def settings(self) -> OutsideReconstructionLossSettings:
        """Return the settings the plugin builds this objective from."""
        return OutsideReconstructionLossSettings(decoder=self.decoder)

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the contextual outside vectors this objective consumes."""
        return frozenset({"outside_cells"})
