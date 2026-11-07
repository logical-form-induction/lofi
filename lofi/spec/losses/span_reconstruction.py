"""Span-reconstruction loss declaration."""

import enum
from typing import TYPE_CHECKING, Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.loss import LossSpec

if TYPE_CHECKING:
    from lofi.spec.components.model import ModelSection

__all__ = [
    "SpanNormalizationName",
    "SpanReconstructionLossSettings",
    "SpanReconstructionLossSpec",
    "SpanWeightingName",
]


class SpanWeightingName(enum.StrEnum):
    """Names how much of the term each scored span takes."""

    UNIFORM = "uniform"
    MARGINAL = "marginal"


class SpanNormalizationName(enum.StrEnum):
    """Names the denominator of the reconstruction objective."""

    SPAN = "span"
    POSITION = "token"


class SpanReconstructionLossSettings(NamedTuple):
    """The settings a span-reconstruction objective is built from.

    Attributes
    ----------
    decoder : str
        The span-decoder role that scores each span's words.
    weighting : SpanWeightingName
        How much of the term each scored span takes.
    max_span_length : int | None
        Largest span reconstructed, or None for every represented length.
    normalization : SpanNormalizationName
        Whether scored spans or scored token positions form the denominator.
    """

    decoder: str
    weighting: SpanWeightingName
    max_span_length: int | None
    normalization: SpanNormalizationName


class SpanReconstructionLossSpec(LossSpec):
    """Declares reconstruction of each span's words from its chart cell.

    Attributes
    ----------
    kind : Literal["span-reconstruction"]
        Registry tag.
    decoder : str
        Span-decoder role supplied by the model graph.
    weighting : SpanWeightingName
        How much of the term each scored span takes.
    max_span_length : int | None
        Largest span reconstructed, or None for every represented length.
    normalization : SpanNormalizationName
        Whether scored spans or scored token positions form the denominator.
    """

    kind: Literal["span-reconstruction"] = "span-reconstruction"
    decoder: str
    weighting: SpanWeightingName = SpanWeightingName.UNIFORM
    max_span_length: int | None = None
    normalization: SpanNormalizationName = SpanNormalizationName.SPAN

    @dx.validates("decoder")
    @classmethod
    def _decoder_is_named(cls, value: str) -> str:
        """Validate a nonempty span-decoder role."""
        return ensure_name("a span-decoder role", value)

    @dx.validates("max_span_length")
    @classmethod
    def _maximum_is_positive(cls, value: int | None) -> int | None:
        """Validate an optional positive span-length bound."""
        if value is not None and value < 1:
            msg = f"a maximum span length is positive, not {value!r}"
            raise ValueError(msg)

        return value

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

    def settings(self) -> SpanReconstructionLossSettings:
        """Return the settings the plugin builds this objective from."""
        return SpanReconstructionLossSettings(
            decoder=self.decoder,
            weighting=self.weighting,
            max_span_length=self.max_span_length,
            normalization=self.normalization,
        )

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the cells and optional marginal weights this objective consumes."""
        required = {"cells"}

        if self.weighting is SpanWeightingName.MARGINAL:
            required.update({"alpha", "mu"})

        return frozenset(required)
