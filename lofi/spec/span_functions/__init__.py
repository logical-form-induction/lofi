"""Declarations of vector functions used at neural-chart span slots."""

from typing import Literal, NamedTuple, Protocol

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.spec.layers import NormPlacement, dropout_rate

__all__ = [
    "BilinearSpanFunctionSettings",
    "BilinearSpanFunctionSpec",
    "CpBilinearSpanFunctionSettings",
    "CpBilinearSpanFunctionSpec",
    "DeclaredSpanFunction",
    "GatedMlpSpanFunctionSettings",
    "GatedMlpSpanFunctionSpec",
    "HrrSpanFunctionSettings",
    "HrrSpanFunctionSpec",
    "MlpSpanFunctionSettings",
    "MlpSpanFunctionSpec",
    "SpanFunctionDeclaration",
    "SpanFunctionSettings",
    "SpanFunctionSpec",
    "TransformerSpanFunctionSettings",
    "TransformerSpanFunctionSpec",
    "default_span_function",
]


class MlpSpanFunctionSettings(NamedTuple):
    """Resolved settings of a multilayer perceptron span function."""

    layers: int
    activation: str


class BilinearSpanFunctionSettings(NamedTuple):
    """Resolved settings of a full bilinear span function."""


class CpBilinearSpanFunctionSettings(NamedTuple):
    """Resolved settings of a CP-factorized bilinear span function."""

    rank: int


class GatedMlpSpanFunctionSettings(NamedTuple):
    """Resolved settings of a gated multilayer span function."""

    layers: int
    activation: str


class HrrSpanFunctionSettings(NamedTuple):
    """Resolved settings of a holographic binding span function."""

    binding: Literal["hadamard", "circular"]


class TransformerSpanFunctionSettings(NamedTuple):
    """Resolved settings of a self-attentive span function."""

    layers: int
    heads: int
    feedforward_width: int
    activation: str
    dropout: float
    norm_placement: NormPlacement


type SpanFunctionSettings = (
    MlpSpanFunctionSettings
    | BilinearSpanFunctionSettings
    | CpBilinearSpanFunctionSettings
    | GatedMlpSpanFunctionSettings
    | HrrSpanFunctionSettings
    | TransformerSpanFunctionSettings
)
"""Resolved settings of one span function variant."""


class SpanFunctionDeclaration(NamedTuple):
    """One registered span function tag and its immutable settings."""

    kind: str
    settings: SpanFunctionSettings


class DeclaredSpanFunction(Protocol):
    """A span function declaration that lowers at the chart boundary."""

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the registered tag and immutable interior settings."""
        ...


class SpanFunctionSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare one vector function used by a neural chart."""


class MlpSpanFunctionSpec(SpanFunctionSpec):
    """Apply affine maps and a declared activation to concatenated inputs."""

    kind: Literal["mlp"] = "mlp"
    layers: int
    activation: str

    @dx.validates("layers")
    @classmethod
    def _layers_are_positive(cls, value: int) -> int:
        """Return a positive layer count."""
        return ensure_positive("an MLP span function's layer count", value)

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Return a named activation plugin."""
        return ensure_name("an MLP span function's activation", value)

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the MLP plugin declaration."""
        return SpanFunctionDeclaration(
            self.kind, MlpSpanFunctionSettings(self.layers, self.activation)
        )


class BilinearSpanFunctionSpec(SpanFunctionSpec):
    """Apply a full bilinear map to exactly two inputs."""

    kind: Literal["bilinear"] = "bilinear"

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the full bilinear plugin declaration."""
        return SpanFunctionDeclaration(self.kind, BilinearSpanFunctionSettings())


class CpBilinearSpanFunctionSpec(SpanFunctionSpec):
    """Apply a CP-factorized bilinear map to exactly two inputs."""

    kind: Literal["cp-bilinear"] = "cp-bilinear"
    rank: int

    @dx.validates("rank")
    @classmethod
    def _rank_is_positive(cls, value: int) -> int:
        """Return a positive factorization rank."""
        return ensure_positive("a CP-bilinear span function's rank", value)

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the CP-bilinear plugin declaration."""
        return SpanFunctionDeclaration(
            self.kind, CpBilinearSpanFunctionSettings(self.rank)
        )


class GatedMlpSpanFunctionSpec(SpanFunctionSpec):
    """Interpolate a candidate and residual projection with a learned gate."""

    kind: Literal["gated-mlp"] = "gated-mlp"
    layers: int
    activation: str

    @dx.validates("layers")
    @classmethod
    def _layers_are_positive(cls, value: int) -> int:
        """Return a positive layer count."""
        return ensure_positive("a gated MLP span function's layer count", value)

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Return a named activation plugin."""
        return ensure_name("a gated MLP span function's activation", value)

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the gated-MLP plugin declaration."""
        return SpanFunctionDeclaration(
            self.kind, GatedMlpSpanFunctionSettings(self.layers, self.activation)
        )


class HrrSpanFunctionSpec(SpanFunctionSpec):
    """Bind two equal-width vectors holographically."""

    kind: Literal["hrr"] = "hrr"
    binding: Literal["hadamard", "circular"]

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the holographic plugin declaration."""
        return SpanFunctionDeclaration(self.kind, HrrSpanFunctionSettings(self.binding))


class TransformerSpanFunctionSpec(SpanFunctionSpec):
    """Compose projected inputs with a self-attentive encoder block."""

    kind: Literal["transformer"] = "transformer"
    layers: int
    heads: int
    feedforward_width: int
    activation: str
    dropout: float = 0.0
    norm_placement: Literal["pre", "post"] = "post"

    @dx.validates("layers")
    @classmethod
    def _layers_are_positive(cls, value: int) -> int:
        """Return a positive layer count."""
        return ensure_positive("a transformer span function's layer count", value)

    @dx.validates("heads")
    @classmethod
    def _heads_are_positive(cls, value: int) -> int:
        """Return a positive attention-head count."""
        return ensure_positive("a transformer span function's head count", value)

    @dx.validates("feedforward_width")
    @classmethod
    def _feedforward_width_is_positive(cls, value: int) -> int:
        """Return a positive feedforward width."""
        return ensure_positive(
            "a transformer span function's feed-forward width", value
        )

    @dx.validates("activation")
    @classmethod
    def _activation_is_named(cls, value: str) -> str:
        """Return a named activation plugin."""
        return ensure_name("a transformer span function's activation", value)

    @dx.validates("dropout")
    @classmethod
    def _dropout_is_a_rate(cls, value: float) -> float:
        """Return a dropout rate in the half-open unit interval."""
        return dropout_rate("a transformer span function's dropout rate", value)

    def declaration(self) -> SpanFunctionDeclaration:
        """Return the transformer plugin declaration."""
        return SpanFunctionDeclaration(
            self.kind,
            TransformerSpanFunctionSettings(
                self.layers,
                self.heads,
                self.feedforward_width,
                self.activation,
                self.dropout,
                self.norm_placement,
            ),
        )


def default_span_function() -> MlpSpanFunctionSpec:
    """Return the one-layer MLP declaration used by a base chart."""
    return MlpSpanFunctionSpec(layers=1, activation="tanh")
