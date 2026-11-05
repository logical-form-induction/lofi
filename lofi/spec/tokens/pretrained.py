"""Pretrained token encoder declaration."""

from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.artifacts import PretrainedSourceSpec, source_artifact
from lofi.spec.components.model import (
    _DEFAULT_TOKEN_ENCODER_TRAINABLE,
    TokenEncoderSpec,
)

if TYPE_CHECKING:
    from lofi._artifacts import PinnedCheckpoint

__all__ = [
    "ArchitectureName",
    "ContextWindowSettings",
    "ContextWindowSpec",
    "FixedLayerReadoutSettings",
    "FixedLayerReadoutSpec",
    "LastLayerReadoutSettings",
    "LastLayerReadoutSpec",
    "LayerReadoutSettings",
    "LayerReadoutSpec",
    "OverflowName",
    "PretrainedTokenEncoderSettings",
    "PretrainedTokenEncoderSpec",
    "ScalarMixReadoutSettings",
    "ScalarMixReadoutSpec",
]

type ArchitectureName = str
"""Which stack of a checkpoint an encoder runs.

An ``encoder-only`` checkpoint is run whole. A ``t5-encoder`` checkpoint is an
encoder-decoder of the T5 family whose encoder stack is run without its decoder.
"""

type OverflowName = str
"""What an encoder does with an item its window cannot hold whole."""

_DEFAULT_WORD_JOIN: Final = " "


class ContextWindowSettings(NamedTuple):
    """The bound on the rows one item may take.

    Attributes
    ----------
    rows : int | None
        How many rows one item may take, or None to derive the checkpoint's bound.
    overflow : OverflowName
        Whether a longer item is refused or encoded to its longest fitting prefix.
    """

    rows: int | None
    overflow: OverflowName


class ContextWindowSpec(dx.Model, extra="forbid"):
    """Declares how many rows one item may take and what a longer one gets.

    Attributes
    ----------
    rows : int | None
        How many rows one item may take, at least one, or None to derive the
        checkpoint's bound.
    overflow : OverflowName
        Whether a longer item is refused outright or encoded to its longest
        fitting prefix with the dropped words named.
    """

    rows: int | None = None
    overflow: OverflowName = "report"

    @dx.validates("rows")
    @classmethod
    def _rows_are_positive_or_derived(cls, value: int | None) -> int | None:
        """Validate an explicit row bound, or request the checkpoint capability."""
        if value is not None and value <= 0:
            msg = "a window of no rows holds no word"
            raise ValueError(msg)

        return value

    @dx.validates("overflow")
    @classmethod
    def _overflow_is_named(cls, value: str) -> str:
        """Validate the installed overflow-policy plugin name."""
        return ensure_name("an overflow policy", value)

    def settings(self) -> ContextWindowSettings:
        """Return the settings the plugin builds this component from."""
        return ContextWindowSettings(
            rows=self.rows,
            overflow=self.overflow,
        )


_DEFAULT_CONTEXT_WINDOW: Final = ContextWindowSpec()


class LayerReadoutSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how a pretrained stack's hidden layers become token rows."""


class LastLayerReadoutSettings(NamedTuple):
    """Select the stack's final hidden layer."""

    kind: Literal["last"] = "last"


class LastLayerReadoutSpec(LayerReadoutSpec):
    """Declare the final hidden layer as the token representation."""

    kind: Literal["last"] = "last"

    def settings(self) -> LastLayerReadoutSettings:
        """Return the final-layer readout settings."""
        return LastLayerReadoutSettings()


class FixedLayerReadoutSettings(NamedTuple):
    """Select one hidden-state tuple position."""

    layer: int


class FixedLayerReadoutSpec(LayerReadoutSpec):
    """Declare one numbered hidden layer as the token representation."""

    kind: Literal["fixed"] = "fixed"
    layer: int

    @dx.validates("layer")
    @classmethod
    def _layer_is_nonnegative(cls, value: int) -> int:
        """Validate a hidden-state tuple position."""
        if value < 0:
            msg = "a fixed pretrained layer is nonnegative"
            raise ValueError(msg)

        return value

    def settings(self) -> FixedLayerReadoutSettings:
        """Return the selected hidden-state tuple position."""
        return FixedLayerReadoutSettings(self.layer)


class ScalarMixReadoutSettings(NamedTuple):
    """Select a trainable normalized mixture over every hidden layer."""

    kind: Literal["scalar-mix"] = "scalar-mix"


class ScalarMixReadoutSpec(LayerReadoutSpec):
    """Declare a trainable scalar mixture over every hidden layer."""

    kind: Literal["scalar-mix"] = "scalar-mix"

    def settings(self) -> ScalarMixReadoutSettings:
        """Return the scalar-mixture settings."""
        return ScalarMixReadoutSettings()


type LayerReadoutSettings = (
    LastLayerReadoutSettings | FixedLayerReadoutSettings | ScalarMixReadoutSettings
)


class PretrainedTokenEncoderSettings(NamedTuple):
    """The settings a pretrained token encoder is built from.

    Attributes
    ----------
    checkpoint : PinnedCheckpoint
        The release the encoder and its tokenizer are fetched from.
    architecture : ArchitectureName
        Which stack of the checkpoint is run.
    trainable : bool
        Whether the loaded parameters take gradients.
    window : ContextWindowSettings
        The bound on one item's rows.
    """

    checkpoint: PinnedCheckpoint
    architecture: ArchitectureName
    trainable: bool
    window: ContextWindowSettings
    readout: LayerReadoutSettings
    join: str


class PretrainedTokenEncoderSpec(TokenEncoderSpec):
    """Declares a token encoder loaded from a pretrained checkpoint.

    Attributes
    ----------
    kind : Literal["pretrained"]
        Registry tag.
    checkpoint : PretrainedSourceSpec
        The hub repository and commit the encoder and its tokenizer are
        fetched from.
    architecture : ArchitectureName
        Which stack of the checkpoint is run: the whole of an encoder-only
        checkpoint, or the encoder stack of a T5-family encoder-decoder.
    trainable : bool
        Whether the loaded parameters take gradients.
    window : ContextWindowSpec
        The bound on one item's rows.
    """

    kind: Literal["pretrained"] = "pretrained"
    checkpoint: PretrainedSourceSpec
    architecture: ArchitectureName = "encoder-only"
    trainable: bool = _DEFAULT_TOKEN_ENCODER_TRAINABLE
    window: ContextWindowSpec = _DEFAULT_CONTEXT_WINDOW
    readout: LayerReadoutSpec = dx.field(default_factory=LastLayerReadoutSpec)
    join: str = _DEFAULT_WORD_JOIN

    @dx.validates("architecture")
    @classmethod
    def _architecture_is_named(cls, value: str) -> str:
        """Validate the installed encoder-architecture plugin name."""
        return ensure_name("an encoder architecture", value)

    @dx.validates("join")
    @classmethod
    def _join_is_not_empty(cls, value: str) -> str:
        """Validate the string placed between input words."""
        if not isinstance(value, str):
            msg = f"a word join is a string, not {value!r}"
            raise TypeError(msg)

        if not value:
            msg = "a word join is not empty"
            raise ValueError(msg)

        return value

    @dx.model_validator()
    def _loads_a_hub_checkpoint(self) -> Self:
        """Refuse a source that is not a hub repository at one commit."""
        self._checkpoint()

        return self

    @override
    def pretrained_weights(self) -> str:
        """Return the pinned checkpoint the encoder's weights are loaded from."""
        return self._checkpoint().identity

    def _checkpoint(self) -> PinnedCheckpoint:
        """Return the pinned release the declared source names."""
        return source_artifact(self.checkpoint).hub_checkpoint(
            "a pretrained token encoder"
        )

    def settings(self) -> PretrainedTokenEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return PretrainedTokenEncoderSettings(
            checkpoint=self._checkpoint(),
            architecture=self.architecture,
            trainable=self.trainable,
            window=self.window.settings(),
            readout=cast(
                "LastLayerReadoutSpec | FixedLayerReadoutSpec | ScalarMixReadoutSpec",
                self.readout,
            ).settings(),
            join=self.join,
        )
