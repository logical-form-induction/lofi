"""Static token encoder declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.components.model import (
    _DEFAULT_TOKEN_ENCODER_TRAINABLE,
    TokenEncoderSpec,
)

__all__ = [
    "OovInitializationName",
    "StaticTokenEncoderSettings",
    "StaticTokenEncoderSpec",
    "VectorsFormat",
]

type VectorsFormat = str
"""The text layout a file of fixed vectors is read under."""

type OovInitializationName = str
"""How rows absent from a static-vector file are initialized."""


class StaticTokenEncoderSettings(NamedTuple):
    """The settings a static token encoder is built from.

    Attributes
    ----------
    vectors : str
        The absolute path of the file the vectors are read from.
    vectors_format : VectorsFormat
        Which text layout that file is written in.
    trainable : bool
        Whether the loaded embedding rows take gradients.
    """

    vectors: str
    vectors_format: VectorsFormat
    trainable: bool
    lowercase: bool
    oov_init: OovInitializationName
    oov_init_std: float
    seed: int


class StaticTokenEncoderSpec(TokenEncoderSpec):
    """Declares a token encoder backed by a file of fixed vectors.

    Attributes
    ----------
    kind : Literal["static"]
        Registry tag.
    vectors : str
        The file the vectors are read from.
    vectors_format : VectorsFormat
        Which text layout that file is written in.
    trainable : bool
        Whether the loaded embedding rows take gradients.
    """

    kind: Literal["static"] = "static"
    vectors: str
    vectors_format: VectorsFormat = "glove"
    trainable: bool = _DEFAULT_TOKEN_ENCODER_TRAINABLE
    lowercase: bool = False
    oov_init: str = "zeros"
    oov_init_std: float = 0.02
    seed: int = 0

    @dx.validates("vectors")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Validate the vectors file."""
        return ensure_name("a vectors file", value)

    @dx.validates("vectors_format")
    @classmethod
    def _format_is_named(cls, value: str) -> str:
        """Validate the installed vector-format plugin name."""
        return ensure_name("a vectors format", value)

    @dx.validates("oov_init")
    @classmethod
    def _oov_initialization_is_named(cls, value: str) -> str:
        """Validate the installed OOV-initialization plugin name."""
        return ensure_name("an OOV initialization", value)

    @dx.validates("oov_init_std")
    @classmethod
    def _oov_standard_deviation_is_nonnegative(cls, value: float) -> float:
        """Validate the declared OOV initialization scale."""
        checked = ensure_finite("an OOV initialization standard deviation", value)

        if checked < 0:
            msg = "an OOV initialization standard deviation is nonnegative"
            raise ValueError(msg)

        return checked

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int) -> int:
        """Validate the dedicated initialization seed."""
        return ensure_non_negative("an OOV initialization seed", value)

    def settings(self) -> StaticTokenEncoderSettings:
        """Return the settings the plugin builds this component from."""
        return StaticTokenEncoderSettings(
            vectors=lower_package_resource(self.vectors),
            vectors_format=self.vectors_format,
            trainable=self.trainable,
            lowercase=self.lowercase,
            oov_init=self.oov_init,
            oov_init_std=self.oov_init_std,
            seed=self.seed,
        )
