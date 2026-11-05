"""Scalar featurizer declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.data import _DEFAULT_FEATURIZER_PADDING, FeaturizerSpec

__all__ = ["ScalarFeaturizerSettings", "ScalarFeaturizerSpec"]


class ScalarFeaturizerSettings(NamedTuple):
    """The settings a scalar-response featurizer is built from.

    Attributes
    ----------
    padding : str
        Reserved lexical padding symbol.
    """

    padding: str


class ScalarFeaturizerSpec(FeaturizerSpec):
    """Declares scalar-response featurization.

    Attributes
    ----------
    kind : Literal["scalar"]
        Registry tag.
    padding : str
        Reserved lexical padding symbol.
    """

    kind: Literal["scalar"] = "scalar"
    padding: str = _DEFAULT_FEATURIZER_PADDING

    @dx.validates("padding")
    @classmethod
    def _padding_is_present(cls, value: str) -> str:
        """Validate the padding symbol."""
        return ensure_name("featurizer setting", value)

    def settings(self) -> ScalarFeaturizerSettings:
        """Return the settings the plugin builds this featurizer from."""
        return ScalarFeaturizerSettings(padding=self.padding)
