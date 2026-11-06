"""Derivation featurizer declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.data import _DEFAULT_FEATURIZER_PADDING, FeaturizerSpec

__all__ = ["DerivationFeaturizerSettings", "DerivationFeaturizerSpec"]


class DerivationFeaturizerSettings(NamedTuple):
    """The settings a gold derivation featurizer is built from.

    Attributes
    ----------
    padding : str
        Reserved lexical padding symbol.
    """

    padding: str


class DerivationFeaturizerSpec(FeaturizerSpec):
    """Declares gold derivation featurization.

    Attributes
    ----------
    kind : Literal["derivation"]
        Registry tag.
    padding : str
        Reserved lexical padding symbol.
    """

    kind: Literal["derivation"] = "derivation"
    padding: str = _DEFAULT_FEATURIZER_PADDING

    @dx.validates("padding")
    @classmethod
    def _padding_is_present(cls, value: str) -> str:
        """Validate the padding symbol."""
        return ensure_name("featurizer setting", value)

    def settings(self) -> DerivationFeaturizerSettings:
        """Return the settings the plugin builds this featurizer from."""
        return DerivationFeaturizerSettings(padding=self.padding)
