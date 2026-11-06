"""Unlabeled featurizer declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.data import _DEFAULT_FEATURIZER_PADDING, FeaturizerSpec

__all__ = ["UnlabeledFeaturizerSettings", "UnlabeledFeaturizerSpec"]


class UnlabeledFeaturizerSettings(NamedTuple):
    """The settings a unlabeled-item featurizer is built from.

    Attributes
    ----------
    padding : str
        Reserved lexical padding symbol.
    """

    padding: str


class UnlabeledFeaturizerSpec(FeaturizerSpec):
    """Declares unlabeled-item featurization.

    Attributes
    ----------
    kind : Literal["unlabeled"]
        Registry tag.
    padding : str
        Reserved lexical padding symbol.
    """

    kind: Literal["unlabeled"] = "unlabeled"
    padding: str = _DEFAULT_FEATURIZER_PADDING

    @dx.validates("padding")
    @classmethod
    def _padding_is_present(cls, value: str) -> str:
        """Validate the padding symbol."""
        return ensure_name("featurizer setting", value)

    def settings(self) -> UnlabeledFeaturizerSettings:
        """Return the settings the plugin builds this featurizer from."""
        return UnlabeledFeaturizerSettings(padding=self.padding)
