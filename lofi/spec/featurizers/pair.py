"""Pair featurizer declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.data import _DEFAULT_FEATURIZER_PADDING, FeaturizerSpec

__all__ = ["PairFeaturizerSettings", "PairFeaturizerSpec", "PairSelection"]

type PairSelection = str
"""Pair-valued supervision arm selected by a pair featurizer."""


class PairFeaturizerSettings(NamedTuple):
    """The settings a pair featurizer is built from.

    Attributes
    ----------
    padding : str
        Reserved lexical padding symbol.
    selection : PairSelection
        Pair-valued supervision arm to read.
    """

    padding: str
    selection: PairSelection


class PairFeaturizerSpec(FeaturizerSpec):
    """Declares pair-valued featurization.

    Attributes
    ----------
    kind : Literal["pair"]
        Registry tag.
    padding : str
        Reserved lexical padding symbol.
    selection : str
        Pair-valued supervision arm to read.
    """

    kind: Literal["pair"] = "pair"
    padding: str = _DEFAULT_FEATURIZER_PADDING
    selection: str = "forced-choice"

    @dx.validates("padding")
    @classmethod
    def _padding_is_present(cls, value: str) -> str:
        """Validate the padding symbol."""
        return ensure_name("featurizer setting", value)

    @dx.validates("selection")
    @classmethod
    def _selection_is_named(cls, value: str) -> str:
        """Validate the installed pair-selection plugin name."""
        return ensure_name("a pair selection", value)

    def settings(self) -> PairFeaturizerSettings:
        """Return the settings the plugin builds this featurizer from."""
        return PairFeaturizerSettings(padding=self.padding, selection=self.selection)
