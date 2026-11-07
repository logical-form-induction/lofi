"""Span-constraint featurizer declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.data import _DEFAULT_FEATURIZER_PADDING, FeaturizerSpec

__all__ = ["SpanConstraintFeaturizerSettings", "SpanConstraintFeaturizerSpec"]


class SpanConstraintFeaturizerSettings(NamedTuple):
    """The settings a span-constraint featurizer is built from.

    Attributes
    ----------
    padding : str
        Reserved lexical padding symbol.
    """

    padding: str


class SpanConstraintFeaturizerSpec(FeaturizerSpec):
    """Declares span-constraint featurization.

    Attributes
    ----------
    kind : Literal["span-constraint"]
        Registry tag.
    padding : str
        Reserved lexical padding symbol.
    """

    kind: Literal["span-constraint"] = "span-constraint"
    padding: str = _DEFAULT_FEATURIZER_PADDING

    @dx.validates("padding")
    @classmethod
    def _padding_is_present(cls, value: str) -> str:
        """Validate the padding symbol."""
        return ensure_name("featurizer setting", value)

    def settings(self) -> SpanConstraintFeaturizerSettings:
        """Return the settings the plugin builds this featurizer from."""
        return SpanConstraintFeaturizerSettings(padding=self.padding)
