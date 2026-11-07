"""Flat span encoder declaration."""

from typing import Final, Literal, NamedTuple, override

import didactic.api as dx

from lofi.spec._ports import Dependency
from lofi.spec.components.model import ParserSpec

__all__ = [
    "FlatParserSettings",
    "FlatParserSpec",
    "FlatProjectionInitialization",
    "FlatSpanFeature",
]

type FlatSpanFeature = Literal["left", "right", "bilinear"]
"""A projected endpoint feature contributing to a flat span cell."""

type FlatProjectionInitialization = Literal["identity", "xavier-uniform"]
"""The declared initialization of the endpoint projection."""

_DEPENDENCIES: Final[tuple[Dependency, ...]] = (Dependency("subword_pooler", "leaves"),)
"""The pooled leaves whose endpoints represent each span."""


class FlatParserSettings(NamedTuple):
    """The settings a flat span encoder is built from.

    Attributes
    ----------
    width : int
        The width of each span representation.
    features : tuple[FlatSpanFeature, ...]
        The projected endpoint features summed into each span cell.
    projection_initialization : FlatProjectionInitialization
        How the endpoint projection is initialized.
    """

    width: int
    features: tuple[FlatSpanFeature, ...]
    projection_initialization: FlatProjectionInitialization


class FlatParserSpec(ParserSpec):
    """Declares a chart that represents each span from its endpoint words.

    Attributes
    ----------
    kind : Literal["flat"]
        Registry tag.
    width : int
        The width of each span representation.
    features : tuple[FlatSpanFeature, ...]
        The projected endpoint features summed into each span cell.
    projection_initialization : FlatProjectionInitialization
        How the endpoint projection is initialized.
    """

    kind: Literal["flat"] = "flat"
    width: int
    features: tuple[FlatSpanFeature, ...]
    projection_initialization: FlatProjectionInitialization

    __axioms__ = (
        dx.axiom("width > 0", message="a span cell has at least one coordinate"),
    )

    @dx.validates("features")
    @classmethod
    def _features_are_nonempty_and_distinct(
        cls, features: tuple[FlatSpanFeature, ...]
    ) -> tuple[FlatSpanFeature, ...]:
        """Return a nonempty feature sequence without repeated contributions."""
        if not features:
            msg = "a flat span cell needs at least one endpoint feature"
            raise ValueError(msg)
        if len(set(features)) != len(features):
            msg = f"a flat span cell repeats an endpoint feature: {features!r}"
            raise ValueError(msg)
        return features

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the pooled leaves this span encoder reads."""
        return _DEPENDENCIES

    def chart_outputs(self) -> frozenset[str]:
        """Return the fixed weights and endpoint vectors the control provides."""
        return frozenset({"inside", "cells"})

    def settings(self) -> FlatParserSettings:
        """Return the settings the plugin builds this component from."""
        return FlatParserSettings(
            width=self.width,
            features=self.features,
            projection_initialization=self.projection_initialization,
        )
