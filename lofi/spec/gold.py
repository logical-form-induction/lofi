"""Declarations for reducing repeated corpus responses to gold labels."""

from typing import Literal, NamedTuple

from lofi.spec.components.data import GoldAggregationSpec

__all__ = ["MajorityGoldSettings", "MajorityGoldSpec"]


class MajorityGoldSettings(NamedTuple):
    """The settings of majority aggregation, which has no free parameter."""


class MajorityGoldSpec(GoldAggregationSpec):
    """Choose the unique most frequent categorical relation."""

    kind: Literal["majority"] = "majority"

    def settings(self) -> MajorityGoldSettings:
        """Return the parameter-free majority settings."""
        return MajorityGoldSettings()
