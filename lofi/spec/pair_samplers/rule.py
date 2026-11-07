"""Rule-inventory pair-sampler declaration."""

from typing import Literal, NamedTuple

from lofi.spec.components.trainer import PairSamplerSpec

__all__ = ["RulePairSamplerSettings", "RulePairSamplerSpec"]


class RulePairSamplerSettings(NamedTuple):
    """Resolved settings for sampling declared rule instances."""


class RulePairSamplerSpec(PairSamplerSpec):
    """Declare bounded sampling from every generative rule arity."""

    kind: Literal["rule"] = "rule"

    def settings(self) -> RulePairSamplerSettings:
        """Return the settings the plugin builds this sampler from."""
        return RulePairSamplerSettings()
