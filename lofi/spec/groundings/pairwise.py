"""The one pairwise grounding declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, cast, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.control_variate import (
    ControlVariateDeclaration,
    ControlVariateSettingsDeclaration,
    ControlVariateSpec,
)
from lofi.reward import InferenceReward, RewardSpec
from lofi.spec.components.model import GroundingSpec, PairScorerSpec
from lofi.spec.groundings._policies import (
    _DEFAULT_GROUNDING_COMPOSITION,
    CompositionDeclaration,
    CompositionRuleSpec,
    CompositionSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.components.model import ReadoutSpec

__all__ = [
    "PairwiseGroundingSettings",
    "PairwiseGroundingSpec",
]


class PairwiseGroundingSettings(NamedTuple):
    """Resolved pairwise grounding settings."""

    premise: str
    hypothesis: str
    composition: CompositionSettings
    control_variate: ControlVariateDeclaration | None
    reward: InferenceReward | None


class PairwiseGroundingSpec(GroundingSpec):
    """Declare evidence composition over retained premise and hypothesis forms."""

    kind: Literal["pairwise"] = "pairwise"
    premise: str
    hypothesis: str
    composition: CompositionRuleSpec = _DEFAULT_GROUNDING_COMPOSITION
    control_variate: ControlVariateSpec | None = None
    reward: RewardSpec | None = None
    scorer: PairScorerSpec

    @dx.validates("premise", "hypothesis")
    @classmethod
    def _readout_is_named(cls, readout: str) -> str:
        """Return a named form readout role."""
        return ensure_name("a pairwise form readout", readout)

    def settings(self) -> PairwiseGroundingSettings:
        """Return the grounding's interior settings."""
        composition = cast("CompositionDeclaration", self.composition)
        return PairwiseGroundingSettings(
            self.premise,
            self.hypothesis,
            composition.settings(),
            (
                None
                if self.control_variate is None
                else ControlVariateDeclaration(
                    cast("str", self.control_variate.kind),
                    cast(
                        "ControlVariateSettingsDeclaration", self.control_variate
                    ).settings(),
                )
            ),
            None if self.reward is None else self.reward.build(),
        )

    @override
    def reward_declared(self) -> bool:
        """Return whether this grounding declares proof payoffs."""
        return self.reward is not None

    @override
    def form_readouts(self) -> tuple[str, str]:
        """Return the readouts supplying the two grounded sides."""
        return self.premise, self.hypothesis

    @override
    def validate_readouts(self, readouts: Mapping[str, ReadoutSpec], /) -> None:
        """Require one common estimate regime and its exact control-variate policy."""
        names = self.form_readouts()
        missing = tuple(name for name in names if name not in readouts)

        if missing:
            msg = f"pairwise grounding names absent form readouts {missing!r}"
            raise ValueError(msg)

        estimates = tuple(readouts[name].estimate_kind() for name in names)

        if any(estimate is None for estimate in estimates):
            msg = f"pairwise grounding reads no complete terms from {names!r}"
            raise ValueError(msg)

        if len(set(estimates)) != 1:
            msg = f"pairwise grounding mixes readout estimates {estimates!r}"
            raise ValueError(msg)

        sampled = estimates[0] == "sampled"

        if sampled != (self.control_variate is not None):
            expected = "a control variate" if sampled else "no control variate"
            msg = f"pairwise grounding over {estimates[0]} forms requires {expected}"
            raise ValueError(msg)

    @override
    def nested_components(self) -> Mapping[str, PairScorerSpec]:
        """Return the scorer as an independently registered nested component."""
        return MappingProxyType({"scorer": self.scorer})
