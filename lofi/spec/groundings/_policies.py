"""Composition rules for inference groundings."""

from typing import Final, Literal, NamedTuple, Protocol

import didactic.api as dx

from lofi._checks import ensure_finite

__all__ = [
    "BestCompositionSpec",
    "CompositionDeclaration",
    "CompositionRuleSpec",
    "CompositionSettings",
    "MarginalCompositionSpec",
    "TemperedCompositionSpec",
]


class CompositionSettings(NamedTuple):
    """Resolved boundary settings for one composition rule."""

    rule: Literal["marginal", "best", "tempered"]
    temperature: float | None


class CompositionDeclaration(Protocol):
    """A composition-rule declaration, of whichever variant."""

    def settings(self) -> CompositionSettings:
        """Return the rule's validated boundary settings."""
        ...


class CompositionRuleSpec(dx.TaggedUnion, discriminator="rule", extra="forbid"):
    """Declares the rule a grounding composes verdicts under."""


class MarginalCompositionSpec(CompositionRuleSpec):
    """Declares composition by expectation over the decided pairs."""

    rule: Literal["marginal"] = "marginal"

    def settings(self) -> CompositionSettings:
        """Return marginal composition settings."""
        return CompositionSettings("marginal", None)


_DEFAULT_GROUNDING_COMPOSITION: Final[CompositionRuleSpec] = MarginalCompositionSpec()


class BestCompositionSpec(CompositionRuleSpec):
    """Declares composition by the heaviest decided pair alone."""

    rule: Literal["best"] = "best"

    def settings(self) -> CompositionSettings:
        """Return best-candidate composition settings."""
        return CompositionSettings("best", None)


class TemperedCompositionSpec(CompositionRuleSpec):
    """Declares composition by expectation at a temperature.

    Attributes
    ----------
    rule : Literal["tempered"]
        Registry tag.
    temperature : float
        The positive, finite temperature the pair weights are divided by.
    """

    rule: Literal["tempered"] = "tempered"
    temperature: float

    @dx.validates("temperature")
    @classmethod
    def _is_positive(cls, value: float) -> float:
        """Validate the temperature."""
        held = ensure_finite("a temperature", value)

        if held <= 0.0:
            msg = f"a temperature is positive, not {value!r}"

            raise ValueError(msg)

        return held

    def settings(self) -> CompositionSettings:
        """Return tempered composition settings."""
        return CompositionSettings("tempered", self.temperature)
