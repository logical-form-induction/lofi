"""Held-out grammaticality and calibration declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = ["GrammaticalityEvaluatorSettings", "GrammaticalityEvaluatorSpec"]


_METRICS = (
    "grammaticality-roc-auc",
    "grammaticality-pr-auc",
    "grammaticality-accuracy-at-youden",
    "grammaticality-accuracy-at-half",
    "grammaticality-mcc",
    "grammaticality-noise-adjusted-auc",
    "grammaticality-auc-ceiling",
    "grammaticality-calibration-ece",
    "grammaticality-calibration-brier",
)


class GrammaticalityEvaluatorSettings(NamedTuple):
    """Validated settings for held-out sentence judgments."""

    readout: str
    output: str
    flip_rate: float
    bins: int
    conventions: ConventionSettings
    metrics: tuple[str, ...]


class GrammaticalityEvaluatorSpec(EvaluatorSpec):
    """Declare the grammaticality battery over a root-mass readout."""

    kind: Literal["grammaticality"] = "grammaticality"
    readout: str
    output: str
    flip_rate: float
    bins: int
    conventions: ConventionSpec

    @dx.validates("readout", "output")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        return ensure_name("a grammaticality span-readout role", value)

    @dx.validates("flip_rate")
    @classmethod
    def _flip_rate_is_informative(cls, value: float) -> float:
        held = ensure_finite("a grammaticality label-flip rate", value)
        if not 0.0 <= held < 0.5:
            msg = f"a grammaticality label-flip rate must lie in [0, 0.5): {value!r}"
            raise ValueError(msg)
        return held

    @dx.validates("bins")
    @classmethod
    def _bins_are_positive(cls, value: int) -> int:
        held = ensure_non_negative("a calibration bin count", value)
        if held == 0:
            msg = "a calibration report declares no bins"
            raise ValueError(msg)
        return held

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the declared root-mass readout dependency."""
        return (Dependency(self.readout, "span-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return the full grammaticality and calibration battery."""
        del labels
        return _METRICS

    def settings(self) -> GrammaticalityEvaluatorSettings:
        """Return immutable plugin settings."""
        return GrammaticalityEvaluatorSettings(
            self.readout,
            self.output,
            self.flip_rate,
            self.bins,
            self.conventions.settings(),
            self.metrics(),
        )
