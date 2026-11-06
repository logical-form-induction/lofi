"""Early stopping declaration."""

from typing import Literal

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_positive

__all__ = ["AccuracyStoppingSpec", "EarlyStoppingSpec"]


class AccuracyStoppingSpec(dx.Model, extra="forbid"):
    """Declare a held-out accuracy target and its optimizer-step ceiling.

    Attributes
    ----------
    metric : str
        Held-out accuracy metric evaluated during pretraining.
    target : float
        Finite target proportion in the closed unit interval.
    max_steps : int
        Positive optimizer-step ceiling when the target is unreachable.
    """

    metric: str
    target: float
    max_steps: int

    @dx.validates("metric")
    @classmethod
    def _metric_is_named(cls, value: str) -> str:
        """Validate the monitored metric name."""
        return ensure_name("metric", value)

    @dx.validates("target")
    @classmethod
    def _target_is_a_proportion(cls, value: float) -> float:
        """Validate the required held-out accuracy."""
        checked = ensure_finite("target", value)

        if not 0.0 <= checked <= 1.0:
            msg = f"target is outside [0, 1]: {value!r}"
            raise ValueError(msg)

        return checked

    @dx.validates("max_steps")
    @classmethod
    def _maximum_is_positive(cls, value: int) -> int:
        """Validate the optimizer-step ceiling."""
        return ensure_positive("max_steps", value)


class EarlyStoppingSpec(dx.Model, extra="forbid"):
    """Declare a stopping rule over the run's fitted retained partition.

    Attributes
    ----------
    metric : str
        Retained metric whose epoch aggregate selects the best checkpoint.
    mode : Literal["min", "max"]
        Whether smaller or larger metric values improve on the best.
    patience : int
        Consecutive nonimproving completed epochs permitted before stopping.
    minimum_delta : float
        Absolute change a candidate must exceed to improve on the best.
    keep_best : int
        Positive number of the newest retained-best checkpoints the run keeps.
    """

    metric: str
    mode: Literal["min", "max"]
    patience: int
    minimum_delta: float
    keep_best: int

    @dx.validates("metric")
    @classmethod
    def _metric_is_named(cls, value: str) -> str:
        """Validate the monitored metric name."""
        return ensure_name("metric", value)

    @dx.validates("patience")
    @classmethod
    def _patience_is_positive(cls, value: int) -> int:
        """Validate the stopping patience."""
        return ensure_positive("patience", value)

    @dx.validates("keep_best")
    @classmethod
    def _retention_is_positive(cls, value: int) -> int:
        """Validate the validation-best retention count."""
        return ensure_positive("keep_best", value)

    @dx.validates("minimum_delta")
    @classmethod
    def _delta_is_nonnegative(cls, value: float) -> float:
        """Validate the absolute improvement threshold."""
        checked = ensure_finite("minimum_delta", value)

        if checked < 0.0:
            msg = f"minimum_delta is negative: {value!r}"
            raise ValueError(msg)

        return checked
