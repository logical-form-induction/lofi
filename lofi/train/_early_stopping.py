"""Absolute-delta early stopping over completed validation epochs."""

from typing import Literal, Never, final, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue

__all__ = [
    "AccuracyStopping",
    "AccuracyStoppingDecision",
    "EarlyStopping",
    "EarlyStoppingState",
    "StoppingDecision",
]


@final
class AccuracyStoppingDecision(FrozenValue):
    """Report whether a held-out accuracy target or its ceiling ended fitting."""

    _noun = "accuracy-stopping decision"
    __slots__ = ("_accuracy", "_met", "_reason", "_stop")

    def __init__(
        self,
        accuracy: float,
        *,
        stop: bool,
        met: bool,
        reason: Literal["target-met", "target-not-met"] | None,
    ) -> None:
        object.__setattr__(self, "_accuracy", ensure_finite("accuracy", accuracy))
        object.__setattr__(self, "_stop", stop)
        object.__setattr__(self, "_met", met)
        object.__setattr__(self, "_reason", reason)

    @property
    def accuracy(self) -> float:
        """Return the held-out accuracy that was observed."""
        return self._accuracy

    @property
    def stop(self) -> bool:
        """Return whether fitting ends at this observation."""
        return self._stop

    @property
    def met(self) -> bool:
        """Return whether the observation met the declared target."""
        return self._met

    @property
    def reason(self) -> Literal["target-met", "target-not-met"] | None:
        """Return the stopping reason, or None while fitting continues."""
        return self._reason


@final
class AccuracyStopping(FrozenValue):
    """Stop pretraining at a held-out accuracy target or a step ceiling."""

    _noun = "accuracy-stopping rule"
    __slots__ = ("_max_steps", "_metric", "_target")

    def __init__(self, metric: str, target: float, max_steps: int) -> None:
        checked = ensure_finite("target", target)

        if not 0.0 <= checked <= 1.0:
            msg = f"target is outside [0, 1]: {target!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_metric", ensure_name("metric", metric))
        object.__setattr__(self, "_target", checked)
        object.__setattr__(self, "_max_steps", ensure_positive("max_steps", max_steps))

    @property
    def metric(self) -> str:
        """Return the monitored held-out metric name."""
        return self._metric

    @property
    def target(self) -> float:
        """Return the required held-out accuracy."""
        return self._target

    @property
    def max_steps(self) -> int:
        """Return the optimizer-step ceiling."""
        return self._max_steps

    def observe(
        self, value: float, completed_steps: int, /
    ) -> AccuracyStoppingDecision:
        """Return the decision after one held-out evaluation."""
        accuracy = ensure_finite(self._metric, value)
        steps = ensure_positive("completed_steps", completed_steps)

        if not 0.0 <= accuracy <= 1.0:
            msg = f"{self._metric} is not an accuracy in [0, 1]: {value!r}"
            raise ValueError(msg)

        if steps > self._max_steps:
            msg = (
                f"completed_steps {steps} exceeds the declared maximum "
                f"{self._max_steps}"
            )
            raise ValueError(msg)

        met = accuracy >= self._target
        exhausted = steps == self._max_steps
        reason: Literal["target-met", "target-not-met"] | None = (
            "target-met" if met else "target-not-met" if exhausted else None
        )
        return AccuracyStoppingDecision(
            accuracy,
            stop=met or exhausted,
            met=met,
            reason=reason,
        )


@final
class EarlyStoppingState(FrozenValue):
    """The best finite observation and failures that followed it."""

    _noun = "early-stopping state"

    __slots__ = ("_best", "_failures")

    def __init__(self, best: float | None = None, failures: int = 0) -> None:
        object.__setattr__(
            self,
            "_best",
            None if best is None else ensure_finite("best", best),
        )
        object.__setattr__(
            self,
            "_failures",
            ensure_non_negative("failures", failures),
        )

    @property
    def best(self) -> float | None:
        """Return the best observation, or None before the first observation."""
        return self._best

    @property
    def failures(self) -> int:
        """Return consecutive observations that did not improve."""
        return self._failures


@final
class StoppingDecision(FrozenValue):
    """The updated stopping state and its two decisions."""

    _noun = "stopping decision"

    __slots__ = ("_improved", "_state", "_stop")

    def __init__(
        self,
        state: EarlyStoppingState,
        *,
        improved: bool,
        stop: bool,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_improved", improved)
        object.__setattr__(self, "_stop", stop)

    @property
    def state(self) -> EarlyStoppingState:
        """Return the state after the observation."""
        return self._state

    @property
    def improved(self) -> bool:
        """Return whether the observation established a new best."""
        return self._improved

    @property
    def stop(self) -> bool:
        """Return whether patience is exhausted."""
        return self._stop


@final
class EarlyStopping(FrozenValue):
    """An absolute-delta stopping rule for one named metric."""

    _noun = "early-stopping rule"

    __slots__ = ("_delta", "_metric", "_mode", "_patience")

    def __init__(
        self,
        metric: str,
        mode: Literal["min", "max"],
        patience: int,
        minimum_delta: float,
    ) -> None:
        checked_delta = ensure_finite("minimum_delta", minimum_delta)

        if mode not in {"min", "max"}:
            msg = f"mode is not 'min' or 'max': {mode!r}"
            raise ValueError(msg)

        if checked_delta < 0.0:
            msg = f"minimum_delta is negative: {minimum_delta!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_metric", ensure_name("metric", metric))
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_patience", ensure_positive("patience", patience))
        object.__setattr__(self, "_delta", checked_delta)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        del name, value
        msg = "an early-stopping rule is immutable"
        raise AttributeError(msg)

    @property
    def metric(self) -> str:
        """Return the monitored metric name."""
        return self._metric

    @property
    def mode(self) -> Literal["min", "max"]:
        """Return whether improvement minimizes or maximizes the metric."""
        return self._mode

    @property
    def patience(self) -> int:
        """Return the permitted consecutive nonimproving observations."""
        return self._patience

    @property
    def minimum_delta(self) -> float:
        """Return the strict absolute improvement threshold."""
        return self._delta

    def observe(
        self,
        state: EarlyStoppingState,
        value: float,
        /,
    ) -> StoppingDecision:
        """Return the stopping decision after one finite observation.

        Parameters
        ----------
        state : EarlyStoppingState
            Best value and failure count before this observation.
        value : float
            Current epoch's monitored validation metric.
        """
        candidate = ensure_finite(self._metric, value)
        best = state.best
        improved = best is None or (
            candidate < best - self._delta
            if self._mode == "min"
            else candidate > best + self._delta
        )
        updated = (
            EarlyStoppingState(candidate)
            if improved
            else EarlyStoppingState(best, state.failures + 1)
        )

        return StoppingDecision(
            updated,
            improved=improved,
            stop=updated.failures >= self._patience,
        )
