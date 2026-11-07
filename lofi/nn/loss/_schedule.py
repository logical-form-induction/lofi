"""Scheduled teacher forcing for supervised decoders."""

from typing import override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.nn.ports import ForwardMode, ForwardOptions

__all__ = ["TeacherForcingSchedule"]


def _checked_saturation(saturation: float) -> float:
    """Return the point of the run at which the whole batch is drawn.

    Parameters
    ----------
    saturation : float
        The fraction of the run elapsed when the drawn fraction reaches one.

    Returns
    -------
    float
        The fraction, widened to float.

    Raises
    ------
    TypeError
        If it is not a number.
    ValueError
        If it is not finite, not above zero or above one.
    """
    checked = ensure_finite("the saturation point", saturation)

    if not 0.0 < checked <= 1.0:
        msg = f"the saturation point lies in (0, 1], not at {saturation}"
        raise ValueError(msg)

    return checked


def _checked_position(step: int, horizon: int) -> tuple[int, int]:
    """Return a position in a run.

    Parameters
    ----------
    step : int
        The step, counting from zero.
    horizon : int
        How many steps the run has.

    Returns
    -------
    tuple[int, int]
        The step and the horizon, unchanged.

    Raises
    ------
    TypeError
        If either is not an integer.
    ValueError
        If the step is negative or the horizon is not above zero.
    """
    checked = ensure_non_negative("a step", step)

    if ensure_non_negative("a horizon", horizon) == 0:
        msg = "a run of no steps has no position to evaluate a schedule at"
        raise ValueError(msg)

    return checked, horizon


class TeacherForcingSchedule:
    """Specifies which pass a supervised decoder runs at each step of a run.

    The sampled fraction rises linearly from zero to one at the saturation step and
    then remains one. A zero fraction uses observing mode; a schedule without a
    saturation step always observes.

    Parameters
    ----------
    saturation : float | None
        The fraction of the run elapsed when the whole batch is drawn, or None for a
        schedule that observes throughout.

    Raises
    ------
    TypeError
        If the saturation point is not a number.
    ValueError
        If it is not finite, not above zero or above one.
    """

    __slots__ = ("_saturation",)

    def __init__(self, saturation: float | None = 0.5) -> None:
        self._saturation = (
            None if saturation is None else _checked_saturation(saturation)
        )

    @property
    def saturation(self) -> float | None:
        """Return the fraction of the run elapsed when the whole batch is drawn."""
        return self._saturation

    @property
    def modes(self) -> frozenset[ForwardMode]:
        """Return the modes of the passes this schedule can name."""
        if self._saturation is None:
            return frozenset({ForwardMode.OBSERVE})

        return frozenset({ForwardMode.OBSERVE, ForwardMode.SAMPLE})

    def rate(self, step: int, *, horizon: int) -> float:
        """Return the fraction of the batch drawn at one step.

        Parameters
        ----------
        step : int
            The step, counting from zero.
        horizon : int
            How many steps the run has.

        Returns
        -------
        float
            The fraction, between zero and one.

        Raises
        ------
        TypeError
            If the step or the horizon is not an integer.
        ValueError
            If the step is negative or the horizon is not above zero.
        """
        at, of = _checked_position(step, horizon)

        if self._saturation is None:
            return 0.0

        return min(1.0, at / (self._saturation * of))

    def at(self, step: int, *, horizon: int) -> ForwardOptions:
        """Return the pass a decoder runs at one step.

        Parameters
        ----------
        step : int
            The step, counting from zero.
        horizon : int
            How many steps the run has.

        Returns
        -------
        ForwardOptions
            An observing pass where nothing is drawn, and otherwise a sampling pass
            drawing the fraction the schedule gives the step.

        Raises
        ------
        TypeError
            If the step or the horizon is not an integer.
        ValueError
            If the step is negative or the horizon is not above zero.
        """
        drawn = self.rate(step, horizon=horizon)

        if drawn == 0.0:
            return ForwardOptions(ForwardMode.OBSERVE)

        return ForwardOptions(ForwardMode.SAMPLE, sampling_rate=drawn)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another schedule saturates at the same point."""
        if type(other) is not TeacherForcingSchedule:
            return NotImplemented

        return self._saturation == other._saturation

    @override
    def __hash__(self) -> int:
        """Return a hash over the saturation point."""
        return hash((TeacherForcingSchedule, self._saturation))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this schedule."""
        return f"TeacherForcingSchedule({self._saturation!r})"
