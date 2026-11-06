"""Typed numeric domains over readings."""

from math import inf
from typing import TYPE_CHECKING, Final, override

import torch

from lofi.nn.sites import DTypeClass
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "LOG_ZERO",
    "LogProbabilities",
    "Probabilities",
    "as_log_probabilities",
    "as_probabilities",
    "ensure_declared",
    "entropy",
    "exponentiate",
    "logarithm",
    "mass",
]

LOG_ZERO: Final[float] = -inf
"""The number an invalid position of a log probability reading holds."""

_SLACK: Final[float] = 1e-6
"""Numeric slack for range checks."""


def _beyond(outside: Tensor, bound: str) -> str | None:
    """Return the out-of-range position count, or None if every position is valid.

    Parameters
    ----------
    outside : Tensor
        True at each position that left the range.
    bound : str
        What those positions are beyond, as a phrase a message ends with.

    Returns
    -------
    str | None
        A phrase naming how many positions left the range, or None where none did.
    """
    count = int(outside.sum())

    return None if count == 0 else f"holds {count} live positions {bound}"


def _contradiction(observation: Observation, declared: DTypeClass) -> str | None:
    """Return how a reading contradicts what its point declared, or None.

    Parameters
    ----------
    observation : Observation
        The reading to hold to the declaration.
    declared : DTypeClass
        What the numbers at the point were declared to mean.

    Returns
    -------
    str | None
        A phrase naming the contradiction, to follow the point in a message, or None
        where the reading is within its declared class.
    """
    values = observation.tensor

    if declared is DTypeClass.MASK:
        return (
            None
            if values.dtype is torch.bool
            else f"marks nothing, since it holds {values.dtype} and not truth values"
        )

    if (
        declared in {DTypeClass.LOG_PROBABILITY, DTypeClass.PROBABILITY}
        and not values.is_floating_point()
    ):
        return f"holds {values.dtype} rather than floating-point numbers"

    live = observation.covering()

    if declared is DTypeClass.LOG_PROBABILITY:
        return _beyond(live & ~(values <= _SLACK), "above zero or not a number")

    if declared is DTypeClass.PROBABILITY:
        return _beyond(
            live & ~((values >= -_SLACK) & (values <= 1.0 + _SLACK)),
            "outside the unit interval",
        )

    if declared is DTypeClass.INTEGER and values.is_floating_point():
        return _beyond(
            live & ~(torch.isfinite(values) & (values.floor() == values)),
            "that are not whole numbers",
        )

    return None


def _fill(declared: DTypeClass) -> float | None:
    """Return what an invalid position of a reading of one class holds, or None.

    Parameters
    ----------
    declared : DTypeClass
        What the numbers at the point were declared to mean.

    Returns
    -------
    float | None
        The number an invalid position holds, or None where the class fixes no such
        number.
    """
    if declared is DTypeClass.LOG_PROBABILITY:
        return LOG_ZERO

    if declared is DTypeClass.PROBABILITY:
        return 0.0

    return None


def _misfilled(observation: Observation, declared: DTypeClass) -> str | None:
    """Return how a reading fills its invalid positions wrongly, or None.

    Parameters
    ----------
    observation : Observation
        The reading to hold to the declaration.
    declared : DTypeClass
        What the numbers at the point were declared to mean.

    Returns
    -------
    str | None
        A phrase naming how many invalid positions are filled wrongly and what the
        first of them holds, to follow the point in a message, or None where every
        invalid position holds what the class fixes.
    """
    fill = _fill(declared)

    if fill is None or observation.mask is None:
        return None

    values = observation.tensor
    wrong = ~observation.covering() & (values != fill)
    count = int(wrong.sum())

    if count == 0:
        return None

    return (
        f"fills {count} invalid positions with {float(values[wrong][0])} "
        f"rather than {fill}"
    )


def ensure_declared(observation: Observation, declared: DTypeClass) -> None:
    """Hold one reading to the numeric class its point declared.

    Parameters
    ----------
    observation : Observation
        The reading to hold to the declaration.
    declared : DTypeClass
        What the numbers at the point were declared to mean.

    Raises
    ------
    ValueError
        If the reading is not stored as the declared class requires, a live position
        lies outside that class, or an invalid position holds a number other than the
        one that class fixes.
    """
    wrong = _contradiction(observation, declared)

    if wrong is None:
        wrong = _misfilled(observation, declared)

    if wrong is not None:
        msg = f"the reading at {observation.site!r} {wrong}"
        raise ValueError(msg)


def _ensure_reading(reading: Observation, declared: DTypeClass) -> None:
    """Refuse anything but a reading within one numeric class.

    Parameters
    ----------
    reading : Observation
        The value a domain is being built over.
    declared : DTypeClass
        The class that domain holds its readings to.

    Raises
    ------
    TypeError
        If the value is not an observation.
    ValueError
        If the reading is not stored as the class requires, a live position lies
        outside the class, or an invalid position holds a number other than the one
        the class fixes.
    """
    if not isinstance(reading, Observation):
        msg = f"a numeric domain is built over an observation, not {reading!r}"
        raise TypeError(msg)

    ensure_declared(reading, declared)


class LogProbabilities:
    """Holds a reading whose numbers are logarithms of probabilities.

    Parameters
    ----------
    reading : Observation
        What was read, stored as floating-point numbers, whose live positions lie at
        or below zero and whose invalid positions hold the log of zero.

    Attributes
    ----------
    reading : Observation
        What was read.
    tensor : Tensor
        The numbers read.
    axes : tuple[str, ...]
        The name of each axis, outermost first.
    mask : Tensor | None
        Which positions of the outermost axes hold data, or None where all do.

    Raises
    ------
    TypeError
        If the value is not an observation.
    ValueError
        If the reading is not stored as floating-point numbers, a live position lies
        above zero or holds no number, or an invalid position holds a number other
        than the log of zero.
    """

    __slots__ = ("_reading",)

    def __init__(self, reading: Observation) -> None:
        _ensure_reading(reading, DTypeClass.LOG_PROBABILITY)

        self._reading = reading

    @property
    def reading(self) -> Observation:
        """Return what was read."""
        return self._reading

    @property
    def tensor(self) -> Tensor:
        """Return the numbers read."""
        return self._reading.tensor

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""
        return self._reading.axes

    @property
    def mask(self) -> Tensor | None:
        """Return which positions of the outermost axes hold data, or None."""
        return self._reading.mask

    def covering(self) -> Tensor:
        """Return which positions hold data, at the reading's own shape.

        Returns
        -------
        Tensor
            Of the reading's rank and shape, holding truth values, true where the
            reading holds data.
        """
        return self._reading.covering()

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this domain value."""
        return f"LogProbabilities({self._reading!r})"


class Probabilities:
    """Holds a reading whose numbers are probabilities.

    Parameters
    ----------
    reading : Observation
        What was read, stored as floating-point numbers, whose live positions lie
        between zero and one and whose invalid positions hold zero.

    Attributes
    ----------
    reading : Observation
        What was read.
    tensor : Tensor
        The numbers read.
    axes : tuple[str, ...]
        The name of each axis, outermost first.
    mask : Tensor | None
        Which positions of the outermost axes hold data, or None where all do.

    Raises
    ------
    TypeError
        If the value is not an observation.
    ValueError
        If the reading is not stored as floating-point numbers, a live position lies
        outside the unit interval or holds no number, or an invalid position holds a
        number other than zero.
    """

    __slots__ = ("_reading",)

    def __init__(self, reading: Observation) -> None:
        _ensure_reading(reading, DTypeClass.PROBABILITY)

        self._reading = reading

    @property
    def reading(self) -> Observation:
        """Return what was read."""
        return self._reading

    @property
    def tensor(self) -> Tensor:
        """Return the numbers read."""
        return self._reading.tensor

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""
        return self._reading.axes

    @property
    def mask(self) -> Tensor | None:
        """Return which positions of the outermost axes hold data, or None."""
        return self._reading.mask

    def covering(self) -> Tensor:
        """Return which positions hold data, at the reading's own shape.

        Returns
        -------
        Tensor
            Of the reading's rank and shape, holding truth values, true where the
            reading holds data.
        """
        return self._reading.covering()

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this domain value."""
        return f"Probabilities({self._reading!r})"


def _ensure_class(
    observation: Observation, declared: DTypeClass, wanted: DTypeClass
) -> None:
    """Refuse a reading whose point declared some other numeric class.

    Parameters
    ----------
    observation : Observation
        The reading being taken into a domain.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.
    wanted : DTypeClass
        What the domain it is being taken into holds.

    Raises
    ------
    TypeError
        If the declaration is not a dtype class.
    ValueError
        If the point declared some other class.
    """
    if not isinstance(declared, DTypeClass):
        msg = f"what a point declares is a DTypeClass, not {declared!r}"
        raise TypeError(msg)

    if declared is not wanted:
        msg = (
            f"{observation.site!r} declares {declared.value} numbers, so what was "
            f"read there is not {wanted.value}"
        )
        raise ValueError(msg)


def as_log_probabilities(
    observation: Observation, declared: DTypeClass
) -> LogProbabilities:
    """Take one reading into the log probability domain.

    Parameters
    ----------
    observation : Observation
        The reading to take.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.

    Returns
    -------
    LogProbabilities
        The reading, in the log probability domain.

    Raises
    ------
    TypeError
        If the value is not an observation, or the declaration is not a dtype class.
    ValueError
        If the point declared some other class, the reading is not stored as
        floating-point numbers, a live position lies above zero or holds no number, or
        an invalid position holds a number other than the log of zero.
    """
    _ensure_class(observation, declared, DTypeClass.LOG_PROBABILITY)

    return LogProbabilities(observation)


def as_probabilities(observation: Observation, declared: DTypeClass) -> Probabilities:
    """Take one reading into the probability domain.

    Parameters
    ----------
    observation : Observation
        The reading to take.
    declared : DTypeClass
        What the point it was read at declared its numbers to mean.

    Returns
    -------
    Probabilities
        The reading, in the probability domain.

    Raises
    ------
    TypeError
        If the value is not an observation, or the declaration is not a dtype class.
    ValueError
        If the point declared some other class, the reading is not stored as
        floating-point numbers, a live position lies outside the unit interval or holds
        no number, or an invalid position holds a number other than zero.
    """
    _ensure_class(observation, declared, DTypeClass.PROBABILITY)

    return Probabilities(observation)


def exponentiate(values: LogProbabilities) -> Probabilities:
    """Return each log probability in one reading as a probability.

    Parameters
    ----------
    values : LogProbabilities
        The log probabilities to raise.

    Returns
    -------
    Probabilities
        The same distribution in the probability domain, at the same point and over the
        same axes. An invalid position holds zero, and a live position is held at or
        below one.

    Raises
    ------
    TypeError
        If the value is not a log probability reading.
    """
    if not isinstance(values, LogProbabilities):
        msg = f"this raises log probabilities, not {values!r}"
        raise TypeError(msg)

    reading = values.reading

    return Probabilities(
        Observation(
            reading.site,
            reading.tensor.exp().clamp(max=1.0),
            axes=reading.axes,
            mask=reading.mask,
        )
    )


def logarithm(values: Probabilities) -> LogProbabilities:
    """Return each probability in one reading as a log probability.

    Parameters
    ----------
    values : Probabilities
        The probabilities to take the logarithm of.

    Returns
    -------
    LogProbabilities
        The same distribution in the log probability domain, at the same point and over
        the same axes. An invalid position holds the log of zero, and a live position is
        held to the unit interval before its logarithm is taken, so none is above zero.

    Raises
    ------
    TypeError
        If the value is not a probability reading.
    """
    if not isinstance(values, Probabilities):
        msg = f"this takes the logarithm of probabilities, not {values!r}"
        raise TypeError(msg)

    reading = values.reading

    return LogProbabilities(
        Observation(
            reading.site,
            reading.tensor.clamp(min=0.0, max=1.0).log(),
            axes=reading.axes,
            mask=reading.mask,
        )
    )


def _summable(values: Probabilities, over: str) -> tuple[Tensor, int]:
    """Return the numbers of one reading and which axis of them the named one is.

    Parameters
    ----------
    values : Probabilities
        The probabilities to prepare.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    tuple[Tensor, int]
        The numbers read, each live position held to the unit interval and every
        invalid position holding zero, and which axis of them the named one is.

    Raises
    ------
    TypeError
        If the value is not a probability reading.
    KeyError
        If the reading declares no axis of that name.
    """
    if not isinstance(values, Probabilities):
        msg = f"this reads probabilities, not {values!r}"
        raise TypeError(msg)

    reading = values.reading

    return (reading.tensor.clamp(min=0.0, max=1.0), reading.position(over))


def mass(values: Probabilities, *, over: str) -> Tensor:
    """Return how much probability each distribution assigns along one axis.

    Parameters
    ----------
    values : Probabilities
        The probabilities to sum.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Tensor
        The sum along that axis, over the reading's other axes in their own order, to
        which an invalid position adds nothing because it holds zero.

    Raises
    ------
    TypeError
        If the value is not a probability reading.
    KeyError
        If the reading declares no axis of that name.
    """
    numbers, position = _summable(values, over)

    return numbers.sum(dim=position)


def entropy(values: Probabilities, *, over: str) -> Tensor:
    """Return the entropy of each distribution along one axis, in nats.

    Parameters
    ----------
    values : Probabilities
        The probabilities to measure.
    over : str
        The axis the outcomes run along, named as the reading declares it.

    Returns
    -------
    Tensor
        The entropy over that axis, over the reading's other axes in their own order,
        to which an outcome of probability zero and an invalid position both contribute
        nothing.

    Raises
    ------
    TypeError
        If the value is not a probability reading.
    KeyError
        If the reading declares no axis of that name.
    """
    numbers, position = _summable(values, over)

    return -torch.xlogy(numbers, numbers).sum(dim=position)
