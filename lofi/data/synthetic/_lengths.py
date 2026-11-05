"""Distributions a synthetic sample draws its token counts from."""

from math import exp, fsum
from typing import TYPE_CHECKING, Literal, Protocol, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping
    from random import Random

__all__ = [
    "LengthDistribution",
    "LengthOutOfSupportError",
    "NegativeBinomialLength",
    "OutOfSupportPolicy",
    "TabularLength",
    "UniformLength",
]

type OutOfSupportPolicy = Literal["resample", "clip", "error"]
"""How a draw outside a distribution's declared bounds is handled."""


class LengthOutOfSupportError(ValueError):
    """Raised when a length draw leaves its declared support under error policy."""


class LengthDistribution(Protocol):
    """Supplies bounded token-count draws and their theoretical moments."""

    @property
    def min_length(self) -> int:
        """Return the least supported token count."""
        ...

    @property
    def max_length(self) -> int:
        """Return the greatest supported token count."""
        ...

    @property
    def mean(self) -> float:
        """Return the expected token count."""
        ...

    @property
    def variance(self) -> float:
        """Return the token-count variance."""
        ...

    @property
    def out_of_support(self) -> OutOfSupportPolicy:
        """Return the policy for a raw draw outside the bounds."""
        ...

    def draw(self, random: Random) -> int:
        """Draw one raw token count."""
        ...


def _bounds(min_length: int, max_length: int) -> tuple[int, int]:
    """Return a positive lower bound and an upper bound at least as large."""
    lower = ensure_non_negative("min_length", min_length)
    upper = ensure_non_negative("max_length", max_length)

    if lower == 0:
        msg = "min_length is not positive: 0"
        raise ValueError(msg)

    if lower > upper:
        msg = f"min_length {lower} exceeds max_length {upper}"
        raise ValueError(msg)

    return lower, upper


def _poisson(random: Random, rate: float) -> int:
    """Draw a Poisson count of the given rate.

    Parameters
    ----------
    random : Random
        The seeded generator the draw consumes.
    rate : float
        The mean of the count, zero or more.
    """
    drawn = 0
    remaining = rate

    # exp(-rate) underflows past about 745, so a large rate is drawn as a
    # sum of counts at rates it does not underflow at
    while remaining > 0.0:
        step = min(remaining, 500.0)
        remaining -= step
        threshold = exp(-step)
        product = random.random()

        while product > threshold:
            drawn += 1
            product *= random.random()

    return drawn


class UniformLength(FrozenValue):
    """Token counts equally likely at every value between two bounds.

    Parameters
    ----------
    min_length : int
        The fewest tokens a drawn count may be.
    max_length : int
        The most tokens a drawn count may be.

    Raises
    ------
    TypeError
        If a bound is not an integer.
    ValueError
        If the lower bound is not positive or the bounds cross.
    """

    __slots__ = ("_max_length", "_min_length")

    _noun = "uniform length distribution"

    def __init__(self, min_length: int, max_length: int) -> None:
        lower, upper = _bounds(min_length, max_length)
        object.__setattr__(self, "_min_length", lower)
        object.__setattr__(self, "_max_length", upper)

    @property
    def min_length(self) -> int:
        """Return the fewest tokens a drawn count may be."""
        return self._min_length

    @property
    def max_length(self) -> int:
        """Return the most tokens a drawn count may be."""
        return self._max_length

    @property
    def mean(self) -> float:
        """Return the expected count."""
        return (self._min_length + self._max_length) / 2

    @property
    def variance(self) -> float:
        """Return the variance of the count."""
        width = self._max_length - self._min_length + 1

        return (width * width - 1) / 12

    @property
    def out_of_support(self) -> OutOfSupportPolicy:
        """Return error, since a uniform draw cannot leave its bounds."""
        return "error"

    def draw(self, random: Random) -> int:
        """Draw one count.

        Parameters
        ----------
        random : Random
            The seeded generator the draw consumes.
        """
        return random.randint(self._min_length, self._max_length)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another distribution has the same bounds."""
        if type(other) is not UniformLength:
            return NotImplemented

        return (self._min_length, self._max_length) == (
            other._min_length,
            other._max_length,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the bounds."""
        return hash((UniformLength, self._min_length, self._max_length))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this distribution."""
        return f"UniformLength({self._min_length!r}, {self._max_length!r})"


class NegativeBinomialLength(FrozenValue):
    """Token counts a negative binomial above a lower bound, up to an upper one.

    A drawn count is the lower bound plus the number of failures before the
    ``r``-th success in trials that each succeed with probability ``p``. The
    mean and the variance are those of the unbounded distribution. The upper
    bound is the most tokens a caller tabulates, and a draw beyond it is
    returned as drawn, for the caller to discard.

    Parameters
    ----------
    r : float
        The number of successes the trials run until, positive and not
        necessarily whole.
    p : float
        The probability each trial succeeds with, strictly between zero and
        one.
    min_length : int
        The fewest tokens a drawn count may be.
    max_length : int
        The most tokens a count is tabulated at.

    Raises
    ------
    TypeError
        If a shape parameter is not numeric or a bound is not an integer.
    ValueError
        If a shape parameter is not finite, ``r`` is not positive, ``p`` is
        not strictly between zero and one, the lower bound is not positive
        or the bounds cross.
    """

    __slots__ = ("_max_length", "_min_length", "_out_of_support", "_p", "_r")

    _noun = "negative-binomial length distribution"

    def __init__(
        self,
        r: float,
        p: float,
        *,
        min_length: int = 1,
        max_length: int,
        out_of_support: OutOfSupportPolicy = "resample",
    ) -> None:
        held_r = ensure_finite("r", r)
        held_p = ensure_finite("p", p)

        if held_r <= 0.0:
            msg = f"r is not positive: {r!r}"
            raise ValueError(msg)

        if not 0.0 < held_p < 1.0:
            msg = f"p is not strictly between zero and one: {p!r}"
            raise ValueError(msg)

        if out_of_support not in ("resample", "clip", "error"):
            msg = f"unknown out-of-support policy: {out_of_support!r}"
            raise ValueError(msg)

        lower, upper = _bounds(min_length, max_length)
        object.__setattr__(self, "_r", held_r)
        object.__setattr__(self, "_p", held_p)
        object.__setattr__(self, "_min_length", lower)
        object.__setattr__(self, "_max_length", upper)
        object.__setattr__(self, "_out_of_support", out_of_support)

    @property
    def r(self) -> float:
        """Return the number of successes the trials run until."""
        return self._r

    @property
    def p(self) -> float:
        """Return the probability each trial succeeds with."""
        return self._p

    @property
    def min_length(self) -> int:
        """Return the fewest tokens a drawn count may be."""
        return self._min_length

    @property
    def max_length(self) -> int:
        """Return the most tokens a count is tabulated at."""
        return self._max_length

    @property
    def mean(self) -> float:
        """Return the expected count under the unbounded distribution."""
        return self._min_length + self._r * (1.0 - self._p) / self._p

    @property
    def variance(self) -> float:
        """Return the variance of the count under the unbounded distribution."""
        return self._r * (1.0 - self._p) / (self._p * self._p)

    @property
    def out_of_support(self) -> OutOfSupportPolicy:
        """Return how a raw draw beyond the tabulated bound is handled."""
        return self._out_of_support

    def draw(self, random: Random) -> int:
        """Draw one count, which may exceed the upper bound.

        Parameters
        ----------
        random : Random
            The seeded generator the draw consumes.
        """
        rate = random.gammavariate(self._r, (1.0 - self._p) / self._p)

        return self._min_length + _poisson(random, rate)

    def _fields(self) -> tuple[float, float, int, int, OutOfSupportPolicy]:
        """Return the fields the distribution is equal on."""
        return (
            self._r,
            self._p,
            self._min_length,
            self._max_length,
            self._out_of_support,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another distribution has the same shape and bounds."""
        if type(other) is not NegativeBinomialLength:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the shape parameters and the bounds."""
        return hash((NegativeBinomialLength, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this distribution."""
        return (
            f"NegativeBinomialLength({self._r!r}, {self._p!r}, "
            f"min_length={self._min_length!r}, max_length={self._max_length!r}, "
            f"out_of_support={self._out_of_support!r})"
        )


class TabularLength(FrozenValue):
    """A finite token-count distribution declared by nonnegative masses."""

    __slots__ = ("_lengths", "_masses", "_mean", "_variance")

    _noun = "tabular length distribution"

    def __init__(self, masses: Mapping[int, float]) -> None:
        checked: list[tuple[int, float]] = []

        for length, mass in masses.items():
            held_length = ensure_non_negative("a tabular length", length)

            if held_length == 0:
                msg = "a tabular length is not positive: 0"
                raise ValueError(msg)

            held_mass = ensure_finite(f"the mass at length {length!r}", mass)

            if held_mass < 0.0:
                msg = f"the mass at length {length!r} is negative: {mass!r}"
                raise ValueError(msg)

            checked.append((held_length, held_mass))

        checked.sort()
        total = fsum(mass for _, mass in checked)

        if total <= 0.0:
            msg = "a tabular length distribution has no positive mass"
            raise ValueError(msg)

        lengths = tuple(length for length, _ in checked)
        normalized = tuple(mass / total for _, mass in checked)
        mean = fsum(
            length * mass for length, mass in zip(lengths, normalized, strict=True)
        )
        variance = fsum(
            (length - mean) ** 2 * mass
            for length, mass in zip(lengths, normalized, strict=True)
        )
        object.__setattr__(self, "_lengths", lengths)
        object.__setattr__(self, "_masses", normalized)
        object.__setattr__(self, "_mean", mean)
        object.__setattr__(self, "_variance", variance)

    @property
    def min_length(self) -> int:
        """Return the least length with declared mass."""
        return self._lengths[0]

    @property
    def max_length(self) -> int:
        """Return the greatest length with declared mass."""
        return self._lengths[-1]

    @property
    def mean(self) -> float:
        """Return the expected token count."""
        return self._mean

    @property
    def variance(self) -> float:
        """Return the token-count variance."""
        return self._variance

    @property
    def out_of_support(self) -> OutOfSupportPolicy:
        """Return error, since a tabular draw cannot leave its support."""
        return "error"

    def draw(self, random: Random) -> int:
        """Draw one count in proportion to its declared mass."""
        remaining = random.random()

        for length, mass in zip(self._lengths, self._masses, strict=True):
            remaining -= mass

            if remaining < 0.0:
                return length

        return self._lengths[-1]

    @property
    def masses(self) -> tuple[tuple[int, float], ...]:
        """Return normalized masses in ascending length order."""
        return tuple(zip(self._lengths, self._masses, strict=True))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another table has the same normalized masses."""
        return type(other) is TabularLength and self.masses == other.masses

    @override
    def __hash__(self) -> int:
        """Return a hash over the normalized table."""
        return hash((TabularLength, self.masses))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this distribution."""
        return f"TabularLength({dict(self.masses)!r})"
