"""Equivalence tests for paired differences."""

from enum import Enum
from math import fsum
from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.stats._permutation import _paired, paired_permutation
from lofi.stats._values import PValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.stats._values import Tail

__all__ = ["Equivalence", "EquivalenceResult", "MarginTest", "Verdict"]

_HIGHEST_LEVEL: Final = 0.5
"""The level below which a two-tailed reference reading remains."""


class Verdict(Enum):
    """Records where a test places a difference relative to a declared margin."""

    #: The difference lies beyond a boundary of the margin.
    SEPARATED = "separated"

    #: The difference lies inside the margin on both sides.
    EQUIVALENT_WITHIN_MARGIN = "equivalent-within-margin"

    #: The data do not place the difference against either boundary.
    INDETERMINATE = "indeterminate"


class MarginTest:
    """An equivalence boundary and two one-sided tests.

    Parameters
    ----------
    boundary : float
        The difference this test is taken at, in the statistic's own units.
    inside : PValue
        The p-value whose rejection places the difference on the near side of the
        boundary, meaning within the margin as far as this side goes.
    outside : PValue
        The p-value whose rejection places the difference beyond the boundary.

    Raises
    ------
    TypeError
        If either p-value is not one, or the boundary is not a number.
    ValueError
        If the boundary is not finite, or if the two p-values do not report opposite
        tails, which is how a pair read in one direction twice is refused.
    """

    __slots__ = ("_boundary", "_inside", "_outside")

    def __init__(self, boundary: float, *, inside: PValue, outside: PValue) -> None:
        for side, given in (("inside", inside), ("outside", outside)):
            if not isinstance(given, PValue):
                msg = f"a margin test's {side} p-value is not one: {given!r}"
                raise TypeError(msg)

        self._boundary = ensure_finite("an equivalence boundary", boundary)
        self._inside = inside
        self._outside = outside

        tails = {inside.provenance.tail, outside.provenance.tail}

        # read each boundary in both tails
        if tails != frozenset({"greater", "less"}):
            msg = (
                f"a margin test reads one boundary in the tails {sorted(tails)} "
                f"rather than in opposite ones"
            )
            raise ValueError(msg)

    @property
    def boundary(self) -> float:
        """Return the difference this test is taken at, in the statistic's own units."""
        return self._boundary

    @property
    def inside(self) -> PValue:
        """Return the p-value for the near side of the boundary."""
        return self._inside

    @property
    def outside(self) -> PValue:
        """Return the p-value placing the difference beyond the boundary."""
        return self._outside

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value tests the same boundary the same way."""
        if type(other) is not MarginTest:
            return NotImplemented

        return (
            self._boundary == other._boundary
            and self._inside == other._inside
            and self._outside == other._outside
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the boundary and its two p-values."""
        return hash((MarginTest, self._boundary, self._inside, self._outside))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this test."""
        return (
            f"MarginTest({self._boundary!r}, inside={self._inside!r}, "
            f"outside={self._outside!r})"
        )


class EquivalenceResult:
    """Stores a two-one-sided-tests equivalence result.

    Parameters
    ----------
    observed : float
        The mean per-item difference, in the statistic's own units.
    margin : float
        The half-width of the equivalence region, which is positive.
    alpha : float
        The level each one-sided test is read at.
    lower : MarginTest
        The test at the lower boundary, which is at minus the margin.
    upper : MarginTest
        The test at the upper boundary, which is at the margin.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the observed difference is not finite, if the margin is not positive, if the
        level does not lie strictly between zero and one half, or if a boundary does not
        sit at the margin it is named for.
    """

    __slots__ = ("_alpha", "_lower", "_margin", "_observed", "_upper")

    def __init__(
        self,
        observed: float,
        *,
        margin: float,
        alpha: float,
        lower: MarginTest,
        upper: MarginTest,
    ) -> None:
        for side, given in (("lower", lower), ("upper", upper)):
            if not isinstance(given, MarginTest):
                msg = f"an equivalence result's {side} test is not one: {given!r}"
                raise TypeError(msg)

        self._observed = ensure_finite("an observed difference", observed)
        self._margin = ensure_finite("an equivalence margin", margin)
        self._alpha = ensure_finite("an equivalence level", alpha)
        self._lower = lower
        self._upper = upper

        if self._margin <= 0.0:
            msg = f"an equivalence margin is not positive: {margin!r}"
            raise ValueError(msg)

        if not 0.0 < self._alpha < _HIGHEST_LEVEL:
            msg = f"an equivalence level is not below {_HIGHEST_LEVEL}: {alpha!r}"
            raise ValueError(msg)

        if lower.boundary != -self._margin or upper.boundary != self._margin:
            msg = (
                f"an equivalence margin of {margin!r} was tested at the "
                f"boundaries {lower.boundary!r} and {upper.boundary!r}"
            )
            raise ValueError(msg)

    @property
    def observed(self) -> float:
        """Return the mean per-item difference, in the statistic's own units."""
        return self._observed

    @property
    def margin(self) -> float:
        """Return the half-width of the equivalence region."""
        return self._margin

    @property
    def alpha(self) -> float:
        """Return the test level."""
        return self._alpha

    @property
    def lower(self) -> MarginTest:
        """Return the test at minus the margin."""
        return self._lower

    @property
    def upper(self) -> MarginTest:
        """Return the test at the margin."""
        return self._upper

    @property
    def verdict(self) -> Verdict:
        """Return the equivalence decision."""
        if (
            self._lower.outside.value <= self._alpha
            or self._upper.outside.value <= self._alpha
        ):
            return Verdict.SEPARATED

        if (
            self._lower.inside.value <= self._alpha
            and self._upper.inside.value <= self._alpha
        ):
            return Verdict.EQUIVALENT_WITHIN_MARGIN

        return Verdict.INDETERMINATE

    def _identity(self) -> tuple[float, float, float, MarginTest, MarginTest]:
        """Return the five fields an equivalence result is equal on."""
        return (
            self._observed,
            self._margin,
            self._alpha,
            self._lower,
            self._upper,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same five fields."""
        if type(other) is not EquivalenceResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the result is equal on."""
        return hash((EquivalenceResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"EquivalenceResult({self._observed!r}, margin={self._margin!r}, "
            f"alpha={self._alpha!r}, lower={self._lower!r}, "
            f"upper={self._upper!r})"
        )


class Equivalence:
    """Tests a paired difference against an equivalence margin.

    Parameters
    ----------
    margin : float
        How large a difference still counts as negligible, in the statistic's own units.
    replicates : int
        How many reassignments each one-sided test draws.
    seed : int
        The seed a draw runs under.
    level : float
        The level each one-sided test is read at.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the margin is not positive, if the replicate budget is zero, if the seed is
        negative, or if the level does not lie strictly between zero and one half.
    """

    __slots__ = ("_level", "_margin", "_replicates", "_seed")

    def __init__(
        self,
        *,
        margin: float,
        replicates: int,
        seed: int,
        level: float,
    ) -> None:
        self._margin = ensure_finite("an equivalence margin", margin)
        self._replicates = ensure_non_negative("a replicate budget", replicates)
        self._seed = ensure_non_negative("a permutation seed", seed)
        self._level = ensure_finite("an equivalence level", level)

        if self._margin <= 0.0:
            msg = f"an equivalence margin is not positive: {margin!r}"
            raise ValueError(msg)

        if self._replicates == 0:
            msg = "an equivalence test was given no replicate budget"
            raise ValueError(msg)

        if not 0.0 < self._level < _HIGHEST_LEVEL:
            msg = f"an equivalence level is not below {_HIGHEST_LEVEL}: {level!r}"
            raise ValueError(msg)

    @property
    def margin(self) -> float:
        """Return how large a difference still counts as negligible."""
        return self._margin

    @property
    def replicates(self) -> int:
        """Return how many reassignments each one-sided test draws."""
        return self._replicates

    @property
    def seed(self) -> int:
        """Return the sampling seed."""
        return self._seed

    @property
    def level(self) -> float:
        """Return the test level."""
        return self._level

    def _at(
        self,
        shifted: Sequence[float],
        right: Sequence[float],
        boundary: float,
        *,
        groups: Sequence[str] | None,
        near: Tail,
        far: Tail,
    ) -> MarginTest:
        """Return both tails of the reference distribution at one boundary.

        Parameters
        ----------
        shifted : Sequence[float]
            The left condition's scores moved so that the boundary falls at a mean
            difference of zero.
        right : Sequence[float]
            The right condition's scores, unchanged.
        boundary : float
            The difference the test is taken at.
        groups : Sequence[str] | None
            The group each item belongs to, in item order, or None where every
            item receives its own sign.
        near : Tail
            The tail whose rejection places the difference on the boundary's near
            side.
        far : Tail
            The tail whose rejection places it beyond the boundary.

        Returns
        -------
        MarginTest
            The two readings of this boundary.
        """
        return MarginTest(
            boundary,
            inside=paired_permutation(
                shifted,
                right,
                replicates=self._replicates,
                seed=self._seed,
                tail=near,
                groups=groups,
            ),
            outside=paired_permutation(
                shifted,
                right,
                replicates=self._replicates,
                seed=self._seed,
                tail=far,
                groups=groups,
            ),
        )

    def compare(
        self,
        left: Sequence[float],
        right: Sequence[float],
        /,
        *,
        groups: Sequence[str] | None = None,
    ) -> EquivalenceResult:
        """Test the paired difference for equivalence.

        Parameters
        ----------
        left : Sequence[float]
            One condition's score for each item, in item order.
        right : Sequence[float]
            The other condition's score for the same items, in the same order.
        groups : Sequence[str] | None
            The group each item belongs to, in item order, or None where every
            item receives its own sign.

        Returns
        -------
        EquivalenceResult
            The observed mean difference, margin, level, boundary tests, and
            verdict.

        Raises
        ------
        TypeError
            If a score or group label is of the wrong type.
        ValueError
            If a score is not finite, the sequences differ in length, there are no
            items, or the grouping is invalid.
        """
        one, other = _paired(left, right)
        observed = fsum(
            first - second for first, second in zip(one, other, strict=True)
        ) / len(one)

        return EquivalenceResult(
            observed,
            margin=self._margin,
            alpha=self._level,
            lower=self._at(
                tuple(score + self._margin for score in one),
                other,
                -self._margin,
                groups=groups,
                near="greater",
                far="less",
            ),
            upper=self._at(
                tuple(score - self._margin for score in one),
                other,
                self._margin,
                groups=groups,
                near="less",
                far="greater",
            ),
        )
