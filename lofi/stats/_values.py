"""Statistical results and provenance values."""

from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue

__all__ = [
    "NULL_FAMILIES",
    "TAILS",
    "BootstrapResult",
    "Decisions",
    "Dependence",
    "Interval",
    "NullFamily",
    "NullProvenance",
    "PValue",
    "Tail",
    "p_value_against_replicates",
]


type Tail = Literal["greater", "less", "two-sided"]
"""Specifies which direction of the reference distribution counts as extreme."""

TAILS: Final[tuple[Tail, ...]] = ("greater", "less", "two-sided")
"""Declared tails in report order."""


type NullFamily = Literal["permutation", "bootstrap-maxima", "matched"]
"""Defines how a null generates reference statistics."""

NULL_FAMILIES: Final[tuple[NullFamily, ...]] = (
    "permutation",
    "bootstrap-maxima",
    "matched",
)
"""Declared null families in report order."""


class Dependence(Enum):
    """The dependence regime for a family of tests."""

    #: Tests are statistically independent.
    INDEPENDENT = "independent"

    #: Tests are positively regression dependent on the true null subset.
    PRDS = "prds"

    #: Tests may have arbitrary dependence.
    ARBITRARY = "arbitrary"


def _criteria(given: Mapping[str, str | int] | None) -> Mapping[str, str | int]:
    """Return the matching criteria as a read-only mapping of names to levels.

    Parameters
    ----------
    given : Mapping[str, str | int] | None
        What the null matched its reference set on, or None where it matched on nothing.

    Returns
    -------
    Mapping[str, str | int]
        A copy, behind a read-only view.

    Raises
    ------
    TypeError
        If a criterion is neither a name nor a count.
    ValueError
        If a criterion's name or its level is empty, or a count is negative.
    """
    checked: dict[str, str | int] = {}

    for key, level in (given or {}).items():
        field = ensure_name("a matching criterion", key)

        if isinstance(level, bool) or not isinstance(level, str | int):
            msg = f"the level of criterion {key!r} is neither a name nor a count"
            raise TypeError(msg)

        checked[field] = (
            ensure_name(f"the level of criterion {key!r}", level)
            if isinstance(level, str)
            else ensure_non_negative(f"the level of criterion {key!r}", level)
        )

    return MappingProxyType(checked)


class NullProvenance:
    """Stores p-value provenance.

    Parameters
    ----------
    null : str
        The name of the null model that ran.
    family : NullFamily
        How that null drew its reference statistics.
    replicates : int
        How many reference statistics the p-value was computed against.
    tail : Tail
        Which direction counted as extreme.
    exact : bool
        Whether the reference distribution was enumerated rather than sampled.
    seed : int | None
        The seed the draw ran under, absent when nothing was drawn.
    matching : Mapping[str, str | int] | None
        What the reference set was matched to the observation on, empty where it was
        matched on nothing.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the name is empty, if the family or the tail is not one this package
        declares, if the replicate count is zero, if a sampled null has no seed, or if
        an enumerated one has a seed.
    """

    __slots__ = (
        "_exact",
        "_family",
        "_matching",
        "_null",
        "_replicates",
        "_seed",
        "_tail",
    )

    def __init__(
        self,
        null: str,
        family: NullFamily,
        replicates: int,
        *,
        tail: Tail,
        exact: bool,
        seed: int | None,
        matching: Mapping[str, str | int] | None = None,
    ) -> None:
        if not isinstance(exact, bool):
            msg = f"a null's exactness is not a boolean: {exact!r}"
            raise TypeError(msg)

        if family not in NULL_FAMILIES:
            declared = ", ".join(NULL_FAMILIES)
            msg = f"a null's family is {family!r}, not one of {declared}"
            raise ValueError(msg)

        if tail not in TAILS:
            declared = ", ".join(TAILS)
            msg = f"a null's tail is {tail!r}, not one of {declared}"
            raise ValueError(msg)

        self._null = ensure_name("a null's name", null)
        self._family: NullFamily = family
        self._replicates = ensure_non_negative("a null's replicate count", replicates)
        self._tail: Tail = tail
        self._exact = exact
        self._seed = (
            None if seed is None else ensure_non_negative("a null's seed", seed)
        )
        self._matching = _criteria(matching)

        if self._replicates == 0:
            msg = "a null produced no reference statistics at all"
            raise ValueError(msg)

        if self._exact and self._seed is not None:
            msg = f"an enumerated null reports a seed: {seed!r}"
            raise ValueError(msg)

        if not self._exact and self._seed is None:
            msg = f"the sampled null {null!r} reports no seed"
            raise ValueError(msg)

    @property
    def null(self) -> str:
        """Return the name of the null model that ran."""
        return self._null

    @property
    def family(self) -> NullFamily:
        """Return how that null drew its reference statistics."""
        return self._family

    @property
    def replicates(self) -> int:
        """Return how many reference statistics the p-value was computed against."""
        return self._replicates

    @property
    def tail(self) -> Tail:
        """Return which direction counted as extreme."""
        return self._tail

    @property
    def exact(self) -> bool:
        """Return whether the reference distribution was enumerated."""
        return self._exact

    @property
    def seed(self) -> int | None:
        """Return the seed the draw ran under, if a draw ran."""
        return self._seed

    @property
    def matching(self) -> Mapping[str, str | int]:
        """Return matching criteria as a read-only view."""
        return self._matching

    def to_json(self) -> dict[str, JsonValue]:
        """Return the provenance as a JSON object naming every field."""
        return {
            "name": self._null,
            "family": self._family,
            "replicates": self._replicates,
            "tail": self._tail,
            "exact": self._exact,
            "seed": self._seed,
            "matching": dict(self._matching),
        }

    def _identity(
        self,
    ) -> tuple[
        str, NullFamily, int, Tail, bool, int | None, tuple[tuple[str, str | int], ...]
    ]:
        """Return the seven fields a provenance is equal on, with criteria ordered."""
        return (
            self._null,
            self._family,
            self._replicates,
            self._tail,
            self._exact,
            self._seed,
            tuple(sorted(self._matching.items())),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same seven fields."""
        if type(other) is not NullProvenance:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the seven fields the provenance is equal on."""
        return hash((NullProvenance, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this provenance."""
        return (
            f"NullProvenance({self._null!r}, {self._family!r}, "
            f"{self._replicates!r}, tail={self._tail!r}, exact={self._exact!r}, "
            f"seed={self._seed!r}, matching={dict(self._matching)!r})"
        )


class _NullTest:
    """The capability produced by a completed null test."""

    __slots__ = ()


_A_NULL_TEST: Final = _NullTest()
"""The constructor token for a completed null test."""


class PValue:
    """The probability of a statistic this extreme under a stated null.

    Parameters
    ----------
    value : float
        The tail probability, which is strictly above zero, since a finite reference
        distribution cannot establish that nothing is more extreme.
    provenance : NullProvenance
        Which null produced it, and how.
    test : _NullTest
        The capability of the null test that produced it.

    Raises
    ------
    TypeError
        If the capability is not the one a null test produces, if the provenance is not
        one, or if the value is not a number.
    ValueError
        If the value is not finite, or does not lie above zero and at or below one.
    """

    __slots__ = ("_provenance", "_value")

    def __init__(
        self, value: float, provenance: NullProvenance, test: _NullTest
    ) -> None:
        if test is not _A_NULL_TEST:
            msg = (
                "a p-value is produced by testing an observed statistic against "
                "a null model, and has no other constructor"
            )
            raise TypeError(msg)

        if not isinstance(provenance, NullProvenance):
            msg = f"a p-value's provenance is not one: {provenance!r}"
            raise TypeError(msg)

        self._value = ensure_finite("a p-value", value)
        self._provenance = provenance

        if not 0.0 < self._value <= 1.0:
            msg = f"a p-value is outside the open unit interval: {value!r}"
            raise ValueError(msg)

    @property
    def value(self) -> float:
        """Return the tail probability."""
        return self._value

    @property
    def provenance(self) -> NullProvenance:
        """Return p-value provenance."""
        return self._provenance

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same probability under the same null."""
        if type(other) is not PValue:
            return NotImplemented

        return self._value == other._value and self._provenance == other._provenance

    @override
    def __hash__(self) -> int:
        """Return a hash over the probability and its provenance."""
        return hash((PValue, self._value, self._provenance))

    @override
    def __repr__(self) -> str:
        """Return the probability and the null it was tested under."""
        return f"PValue(value={self._value!r}, provenance={self._provenance!r})"


def _tail_probability(
    observed: float, replicates: Sequence[float], *, tail: Tail, exact: bool
) -> float:
    """Return the mass of the reference distribution at or beyond the observation.

    Parameters
    ----------
    observed : float
        The statistic actually seen.
    replicates : Sequence[float]
        The reference statistics, in any order.
    tail : Tail
        Which direction counts as extreme.
    exact : bool
        Whether the reference distribution was enumerated rather than sampled.

    Returns
    -------
    float
        The tail probability.
    """
    drawn = len(replicates)
    at_or_above = sum(1 for replicate in replicates if replicate >= observed)
    at_or_below = sum(1 for replicate in replicates if replicate <= observed)

    if exact:
        greater = at_or_above / drawn
        less = at_or_below / drawn

    else:
        greater = (1 + at_or_above) / (1 + drawn)
        less = (1 + at_or_below) / (1 + drawn)

    if tail == "greater":
        return greater

    if tail == "less":
        return less

    return min(1.0, 2.0 * min(greater, less))


def p_value_against_replicates(
    observed: float,
    replicates: Sequence[float],
    *,
    tail: Tail,
    null: str,
    family: NullFamily,
    exact: bool,
    seed: int | None,
    matching: Mapping[str, str | int] | None = None,
) -> PValue:
    """Return the p-value of an observed statistic against a null's replicates.

    Parameters
    ----------
    observed : float
        The statistic actually seen.
    replicates : Sequence[float]
        The statistic recomputed under the null, in any order.
    tail : Tail
        Which direction counts as extreme.
    null : str
        The name of the null model that drew them.
    family : NullFamily
        How that null drew them.
    exact : bool
        Whether the reference distribution enumerates the null's space rather than
        sampling from it.
    seed : int | None
        The seed the draw ran under, which an enumeration does not have.
    matching : Mapping[str, str | int] | None
        What the reference set was matched to the observation on, if anything.

    Returns
    -------
    PValue
        The tail probability, recording the provenance of this test.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If a statistic is nonfinite; no replicates are supplied; ``family`` or ``tail``
        is unknown; a sampled null lacks a seed; or an exact reference distribution
        omits ``observed``.
    """
    at = ensure_finite("an observed statistic", observed)
    reference = tuple(
        ensure_finite("a null replicate", replicate) for replicate in replicates
    )

    if not reference:
        msg = f"the null {null!r} drew no replicates to compare against"
        raise ValueError(msg)

    provenance = NullProvenance(
        null,
        family,
        len(reference),
        tail=tail,
        exact=exact,
        seed=seed,
        matching=matching,
    )
    probability = _tail_probability(at, reference, tail=tail, exact=exact)

    if probability <= 0.0:
        msg = (
            f"the enumerated null {null!r} does not contain the observed "
            f"statistic {observed!r}"
        )
        raise ValueError(msg)

    return PValue(probability, provenance, _A_NULL_TEST)


class Decisions:
    """Stores a multiple comparison decision.

    Parameters
    ----------
    correction : str
        The name of the procedure that ran.
    alpha : float
        The level the family was controlled at.
    dependence : Dependence
        The dependence regime the procedure was applied under.
    raw : Sequence[PValue]
        The p-values handed to the procedure, in the order they were handed in.
    adjusted : Sequence[float]
        The corrected p-values, one per raw p-value and in the same order.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the name is empty, if alpha does not lie strictly inside the unit interval,
        if the two sequences differ in length, or if an adjusted value is outside the
        unit interval.
    """

    __slots__ = ("_adjusted", "_alpha", "_correction", "_dependence", "_raw", "_reject")

    def __init__(
        self,
        correction: str,
        alpha: float,
        dependence: Dependence,
        raw: Sequence[PValue],
        adjusted: Sequence[float],
    ) -> None:
        if not isinstance(dependence, Dependence):
            msg = f"a decision's dependence regime is not one: {dependence!r}"
            raise TypeError(msg)

        self._raw = tuple(raw)

        for entry in self._raw:
            if not isinstance(entry, PValue):
                msg = (
                    f"a correction was handed {entry!r}, which is not a p-value. "
                    f"only a null test produces one"
                )
                raise TypeError(msg)

        self._correction = ensure_name("a correction's name", correction)
        self._alpha = ensure_finite("a correction's alpha", alpha)
        self._dependence = dependence
        self._adjusted = tuple(
            ensure_finite("an adjusted p-value", value) for value in adjusted
        )

        if not 0.0 < self._alpha < 1.0:
            msg = f"a correction's alpha is not a proportion: {alpha!r}"
            raise ValueError(msg)

        if len(self._raw) != len(self._adjusted):
            msg = (
                f"a correction returned {len(self._adjusted)} adjusted values "
                f"for {len(self._raw)} p-values"
            )
            raise ValueError(msg)

        for value in self._adjusted:
            if not 0.0 <= value <= 1.0:
                msg = f"an adjusted p-value is outside the unit interval: {value!r}"
                raise ValueError(msg)

        self._reject = tuple(value <= self._alpha for value in self._adjusted)

    @property
    def correction(self) -> str:
        """Return the name of the procedure that ran."""
        return self._correction

    @property
    def alpha(self) -> float:
        """Return the level the family was controlled at."""
        return self._alpha

    @property
    def dependence(self) -> Dependence:
        """Return the dependence regime the procedure was applied under."""
        return self._dependence

    @property
    def raw(self) -> tuple[PValue, ...]:
        """Return input p-values in original order."""
        return self._raw

    @property
    def adjusted(self) -> tuple[float, ...]:
        """Return the corrected p-values, in the same order."""
        return self._adjusted

    @property
    def reject(self) -> tuple[bool, ...]:
        """Return which hypotheses are rejected, in the same order."""
        return self._reject

    @property
    def n_rejected(self) -> int:
        """Return how many hypotheses are rejected."""
        return sum(self._reject)

    def _identity(
        self,
    ) -> tuple[str, float, Dependence, tuple[PValue, ...], tuple[float, ...]]:
        """Return the five fields a decision record is equal on."""
        return (
            self._correction,
            self._alpha,
            self._dependence,
            self._raw,
            self._adjusted,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same five fields."""
        if type(other) is not Decisions:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the record is equal on."""
        return hash((Decisions, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return (
            f"Decisions({self._correction!r}, {self._alpha!r}, "
            f"{self._dependence!r}, {self._raw!r}, {self._adjusted!r})"
        )


class Interval(FrozenValue):
    """A closed finite interval, optionally carrying nominal coverage.

    Parameters
    ----------
    low : float
        The lower bound.
    high : float
        The upper bound.
    level : float | None
        The nominal coverage, as a proportion, when this is a confidence
        interval. Domain intervals leave it absent.

    Raises
    ------
    TypeError
        If a bound or the level is not a number.
    ValueError
        If a bound is not finite, if the upper bound falls below the lower one, or if
        the level does not lie strictly inside the unit interval.
    """

    _noun = "interval"

    __slots__ = ("_high", "_level", "_low")

    def __init__(self, low: float, high: float, level: float | None = None) -> None:
        held_low = ensure_finite("an interval's lower bound", low)
        held_high = ensure_finite("an interval's upper bound", high)
        held_level = (
            None
            if level is None
            else ensure_finite("an interval's coverage level", level)
        )

        if held_high < held_low:
            msg = f"an interval ends below where it starts: [{low!r}, {high!r}]"
            raise ValueError(msg)

        if held_level is not None and not 0.0 < held_level < 1.0:
            msg = f"an interval's coverage level is not a proportion: {level!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_low", held_low)
        object.__setattr__(self, "_high", held_high)
        object.__setattr__(self, "_level", held_level)

    @property
    def low(self) -> float:
        """Return the lower bound."""
        return self._low

    @property
    def high(self) -> float:
        """Return the upper bound."""
        return self._high

    @property
    def level(self) -> float | None:
        """Return nominal coverage, or ``None`` for a domain interval."""
        return self._level

    @property
    def width(self) -> float:
        """Return the distance between the bounds."""
        return self._high - self._low

    def covers(self, value: float) -> bool:
        """Return whether a value falls between the bounds, which are inclusive.

        Parameters
        ----------
        value : float
            The number to test.

        Returns
        -------
        bool
            Whether the value lies within the interval.

        Raises
        ------
        TypeError
            If the value is not a number.
        ValueError
            If the value is not finite.
        """
        number = ensure_finite("a tested value", value)

        return self._low <= number <= self._high

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is an interval with the same three parts."""
        if type(other) is not Interval:
            return NotImplemented

        return (
            self._low == other._low
            and self._high == other._high
            and self._level == other._level
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the bounds and the level."""
        return hash((Interval, self._low, self._high, self._level))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this interval."""
        suffix = "" if self._level is None else f", {self._level!r}"
        return f"Interval({self._low!r}, {self._high!r}{suffix})"


def copy_validated_interval(low: float, high: float, level: float | None) -> Interval:
    """Copy interval fields already validated at a boundary."""
    interval = object.__new__(Interval)
    object.__setattr__(interval, "_low", low)
    object.__setattr__(interval, "_high", high)
    object.__setattr__(interval, "_level", level)
    return interval


class BootstrapResult:
    """Stores a bootstrap estimate and its provenance.

    Parameters
    ----------
    observed : float
        The statistic on the data as it came in.
    unit : str
        What one resampled unit is.
    units : int
        How many of them stood behind the estimate.
    replicates : int
        How many resamples the intervals were read off.
    seed : int
        The seed the resampling ran under.
    percentile : Interval
        The interval read straight off the ordered replicates.
    bca : Interval
        The bias-corrected and accelerated interval.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type, which is how a tensor, a module
        or an autograd node is refused.
    ValueError
        If the observed statistic is not finite, if the unit's name is empty, if no
        units or no replicates stood behind the estimate, if the seed is negative, or if
        the two intervals claim different coverage.
    """

    __slots__ = (
        "_bca",
        "_observed",
        "_percentile",
        "_replicates",
        "_seed",
        "_unit",
        "_units",
    )

    def __init__(
        self,
        observed: float,
        unit: str,
        *,
        units: int,
        replicates: int,
        seed: int,
        percentile: Interval,
        bca: Interval,
    ) -> None:
        for name, given in (("percentile", percentile), ("accelerated", bca)):
            if not isinstance(given, Interval) or given.level is None:
                msg = f"a bootstrap's {name} interval is not one: {given!r}"
                raise TypeError(msg)

        self._observed = ensure_finite("a bootstrapped statistic", observed)
        self._unit = ensure_name("a resampling unit", unit)
        self._units = ensure_non_negative("a resampled unit count", units)
        self._replicates = ensure_non_negative(
            "a bootstrap replicate count", replicates
        )
        self._seed = ensure_non_negative("a bootstrap seed", seed)
        self._percentile = percentile
        self._bca = bca

        if self._units == 0:
            msg = f"a bootstrap over no {unit!r} at all reports an estimate"
            raise ValueError(msg)

        if self._replicates == 0:
            msg = "a bootstrap drew no resamples at all"
            raise ValueError(msg)

        if self._percentile.level != self._bca.level:
            msg = (
                f"a bootstrap's two intervals claim different coverage: "
                f"{self._percentile.level!r} and {self._bca.level!r}"
            )
            raise ValueError(msg)

    @property
    def observed(self) -> float:
        """Return the observed statistic."""
        return self._observed

    @property
    def unit(self) -> str:
        """Return the resampling unit."""
        return self._unit

    @property
    def units(self) -> int:
        """Return the resampling-unit count."""
        return self._units

    @property
    def replicates(self) -> int:
        """Return the resample count."""
        return self._replicates

    @property
    def seed(self) -> int:
        """Return the seed the resampling ran under."""
        return self._seed

    @property
    def percentile(self) -> Interval:
        """Return the percentile interval."""
        return self._percentile

    @property
    def bca(self) -> Interval:
        """Return the bias-corrected and accelerated interval."""
        return self._bca

    @property
    def level(self) -> float:
        """Return the coverage both intervals claim."""
        level = self._percentile.level

        if level is None:
            msg = "a bootstrap result lost its validated coverage level"
            raise RuntimeError(msg)

        return level

    def _identity(
        self,
    ) -> tuple[float, str, int, int, int, Interval, Interval]:
        """Return the seven fields a bootstrap result is equal on."""
        return (
            self._observed,
            self._unit,
            self._units,
            self._replicates,
            self._seed,
            self._percentile,
            self._bca,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same seven fields."""
        if type(other) is not BootstrapResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the seven fields the result is equal on."""
        return hash((BootstrapResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"BootstrapResult({self._observed!r}, {self._unit!r}, "
            f"units={self._units!r}, replicates={self._replicates!r}, "
            f"seed={self._seed!r}, percentile={self._percentile!r}, "
            f"bca={self._bca!r})"
        )
