"""Reference distributions under declared null models."""

from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.stats._values import TAILS, PValue, p_value_against_replicates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi.stats._ports import Statistic
    from lofi.stats._values import NullFamily, Tail

__all__ = ["BootstrapMaxima", "LabelPermutation", "MatchedDraw"]

_RESAMPLED_MINIMUM: Final = 2
"""The minimum unit count for rearrangement or resampling."""

_MATCHED_MINIMUM: Final = 1
"""The minimum unit count for a matched draw."""


def _units[Element](
    null: str, observed: Sequence[Element], minimum: int
) -> tuple[Element, ...]:
    """Validate and return observed units.

    Parameters
    ----------
    null : str
        The name of the null being run, for the message.
    observed : Sequence[Element]
        The data as it came in.
    minimum : int
        How many units this null needs.

    Returns
    -------
    tuple[Element, ...]
        The units, in the order they came in.

    Raises
    ------
    ValueError
        If there are fewer units than the null needs.
    """
    units = tuple(observed)

    if len(units) < minimum:
        msg = (
            f"the null {null!r} needs at least {minimum} units and was given "
            f"{len(units)}"
        )
        raise ValueError(msg)

    return units


def _draw(null: str, replicates: int, seed: int) -> tuple[int, int]:
    """Validate and return a replicate budget and seed.

    Parameters
    ----------
    null : str
        The name of the null being constructed, for the message.
    replicates : int
        How many reference statistics it will draw.
    seed : int
        The seed the draw runs under.

    Returns
    -------
    tuple[int, int]
        The budget and the seed.

    Raises
    ------
    TypeError
        If the budget or the seed is not an integer.
    ValueError
        If the budget is zero, or the seed is negative.
    """
    budget = ensure_non_negative("a replicate budget", replicates)
    drawn_under = ensure_non_negative("a null's seed", seed)

    if budget == 0:
        msg = f"the null {null!r} was given no replicate budget"
        raise ValueError(msg)

    return budget, drawn_under


class LabelPermutation[Element]:
    """Permutes assignments among observed units.

    Parameters
    ----------
    replicates : int
        How many orderings to draw.
    seed : int
        The seed the draw runs under, so a published p-value can be recomputed.

    Raises
    ------
    TypeError
        If the budget or the seed is not an integer.
    ValueError
        If the budget is zero, or the seed is negative.
    """

    __slots__ = ("_replicates", "_seed")

    def __init__(self, *, replicates: int, seed: int) -> None:
        self._replicates, self._seed = _draw("label permutation", replicates, seed)

    @property
    def name(self) -> str:
        """Return the null's name, unique within the report it appears in."""
        return "label permutation"

    @property
    def family(self) -> NullFamily:
        """Return how this null draws its reference statistics."""
        return "permutation"

    @property
    def replicates(self) -> int | None:
        """Return how many orderings it draws."""
        return self._replicates

    @property
    def seed(self) -> int | None:
        """Return the sampling seed."""
        return self._seed

    @property
    def matching(self) -> Mapping[str, str | int]:
        """Return no matching criterion."""
        return MappingProxyType({})

    def reference(
        self, statistic: Statistic[Sequence[Element]], observed: Sequence[Element], /
    ) -> tuple[float, ...]:
        """Return the statistic under each drawn ordering, in the order drawn.

        Parameters
        ----------
        statistic : Statistic[Sequence[Element]]
            The summary recomputed over each ordering.
        observed : Sequence[Element]
            The data as it came in.

        Returns
        -------
        tuple[float, ...]
            One statistic per replicate, each finite.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a value that is not finite or there are fewer
            than two units to shuffle.
        """
        units = _units(self.name, observed, _RESAMPLED_MINIMUM)
        rng = Random(self._seed)
        size = len(units)

        return tuple(
            ensure_finite("a null replicate", statistic(rng.sample(units, size)))
            for _ in range(self._replicates)
        )

    def test(
        self,
        statistic: Statistic[Sequence[Element]],
        observed: Sequence[Element],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        """Return the p-value of the statistic on this data under this null.

        Parameters
        ----------
        statistic : Statistic[Sequence[Element]]
            The summary evaluated on the observed and drawn sample orderings.
        observed : Sequence[Element]
            The data as it actually came in.
        tail : Tail
            Which direction of the reference distribution counts as extreme.

        Returns
        -------
        PValue
            The tail probability with the null's name, family, replicate count,
            and seed.

        Raises
        ------
        TypeError
            If the statistic returns something that is not a number.
        ValueError
            If the statistic returns a nonfinite value, there are fewer than two
            units to shuffle, or the tail is undeclared.
        """
        units = _units(self.name, observed, _RESAMPLED_MINIMUM)

        return p_value_against_replicates(
            ensure_finite("an observed statistic", statistic(units)),
            self.reference(statistic, units),
            tail=tail,
            null=self.name,
            family=self.family,
            exact=False,
            seed=self._seed,
        )


class BootstrapMaxima[Element]:
    """A max-statistic bootstrap null.

    Parameters
    ----------
    statistics : Sequence[Statistic[Sequence[Element]]]
        The family the maximum is taken over, in any order.
    replicates : int
        How many resamples to draw.
    seed : int
        The seed the draw runs under, so a published p-value can be recomputed.

    Raises
    ------
    TypeError
        If a member of the family is not callable, or if the budget or the seed is not
        an integer.
    ValueError
        If the family is empty, if the budget is zero, or if the seed is negative.
    """

    __slots__ = ("_replicates", "_seed", "_statistics")

    def __init__(
        self,
        statistics: Sequence[Statistic[Sequence[Element]]],
        /,
        *,
        replicates: int,
        seed: int,
    ) -> None:
        self._statistics = tuple(statistics)
        self._replicates, self._seed = _draw("bootstrap maxima", replicates, seed)

        if not self._statistics:
            msg = "a bootstrap maxima null was given no family to take maxima over"
            raise ValueError(msg)

        for member in self._statistics:
            if not callable(member):
                msg = (
                    f"a member of a bootstrap maxima family is not callable: {member!r}"
                )
                raise TypeError(msg)

    @property
    def name(self) -> str:
        """Return the null's name, unique within the report it appears in."""
        return "bootstrap maxima"

    @property
    def family(self) -> NullFamily:
        """Return how this null draws its reference statistics."""
        return "bootstrap-maxima"

    @property
    def replicates(self) -> int | None:
        """Return how many resamples it draws."""
        return self._replicates

    @property
    def seed(self) -> int | None:
        """Return the sampling seed."""
        return self._seed

    @property
    def matching(self) -> Mapping[str, str | int]:
        """Return the size of the family the extreme is taken over."""
        return MappingProxyType({"family": len(self._statistics)})

    def _observed(self, units: tuple[Element, ...]) -> tuple[float, ...]:
        """Return observed statistics for the family.

        Parameters
        ----------
        units : tuple[Element, ...]
            The data as it came in.

        Returns
        -------
        tuple[float, ...]
            One finite value per member, in the family's own order.
        """
        return tuple(
            ensure_finite("an observed statistic", member(units))
            for member in self._statistics
        )

    def _deviations(self, units: tuple[Element, ...]) -> Iterator[tuple[float, ...]]:
        """Yield centered family statistics for each draw.

        Parameters
        ----------
        units : tuple[Element, ...]
            The data as it came in.

        Yields
        ------
        tuple[float, ...]
            One departure per member of the family, in the family's own order, for one
            resample of the units drawn with replacement.
        """
        at = self._observed(units)
        rng = Random(self._seed)
        size = len(units)

        for _ in range(self._replicates):
            resample = rng.choices(units, k=size)

            yield tuple(
                ensure_finite("a null replicate", member(resample)) - value
                for member, value in zip(self._statistics, at, strict=True)
            )

    def reference(
        self, observed: Sequence[Element], /, *, tail: Tail
    ) -> tuple[float, ...]:
        """Return the family's extreme departure under each resample.

        Parameters
        ----------
        observed : Sequence[Element]
            The data as it came in.
        tail : Tail
            The direction that determines which family extreme is kept.

        Returns
        -------
        tuple[float, ...]
            One value per resample for a one-sided tail, and two of opposite sign per
            resample for the two-sided one.

        Raises
        ------
        TypeError
            If a member of the family returns something that is not a number.
        ValueError
            If a member returns a value that is not finite, if there are fewer than two
            units to resample, or if the tail is not a declared one.
        """
        if tail not in TAILS:
            declared = ", ".join(TAILS)
            msg = (
                f"the null {self.name!r} was asked for the {tail!r} tail, not "
                f"one of {declared}"
            )
            raise ValueError(msg)

        units = _units(self.name, observed, _RESAMPLED_MINIMUM)

        if tail == "greater":
            return tuple(max(drawn) for drawn in self._deviations(units))

        if tail == "less":
            return tuple(min(drawn) for drawn in self._deviations(units))

        symmetric: list[float] = []

        for drawn in self._deviations(units):
            extreme = max(abs(departure) for departure in drawn)
            symmetric.extend((extreme, -extreme))

        return tuple(symmetric)

    def test(
        self,
        statistic: Statistic[Sequence[Element]],
        observed: Sequence[Element],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        """Return the family-wise p-value of one member of the family.

        Parameters
        ----------
        statistic : Statistic[Sequence[Element]]
            The member being tested.
        observed : Sequence[Element]
            The data as it actually came in.
        tail : Tail
            Which direction of the reference distribution counts as extreme.

        Returns
        -------
        PValue
            The tail probability with the null's name, family, seed, and family
            size.

        Raises
        ------
        TypeError
            If a member of the family returns something that is not a number.
        ValueError
            If the statistic is outside the family, a member returns a nonfinite
            value, there are fewer than two units, or the tail is undeclared.
        """
        if statistic not in self._statistics:
            msg = (
                f"the statistic tested against {self.name!r} is not one of the "
                f"{len(self._statistics)} the maxima are taken over"
            )
            raise ValueError(msg)

        units = _units(self.name, observed, _RESAMPLED_MINIMUM)

        return p_value_against_replicates(
            ensure_finite("an observed statistic", statistic(units)),
            self.reference(units, tail=tail),
            tail=tail,
            null=self.name,
            family=self.family,
            exact=False,
            seed=self._seed,
            matching=self.matching,
        )


class MatchedDraw[Element]:
    """A matched-sampling null.

    Parameters
    ----------
    criterion : str
        What the surrogates are matched on, as a name a report can print.
    pool : Sequence[Element]
        The units surrogates are drawn from.
    stratum : Callable[[Element], str]
        The stratum of a unit, as a name.
    replicates : int
        How many matched draws to take.
    seed : int
        The seed the draw runs under, so a published p-value can be recomputed.

    Raises
    ------
    TypeError
        If the criterion or a stratum is not a string, or if the budget or the seed is
        not an integer.
    ValueError
        If the criterion or a stratum is empty, if the pool is empty, if the budget is
        zero, or if the seed is negative.
    """

    __slots__ = (
        "_criterion",
        "_matching",
        "_replicates",
        "_seed",
        "_strata",
        "_stratum",
    )

    def __init__(
        self,
        criterion: str,
        pool: Sequence[Element],
        stratum: Callable[[Element], str],
        /,
        *,
        replicates: int,
        seed: int,
    ) -> None:
        self._criterion = ensure_name("a matching criterion", criterion)
        self._stratum = stratum
        self._replicates, self._seed = _draw(self.name, replicates, seed)
        gathered: dict[str, list[Element]] = {}

        for member in pool:
            gathered.setdefault(ensure_name("a stratum", stratum(member)), []).append(
                member
            )

        if not gathered:
            msg = f"the null {self.name!r} was given an empty pool to draw from"
            raise ValueError(msg)

        self._strata = {key: tuple(members) for key, members in gathered.items()}
        self._matching: Mapping[str, str | int] = MappingProxyType(
            {
                "criterion": self._criterion,
                "strata": len(self._strata),
                "pool": sum(len(members) for members in self._strata.values()),
            }
        )

    @property
    def name(self) -> str:
        """Return the null's name, unique within the report it appears in."""
        return f"matched on {self._criterion}"

    @property
    def family(self) -> NullFamily:
        """Return how this null draws its reference statistics."""
        return "matched"

    @property
    def replicates(self) -> int | None:
        """Return how many matched draws it takes."""
        return self._replicates

    @property
    def seed(self) -> int | None:
        """Return the sampling seed."""
        return self._seed

    @property
    def matching(self) -> Mapping[str, str | int]:
        """Return the matching criterion and source-pool summary."""
        return self._matching

    def _surrogate(self, keys: Sequence[str], rng: Random) -> list[Element]:
        """Draw one matched unit per observed stratum.

        Parameters
        ----------
        keys : Sequence[str]
            The stratum of each observed unit, in the units' order.
        rng : Random
            The generator the draw runs on.

        Returns
        -------
        list[Element]
            One drawn unit per observed unit, in the same order.
        """
        return [rng.choice(self._strata[key]) for key in keys]

    def _keys(self, units: tuple[Element, ...]) -> tuple[str, ...]:
        """Validate and return observed-unit strata.

        Parameters
        ----------
        units : tuple[Element, ...]
            The data as it came in.

        Returns
        -------
        tuple[str, ...]
            One stratum name per unit, in the units' order.

        Raises
        ------
        TypeError
            If a stratum is not a string.
        ValueError
            If a stratum is empty, or if the pool holds nothing in it.
        """
        keys = tuple(ensure_name("a stratum", self._stratum(unit)) for unit in units)
        missing = sorted({key for key in keys if key not in self._strata})

        if missing:
            listed = ", ".join(repr(key) for key in missing)
            msg = f"the null {self.name!r} draws from a pool holding no {listed}"
            raise ValueError(msg)

        return keys

    def reference(
        self, statistic: Statistic[Sequence[Element]], observed: Sequence[Element], /
    ) -> tuple[float, ...]:
        """Return the statistic over each matched draw, in the order drawn.

        Parameters
        ----------
        statistic : Statistic[Sequence[Element]]
            The summary recomputed over each matched draw.
        observed : Sequence[Element]
            The data as it came in.

        Returns
        -------
        tuple[float, ...]
            One statistic per replicate, each finite.

        Raises
        ------
        TypeError
            If a stratum is not a string or the statistic returns a nonnumber.
        ValueError
            If a stratum is empty, if the pool holds nothing in one of them, if the
            statistic returns a nonfinite value, or if there are no units.
        """
        units = _units(self.name, observed, _MATCHED_MINIMUM)
        keys = self._keys(units)
        rng = Random(self._seed)

        return tuple(
            ensure_finite("a null replicate", statistic(self._surrogate(keys, rng)))
            for _ in range(self._replicates)
        )

    def test(
        self,
        statistic: Statistic[Sequence[Element]],
        observed: Sequence[Element],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        """Return the p-value of the statistic on this data under this null.

        Parameters
        ----------
        statistic : Statistic[Sequence[Element]]
            The summary being tested, evaluated on the data as it came in and once per
            matched draw.
        observed : Sequence[Element]
            The data as it actually came in.
        tail : Tail
            Which direction of the reference distribution counts as extreme.

        Returns
        -------
        PValue
            The tail probability with the null's name, family, replicate count,
            seed, and matching criterion.

        Raises
        ------
        TypeError
            If a stratum is not a string or the statistic returns a nonnumber.
        ValueError
            If a stratum is empty, if the pool holds nothing in one of them, if the
            statistic returns a nonfinite value, if there are no units, or if the
            tail is undeclared.
        """
        units = _units(self.name, observed, _MATCHED_MINIMUM)

        return p_value_against_replicates(
            ensure_finite("an observed statistic", statistic(units)),
            self.reference(statistic, units),
            tail=tail,
            null=self.name,
            family=self.family,
            exact=False,
            seed=self._seed,
            matching=self._matching,
        )
