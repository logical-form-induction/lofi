"""Statistical protocols."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.stats._values import (
        Decisions,
        Dependence,
        NullFamily,
        PValue,
        Tail,
    )

__all__ = ["Correction", "NullModel", "Statistic"]


type Statistic[Data] = Callable[[Data], float]
"""Defines a summary of one arrangement of the data, as a number."""


@runtime_checkable
class NullModel[Data](Protocol):
    """A null model for a statistic.

    Attributes
    ----------
    name : str
        The null's name, unique within the report it appears in.
    family : NullFamily
        How this null draws its reference statistics.
    replicates : int | None
        How many reference statistics it draws, absent when it enumerates.
    seed : int | None
        The seed the draw runs under, absent when the null enumerates.
    matching : Mapping[str, str | int]
        What the reference set is matched to the observation on, empty for a null that
        matches on nothing.
    """

    @property
    def name(self) -> str:
        """Return the null's name, unique within the report it appears in."""

    @property
    def family(self) -> NullFamily:
        """Return how this null draws its reference statistics."""

    @property
    def replicates(self) -> int | None:
        """Return how many reference statistics it draws, absent when it enumerates."""

    @property
    def seed(self) -> int | None:
        """Return the sampling seed, or ``None`` for enumeration."""

    @property
    def matching(self) -> Mapping[str, str | int]:
        """Return the matching criterion."""

    def test(
        self, statistic: Statistic[Data], observed: Data, /, *, tail: Tail
    ) -> PValue:
        """Return the p-value of the statistic on this data under this null.

        Parameters
        ----------
        statistic : Statistic[Data]
            The summary being tested.
        observed : Data
            The data as it actually came in.
        tail : Tail
            Which direction of the reference distribution counts as extreme.

        Returns
        -------
        PValue
            The tail probability with the null's name, family, reference size,
            sampling mode, seed, and matching criteria.
        """


@runtime_checkable
class Correction(Protocol):
    """An error-rate correction for a family of tests.

    Attributes
    ----------
    name : str
        The procedure's name, unique within the report it appears in.
    valid_under : frozenset[Dependence]
        The dependence regimes this procedure's guarantee holds under.
    """

    @property
    def name(self) -> str:
        """Return the procedure's name, unique within the report it appears in."""

    @property
    def valid_under(self) -> frozenset[Dependence]:
        """Return the dependence regimes this procedure's guarantee holds under."""

    def apply(
        self, pvalues: Sequence[PValue], alpha: float, dependence: Dependence, /
    ) -> Decisions:
        """Correct the p-value family at the given level.

        Parameters
        ----------
        pvalues : Sequence[PValue]
            The family, in the order the caller will read the results back in.
        alpha : float
            The level the family is controlled at.
        dependence : Dependence
            The regime the tests were run under.

        Returns
        -------
        Decisions
            The raw p-values unchanged, the adjusted ones in the same order, and the
            rejection mask the two of those and alpha determine.

        Raises
        ------
        DependenceError
            When the regime is not one this procedure is valid under.
        """
