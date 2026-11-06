"""P-value corrections for multiple comparisons."""

from math import fsum
from typing import TYPE_CHECKING, Final

from lofi.stats._errors import DependenceError
from lofi.stats._values import Decisions, Dependence, PValue

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BenjaminiHochberg",
    "BenjaminiYekutieli",
    "Bonferroni",
    "Holm",
    "false_discovery_correction",
]

_EVERY_REGIME: Final = frozenset(Dependence)
"""All dependence regimes."""

_WEAK_DEPENDENCE: Final = frozenset({Dependence.INDEPENDENT, Dependence.PRDS})
"""Dependence regimes supported by the step-up procedure."""


def _family(
    correction: str,
    pvalues: Sequence[PValue],
    dependence: Dependence,
    valid_under: frozenset[Dependence],
) -> tuple[float, ...]:
    """Validate and return a family of p-values.

    Parameters
    ----------
    correction : str
        The name of the procedure that was asked, for the message.
    pvalues : Sequence[PValue]
        The family, in the order the caller handed it in.
    dependence : Dependence
        The regime the tests were run under.
    valid_under : frozenset[Dependence]
        The regimes the procedure's guarantee holds under.

    Returns
    -------
    tuple[float, ...]
        One tail probability per member, in the order they came in.

    Raises
    ------
    DependenceError
        If the regime is not one the procedure is valid under.
    TypeError
        If the regime is not one of the declared ones, or if a member of the family is
        not a p-value.
    ValueError
        If the family is empty.
    """
    if not isinstance(dependence, Dependence):
        msg = f"a correction's dependence regime is not one: {dependence!r}"
        raise TypeError(msg)

    if dependence not in valid_under:
        raise DependenceError(
            correction=correction, dependence=dependence, valid_under=valid_under
        )

    if not pvalues:
        msg = f"{correction} was handed no p-values to correct"
        raise ValueError(msg)

    for entry in pvalues:
        if not isinstance(entry, PValue):
            msg = (
                f"{correction} was handed {entry!r}, which is not a p-value. "
                f"only a null test produces one"
            )
            raise TypeError(msg)

    return tuple(entry.value for entry in pvalues)


def _single_step(raw: tuple[float, ...]) -> tuple[float, ...]:
    """Apply the Bonferroni adjustment.

    Parameters
    ----------
    raw : tuple[float, ...]
        The tail probabilities, in the order they came in.

    Returns
    -------
    tuple[float, ...]
        One adjusted probability per member, in the same order.
    """
    return tuple(min(1.0, len(raw) * probability) for probability in raw)


def _step_down(raw: tuple[float, ...]) -> tuple[float, ...]:
    """Apply the Holm adjustment.

    Parameters
    ----------
    raw : tuple[float, ...]
        The tail probabilities, in the order they came in.

    Returns
    -------
    tuple[float, ...]
        One adjusted probability per member, in the same order.
    """
    size = len(raw)
    adjusted = [0.0] * size
    running = 0.0

    for rank, index in enumerate(sorted(range(size), key=lambda at: raw[at])):
        running = max(running, min(1.0, (size - rank) * raw[index]))
        adjusted[index] = running

    return tuple(adjusted)


def _step_up(raw: tuple[float, ...], penalty: float) -> tuple[float, ...]:
    """Apply the Benjamini-Hochberg adjustment.

    Parameters
    ----------
    raw : tuple[float, ...]
        The tail probabilities, in the order they came in.
    penalty : float
        The factor the whole family is inflated by before ranking.

    Returns
    -------
    tuple[float, ...]
        One adjusted probability per member, in the same order.
    """
    size = len(raw)
    adjusted = [0.0] * size
    running = 1.0
    descending = sorted(range(size), key=lambda at: raw[at], reverse=True)

    for position, index in enumerate(descending):
        rank = size - position
        running = min(running, 1.0, penalty * size * raw[index] / rank)
        adjusted[index] = running

    return tuple(adjusted)


def _harmonic(size: int) -> float:
    """Return the sum of the reciprocals of the first counting numbers.

    Parameters
    ----------
    size : int
        How many terms to sum, which is the size of the family.

    Returns
    -------
    float
        The harmonic sum, which is one for a family of one.
    """
    return fsum(1.0 / term for term in range(1, size + 1))


class Bonferroni:
    """The single-step family-wise error-rate correction."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the procedure's name, unique within the report it appears in."""
        return "bonferroni"

    @property
    def valid_under(self) -> frozenset[Dependence]:
        """Return the dependence regimes this procedure's guarantee holds under."""
        return _EVERY_REGIME

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
            rejection mask those and alpha determine.

        Raises
        ------
        DependenceError
            When the regime is not one this procedure is valid under.
        """
        raw = _family(self.name, pvalues, dependence, self.valid_under)

        return Decisions(self.name, alpha, dependence, pvalues, _single_step(raw))


class Holm:
    """The step-down family-wise error-rate correction."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the procedure's name, unique within the report it appears in."""
        return "holm"

    @property
    def valid_under(self) -> frozenset[Dependence]:
        """Return the dependence regimes this procedure's guarantee holds under."""
        return _EVERY_REGIME

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
            rejection mask those and alpha determine.

        Raises
        ------
        DependenceError
            When the regime is not one this procedure is valid under.
        """
        raw = _family(self.name, pvalues, dependence, self.valid_under)

        return Decisions(self.name, alpha, dependence, pvalues, _step_down(raw))


class BenjaminiHochberg:
    """The step-up false discovery rate correction."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the procedure's name, unique within the report it appears in."""
        return "benjamini-hochberg"

    @property
    def valid_under(self) -> frozenset[Dependence]:
        """Return the dependence regimes this procedure's guarantee holds under."""
        return _WEAK_DEPENDENCE

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
            rejection mask those and alpha determine.

        Raises
        ------
        DependenceError
            When the regime is arbitrary dependence.
        """
        raw = _family(self.name, pvalues, dependence, self.valid_under)

        return Decisions(self.name, alpha, dependence, pvalues, _step_up(raw, 1.0))


class BenjaminiYekutieli:
    """Controls false discovery under arbitrary dependence."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the procedure's name, unique within the report it appears in."""
        return "benjamini-yekutieli"

    @property
    def valid_under(self) -> frozenset[Dependence]:
        """Return the dependence regimes this procedure's guarantee holds under."""
        return _EVERY_REGIME

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
            rejection mask those and alpha determine.

        Raises
        ------
        DependenceError
            When the regime is not one this procedure is valid under.
        """
        raw = _family(self.name, pvalues, dependence, self.valid_under)
        penalty = _harmonic(len(raw))

        return Decisions(self.name, alpha, dependence, pvalues, _step_up(raw, penalty))


def false_discovery_correction(
    dependence: Dependence,
) -> BenjaminiHochberg | BenjaminiYekutieli:
    """Return the false-discovery procedure whose guarantee covers a regime.

    Parameters
    ----------
    dependence : Dependence
        The regime the tests were run under.

    Returns
    -------
    BenjaminiHochberg | BenjaminiYekutieli
        A procedure valid under that regime.

    Raises
    ------
    TypeError
        If the regime is not one of the declared ones.
    """
    if not isinstance(dependence, Dependence):
        msg = f"a correction's dependence regime is not one: {dependence!r}"
        raise TypeError(msg)

    if dependence in _WEAK_DEPENDENCE:
        return BenjaminiHochberg()

    return BenjaminiYekutieli()
